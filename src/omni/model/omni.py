"""OmniModel: one decoder-only transformer over 1 text + n_codebooks audio streams.

The model always consumes DELAYED grids (see `omni.streams.apply_delay`). The
input at each step is the sum of per-stream token embeddings plus a channel
(turn owner) embedding; the output is one text head and n_codebooks audio heads
read from the same final hidden state. Heads run in model dtype and logits are
cast to float32.

v2 extensions (all OFF by default; defaults reproduce v1 exactly):
- `cfg.use_depth` (with `audio_delay_mode="flat"`): the parallel audio heads are
  replaced by a `DepthTransformer` that predicts one frame's codebooks
  sequentially from the backbone hidden state. `forward` stays teacher-forced
  and returns the same `ModelOutput` shapes; decoding goes through
  `prefill_hidden`/`step_hidden` + `depth.sample`.
- `cfg.duplex`: grids carry a second, input-only user audio group
  (rows n_q+1..2n_q) embedded via `user_audio_embs`; only the assistant group
  (rows 1..n_q) is ever a loss target.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from ..streams import audio_pad_id
from .layers import Block, DepthTransformer, KVCache, RMSNorm, precompute_rope


@dataclass
class ModelOutput:
    """Full-sequence logits, always float32.

    ``audio_positions`` (training-only, depth amortization): when set, the
    audio logits cover ONLY those predictor positions ``[P]`` (audio_logits is
    ``[B, n_q, P, Va]`` and position ``p`` predicts grid column ``p + 1``);
    None means all T positions (audio_logits ``[B, n_q, T, Va]``)."""

    text_logits: torch.Tensor  # float32 [B, T, text_vocab_size]
    audio_logits: torch.Tensor  # float32 [B, n_q, T|P, audio_vocab_size]
    audio_positions: torch.Tensor | None = None  # long [P] or None


def _depth_positions(
    T: int, ratio: float, training: bool, device: torch.device
) -> torch.Tensor | None:
    """CSM-style depth-loss amortization: the random predictor-position subset
    to run the depth transformer on (sorted long [P]), or None for all
    positions (eval mode, ratio 1.0, or degenerate T)."""
    if not training or ratio >= 1.0 or T <= 1:
        return None
    P = max(1, int(round(ratio * (T - 1))))
    if P >= T - 1:
        return None
    return torch.randperm(T - 1, device=device)[:P].sort().values


def multistream_loss(
    out: ModelOutput,
    grid: torch.Tensor,
    loss_mask: torch.Tensor,
    n_codebooks: int,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Next-token cross-entropy over a DELAYED grid; shared by every model class.

    Logits at position p predict grid position p+1: position slice ``:-1``
    scores targets grid[:, :, 1:], gated by loss_mask[:, :, 1:] (with
    ``out.audio_positions`` set, the audio logits already sit at that position
    subset). grid long [B, S, T]; loss_mask bool [B, S, T]; weights =
    (text_w, audio_w, semantic_extra_w_on_cb0), so head k weighs
    audio_w * (semantic_w if k == 0 else 1).

    Audio targets/masks are ALWAYS the assistant group — audio head k scores
    stream row 1+k. Duplex user rows (n_q+1..2n_q) are input-only and never
    contribute, whatever their mask says.

    Per-head mean CE over that head's masked targets; the total divides by
    the CONSTANT configured weight sum ``text_w + audio_w * (sem_w + n_q - 1)``
    so per-head gradient scale does not swing with the batch's task
    composition (heads without targets contribute exactly 0). Implemented
    with ignore_index CE — no per-head host syncs in the graph; ONE host
    transfer collects all head counts for the metrics dict. Returns
    (total, metrics): {"loss", "loss/text", "loss/audio_k", ...} detached;
    per-head keys only appear when that head had targets.
    """
    text_w, audio_w, sem_w = weights
    sel = out.audio_positions

    def head_sums(logits: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(sum CE over masked targets, target count) for stream row s.

        logits [B, T*, V] fp32 — T* = T for text, T or P for audio; audio
        subset positions p score column p+1 via `sel + 1`.
        """
        if s > 0 and sel is not None:
            m = loss_mask[:, s].index_select(1, sel + 1)
            tgt = grid[:, s].index_select(1, sel + 1)
            lg = logits
        else:
            m = loss_mask[:, s, 1:]
            tgt = grid[:, s, 1:]
            lg = logits[:, :-1]
        tgt = tgt.masked_fill(~m, -100)
        ce_sum = F.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        return ce_sum, m.sum()

    names = ["loss/text"] + [f"loss/audio_{k}" for k in range(n_codebooks)]
    head_w = [text_w] + [audio_w * (sem_w if k == 0 else 1.0) for k in range(n_codebooks)]
    sums: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    ce, n = head_sums(out.text_logits, 0)
    sums.append(ce)
    counts.append(n)
    for k in range(n_codebooks):
        ce, n = head_sums(out.audio_logits[:, k], 1 + k)
        sums.append(ce)
        counts.append(n)

    counts_t = torch.stack(counts)
    denom = text_w + audio_w * (sem_w + max(0, n_codebooks - 1))
    if denom <= 0.0:  # all-zero weights: avoid 0/0, loss is a true zero
        denom = 1.0
    w_t = torch.tensor(head_w, dtype=sums[0].dtype, device=sums[0].device)
    means_t = torch.stack(sums) / counts_t.clamp(min=1)
    # exact-zero anchor: keeps the graph connected to every head even when a
    # batch has no targets at all (CE over all-ignored targets can detach)
    anchor = (out.text_logits.sum() + out.audio_logits.sum()) * 0.0
    total = (w_t * means_t).sum() / denom + anchor

    metrics: dict[str, torch.Tensor] = {"loss": total.detach()}
    for name, mean, cnt in zip(names, means_t.detach(), counts_t.tolist()):  # one sync
        if cnt > 0:
            metrics[name] = mean
    return total, metrics


def _split_decay_params(
    model: nn.Module, names: list[str] | None = None
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split (a subset of) a model's trainable params into (decay, no_decay).

    no_decay: 1-D parameters (norm gains, biases) and every ``nn.Embedding``
    weight — standard practice excludes both from weight decay. ``names``
    restricts the split to those parameter FQNs (None = all trainable).
    Tied parameters appear once (named_parameters dedupes).
    """
    emb_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }
    wanted = set(names) if names is not None else None
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for n, p in model.named_parameters():
        if not p.requires_grad or (wanted is not None and n not in wanted):
            continue
        (no_decay if p.ndim < 2 or id(p) in emb_ids else decay).append(p)
    return decay, no_decay


class OmniModel(nn.Module):
    """Llama-style multistream decoder (see docs/DESIGN.md §1, §3).

    Components: text_emb [text_vocab_size, D]; audio_embs: n_codebooks x
    [audio_vocab_size, D]; channel_emb [2, D]; n_layers pre-norm Blocks; final
    RMSNorm; text_head (weight-tied to text_emb when cfg.tie_text_head) and
    n_codebooks audio heads, all bias-free.

    v2: when cfg.use_depth there are NO audio_heads; `self.depth` (a
    DepthTransformer sharing `audio_embs`) produces the audio logits. When
    cfg.duplex, `self.user_audio_embs` (n_codebooks extra tables) embed the
    input-only user audio group.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.text_emb = nn.Embedding(cfg.text_vocab_size, d)
        self.audio_embs = nn.ModuleList(
            nn.Embedding(cfg.audio_vocab_size, d) for _ in range(cfg.n_codebooks)
        )
        self.channel_emb = nn.Embedding(2, d)
        self.user_audio_embs: nn.ModuleList | None = (
            nn.ModuleList(
                nn.Embedding(cfg.audio_vocab_size, d) for _ in range(cfg.n_codebooks)
            )
            if cfg.duplex
            else None
        )
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.text_head = nn.Linear(d, cfg.text_vocab_size, bias=False)
        self.audio_heads: nn.ModuleList | None
        self.depth: DepthTransformer | None
        if cfg.use_depth:
            self.audio_heads = None
            self.depth = DepthTransformer(cfg, self.audio_embs)
        else:
            self.audio_heads = nn.ModuleList(
                nn.Linear(d, cfg.audio_vocab_size, bias=False)
                for _ in range(cfg.n_codebooks)
            )
            self.depth = None
        if cfg.tie_text_head:
            self.text_head.weight = self.text_emb.weight
        # fp32 rope table [max_frames, head_dim//2, 2]. Kept as a plain tensor
        # attribute (NOT a buffer) so Module.to(dtype) can never downcast it;
        # moved to the input device lazily in _rope_on.
        self._rope = precompute_rope(cfg.head_dim, cfg.max_frames, cfg.rope_theta)
        self.init_weights()

    # ------------------------------------------------------------------ utils
    def _rope_on(self, device: torch.device) -> torch.Tensor:
        if self._rope.device != device:
            self._rope = self._rope.to(device)
        return self._rope

    def _head_logits(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B, D] (post final norm) -> (text [B, Vt], audio [B, n_q, Va]) fp32."""
        assert self.audio_heads is not None  # guarded by the use_depth checks
        text = self.text_head(h).float()
        audio = torch.stack([head(h) for head in self.audio_heads], dim=1).float()
        return text, audio

    # ------------------------------------------------------------- embeddings
    def embed(self, grid: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        """Sum-of-streams input embedding.

        grid long [B, S, T] (DELAYED), channel long [B, T] in {0, 1}
        -> [B, T, D]: text_emb(grid[:, 0]) + sum_k audio_embs[k](grid[:, 1+k])
        + channel_emb(channel). Duplex grids (S = 1 + 2*n_q) additionally sum
        the user rows n_q+1..2n_q via user_audio_embs.
        """
        assert grid.shape[1] == self.cfg.n_streams, (
            f"grid has {grid.shape[1]} streams, model expects {self.cfg.n_streams}"
        )
        assert channel.shape == (grid.shape[0], grid.shape[2]), (
            f"channel shape {tuple(channel.shape)} must be [B, T] = "
            f"({grid.shape[0]}, {grid.shape[2]})"
        )
        h = self.text_emb(grid[:, 0])
        for k, emb in enumerate(self.audio_embs):
            h = h + emb(grid[:, 1 + k])
        if self.user_audio_embs is not None:
            n_q = self.cfg.n_codebooks
            for k, emb in enumerate(self.user_audio_embs):
                h = h + emb(grid[:, 1 + n_q + k])
        return h + self.channel_emb(channel)

    # ----------------------------------------------------------- training path
    def forward(self, grid: torch.Tensor, channel: torch.Tensor) -> ModelOutput:
        """Full-sequence pass, no KV cache; positions 0..T-1.

        grid long [B, S, T] DELAYED, channel long [B, T]. Returns float32
        logits: text [B, T, Vt], audio [B, n_q, T, Va]. Per-block gradient
        checkpointing (use_reentrant=False) when cfg.grad_checkpoint and
        self.training.

        With cfg.use_depth the audio logits come from ONE vectorized
        depth-transformer call over all B*T positions: the depth input at
        position p is teacher-forced with the frame being predicted there,
        `grid[:, 1:1+n_q, p+1]` (flat delays put a frame's codebooks on one
        column). The last position has no p+1; it gets an AUDIO_PAD teacher and
        is always loss-masked. Output shapes are unchanged.
        """
        h = self.embed(grid, channel)
        rope = self._rope_on(h.device)
        use_ckpt = self.cfg.grad_checkpoint and self.training
        for block in self.blocks:
            if use_ckpt:
                h = checkpoint(block, h, rope, use_reentrant=False)
            else:
                h = block(h, rope)
        h = self.norm(h)
        text_logits = self.text_head(h).float()
        sel: torch.Tensor | None = None
        if self.depth is not None:
            B, T, d = h.shape
            n_q = self.cfg.n_codebooks
            sel = _depth_positions(T, self.cfg.depth_loss_ratio, self.training, h.device)
            if sel is None:
                apad = audio_pad_id(self.cfg.audio_codec_vocab)
                teacher = torch.cat(
                    [grid[:, 1 : 1 + n_q, 1:], grid.new_full((B, n_q, 1), apad)], dim=2
                )  # [B, n_q, T]: frame targets shifted so position p sees col p+1
                teacher = teacher.permute(0, 2, 1).reshape(B * T, n_q)
                depth_logits = self.depth(h.reshape(B * T, d), teacher)  # [B*T, n_q, Va]
                audio_logits = depth_logits.view(B, T, n_q, -1).permute(0, 2, 1, 3)
            else:
                # CSM-style amortization: depth loss on a random position
                # subset (cfg.depth_loss_ratio of the frames) — the teacher at
                # subset position p is column p+1, always in range (p <= T-2)
                P = int(sel.shape[0])
                h_sel = h.index_select(1, sel)  # [B, P, d]
                teacher = grid[:, 1 : 1 + n_q].index_select(2, sel + 1)  # [B, n_q, P]
                teacher = teacher.permute(0, 2, 1).reshape(B * P, n_q)
                depth_logits = self.depth(h_sel.reshape(B * P, d), teacher)
                audio_logits = depth_logits.view(B, P, n_q, -1).permute(0, 2, 1, 3)
        else:
            audio_logits = torch.stack(
                [head(h) for head in self.audio_heads], dim=1
            ).float()
        return ModelOutput(
            text_logits=text_logits, audio_logits=audio_logits, audio_positions=sel
        )

    def loss(
        self,
        out: ModelOutput,
        grid: torch.Tensor,
        loss_mask: torch.Tensor,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Next-token cross-entropy over the delayed grid; see
        :func:`multistream_loss` (identical semantics, shared with the v6
        pretrained-backbone model)."""
        return multistream_loss(out, grid, loss_mask, self.cfg.n_codebooks, weights)

    # ------------------------------------------------------------ decode path
    def new_cache(
        self,
        batch: int,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> KVCache:
        """Fresh decode cache for this model (generators call this instead of
        a concrete cache class, so backbone-backed models can supply their
        own cache type)."""
        return KVCache.allocate(self.cfg, batch, device, dtype)

    def _decode_last_hidden(
        self, grid: torch.Tensor, channel: torch.Tensor, cache: KVCache
    ) -> torch.Tensor:
        """Run grid [B, S, T] / channel [B, T] through the cached backbone at
        absolute positions cache.pos..cache.pos+T-1; return the LAST position's
        post-final-norm hidden state [B, D] (model dtype)."""
        assert len(cache.layers) == len(self.blocks), "cache/model layer count mismatch"
        pos = cache.pos
        h = self.embed(grid, channel)
        rope = self._rope_on(h.device)
        for block, layer_cache in zip(self.blocks, cache.layers):
            h = block(h, rope, layer_cache, pos)
        return self.norm(h[:, -1])

    def _no_parallel_heads(self, name: str) -> None:
        if self.cfg.use_depth:
            raise RuntimeError(
                f"{name}() returns parallel audio-head logits, but this model uses "
                "the depth transformer (cfg.use_depth=True): a frame's codebooks "
                "are predicted sequentially. Use prefill_hidden()/step_hidden() "
                "and self.depth.sample(hidden, sample_fn) instead."
            )

    def prefill(
        self, grid: torch.Tensor, channel: torch.Tensor, cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume a DELAYED prompt through the KV cache.

        grid long [B, S, T], channel long [B, T]; absolute positions
        cache.pos..cache.pos+T-1 (0 on a fresh cache). Returns logits for the
        LAST position: (text [B, Vt], audio [B, n_q, Va]), both float32.
        Raises RuntimeError on depth models (use prefill_hidden).
        """
        self._no_parallel_heads("prefill")
        return self._head_logits(self._decode_last_hidden(grid, channel, cache))

    def step(
        self, tokens: torch.Tensor, channel: torch.Tensor, cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode step at absolute position cache.pos.

        tokens long [B, S] (the DELAYED input column), channel long [B].
        Returns (text [B, Vt], audio [B, n_q, Va]) float32 logits predicting
        position cache.pos+1's tokens. Raises RuntimeError on depth models
        (use step_hidden).
        """
        self._no_parallel_heads("step")
        assert tokens.ndim == 2 and channel.ndim == 1, "step takes tokens [B, S], channel [B]"
        h = self._decode_last_hidden(tokens.unsqueeze(-1), channel.unsqueeze(-1), cache)
        return self._head_logits(h)

    def prefill_hidden(
        self, grid: torch.Tensor, channel: torch.Tensor, cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `prefill` but returns (text_logits [B, Vt] float32, hidden
        [B, D] post-final-norm, model dtype) for the LAST position. Works for
        ALL models; on depth models feed `hidden` to `self.depth`."""
        h = self._decode_last_hidden(grid, channel, cache)
        return self.text_head(h).float(), h

    def step_hidden(
        self, tokens: torch.Tensor, channel: torch.Tensor, cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `step` but returns (text_logits [B, Vt] float32, hidden [B, D]
        post-final-norm, model dtype) predicting position cache.pos+1. Works
        for ALL models; on depth models feed `hidden` to `self.depth`."""
        assert tokens.ndim == 2 and channel.ndim == 1, (
            "step_hidden takes tokens [B, S], channel [B]"
        )
        h = self._decode_last_hidden(tokens.unsqueeze(-1), channel.unsqueeze(-1), cache)
        return self.text_head(h).float(), h

    # ------------------------------------------------------------------ admin
    def init_weights(self) -> None:
        """Init: linears normal(0, cfg.init_std); attn out-proj and mlp down-proj
        std scaled by 1/sqrt(2*n_layers); embeddings normal(0, 0.02); norms 1.
        Depth-transformer params get the same recipe via its own init_weights
        (out-projections scaled by 1/sqrt(2*depth_n_layers)).

        In-place, so text-head tying is preserved; embeddings are written last,
        so a tied text head ends with the embedding init (std 0.02) and the
        depth transformer's shared audio_embs match the backbone's.
        """
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
        out_std = std / math.sqrt(2.0 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=out_std)
            nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=out_std)
        if self.depth is not None:
            self.depth.init_weights()
        embs = [self.text_emb, *self.audio_embs, self.channel_emb]
        if self.user_audio_embs is not None:
            embs.extend(self.user_audio_embs)
        for emb in embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def optim_param_groups(self, train_cfg) -> list[dict]:
        """AdamW param groups at train.lr, split so weight decay skips what it
        should never touch: RMSNorm gains / any 1-D parameter, and every
        embedding table (including a tied text head). Same duck-typed hook the
        Trainer uses for HFOmniModel."""
        decay, no_decay = _split_decay_params(self)
        groups: list[dict] = [{"params": decay, "lr": train_cfg.lr}]
        if no_decay:
            groups.append({"params": no_decay, "lr": train_cfg.lr, "weight_decay": 0.0})
        return groups

    def param_counts(self) -> dict[str, int]:
        """{"total", "non_embedding"}; tied/shared parameters counted once;
        non_embedding excludes the token embedding tables (text, audio,
        user audio, channel). The depth transformer's tiny pos_emb counts as
        non-embedding."""
        total = sum(p.numel() for p in self.parameters())
        emb = (
            self.text_emb.weight.numel()
            + sum(e.weight.numel() for e in self.audio_embs)
            + self.channel_emb.weight.numel()
        )
        if self.user_audio_embs is not None:
            emb += sum(e.weight.numel() for e in self.user_audio_embs)
        return {"total": total, "non_embedding": total - emb}

    def save_pretrained(self, save_dir: str | Path) -> None:
        """Write <save_dir>/model.safetensors + config.yaml (flat ModelConfig
        fields, v2 fields included). Tied/shared duplicates (text_head.weight,
        the depth transformer's shared audio_embs) are dropped by safetensors'
        save_model and re-bound on load."""
        from safetensors.torch import save_model

        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        save_model(self, str(p / "model.safetensors"))
        with open(p / "config.yaml", "w") as f:
            yaml.safe_dump(dataclasses.asdict(self.cfg), f, sort_keys=False)

    @classmethod
    def from_pretrained(
        cls, save_dir: str | Path, map_location: str | torch.device = "cpu"
    ) -> "OmniModel":
        """Load a model saved by save_pretrained (or Trainer.export_model).

        Rebuilds ModelConfig from config.yaml (unknown keys ignored, so v1
        checkpoints load with v2 defaults), loads weights strictly (modulo
        tied/shared-weight dedup), returns the model on map_location, in
        eval-able fp32 state.
        """
        from safetensors.torch import load_model

        p = Path(save_dir)
        with open(p / "config.yaml") as f:
            d = yaml.safe_load(f) or {}
        names = {f.name for f in dataclasses.fields(ModelConfig)}
        cfg = ModelConfig(**{k: v for k, v in d.items() if k in names})
        model = cls(cfg)
        load_model(model, str(p / "model.safetensors"))
        return model.to(map_location)
