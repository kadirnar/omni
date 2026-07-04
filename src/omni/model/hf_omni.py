"""HFOmniModel: a pretrained HF causal-LM as the temporal backbone (DESIGN_V6).

Same public surface as :class:`omni.model.omni.OmniModel` — ``embed`` /
``forward`` / ``loss`` / ``prefill[_hidden]`` / ``step[_hidden]`` /
``new_cache`` / ``save_pretrained`` / ``from_pretrained`` — so the Trainer,
OmniGenerator, DuplexGenerator, chat, and serve run against either class.

Layout (DESIGN_V6 §2–3):

- The backbone (Qwen3 / Llama 3.x / Gemma, any ``AutoModelForCausalLM``) is
  driven through ``inputs_embeds`` and its own KV cache; its RoPE, norms, and
  attention are untouched. One grid step = one backbone position.
- Text ids live in the omni id space: ids 0..63 are the reserved specials
  (fresh 64-row embedding + 64-way head), ids >= 64 are backbone token
  ``id - 64`` (backbone embedding + pretrained ``lm_head``). Text logits are
  ``cat([special_head(h), lm_head(h)])``, so ``text_vocab_size`` is
  ``64 + backbone embedding rows``.
- Audio: new per-codebook embedding tables at the backbone's hidden size,
  summed into the input embedding together with the channel embedding
  (zero-init so step-0 behavior matches the text LLM); output through the V3
  DepthTransformer (flat delays) or parallel linear heads (stagger).

Backbone-derived ``ModelConfig`` fields (d_model, layer/head counts,
text_vocab_size, ...) are overwritten on ``self.cfg`` at construction, so
callers can do ``cfg.model = model.cfg`` exactly like with OmniModel
checkpoints. Nothing here touches the network unless ``cfg.backbone_id`` is
resolved (i.e. no ``backbone=`` injection) — same explicit-download rule as
``--codec mimi``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ..config import ModelConfig
from ..streams import N_RESERVED_SPECIALS, audio_pad_id
from .layers import DepthTransformer
from .omni import ModelOutput, _depth_positions, _split_decay_params, multistream_loss


class HFCache:
    """Decode cache for HFOmniModel: a ``transformers`` cache + the filled
    length, mirroring ``KVCache.pos`` so generator-side assertions hold."""

    def __init__(self, hf_cache) -> None:
        self.hf = hf_cache
        self.pos = 0


def _load_backbone(model_id: str, dtype: str) -> nn.Module:
    """Download/instantiate the pretrained causal LM (explicit user path)."""
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    except TypeError:  # older transformers spell the kwarg torch_dtype
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)


class HFOmniModel(nn.Module):
    """Multistream decoder over a pretrained HF backbone (DESIGN_V6).

    New (randomly initialized, always trainable) modules: ``special_emb``
    [64, D] + ``special_head`` [D, 64] for the omni specials; ``audio_embs``
    (n_codebooks x [audio_vocab_size, D], shared with the depth transformer);
    ``channel_emb`` [2, D] (zero-init); ``user_audio_embs`` when duplex;
    ``depth`` (flat mode) or ``audio_heads`` (stagger). ``cfg.freeze_backbone``
    freezes every backbone parameter; ``cfg.lora_rank > 0`` additionally
    injects LoRA adapters into the backbone (optional dep ``peft``).
    """

    def __init__(self, cfg: ModelConfig, backbone: nn.Module | None = None):
        super().__init__()
        assert cfg.backbone_id is not None or backbone is not None, (
            "HFOmniModel needs cfg.backbone_id (hub download) or an injected backbone"
        )
        if backbone is None:
            backbone = _load_backbone(cfg.backbone_id, cfg.backbone_dtype)
        self.backbone = backbone

        emb = backbone.get_input_embeddings()
        d = int(emb.embedding_dim)
        bcfg = getattr(backbone, "config", None)
        # Re-derive the structural fields from the backbone so self.cfg
        # describes the real model (generator checks, save/load, KV math).
        self.cfg = cfg = dataclasses.replace(
            cfg,
            d_model=d,
            text_vocab_size=N_RESERVED_SPECIALS + int(emb.num_embeddings),
            n_layers=int(getattr(bcfg, "num_hidden_layers", cfg.n_layers)),
            n_heads=int(getattr(bcfg, "num_attention_heads", cfg.n_heads)),
            n_kv_heads=int(
                getattr(bcfg, "num_key_value_heads", None)
                or getattr(bcfg, "num_attention_heads", cfg.n_kv_heads)
            ),
            d_ff=int(getattr(bcfg, "intermediate_size", cfg.d_ff)),
            tie_text_head=bool(getattr(bcfg, "tie_word_embeddings", cfg.tie_text_head)),
        )

        self.special_emb = nn.Embedding(N_RESERVED_SPECIALS, d)
        self.special_head = nn.Linear(d, N_RESERVED_SPECIALS, bias=False)
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
        self.init_weights()
        # New modules follow the backbone's parameter dtype so the summed
        # input embedding and the depth/head matmuls stay in one dtype.
        bb_dtype = next(backbone.parameters()).dtype
        for m in self._new_modules():
            m.to(bb_dtype)

        if cfg.freeze_backbone:
            self.backbone.requires_grad_(False)
        if cfg.lora_rank > 0:
            self._apply_lora()
        if cfg.grad_checkpoint and hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:  # older transformers take no kwargs
                self.backbone.gradient_checkpointing_enable()

    # ------------------------------------------------------------------ setup
    def _new_modules(self) -> list[nn.Module]:
        """Every non-backbone module (the 'adapter' set)."""
        mods: list[nn.Module] = [
            self.special_emb, self.special_head, self.audio_embs, self.channel_emb,
        ]
        if self.user_audio_embs is not None:
            mods.append(self.user_audio_embs)
        mods.append(self.depth if self.depth is not None else self.audio_heads)
        return mods

    def _apply_lora(self) -> None:
        """Inject LoRA adapters into the backbone's attention/MLP linears."""
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "cfg.lora_rank > 0 needs the 'peft' package; "
                "pip install 'omni[lora]' or set model.lora_rank=0"
            ) from e
        lcfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules="all-linear",
        )
        inject_adapter_in_model(lcfg, self.backbone)

    def init_weights(self) -> None:
        """Init the NEW modules only (the backbone keeps its pretrained
        weights): embeddings normal(0, 0.02) except channel_emb = 0 (so the
        summed input starts as pure text-LM behavior for text-only steps),
        heads normal(0, cfg.init_std), depth via its own recipe."""
        std = self.cfg.init_std
        nn.init.normal_(self.special_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.special_head.weight, mean=0.0, std=std)
        nn.init.zeros_(self.channel_emb.weight)
        embs = list(self.audio_embs)
        if self.user_audio_embs is not None:
            embs.extend(self.user_audio_embs)
        for e in embs:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        if self.depth is not None:
            self.depth.init_weights()
        if self.audio_heads is not None:
            for h in self.audio_heads:
                nn.init.normal_(h.weight, mean=0.0, std=std)

    # ------------------------------------------------------------- embeddings
    def embed(self, grid: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        """Sum-of-streams input embedding (same contract as OmniModel.embed).

        grid long [B, S, T] (DELAYED), channel long [B, T] in {0, 1}
        -> [B, T, D]. Text row: ids < 64 hit `special_emb`, ids >= 64 hit the
        backbone table at `id - 64`.
        """
        assert grid.shape[1] == self.cfg.n_streams, (
            f"grid has {grid.shape[1]} streams, model expects {self.cfg.n_streams}"
        )
        assert channel.shape == (grid.shape[0], grid.shape[2]), (
            f"channel shape {tuple(channel.shape)} must be [B, T] = "
            f"({grid.shape[0]}, {grid.shape[2]})"
        )
        t = grid[:, 0]
        is_special = t < N_RESERVED_SPECIALS
        bb = self.backbone.get_input_embeddings()(
            (t - N_RESERVED_SPECIALS).clamp(min=0)
        )
        sp = self.special_emb(t.clamp(max=N_RESERVED_SPECIALS - 1))
        h = torch.where(is_special.unsqueeze(-1), sp, bb)
        for k, emb in enumerate(self.audio_embs):
            h = h + emb(grid[:, 1 + k])
        if self.user_audio_embs is not None:
            n_q = self.cfg.n_codebooks
            for k, emb in enumerate(self.user_audio_embs):
                h = h + emb(grid[:, 1 + n_q + k])
        return h + self.channel_emb(channel)

    def _text_logits(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., D] -> float32 [..., 64 + backbone_vocab]: special head ids
        0..63 then the pretrained lm_head."""
        lm_head = self.backbone.get_output_embeddings()
        return torch.cat([self.special_head(h), lm_head(h)], dim=-1).float()

    def _hidden_states(
        self,
        h: torch.Tensor,
        cache: HFCache | None = None,
    ) -> torch.Tensor:
        """Run inputs_embeds h [B, T, D] through the backbone decoder stack;
        returns the post-final-norm hidden states [B, T, D] (backbone dtype).
        With `cache`, positions are cache.pos..cache.pos+T-1 and cache.pos
        advances by T."""
        decoder = self.backbone.get_decoder()
        B, T, _ = h.shape
        if cache is None:
            out = decoder(inputs_embeds=h, use_cache=False)
            return out.last_hidden_state
        cache_position = torch.arange(cache.pos, cache.pos + T, device=h.device)
        out = decoder(
            inputs_embeds=h,
            past_key_values=cache.hf,
            use_cache=True,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0).expand(B, -1),
        )
        cache.pos += T
        cache.hf = out.past_key_values  # legacy versions return a new object
        return out.last_hidden_state

    # ----------------------------------------------------------- training path
    def forward(self, grid: torch.Tensor, channel: torch.Tensor) -> ModelOutput:
        """Full-sequence teacher-forced pass; identical contract to
        OmniModel.forward (float32 logits: text [B, T, Vt], audio
        [B, n_q, T, Va]; depth models get ONE vectorized depth call over all
        B*T positions with the frame at p+1 as teacher)."""
        h = self.embed(grid, channel)
        h = self._hidden_states(h)
        text_logits = self._text_logits(h)
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
                # CSM-style amortization (cfg.depth_loss_ratio of the frames)
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
        """Shared multistream CE; see :func:`omni.model.omni.multistream_loss`."""
        return multistream_loss(out, grid, loss_mask, self.cfg.n_codebooks, weights)

    # ------------------------------------------------------------ decode path
    def new_cache(
        self,
        batch: int,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> HFCache:
        """Fresh backbone KV cache (DynamicCache: batch/device/dtype follow
        the first forward, so the arguments are accepted for signature parity
        and ignored)."""
        del batch, device, dtype
        from transformers import DynamicCache

        return HFCache(DynamicCache())

    def _decode_last_hidden(
        self, grid: torch.Tensor, channel: torch.Tensor, cache: HFCache
    ) -> torch.Tensor:
        """grid [B, S, T] / channel [B, T] through the cached backbone at
        absolute positions cache.pos..cache.pos+T-1 -> last position's
        post-final-norm hidden [B, D] (backbone dtype)."""
        h = self._hidden_states(self.embed(grid, channel), cache)
        return h[:, -1]

    def _head_logits(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.audio_heads is not None  # guarded by the use_depth checks
        text = self._text_logits(h)
        audio = torch.stack([head(h) for head in self.audio_heads], dim=1).float()
        return text, audio

    def _no_parallel_heads(self, name: str) -> None:
        if self.cfg.use_depth:
            raise RuntimeError(
                f"{name}() returns parallel audio-head logits, but this model uses "
                "the depth transformer (cfg.use_depth=True): a frame's codebooks "
                "are predicted sequentially. Use prefill_hidden()/step_hidden() "
                "and self.depth.sample(hidden, sample_fn) instead."
            )

    def prefill(
        self, grid: torch.Tensor, channel: torch.Tensor, cache: HFCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as OmniModel.prefill (last-position float32 logits)."""
        self._no_parallel_heads("prefill")
        return self._head_logits(self._decode_last_hidden(grid, channel, cache))

    def step(
        self, tokens: torch.Tensor, channel: torch.Tensor, cache: HFCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as OmniModel.step."""
        self._no_parallel_heads("step")
        assert tokens.ndim == 2 and channel.ndim == 1, "step takes tokens [B, S], channel [B]"
        h = self._decode_last_hidden(tokens.unsqueeze(-1), channel.unsqueeze(-1), cache)
        return self._head_logits(h)

    def prefill_hidden(
        self, grid: torch.Tensor, channel: torch.Tensor, cache: HFCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as OmniModel.prefill_hidden: (text_logits [B, Vt]
        float32, hidden [B, D] post-final-norm)."""
        h = self._decode_last_hidden(grid, channel, cache)
        return self._text_logits(h), h

    def step_hidden(
        self, tokens: torch.Tensor, channel: torch.Tensor, cache: HFCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as OmniModel.step_hidden."""
        assert tokens.ndim == 2 and channel.ndim == 1, (
            "step_hidden takes tokens [B, S], channel [B]"
        )
        h = self._decode_last_hidden(tokens.unsqueeze(-1), channel.unsqueeze(-1), cache)
        return self._text_logits(h), h

    # ------------------------------------------------------------------ admin
    def optim_param_groups(self, train_cfg) -> list[dict]:
        """AdamW param groups: new modules at train.lr; trainable backbone
        parameters (full finetune or LoRA) at train.backbone_lr. Frozen
        parameters are excluded entirely, and each lr tier is further split so
        weight decay skips 1-D params (norm gains) and embedding tables."""
        new_names = [
            n for n, p in self.named_parameters()
            if not n.startswith("backbone.") and p.requires_grad
        ]
        bb_names = [
            n for n, p in self.named_parameters()
            if n.startswith("backbone.") and p.requires_grad
        ]
        groups: list[dict] = []
        for names, lr in ((new_names, train_cfg.lr), (bb_names, train_cfg.backbone_lr)):
            if not names:
                continue
            decay, no_decay = _split_decay_params(self, names)
            if decay:
                groups.append({"params": decay, "lr": lr})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    def param_counts(self) -> dict[str, int]:
        """{"total", "non_embedding", "trainable"}; shared tensors counted
        once; non_embedding excludes every token-embedding table (backbone
        input embeddings included)."""
        seen: set[int] = set()
        total = trainable = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        emb = (
            self.backbone.get_input_embeddings().weight.numel()
            + self.special_emb.weight.numel()
            + sum(e.weight.numel() for e in self.audio_embs)
            + self.channel_emb.weight.numel()
        )
        if self.user_audio_embs is not None:
            emb += sum(e.weight.numel() for e in self.user_audio_embs)
        return {"total": total, "non_embedding": total - emb, "trainable": trainable}

    # keys never saved: backbone weights (referenced by backbone_id, except
    # LoRA adapters) and the depth transformer's aliases of the shared
    # audio_embs (re-bound at construction).
    @staticmethod
    def _is_adapter_key(key: str) -> bool:
        if key.startswith("depth.audio_embs."):
            return False
        if key.startswith("backbone."):
            return "lora_" in key
        return True

    def save_pretrained(
        self, save_dir: str | Path, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        """Write <save_dir>/adapters.safetensors (new modules + LoRA only —
        the pretrained backbone is referenced via config.yaml's backbone_id,
        not copied) + config.yaml (flat ModelConfig). Pass ``state_dict`` to
        export a consolidated dict gathered elsewhere (Trainer/FSDP2).

        NOTE: with ``freeze_backbone=false`` (full finetune) the updated
        backbone weights are NOT captured here; merge/export them separately
        (DESIGN_V6 §8 --export-full, queue item)."""
        from safetensors.torch import save_file

        if state_dict is None:
            state_dict = self.state_dict()
        adapters = {
            k: v.detach().to("cpu", torch.float32).contiguous()
            for k, v in state_dict.items()
            if self._is_adapter_key(k)
        }
        if not self.cfg.freeze_backbone and self.cfg.lora_rank == 0:
            import warnings

            warnings.warn(
                "save_pretrained on a fully-finetuned backbone saves only the "
                "adapter modules; the updated backbone weights are NOT included",
                stacklevel=2,
            )
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        save_file(adapters, str(p / "adapters.safetensors"))
        with open(p / "config.yaml", "w") as f:
            yaml.safe_dump(dataclasses.asdict(self.cfg), f, sort_keys=False)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str | Path,
        backbone: nn.Module | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "HFOmniModel":
        """Load a model saved by save_pretrained: rebuild ModelConfig from
        config.yaml, construct (downloading the backbone unless one is
        injected), and load the adapter weights strictly."""
        from safetensors.torch import load_file

        p = Path(save_dir)
        with open(p / "config.yaml") as f:
            d = yaml.safe_load(f) or {}
        names = {f.name for f in dataclasses.fields(ModelConfig)}
        cfg = ModelConfig(**{k: v for k, v in d.items() if k in names})
        model = cls(cfg, backbone=backbone)
        sd = load_file(str(p / "adapters.safetensors"))
        target_dtype = next(model.backbone.parameters()).dtype
        sd = {k: v.to(target_dtype) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected adapter keys: {unexpected[:5]}"
        bad = [k for k in missing if cls._is_adapter_key(k)]
        assert not bad, f"missing adapter keys: {bad[:5]}"
        return model.to(map_location)
