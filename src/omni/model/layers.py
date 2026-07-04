"""Transformer building blocks for the omni multistream decoder.

Llama-style pieces: RMSNorm (fp32-internal), rotary position embeddings applied
in fp32, GQA attention (KV heads repeat_interleaved up to n_heads), SwiGLU MLP,
a pre-norm Block, and a preallocated per-layer KV cache for frame-synchronous
decoding. Plus the v2 DepthTransformer: a small causal transformer over the
n_codebooks positions of ONE frame (flat delay mode), predicting the codebooks
sequentially from the backbone hidden state (CSM/Moshi-style).

Shape symbols: B batch, T steps (DELAYED-grid positions), D d_model, H n_heads,
Hkv n_kv_heads, hd head_dim; for the depth transformer Dd depth_d_model and
n_q n_codebooks. No linear layer has a bias.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (no bias). [..., dim] -> [..., dim].

    Statistics and the weight multiply run in float32; the result is cast back
    to the input dtype (bf16-autocast safe).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def precompute_rope(
    head_dim: int,
    max_pos: int,
    theta: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Rotary embedding table: float32 [max_pos, head_dim // 2, 2].

    [p, i, 0] = cos(p * theta^(-2i/head_dim)), [p, i, 1] = the matching sin.
    Always float32 regardless of the default dtype.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)  # [max_pos, head_dim//2]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor, pos: int = 0) -> torch.Tensor:
    """Rotate adjacent channel pairs of x by absolute positions pos..pos+T-1.

    x: [B, H, T, head_dim]; rope: float32 [max_pos, head_dim//2, 2] from
    `precompute_rope`. Math in float32, result cast back to x.dtype.
    """
    B, H, T, hd = x.shape
    r = rope[pos : pos + T]  # [T, hd//2, 2]
    assert r.shape[0] == T, (
        f"rope table too short: need positions {pos}..{pos + T - 1}, have {rope.shape[0]}"
    )
    cos = r[..., 0]  # [T, hd//2], broadcasts over B, H
    sin = r[..., 1]
    xf = x.float().reshape(B, H, T, hd // 2, 2)
    x1, x2 = xf[..., 0], xf[..., 1]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.reshape(B, H, T, hd).to(x.dtype)


class LayerKVCache:
    """Preallocated KV cache for one attention layer.

    k, v: [B, Hkv, max_frames, head_dim]; pos = filled prefix length (steps
    already appended).
    """

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v
        self.pos = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k_new/v_new [B, Hkv, T, hd] at pos..pos+T-1 and advance pos.

        Returns views (k[:, :, :pos+T], v[:, :, :pos+T]) over the post-append
        filled prefix.
        """
        T = k_new.shape[2]
        assert self.pos + T <= self.k.shape[2], (
            f"KV cache overflow: pos {self.pos} + T {T} > max_frames {self.k.shape[2]}"
        )
        self.k[:, :, self.pos : self.pos + T] = k_new
        self.v[:, :, self.pos : self.pos + T] = v_new
        self.pos += T
        return self.k[:, :, : self.pos], self.v[:, :, : self.pos]


class KVCache:
    """One LayerKVCache per transformer block.

    `pos` is read-only, derived from layer 0 (all layers stay in sync because
    the model appends to every layer once per forward). Allocate a fresh cache
    per generation; there is no reset.
    """

    layers: list[LayerKVCache]

    def __init__(self, layers: list[LayerKVCache]):
        self.layers = layers

    @property
    def pos(self) -> int:
        return self.layers[0].pos if self.layers else 0

    @classmethod
    def allocate(
        cls,
        cfg: ModelConfig,
        batch: int,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> "KVCache":
        """Zero tensors [batch, n_kv_heads, cfg.max_frames, head_dim] per layer."""
        shape = (batch, cfg.n_kv_heads, cfg.max_frames, cfg.head_dim)
        layers = [
            LayerKVCache(
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(cfg.n_layers)
        ]
        return cls(layers)


class Attention(nn.Module):
    """Causal GQA self-attention, no biases.

    KV heads are expanded to n_heads via repeat_interleave; RoPE applied to q/k
    in fp32 at absolute positions.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.wq = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        cache: LayerKVCache | None = None,
        pos: int = 0,
    ) -> torch.Tensor:
        """x: [B, T, D] at absolute delayed-grid positions pos..pos+T-1 -> [B, T, D].

        cache None (training): full-sequence SDPA with is_causal=True (pos must
        be 0). cache set (decode): append k/v then attend over the filled
        prefix [:, :, :pos+T]; T == 1 needs no mask, prefill (T > 1, empty
        cache) uses is_causal=True, and chunked prefill over a non-empty cache
        gets an explicit position-aligned causal mask.
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, rope, pos)
        k = apply_rope(k, rope, pos)

        if cache is not None:
            assert cache.pos == pos, (
                f"cache.pos ({cache.pos}) must equal the rope offset ({pos})"
            )
            k, v = cache.append(k, v)  # [B, Hkv, pos+T, hd]

        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if cache is None or (pos == 0 and T > 1):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # single decode step: attend over the whole filled prefix, no mask
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # chunk appended onto existing context: is_causal would misalign the
            # diagonal for non-square [T, pos+T] scores, so build the mask.
            idx_q = torch.arange(T, device=x.device)
            idx_k = torch.arange(pos + T, device=x.device)
            mask = idx_k[None, :] <= (pos + idx_q)[:, None]  # True = may attend
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: w2(silu(w1 x) * w3 x). [B, T, D] -> [B, T, D]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(norm(x)), then x + MLP(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        cache: LayerKVCache | None = None,
        pos: int = 0,
    ) -> torch.Tensor:
        """[B, T, D] -> [B, T, D]; pos = absolute position of x[:, 0]."""
        h = self.attn(self.attn_norm(x), rope, cache, pos)
        if self.dropout_p > 0.0 and self.training:
            h = F.dropout(h, self.dropout_p)
        x = x + h
        h = self.mlp(self.mlp_norm(x))
        if self.dropout_p > 0.0 and self.training:
            h = F.dropout(h, self.dropout_p)
        return x + h


# ---------------------------------------------------------------------------
# Depth transformer (v2, flat delay mode): per-frame sequential codebook decoder
# ---------------------------------------------------------------------------
class _DepthAttention(nn.Module):
    """Causal MHA over the (tiny) codebook axis. NO RoPE, no biases, no cache.

    [N, K, Dd] -> [N, K, Dd] with K <= n_codebooks; position identity comes
    from the learned pos_emb added to the inputs, so plain is_causal SDPA
    suffices.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.depth_d_model % cfg.depth_n_heads == 0
        self.n_heads = cfg.depth_n_heads
        self.head_dim = cfg.depth_d_model // cfg.depth_n_heads
        d = cfg.depth_d_model
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, K, _ = x.shape
        q = self.wq(x).view(N, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(N, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(N, K, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(N, K, self.n_heads * self.head_dim))


class _DepthBlock(nn.Module):
    """Pre-norm block for the depth transformer: SwiGLU d_ff = 4 * depth_d_model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.depth_d_model
        self.attn_norm = RMSNorm(d, cfg.norm_eps)
        self.attn = _DepthAttention(cfg)
        self.mlp_norm = RMSNorm(d, cfg.norm_eps)
        self.w1 = nn.Linear(d, 4 * d, bias=False)
        self.w3 = nn.Linear(d, 4 * d, bias=False)
        self.w2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, K, Dd] -> [N, K, Dd]."""
        x = x + self.attn(self.attn_norm(x))
        h = self.mlp_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class DepthTransformer(nn.Module):
    """Small causal transformer over the codebook positions of one frame.

    Used with `audio_delay_mode="flat"`: the backbone predicts a whole frame's
    hidden state at once and this module rolls out its n_codebooks codes
    sequentially. Shares the backbone's audio embedding tables (`audio_embs` is
    the SAME ModuleList object, no copies; deduped by nn.Module/safetensors).

    Position-k input (k in 0..n_q-1): `in_proj(h)` for k == 0 else
    `in_proj(audio_embs[k-1](code of codebook k-1))`, plus `pos_emb(k)`.
    Head k reads position k and scores codebook k.
    """

    def __init__(self, cfg: ModelConfig, audio_embs: nn.ModuleList):
        super().__init__()
        assert len(audio_embs) == cfg.n_codebooks, (
            f"need {cfg.n_codebooks} shared audio embedding tables, got {len(audio_embs)}"
        )
        self.cfg = cfg
        self.n_q = cfg.n_codebooks
        self.audio_embs = audio_embs  # shared with the backbone (no copies)
        self.in_proj = nn.Linear(cfg.d_model, cfg.depth_d_model, bias=False)
        self.pos_emb = nn.Embedding(cfg.n_codebooks, cfg.depth_d_model)
        self.blocks = nn.ModuleList(_DepthBlock(cfg) for _ in range(cfg.depth_n_layers))
        self.norm = RMSNorm(cfg.depth_d_model, cfg.norm_eps)
        self.heads = nn.ModuleList(
            nn.Linear(cfg.depth_d_model, cfg.audio_vocab_size, bias=False)
            for _ in range(cfg.n_codebooks)
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Own parameters only (the shared audio_embs belong to the backbone):
        linears normal(0, init_std); block out-projections (attn wo, mlp w2)
        scaled by 1/sqrt(2*depth_n_layers); pos_emb std 0.02; norms 1."""
        std = self.cfg.init_std
        for m in (self.in_proj, *self.heads):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        for block in self.blocks:
            for m in (block.attn.wq, block.attn.wk, block.attn.wv, block.w1, block.w3):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            out_std = std / math.sqrt(2.0 * self.cfg.depth_n_layers)
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=out_std)
            nn.init.normal_(block.w2.weight, mean=0.0, std=out_std)
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def _inputs_from_codes(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """h [N, D] + codes [N, K] (codebooks 0..K-1) -> inputs [N, K+1, Dd]
        for positions 0..K (position j > 0 embeds codebook j-1's code)."""
        cols = [self.in_proj(h)]
        for j in range(codes.shape[1]):
            cols.append(self.in_proj(self.audio_embs[j](codes[:, j])))
        x = torch.stack(cols, dim=1)  # [N, K+1, Dd]
        pos = torch.arange(x.shape[1], device=x.device)
        return x + self.pos_emb(pos)

    def forward(self, h: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Teacher-forced pass over all codebook positions of one frame.

        h: [N, d_model] backbone hidden states (post-final-norm); teacher:
        long [N, n_q], the codebook ids of the frame being predicted (position
        k > 0 consumes teacher[:, k-1]; teacher[:, n_q-1] is never read).
        Returns float32 logits [N, n_q, audio_vocab_size]; head k at index k.
        """
        assert teacher.shape == (h.shape[0], self.n_q), (
            f"teacher must be [N, n_q={self.n_q}], got {tuple(teacher.shape)}"
        )
        x = self._inputs_from_codes(h, teacher[:, : self.n_q - 1])  # [N, n_q, Dd]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = [self.heads[k](x[:, k]) for k in range(self.n_q)]
        return torch.stack(logits, dim=1).float()  # [N, n_q, Va]

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        sample_fn: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> torch.Tensor:
        """Sequential decode of one frame's codebooks.

        h: [B, d_model] backbone hidden (post-final-norm); sample_fn(logits
        [B, audio_vocab_size] fp32, codebook_index) -> long [B]. Returns
        long [B, n_q]. The prefix is re-run per codebook (n_q is tiny), which
        matches `forward`'s causal math exactly.
        """
        B = h.shape[0]
        codes = torch.zeros((B, 0), dtype=torch.long, device=h.device)
        for k in range(self.n_q):
            x = self._inputs_from_codes(h, codes)  # [B, k+1, Dd]
            for block in self.blocks:
                x = block(x)
            logits = self.heads[k](self.norm(x[:, -1])).float()  # [B, Va]
            nxt = sample_fn(logits, k).to(dtype=torch.long, device=h.device)
            codes = torch.cat([codes, nxt.reshape(B, 1)], dim=1)
        return codes
