"""Inference/training performance utilities: regional compile, int8, benchmarks.

Nothing here compiles, quantizes, or allocates at import time. torch.compile is
applied per-Block (regional compile, the FSDP2/AC-friendly order from
docs/DESIGN.md) and is a warned no-op on MPS, where inductor-Metal is immature.
"""

from __future__ import annotations

import time
import warnings

import torch

from ..config import OmniConfig
from ..model.omni import OmniModel
from ..streams import BOS, CHANNEL_ASSISTANT, TEXT_PAD, VOICE, VOICE_END, audio_pad_id


def _sync(device: torch.device) -> None:
    """Block until queued kernels finish so wall-clock timing is honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def apply_compile(model: OmniModel, mode: str = "default") -> OmniModel:
    """Regionally torch.compile each transformer Block in place; returns model.

    Per-Block compile keeps graphs small and composes with activation
    checkpointing / FSDP2 (compile before sharding). On MPS this warns and
    no-ops. Note: compiled blocks are wrapped in OptimizedModule, so state-dict
    keys gain a ``_orig_mod.`` prefix — compile for serving/benchmarks, not
    right before ``save_pretrained``.

    v6 backbone models (HFOmniModel) get the same regional treatment on the
    HF decoder's layer list instead of ``model.blocks``.
    """
    device = next(model.parameters()).device
    if device.type == "mps":
        warnings.warn(
            "torch.compile on MPS is immature; apply_compile is a no-op there",
            stacklevel=2,
        )
        return model
    if hasattr(model, "backbone"):  # HFOmniModel: compile the HF decoder layers
        layers = model.backbone.get_decoder().layers
        for i in range(len(layers)):
            layers[i] = torch.compile(layers[i], mode=mode)
        return model
    for i in range(len(model.blocks)):
        model.blocks[i] = torch.compile(model.blocks[i], mode=mode)
    return model


def quantize_int8(model: OmniModel) -> OmniModel:
    """Int8 weight-only quantization via torchao (lazy import); returns model.

    Inference-only: quantized linears do not support meaningful backward.
    Fast int8 kernels are CUDA-centric; CPU works, MPS support is weak.
    """
    try:
        from torchao.quantization import quantize_
    except ImportError as e:
        raise ImportError(
            "quantize_int8 needs the optional dependency torchao "
            "(pip install torchao)"
        ) from e
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        spec = Int8WeightOnlyConfig()
    except ImportError:  # older torchao spelling
        from torchao.quantization import int8_weight_only

        spec = int8_weight_only()
    quantize_(model, spec)
    return model


@torch.inference_mode()
def benchmark_decode(
    model: OmniModel,
    cfg: OmniConfig,
    device: str | torch.device,
    *,
    n_frames: int = 100,
    batch: int = 1,
    voice_frames: int = 0,
) -> dict:
    """Time the frame decode loop: prefill a tiny random prompt, then step.

    Feeds each step the argmax of the previous logits (tokens [batch, S] stay
    on device; no host sync inside the timed loop). Depth models
    (cfg.use_depth) route through prefill_hidden/step_hidden + a greedy
    ``depth.sample``; duplex user rows are fed AUDIO_PAD (input-only). Real
    time factor ``rtf = steps_per_s / 12.5`` (>1 means faster than the
    12.5 Hz frame clock). Returns {"steps_per_s", "rtf", "ms_per_step",
    "prefill_ms", "n_frames", "batch"}.

    voice_frames > 0 (DESIGN_V5): a random ``[<bos> + <voice> + R frames +
    <voice_end>]`` block (R + 2 cols, voice-segment geometry) is prefilled
    into the SAME cache first as its own timed call, adding
    ``"voice_prefill_ms"`` to the result. ``prefill_ms`` and the decode keys
    keep their meaning but run with the prefix in the KV, so comparing runs
    with voice_frames=0 vs >0 measures the per-step cost of the prefix
    (DESIGN_V5 gate: prefill(127) < 40 ms, step delta < 1 ms at quality).
    """
    assert batch >= 1 and n_frames >= 1 and voice_frames >= 0
    device = torch.device(device)
    model = model.to(device).eval()
    mc = cfg.model
    dtype = next(model.parameters()).dtype

    t0 = min(8, max(1, mc.max_frames // 4))  # tiny prompt
    warmup = 3
    v_cols = voice_frames + 2 if voice_frames > 0 else 0  # <bos> + segment
    n_frames = min(n_frames, mc.max_frames - t0 - warmup - v_cols)
    if n_frames < 1:
        raise ValueError(
            f"model.max_frames={mc.max_frames} too small to benchmark "
            f"(prompt {t0} + warmup {warmup} + voice block {v_cols})"
        )

    g = torch.Generator().manual_seed(0)
    text = torch.randint(0, mc.text_vocab_size, (batch, 1, t0), generator=g)
    audio = torch.randint(0, mc.audio_vocab_size, (batch, mc.n_streams - 1, t0), generator=g)
    grid = torch.cat([text, audio], dim=1).to(device)  # [B, S, t0]
    channel = torch.zeros((batch, t0), dtype=torch.long, device=device)
    chan1 = torch.ones((batch,), dtype=torch.long, device=device)
    # Duplex user rows are input-only: feed them AUDIO_PAD in the decode loop.
    user_pad = None
    if mc.duplex:
        user_pad = torch.full(
            (batch, mc.n_codebooks), audio_pad_id(mc.audio_codec_vocab),
            dtype=torch.long, device=device,
        )
    vgrid = vchan = None
    if voice_frames > 0:
        # Voice-prefix block [B, S, R+2] in grids.voice_segment geometry:
        # text <bos> <voice> TEXT_PAD.. <voice_end>; random raw codes on the
        # assistant rows under cols 1..R; APAD elsewhere (duplex user rows
        # too); channel CHANNEL_ASSISTANT.
        apad = audio_pad_id(mc.audio_codec_vocab)
        vtext = torch.full((batch, 1, v_cols), TEXT_PAD, dtype=torch.long)
        vtext[:, 0, 0] = BOS
        vtext[:, 0, 1] = VOICE
        vtext[:, 0, -1] = VOICE_END
        vaudio = torch.full((batch, mc.n_streams - 1, v_cols), apad, dtype=torch.long)
        vaudio[:, : mc.n_codebooks, 1:-1] = torch.randint(
            0, mc.audio_codec_vocab, (batch, mc.n_codebooks, voice_frames), generator=g
        )
        vgrid = torch.cat([vtext, vaudio], dim=1).to(device)  # [B, S, R+2]
        vchan = torch.full(
            (batch, v_cols), CHANNEL_ASSISTANT, dtype=torch.long, device=device
        )

    if mc.use_depth:
        prefill_fn, step_fn = model.prefill_hidden, model.step_hidden

        def next_tokens(tl: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
            # aux = hidden [B, D]: greedy sequential depth decode -> [B, n_q]
            codes = model.depth.sample(aux, lambda lg, k: lg.argmax(-1))
            cols = [tl.argmax(-1, keepdim=True), codes]
            if user_pad is not None:
                cols.append(user_pad)
            return torch.cat(cols, dim=1)  # greedy input column [B, S]
    else:
        prefill_fn, step_fn = model.prefill, model.step

        def next_tokens(tl: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
            # aux = parallel audio-head logits [B, n_q, Va]
            cols = [tl.argmax(-1, keepdim=True), aux.argmax(-1)]
            if user_pad is not None:
                cols.append(user_pad)
            return torch.cat(cols, dim=1)  # greedy input column [B, S]

    cache = model.new_cache(batch, device, dtype)
    voice_prefill_ms = 0.0
    if vgrid is not None:
        # One-time voice-prefix prefill (its logits are irrelevant); the
        # prompt prefill below then appends at cache.pos = R + 2 (the
        # attention layer position-aligns chunked prefills).
        _sync(device)
        tic = time.perf_counter()
        prefill_fn(vgrid, vchan, cache)
        _sync(device)
        voice_prefill_ms = (time.perf_counter() - tic) * 1e3
    _sync(device)
    tic = time.perf_counter()
    tl, aux = prefill_fn(grid, channel, cache)
    _sync(device)
    prefill_ms = (time.perf_counter() - tic) * 1e3

    for _ in range(warmup):  # also absorbs torch.compile of the step shape
        tl, aux = step_fn(next_tokens(tl, aux), chan1, cache)
    _sync(device)
    tic = time.perf_counter()
    for _ in range(n_frames):
        tl, aux = step_fn(next_tokens(tl, aux), chan1, cache)
    _sync(device)
    dt = time.perf_counter() - tic

    steps_per_s = n_frames / dt
    out = {
        "steps_per_s": steps_per_s,
        "rtf": steps_per_s / 12.5,
        "ms_per_step": 1e3 * dt / n_frames,
        "prefill_ms": prefill_ms,
        "n_frames": float(n_frames),
        "batch": float(batch),
    }
    if voice_frames > 0:
        out["voice_prefill_ms"] = voice_prefill_ms
    return out


def benchmark_forward(
    model: OmniModel,
    cfg: OmniConfig,
    device: str | torch.device,
    *,
    batch: int,
    frames: int,
    steps: int = 10,
) -> dict:
    """Time full-sequence forward+backward on a random DELAYED batch.

    grid [batch, S, frames], all-True loss mask. ``tokens_per_s`` counts grid
    positions (batch * frames * steps / elapsed); each position carries
    n_streams stream tokens. Peak memory is reported on CUDA only
    ("peak_mem_mb"). The model is left in eval mode with grads cleared.
    """
    assert batch >= 1 and steps >= 1
    device = torch.device(device)
    model = model.to(device).train()
    mc = cfg.model
    frames = min(frames, mc.max_frames)
    assert frames >= 2, "need at least 2 frames for next-token loss"

    g = torch.Generator().manual_seed(0)
    text = torch.randint(0, mc.text_vocab_size, (batch, 1, frames), generator=g)
    audio = torch.randint(0, mc.audio_vocab_size, (batch, mc.n_streams - 1, frames), generator=g)
    grid = torch.cat([text, audio], dim=1).to(device)  # [B, S, T]
    channel = torch.randint(0, 2, (batch, frames), generator=g).to(device)
    loss_mask = torch.ones((batch, mc.n_streams, frames), dtype=torch.bool, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def one_iter() -> None:
        out = model(grid, channel)
        loss, _ = model.loss(out, grid, loss_mask)
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(2):  # warmup (also absorbs compile)
        one_iter()
    _sync(device)
    tic = time.perf_counter()
    for _ in range(steps):
        one_iter()
    _sync(device)
    dt = time.perf_counter() - tic

    model.zero_grad(set_to_none=True)
    model.eval()
    out = {
        "tokens_per_s": batch * frames * steps / dt,
        "ms_per_iter": 1e3 * dt / steps,
        "batch": float(batch),
        "frames": float(frames),
        "steps": float(steps),
    }
    if device.type == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
    return out
