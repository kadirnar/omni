"""GPU-day preflight: contracts that CANNOT be verified offline.

Every test here downloads weights and/or needs CUDA — run on the training
node BEFORE bulk data prep or the first torchrun:

    RUN_SLOW=1 .venv/bin/pytest tests/test_gpu_preflight.py -v

These pin the assumptions the offline suite takes on faith (2026-07 review):
FakeCodec's frame math must match the real Mimi, the real backbone tokenizer
must round-trip through HFTextTokenizer, and a real backbone must build+step.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def mimi():
    from omni.audio.codec import MimiCodec

    return MimiCodec(n_codebooks=8)  # downloads kyutai/mimi on first use


def test_mimi_frame_count_matches_fakecodec_contract(mimi) -> None:
    """FakeCodec pins T_frames = ceil(T_samples / 1920); prep alignment math
    assumes Mimi agrees for non-multiple-of-1920 inputs."""
    spf = mimi.samples_per_frame
    assert spf == 1920 and mimi.sample_rate == 24_000 and mimi.frame_rate == 12.5
    for n in (1, spf - 1, spf, spf + 1, 3 * spf, 3 * spf + 7):
        wav = torch.sin(torch.linspace(0, 100, n))
        codes = mimi.encode(wav)
        assert codes.shape == (8, math.ceil(n / spf)), (
            f"Mimi frame count for {n} samples diverges from ceil(T/1920) — "
            "FakeCodec-prepared alignment math would shift when swapping codecs"
        )
        assert int(codes.min()) >= 0 and int(codes.max()) < mimi.codec_vocab


def test_mimi_num_quantizers_kwarg(mimi) -> None:
    """n_codebooks must actually limit the quantizers (kwarg name drift in
    transformers would silently return all 32)."""
    wav = torch.sin(torch.linspace(0, 100, 4 * 1920))
    assert mimi.encode(wav).shape[0] == 8
    from omni.audio.codec import MimiCodec

    m32 = MimiCodec(n_codebooks=32)
    assert m32.encode(wav).shape[0] == 32


def test_mimi_decode_roundtrip_shape(mimi) -> None:
    wav = torch.sin(torch.linspace(0, 100, 5 * 1920))
    codes = mimi.encode(wav)
    out = mimi.decode(codes)
    assert out.ndim == 1 and abs(out.shape[0] - wav.shape[0]) <= 2 * 1920


def test_streaming_encoder_window_close_to_batch(mimi) -> None:
    """The serve-side rolling-context encoder should approximate batch codes
    far better than stateless per-chunk encoding (duplex train/serve gap)."""
    from omni.serve.streaming import StreamingEncoder

    spf = mimi.samples_per_frame
    torch.manual_seed(0)
    wav = torch.randn(25 * spf).clamp(-1, 1) * 0.3
    batch = mimi.encode(wav)  # [8, 25]

    enc = StreamingEncoder(mimi, context_s=2.0)
    windowed = torch.stack(
        [enc.feed(wav[i * spf : (i + 1) * spf]) for i in range(25)], dim=1
    )
    stateless = torch.stack(
        [mimi.encode(wav[i * spf : (i + 1) * spf])[:, 0] for i in range(25)], dim=1
    )
    # compare agreement on the semantic codebook after the context fills
    tail = slice(5, 25)
    agree_windowed = (windowed[0, tail] == batch[0, tail]).float().mean()
    agree_stateless = (stateless[0, tail] == batch[0, tail]).float().mean()
    assert agree_windowed >= agree_stateless, (
        f"rolling window ({agree_windowed:.2f}) should beat stateless "
        f"({agree_stateless:.2f}) at matching training-time codes"
    )


def test_hf_tokenizer_real_backbone_roundtrip() -> None:
    """HFTextTokenizer against the real Qwen3 tokenizer: +64 offset, specials
    reserved, shard v2 vocab sizing."""
    from transformers import AutoTokenizer

    from omni.text.tokenizer import HFTextTokenizer

    hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    tok = HFTextTokenizer(hf_tok, model_id="Qwen/Qwen3-1.7B-Base")
    assert tok.vocab_size > 65_536, "expected a >uint16 vocab (shard v2 territory)"
    ids = tok.encode("merhaba dünya, hello world")
    assert ids and all(i >= 64 for i in ids)
    assert "hello world" in tok.decode(ids)


def test_real_backbone_builds_and_steps() -> None:
    """qwen3-1.7b preset: build, adopt cfg, one forward + one cached step."""
    from omni.config import load_config
    from omni.model import build_model

    cfg = load_config("qwen3-1.7b", ["model.n_codebooks=8"])
    model = build_model(cfg.model)  # downloads the backbone
    cfg.model = model.cfg
    S = 1 + cfg.model.n_codebooks
    grid = torch.zeros(1, S, 4, dtype=torch.long)
    grid[:, 0] = 70  # a real text id
    grid[:, 1:] = 5
    channel = torch.zeros(1, 4, dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        out = model(grid, channel)
        assert out.text_logits.shape[-1] == cfg.model.text_vocab_size
        cache = model.new_cache(1, "cpu", next(model.parameters()).dtype)
        tl, h = model.prefill_hidden(grid, channel, cache)
        assert tl.shape == (1, cfg.model.text_vocab_size)
        assert h.shape[-1] == cfg.model.d_model
