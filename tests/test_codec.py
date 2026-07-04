"""Tests for omni.audio.codec: FakeCodec behavior and wav io (offline only)."""

from __future__ import annotations

import math

import pytest
import torch

from omni.audio.codec import AudioCodec, FakeCodec, build_codec, load_wav, save_wav

SR = 24_000
SPF = 1920  # samples per frame at 24 kHz / 12.5 Hz


def _sine(n_frames: int, freq: float = 220.0) -> torch.Tensor:
    """Mono float32 sine [n_frames * SPF] in [-0.5, 0.5]."""
    t = torch.arange(n_frames * SPF, dtype=torch.float32) / SR
    return 0.5 * torch.sin(2 * math.pi * freq * t)


def test_fake_codec_attrs(fake_codec: FakeCodec) -> None:
    assert isinstance(fake_codec, AudioCodec)
    assert fake_codec.sample_rate == SR
    assert fake_codec.frame_rate == 12.5
    assert fake_codec.n_codebooks == 2
    assert fake_codec.codec_vocab == 2048


def test_encode_shapes_unbatched(fake_codec: FakeCodec) -> None:
    codes = fake_codec.encode(_sine(5))
    assert codes.dtype == torch.long
    assert codes.shape == (2, 5)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 2048


def test_encode_decode_batched(fake_codec: FakeCodec) -> None:
    wav = torch.stack([_sine(4, 220.0), _sine(4, 550.0), _sine(4, 990.0)])
    codes = fake_codec.encode(wav)
    assert codes.shape == (3, 2, 4)
    out = fake_codec.decode(codes)
    assert out.shape == (3, 4 * SPF)
    assert out.dtype == torch.float32


def test_encode_deterministic(fake_codec: FakeCodec) -> None:
    wav = _sine(6, 330.0)
    c1 = fake_codec.encode(wav)
    c2 = fake_codec.encode(wav)
    assert torch.equal(c1, c2), "same wav must give same codes"
    fresh = FakeCodec(n_codebooks=2, codec_vocab=2048)
    assert torch.equal(fresh.encode(wav), c1), "determinism across instances"
    other = fake_codec.encode(_sine(6, 700.0))
    assert not torch.equal(other, c1), "different wav should give different codes"


def test_decode_length_and_signal(fake_codec: FakeCodec) -> None:
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 2048, (2, 7), generator=g)
    wav = fake_codec.decode(codes)
    assert wav.shape == (7 * SPF,), "decode length must be frames * 1920"
    assert wav.dtype == torch.float32
    assert torch.isfinite(wav).all()
    assert float(wav.abs().max()) <= 1.0 + 1e-6, "decode output is clamped to [-1, 1]"
    assert float(wav.abs().max()) > 0.0, "decode output must be nonzero"


def test_build_codec_fake() -> None:
    c = build_codec("fake", n_codebooks=2)
    assert isinstance(c, AudioCodec)
    assert c.n_codebooks == 2
    assert c.codec_vocab == 2048
    with pytest.raises(Exception):
        build_codec("not-a-codec")


def test_wav_save_load_roundtrip(tmp_path) -> None:
    wav = _sine(5, 440.0)
    p = tmp_path / "x.wav"
    save_wav(p, wav, SR)
    assert p.exists()
    back = load_wav(p, SR)
    assert back.dtype == torch.float32
    assert back.ndim == 1
    assert back.shape[0] == wav.shape[0]
    assert torch.allclose(back, wav, atol=2e-3)
    # linear resample to half rate roughly halves the length
    half = load_wav(p, SR // 2)
    assert abs(half.shape[0] - wav.shape[0] // 2) <= 2
    assert torch.isfinite(half).all()
