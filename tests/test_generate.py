"""Tests for omni.infer.generate: sample_logits, task smoke tests, reproducibility."""

from __future__ import annotations

import math

import pytest
import torch

from omni.grids import build_tts_prompt, prompt_forced_text
from omni.infer.generate import GenResult, OmniGenerator, sample_logits
from omni.model.omni import OmniModel

SPF = 1920


@pytest.fixture(scope="module")
def tiny_gen(test_cfg, byte_tok) -> OmniGenerator:
    torch.manual_seed(0)
    model = OmniModel(test_cfg.model)
    model.init_weights()
    model.eval()
    return OmniGenerator(model, test_cfg, device="cpu", tokenizer=byte_tok)


def _sine_wav(n_frames: int, freq: float = 220.0) -> torch.Tensor:
    t = torch.arange(n_frames * SPF, dtype=torch.float32) / 24_000
    return 0.5 * torch.sin(2 * math.pi * freq * t)


def test_sample_logits_argmax() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 9, generator=g)
    out = sample_logits(logits, 0.0, 5)
    assert out.dtype == torch.long
    assert out.shape == (5,)
    assert torch.equal(out, logits.argmax(-1)), "temperature <= 0 means argmax"
    assert torch.equal(sample_logits(logits, -1.0, 3), logits.argmax(-1))
    one = sample_logits(logits[0], 0.0, 4)
    assert one.shape == ()
    assert int(one) == int(logits[0].argmax())
    # top_k=1 is argmax regardless of temperature
    got = sample_logits(logits, 1.0, 1, torch.Generator().manual_seed(1))
    assert torch.equal(got, logits.argmax(-1))


def test_sample_logits_seeded_and_topk() -> None:
    logits = torch.randn(64, 33, generator=torch.Generator().manual_seed(2))
    a = sample_logits(logits, 0.9, 10, torch.Generator().manual_seed(7))
    b = sample_logits(logits, 0.9, 10, torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "same generator seed must reproduce samples"
    assert int(a.min()) >= 0 and int(a.max()) < 33
    # samples stay within the top-k support
    k3 = sample_logits(logits, 1.0, 3, torch.Generator().manual_seed(3))
    topk = logits.topk(3, dim=-1).indices
    assert (k3.unsqueeze(-1) == topk).any(-1).all()


def test_tts_smoke(tiny_gen, fake_codec, test_cfg) -> None:
    r = tiny_gen.tts("hello world", fake_codec, max_frames=12, seed=0)
    assert isinstance(r, GenResult)
    assert r.audio_codes.dtype == torch.long
    assert r.audio_codes.ndim == 2
    assert r.audio_codes.shape[0] == test_cfg.model.n_codebooks
    assert r.frames == r.audio_codes.shape[1]
    assert 1 <= r.frames <= 12, "must stop by max_frames"
    assert int(r.audio_codes.min()) >= 0
    assert int(r.audio_codes.max()) < test_cfg.model.audio_codec_vocab, (
        "audio_codes must be raw codes (specials stripped)"
    )
    wav = fake_codec.decode(r.audio_codes)
    assert wav.shape == (r.frames * SPF,)
    assert torch.isfinite(wav).all()


def test_asr_smoke(tiny_gen, fake_codec) -> None:
    r = tiny_gen.asr(_sine_wav(8), fake_codec, max_frames=8, seed=0)
    assert isinstance(r, GenResult)
    assert isinstance(r.text, str), "asr with a tokenizer must return a transcript str"
    assert isinstance(r.text_ids, list)
    assert all(isinstance(i, int) for i in r.text_ids)
    assert r.audio_codes.ndim == 2


def test_s2s_smoke(tiny_gen, fake_codec, test_cfg) -> None:
    r = tiny_gen.s2s(_sine_wav(6), fake_codec, max_frames=10, seed=1)
    assert isinstance(r, GenResult)
    assert r.audio_codes.shape[0] == test_cfg.model.n_codebooks
    assert r.frames == r.audio_codes.shape[1]
    assert r.frames <= 10
    if r.frames:
        assert int(r.audio_codes.min()) >= 0
        assert int(r.audio_codes.max()) < test_cfg.model.audio_codec_vocab
        wav = fake_codec.decode(r.audio_codes)
        assert torch.isfinite(wav).all()


def test_generate_seeded_reproducible(tiny_gen, fake_codec) -> None:
    # max_frames must fit the whole forced monologue (20 ids) — shorter
    # budgets now raise instead of silently speaking a truncated sentence
    r1 = tiny_gen.tts("say something nice", fake_codec, max_frames=24, seed=7)
    r2 = tiny_gen.tts("say something nice", fake_codec, max_frames=24, seed=7)
    assert r1.text_ids == r2.text_ids
    assert r1.frames == r2.frames
    assert torch.equal(r1.audio_codes, r2.audio_codes)
    r3 = tiny_gen.tts("say something nice", fake_codec, max_frames=24, seed=8)
    assert (
        r3.audio_codes.shape != r1.audio_codes.shape
        or not torch.equal(r3.audio_codes, r1.audio_codes)
    ), "different seed should change sampled audio"


def test_generate_direct_prompt(tiny_gen, byte_tok, test_cfg) -> None:
    mcfg = test_cfg.model
    prompt = build_tts_prompt(mcfg.n_codebooks, mcfg.audio_codec_vocab)
    forced = prompt_forced_text("tts", byte_tok.encode("hi"))
    r = tiny_gen.generate(prompt, forced, max_frames=6, seed=0)
    assert isinstance(r, GenResult)
    assert r.frames <= 6
    assert r.audio_codes.shape[0] == mcfg.n_codebooks
