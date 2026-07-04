"""Tests for the 2026-07 review fixes in the data layer: shard format v2,
control-id filtering, weighted RESAMPLING mixtures, anti-aliased resampling,
vocab-mismatch handling, truncation tag hygiene."""

from __future__ import annotations

import math

import pytest
import torch

from omni.audio.codec import resample
from omni.config import load_config
from omni.data.dataset import MixDataset, ShardDataset, _check_meta, build_dataloader
from omni.data.prepare import ShardWriter, _encode_text, _truncate_turn, prepare_fake
from omni.data.synthesize import word_frames_from_alignment
from omni.streams import EMO_PCV, EMO_RSP, Sample


def _sample(n_q: int = 2, T: int = 8, cv: int = 2048, text_ids=None, seed: int = 0) -> Sample:
    g = torch.Generator().manual_seed(seed)
    grid = torch.zeros((1 + n_q, T), dtype=torch.long)
    grid[0] = torch.randint(64, 320, (T,), generator=g)
    if text_ids is not None:
        grid[0, : len(text_ids)] = torch.tensor(text_ids, dtype=torch.long)
    grid[1:] = torch.randint(0, cv, (n_q, T), generator=g)
    return Sample(
        grid=grid,
        loss_mask=torch.ones_like(grid, dtype=torch.bool),
        channel=torch.zeros(T, dtype=torch.long),
        task="tts",
    )


class _StubTok:
    """Deterministic stand-in whose encode can emit reserved ids (< 64)."""

    vocab_size = 200_000

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for w in text.split():
            out.append(4 if w == "<user>" else 2 if w == "<eos>" else 64 + (hash(w) % 1000))
        return out


# ---------------------------------------------------------------------------
# Shard format v2
# ---------------------------------------------------------------------------
def test_shard_v2_roundtrip_big_text_vocab(tmp_path) -> None:
    """text_vocab_size > 65536 (every v6 backbone tokenizer) must write v2
    shards whose uint32 text row round-trips ids far above uint16."""
    big_id = 151_936  # a Qwen3-range text id
    with ShardWriter(
        tmp_path, n_codebooks=2, codec_vocab=2048, text_vocab_size=200_000
    ) as w:
        assert w.version == 2
        samples = [_sample(text_ids=[big_id, 70_000, 64], seed=i) for i in range(3)]
        for s in samples:
            w.add(s)
    ds = ShardDataset(tmp_path)
    assert int(ds.meta["version"]) == 2
    assert len(ds) == 3
    for i, want in enumerate(samples):
        got = ds[i]
        assert torch.equal(got.grid, want.grid)
        assert torch.equal(got.loss_mask, want.loss_mask)
        assert torch.equal(got.channel, want.channel)
    assert int(ds[0].grid[0, 0]) == big_id


def test_shard_v1_kept_for_small_vocabs(tmp_path) -> None:
    with ShardWriter(
        tmp_path, n_codebooks=2, codec_vocab=2048, text_vocab_size=320
    ) as w:
        assert w.version == 1
        w.add(_sample())
    ds = ShardDataset(tmp_path)
    assert int(ds.meta["version"]) == 1
    assert torch.equal(ds[0].grid, _sample().grid)


# ---------------------------------------------------------------------------
# Control-id filtering
# ---------------------------------------------------------------------------
def test_encode_text_drops_reserved_ids() -> None:
    ids = _encode_text(_StubTok(), "say <user> hi <eos> there")
    assert all(i >= 64 for i in ids)
    assert len(ids) == 3  # say / hi / there survive


def test_word_frames_filter_reserved_ids() -> None:
    alignment = [(0.0, 0.1, "<user>"), (0.2, 0.3, "hello")]
    wf = word_frames_from_alignment(alignment, _StubTok(), frame_rate=12.5)
    # the "<user>" word tokenizes entirely to a reserved id -> dropped
    assert len(wf) == 1
    assert all(i >= 64 for _f, ids in wf for i in ids)


# ---------------------------------------------------------------------------
# Weighted mixtures resample (survive any shuffling)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_dirs(tmp_path_factory):
    cfg = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320"])
    a = tmp_path_factory.mktemp("mix_a")
    b = tmp_path_factory.mktemp("mix_b")
    prepare_fake(a, n_samples=30, cfg=cfg, seed=0)
    prepare_fake(b, n_samples=10, cfg=cfg, seed=1)
    return cfg, a, b


def test_mix_weights_are_proportions(two_dirs) -> None:
    _cfg, a, b = two_dirs
    da, db = ShardDataset(a), ShardDataset(b)
    mix = MixDataset([da, db], [0.9, 0.1], seed=0)
    assert len(mix) == 40
    assert mix.counts == [36, 4]  # 0.9/0.1 of L=40, NOT the 30/10 natural sizes
    # multiplicity (not order) carries the proportions -> any shuffle keeps them
    ds_hist = torch.bincount(torch.tensor([int(d) for d in mix._ds_of]), minlength=2)
    assert ds_hist.tolist() == [36, 4]
    # dataset a (30 samples -> 36 entries) repeats: one full pass + 6 extras
    la = [int(j) for d, j in zip(mix._ds_of, mix._local) if int(d) == 0]
    assert len(la) == 36 and len(set(la)) == 30
    # deterministic
    mix2 = MixDataset([ShardDataset(a), ShardDataset(b)], [0.9, 0.1], seed=0)
    assert mix2.counts == mix.counts
    assert [int(x) for x in mix2._local] == [int(x) for x in mix._local]


def test_mix_natural_when_unweighted(two_dirs) -> None:
    _cfg, a, b = two_dirs
    mix = MixDataset([ShardDataset(a), ShardDataset(b)], None, seed=0)
    assert mix.counts == [30, 10]
    assert len(mix) == 40


def test_build_dataloader_list_vs_dict(two_dirs) -> None:
    cfg, a, b = two_dirs
    cfg.data.num_workers = 0
    cfg.data.batch_size = 2
    dl = build_dataloader(cfg, [str(a), str(b)])
    assert dl.dataset.counts == [30, 10]
    dl = build_dataloader(cfg, {str(a): 0.5, str(b): 0.5})
    assert dl.dataset.counts == [20, 20]


# ---------------------------------------------------------------------------
# Anti-aliased resampling
# ---------------------------------------------------------------------------
def _tone(freq: float, sr: int, secs: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(sr * secs), dtype=torch.float64) / sr
    return torch.sin(2 * math.pi * freq * t).to(torch.float32)


def test_resample_kills_aliasing() -> None:
    """A 15 kHz tone at 48 kHz is above the 24 kHz Nyquist: linear
    interpolation folded it to 9 kHz; the sinc resampler must suppress it."""
    out = resample(_tone(15_000, 48_000), 48_000, 24_000)
    spec = torch.fft.rfft(out.double()).abs()
    freqs = torch.fft.rfftfreq(out.shape[0], d=1 / 24_000)
    alias_band = spec[(freqs > 8_500) & (freqs < 9_500)].max()
    assert float(alias_band) < 0.01 * out.shape[0] / 2  # << a full-scale tone


def test_resample_preserves_passband_and_length() -> None:
    x = _tone(1_000, 48_000)
    y = resample(x, 48_000, 24_000)
    assert y.shape[0] == math.ceil(x.shape[0] * 24_000 / 48_000)
    spec = torch.fft.rfft(y.double()).abs() * 2 / y.shape[0]
    freqs = torch.fft.rfftfreq(y.shape[0], d=1 / 24_000)
    peak = spec[(freqs > 900) & (freqs < 1_100)].max()
    assert 0.97 < float(peak) < 1.03  # amplitude preserved
    # identity and empty passthrough
    assert torch.equal(resample(x, 48_000, 48_000), x)
    assert resample(torch.zeros(0), 48_000, 24_000).shape[0] == 0
    # deterministic
    assert torch.equal(resample(x, 48_000, 24_000), y)


def test_resample_odd_ratio() -> None:
    """44.1 kHz -> 24 kHz exercises a non-trivial L/M rational ratio."""
    x = _tone(2_000, 44_100)
    y = resample(x, 44_100, 24_000)
    assert y.shape[0] == math.ceil(x.shape[0] * 24_000 / 44_100)
    spec = torch.fft.rfft(y.double()).abs() * 2 / y.shape[0]
    freqs = torch.fft.rfftfreq(y.shape[0], d=1 / 24_000)
    assert 0.95 < float(spec[(freqs > 1_900) & (freqs < 2_100)].max()) < 1.05


# ---------------------------------------------------------------------------
# Turn truncation never strands a dangling emotion marker
# ---------------------------------------------------------------------------
def test_truncate_turn_strips_dangling_marker() -> None:
    n_q = 2
    user = torch.zeros((n_q, 10), dtype=torch.long)
    asst = torch.zeros((n_q, 5), dtype=torch.long)
    ids = [37, EMO_PCV, 18, EMO_RSP, 25, 100, 101]  # lang, pcv+cls, rsp+cls, text
    out = _truncate_turn(user, ids, asst, budget=14)
    assert out is not None
    _u, text, _a = out
    # the raw cut ([:4]) would end on the bare EMO_RSP marker; it must be gone
    assert text == [37, EMO_PCV, 18]


# ---------------------------------------------------------------------------
# Vocab-size mismatch handling (the padded-embedding backbone trap)
# ---------------------------------------------------------------------------
def test_meta_vocab_relaxed_only_for_backbones() -> None:
    meta = {"n_codebooks": 8, "codec_vocab": 2048, "text_vocab_size": 151_936}
    hf = load_config("tiny", ["model.backbone_id=stub/backbone", "model.text_vocab_size=152000"])
    _check_meta(meta, hf, "dir")  # smaller meta accepted: ids all in range
    scratch = load_config("tiny", ["model.text_vocab_size=152000"])
    with pytest.raises(ValueError, match="text_vocab_size"):
        _check_meta(meta, scratch, "dir")  # from-scratch models stay strict
    too_big = {**meta, "text_vocab_size": 160_000}
    with pytest.raises(ValueError, match="text_vocab_size"):
        _check_meta(too_big, hf, "dir")  # larger meta = out-of-range ids


def test_mixed_dirs_must_share_tokenizer(tmp_path) -> None:
    cfg = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320"])
    cfg.data.num_workers = 0
    cfg.data.batch_size = 2
    dirs = []
    for i, tok_id in enumerate(["hf:qwen", "hf:llama"]):
        d = tmp_path / f"d{i}"
        with ShardWriter(
            d, n_codebooks=2, codec_vocab=2048, text_vocab_size=320, tokenizer_id=tok_id
        ) as w:
            for j in range(4):
                w.add(_sample(seed=j))
        dirs.append(str(d))
    with pytest.raises(ValueError, match="tokenizer_id"):
        build_dataloader(cfg, dirs)
