"""Tests for omni.data: shard roundtrip, collate, samplers, dataloader."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from omni.data.dataset import (
    BucketBatchSampler,
    MixDataset,
    ShardDataset,
    build_dataloader,
    collate,
)
from omni.data.prepare import ShardWriter
from omni.grids import build_asr, build_audiolm, build_s2s, build_textlm, build_tts
from omni.streams import PAD, Sample, apply_delay, audio_pad_id

NQ, CV, TV = 2, 2048, 320
TASKS = {"textlm", "alm", "asr", "tts", "s2s"}


def _codes(T: int, base: int = 0) -> torch.Tensor:
    return (torch.arange(NQ * T, dtype=torch.long).reshape(NQ, T) + base) % CV


def _five_samples() -> list[Sample]:
    return [
        build_textlm([100, 101, 102, 103], NQ, CV),
        build_audiolm(_codes(6, 3), NQ, CV),
        build_asr(_codes(5, 9), [120, 121], NQ, CV),
        build_tts([130, 131, 132], _codes(7, 1), NQ, CV),
        build_s2s([(_codes(4, 2), [140, 141], _codes(5, 6))], NQ, CV),
    ]


def test_shard_writer_dataset_roundtrip(tmp_path) -> None:
    samples = _five_samples()
    writer = ShardWriter(tmp_path, n_codebooks=NQ, codec_vocab=CV, text_vocab_size=TV)
    for s in samples:
        writer.add(s)
    writer.close()

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["version"] == 1
    assert meta["n_codebooks"] == NQ
    assert meta["codec_vocab"] == CV
    assert meta["text_vocab_size"] == TV
    assert meta["n_samples"] == len(samples)
    assert meta["n_shards"] >= 1

    ds = ShardDataset(tmp_path)
    assert len(ds) == len(samples)
    assert list(ds.frames) == [s.n_frames for s in samples]
    for i, s in enumerate(samples):
        r = ds[i]
        assert isinstance(r, Sample)
        assert r.grid.dtype == torch.long and r.loss_mask.dtype == torch.bool
        assert torch.equal(r.grid, s.grid)
        assert torch.equal(r.loss_mask, s.loss_mask)
        assert torch.equal(r.channel, s.channel)
        assert r.task == s.task


def test_fake_shards_dataset(fake_shards, test_cfg) -> None:
    ds = ShardDataset(fake_shards)
    assert len(ds) == 24
    assert ds.meta["n_samples"] == 24
    assert ds.meta["n_codebooks"] == test_cfg.model.n_codebooks
    assert ds.meta["codec_vocab"] == test_cfg.model.audio_codec_vocab
    assert ds.meta["text_vocab_size"] == test_cfg.model.text_vocab_size
    assert len(ds.frames) == 24
    tasks = set()
    for i in range(len(ds)):
        s = ds[i]
        s.validate(test_cfg.model.audio_codec_vocab, test_cfg.model.text_vocab_size)
        assert s.n_frames == ds.frames[i]
        assert s.n_frames <= test_cfg.data.max_sample_frames
        tasks.add(s.task)
    assert tasks <= TASKS
    assert len(tasks) >= 2, "prepare_fake should mix tasks"


def test_collate_shapes_and_padding(test_cfg) -> None:
    apad = audio_pad_id(CV)
    a = build_tts([130, 131, 132], _codes(7, 1), NQ, CV)  # T = 11
    b = build_asr(_codes(5, 9), [120, 121], NQ, CV)       # T = 12 (longest)
    # short hand-built sample whose channel ends on USER to pin channel padding
    g = torch.zeros((1 + NQ, 5), dtype=torch.long)
    g[0] = torch.tensor([64, 65, 66, 67, 68])
    g[1:] = _codes(5, 4)
    c = Sample(
        grid=g,
        loss_mask=torch.ones_like(g, dtype=torch.bool),
        channel=torch.tensor([1, 1, 0, 0, 0]),
        task="tts",
    )
    batch = collate([a, b, c], CV)
    grid, mask, channel = batch["grid"], batch["loss_mask"], batch["channel"]
    Tmax = max(s.n_frames for s in (a, b, c)) + NQ  # +delay slack
    assert grid.shape == (3, 1 + NQ, Tmax) and grid.dtype == torch.long
    assert mask.shape == (3, 1 + NQ, Tmax) and mask.dtype == torch.bool
    assert channel.shape == (3, Tmax) and channel.dtype == torch.long
    for i, s in enumerate((a, b, c)):
        dg, dm, dc = apply_delay(s.grid, s.loss_mask, s.channel, CV)
        L = dg.shape[1]
        assert torch.equal(grid[i, :, :L], dg), f"sample {i}: delayed prefix mismatch"
        assert torch.equal(mask[i, :, :L], dm)
        assert torch.equal(channel[i, :L], dc)
        assert (grid[i, 0, L:] == PAD).all(), "text padding must be PAD"
        assert (grid[i, 1:, L:] == apad).all(), "audio padding must be AUDIO_PAD"
        assert not mask[i, :, L:].any(), "padding is never a loss target"
        assert (channel[i, L:] == dc[-1]).all(), "channel padding repeats last value"


def _flat(batches: list[list[int]]) -> list[int]:
    return [int(i) for b in batches for i in b]


def test_bucket_sampler_deterministic_and_epoch_reshuffle() -> None:
    lengths = [10, 11, 12, 13] * 3 + [100, 101, 102, 103] * 3  # 24 lengths
    s1 = BucketBatchSampler(lengths, 2, shuffle=True, seed=7)
    s2 = BucketBatchSampler(lengths, 2, shuffle=True, seed=7)
    s1.set_epoch(0)
    s2.set_epoch(0)
    b1 = [list(b) for b in s1]
    b2 = [list(b) for b in s2]
    assert b1 == b2, "same seed + epoch must reproduce batches exactly"
    assert len(b1) == 12
    assert all(len(b) == 2 for b in b1)
    assert sorted(_flat(b1)) == list(range(24)), "each index exactly once per epoch"
    s2.set_epoch(1)
    b3 = [list(b) for b in s2]
    assert b3 != b1, "set_epoch must reshuffle"
    assert sorted(_flat(b3)) == list(range(24))


def test_bucket_sampler_batches_length_homogeneous() -> None:
    lengths = [10, 11, 12, 13] * 3 + [100, 101, 102, 103] * 3
    s = BucketBatchSampler(lengths, 2, shuffle=True, seed=1)
    s.set_epoch(0)
    for batch in s:
        ls = [lengths[int(i)] for i in batch]
        assert max(ls) <= 4 * min(ls), f"mixed-length batch: {ls}"


def test_bucket_sampler_ranks_disjoint_equal_count() -> None:
    lengths = list(range(20, 45))  # 25 lengths -> 12 global batches of 2
    per_rank = []
    for r in (0, 1):
        s = BucketBatchSampler(
            lengths, 2, shuffle=True, seed=3, rank=r, world_size=2, drop_last=True
        )
        s.set_epoch(0)
        per_rank.append([list(b) for b in s])
    assert len(per_rank[0]) == len(per_rank[1]) == 6, "equal batch count per rank"
    f0, f1 = set(_flat(per_rank[0])), set(_flat(per_rank[1]))
    assert len(f0) == 12 and len(f1) == 12, "no duplicate index within a rank"
    assert not (f0 & f1), "ranks must see disjoint batches"
    assert (f0 | f1) <= set(range(25))


def test_mix_dataset_deterministic(fake_shards) -> None:
    def make() -> MixDataset:
        return MixDataset(
            [ShardDataset(fake_shards), ShardDataset(fake_shards)], [0.7, 0.3], seed=5
        )

    m1, m2 = make(), make()
    assert len(m1) == 48
    assert len(m1.frames) == 48
    for i in (0, 7, 23, 31, 47):
        a, b = m1[i], m2[i]
        assert torch.equal(a.grid, b.grid), f"MixDataset not deterministic at {i}"
        assert a.task == b.task
        assert m1.frames[i] == a.n_frames, ".frames must match item lengths"


def test_build_dataloader_batches(test_cfg, fake_shards) -> None:
    loader = build_dataloader(test_cfg, [str(fake_shards)])
    batch = next(iter(loader))
    grid, mask, channel = batch["grid"], batch["loss_mask"], batch["channel"]
    B, S, Tp = grid.shape
    assert B == test_cfg.data.batch_size
    assert S == test_cfg.model.n_streams
    assert Tp <= test_cfg.model.max_frames, "delayed length must fit the context"
    assert grid.dtype == torch.long
    assert mask.dtype == torch.bool and mask.shape == (B, S, Tp)
    assert channel.dtype == torch.long and channel.shape == (B, Tp)
    # weighted-dict form of shard_dirs
    loader2 = build_dataloader(test_cfg, {str(fake_shards): 1.0})
    batch2 = next(iter(loader2))
    assert batch2["grid"].shape[1] == S


def test_build_dataloader_meta_mismatch_raises(test_cfg, fake_shards) -> None:
    bad = copy.deepcopy(test_cfg)
    bad.model.n_codebooks = 4  # shards were written with 2
    with pytest.raises((AssertionError, ValueError)):
        build_dataloader(bad, [str(fake_shards)])
