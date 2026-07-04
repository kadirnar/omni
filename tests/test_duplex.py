"""Extensions v2: full-duplex dual streams (INTERFACES.md "Extensions v2")."""

from __future__ import annotations

import copy

import pytest
import torch

from omni import grids, streams
from omni.model.omni import OmniModel

V = 2048


def _tracks(n_q: int, T: int = 30, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    apad = streams.audio_pad_id(V)
    user = torch.full((n_q, T), apad, dtype=torch.long)
    user[:, 2:12] = torch.randint(0, V, (n_q, 10), generator=g)
    asst = torch.full((n_q, T), apad, dtype=torch.long)
    asst[:, 14:26] = torch.randint(0, V, (n_q, 12), generator=g)
    return user, asst


def _duplex_batch(duplex_cfg, T: int = 30, seed: int = 0):
    n_q = duplex_cfg.model.n_codebooks
    user, asst = _tracks(n_q, T, seed)
    s = grids.build_duplex(user, asst, [(14, [70, 71]), (20, [72])], n_q, V)
    dg, dm, dch = streams.apply_delay(
        s.grid, s.loss_mask, s.channel, V,
        mode=duplex_cfg.model.audio_delay_mode, duplex=True,
    )
    return dg.unsqueeze(0), dm.unsqueeze(0), dch.unsqueeze(0)


def test_build_duplex_properties() -> None:
    n_q, T = 2, 30
    user, asst = _tracks(n_q, T)
    s = grids.build_duplex(user, asst, [(5, [70, 71]), (5, [72])], n_q, V)
    S, L = s.grid.shape
    assert (S, L) == (1 + 2 * n_q, T + 2)
    assert s.task == "duplex"
    assert s.grid[0, 0] == streams.BOS and s.grid[0, -1] == streams.EOS
    # groups: assistant rows then user rows, frame f at col f+1
    assert torch.equal(s.grid[1 : 1 + n_q, 1 : 1 + T], asst)
    assert torch.equal(s.grid[1 + n_q :, 1 : 1 + T], user)
    # masks: text ON after col 0, assistant ON over the timeline, user always OFF
    assert s.loss_mask[0, 1:].all() and not s.loss_mask[0, 0]
    assert s.loss_mask[1 : 1 + n_q, 1 : 1 + T].all()
    assert not s.loss_mask[1 + n_q :].any()
    # colliding words shift right: frames 5,5 -> cols 6,7 then 8
    assert s.grid[0, 6] == 70 and s.grid[0, 7] == 71 and s.grid[0, 8] == 72
    assert s.grid[0, 9] == streams.TEXT_PAD
    with pytest.raises(AssertionError):
        grids.build_duplex(user, asst, [(T, [70])], n_q, V)  # frame out of range
    with pytest.raises(AssertionError):
        bad = asst.clone()
        bad[0, 0] = streams.audio_eos_id(V)  # only raw codes / APAD allowed
        grids.build_duplex(user, bad, [], n_q, V)


def test_duplex_delay_roundtrip(duplex_cfg) -> None:
    dg, dm, dch = _duplex_batch(duplex_cfg)
    S = duplex_cfg.model.n_streams
    n_q = duplex_cfg.model.n_codebooks
    assert dg.shape[1] == S
    back = streams.undelay(dg[0], mode=duplex_cfg.model.audio_delay_mode, duplex=True)
    assert back.shape == (S, dg.shape[2] - duplex_cfg.model.max_delay)
    # user group delayed identically to assistant group
    dl = streams.stream_delays(n_q, duplex_cfg.model.audio_delay_mode, duplex=True)
    assert dl[1 : 1 + n_q] == dl[1 + n_q :]


def test_duplex_shards_and_collate(duplex_cfg, duplex_shards) -> None:
    from omni.data.dataset import ShardDataset, collate

    ds = ShardDataset(duplex_shards)
    assert len(ds) > 0 and ds.meta.get("duplex") is True
    S = duplex_cfg.model.n_streams
    s = ds[0]
    assert s.grid.shape[0] == S and s.task == "duplex"
    assert not s.loss_mask[1 + duplex_cfg.model.n_codebooks :].any()
    batch = collate([ds[0], ds[min(1, len(ds) - 1)]], V,
                    mode=duplex_cfg.model.audio_delay_mode, duplex=True)
    B, S2, Tp = batch["grid"].shape
    assert (B, S2) == (2, S)
    assert batch["loss_mask"].shape == (2, S, Tp) and batch["channel"].shape == (2, Tp)


def test_duplex_dataloader_mismatch(test_cfg, duplex_shards) -> None:
    from omni.data.dataset import build_dataloader

    with pytest.raises(ValueError):
        build_dataloader(test_cfg, [str(duplex_shards)])  # non-duplex cfg, duplex shards


def test_duplex_model_forward_loss(duplex_cfg) -> None:
    torch.manual_seed(0)
    model = OmniModel(duplex_cfg.model)
    model.init_weights()
    model.eval()
    assert model.user_audio_embs is not None
    grid, mask, chan = _duplex_batch(duplex_cfg)
    out = model(grid, chan)
    n_q = duplex_cfg.model.n_codebooks
    assert out.audio_logits.shape[1] == n_q  # assistant group only
    loss, metrics = model.loss(out, grid, mask)
    assert torch.isfinite(loss)
    assert f"loss/audio_{n_q - 1}" in metrics and f"loss/audio_{n_q}" not in metrics


def test_duplex_generator_steps_and_determinism(duplex_cfg) -> None:
    from omni.infer.duplex import DuplexGenerator, DuplexStep

    def run(seed: int):
        torch.manual_seed(0)
        model = OmniModel(duplex_cfg.model)
        model.init_weights()
        model.eval()
        gen = DuplexGenerator(model, duplex_cfg, device="cpu", seed=seed)
        gen.reset()
        g = torch.Generator().manual_seed(9)
        n_q = duplex_cfg.model.n_codebooks
        steps = []
        for t in range(12):
            frame = torch.randint(0, V, (n_q,), generator=g) if t % 3 else None
            steps.append(gen.step(frame))
        return steps

    steps = run(seed=7)
    assert all(isinstance(s, DuplexStep) for s in steps)
    frames = [s.assistant_frame for s in steps if s.assistant_frame is not None]
    assert frames, "pipeline never produced an assistant frame in 12 ticks"
    n_q = duplex_cfg.model.n_codebooks
    for f in frames:
        assert f.shape == (n_q,) and f.dtype == torch.long
        assert int(f.max()) < V + 3 and int(f.min()) >= 0
    # None only while the delay pipeline fills (a prefix), then steady frames
    got_frame = [s.assistant_frame is not None for s in steps]
    first = got_frame.index(True)
    assert first <= duplex_cfg.model.max_delay + 2
    assert all(got_frame[first:]), "assistant frames must be contiguous once flowing"
    steps2 = run(seed=7)
    for a, b in zip(steps, steps2):
        assert a.text_id == b.text_id
        assert (a.assistant_frame is None) == (b.assistant_frame is None)
        if a.assistant_frame is not None:
            assert torch.equal(a.assistant_frame, b.assistant_frame)


def test_duplex_run_file(duplex_cfg, fake_codec) -> None:
    from omni.infer.duplex import DuplexGenerator

    torch.manual_seed(1)
    model = OmniModel(duplex_cfg.model)
    model.init_weights()
    model.eval()
    wav = torch.sin(torch.linspace(0, 800, fake_codec.samples_per_frame * 15))
    n_in = fake_codec.encode(wav).shape[1]
    gen = DuplexGenerator(model, duplex_cfg, device="cpu", seed=3)
    text, out_wav = gen.run_file(wav, fake_codec)
    assert isinstance(text, str)
    assert out_wav.shape[0] == n_in * fake_codec.samples_per_frame
    assert torch.isfinite(out_wav).all()


def test_omnigenerator_rejects_duplex(duplex_cfg) -> None:
    from omni.infer.generate import OmniGenerator

    model = OmniModel(duplex_cfg.model)
    with pytest.raises(ValueError):
        OmniGenerator(model, duplex_cfg, device="cpu")


def test_duplex_e2e_train_drops_loss(duplex_cfg, duplex_shards, tmp_path) -> None:
    from omni.data.dataset import build_dataloader
    from omni.train.loop import Trainer

    cfg = copy.deepcopy(duplex_cfg)
    cfg.train.max_steps = 20
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.lr = 1e-3
    cfg.train.warmup_steps = 2
    cfg.train.ckpt_dir = str(tmp_path / "ckpt")
    cfg.train.save_every = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.log_every = 5
    torch.manual_seed(2)
    model = OmniModel(cfg.model)
    model.init_weights()
    loader = build_dataloader(cfg, [str(duplex_shards)])
    probe = next(iter(loader))
    model.eval()
    with torch.no_grad():
        out = model(probe["grid"], probe["channel"])
        before, _ = model.loss(out, probe["grid"], probe["loss_mask"])
    Trainer(cfg, model, loader).fit()
    model.eval()
    with torch.no_grad():
        out = model(probe["grid"], probe["channel"])
        after, _ = model.loss(out, probe["grid"], probe["loss_mask"])
    assert after.item() < before.item(), (before.item(), after.item())


def test_benchmarks_handle_duplex_streams(duplex_cfg) -> None:
    """perf benchmarks must build n_streams-row grids (S = 1 + 2*n_q for duplex;
    INTERFACES.md v2: benchmark_decode works for all configs)."""
    from omni.optim.perf import benchmark_decode, benchmark_forward

    torch.manual_seed(0)
    model = OmniModel(duplex_cfg.model)
    res = benchmark_decode(model, duplex_cfg, "cpu", n_frames=2)
    assert set(res) == {
        "steps_per_s", "rtf", "ms_per_step", "prefill_ms", "n_frames", "batch",
    }
    fwd = benchmark_forward(model, duplex_cfg, "cpu", batch=1, frames=4, steps=1)
    assert fwd["tokens_per_s"] > 0
