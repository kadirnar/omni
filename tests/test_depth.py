"""Extensions v2: flat delay mode + depth transformer (INTERFACES.md "Extensions v2")."""

from __future__ import annotations

import copy

import pytest
import torch

from omni import grids, streams
from omni.config import ModelConfig
from omni.model.layers import KVCache
from omni.model.omni import OmniModel

V = 2048


def _tts_batch(cfg, frames: int = 40, seed: int = 0):
    """One delayed flat-mode TTS sample as a [1, S, T'] batch."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, V, (cfg.model.n_codebooks, frames), generator=g)
    text = [64 + (i % 200) for i in range(12)]
    s = grids.build_tts(text, codes, cfg.model.n_codebooks, V)
    dg, dm, dch = streams.apply_delay(
        s.grid, s.loss_mask, s.channel, V, mode=cfg.model.audio_delay_mode
    )
    return dg.unsqueeze(0), dm.unsqueeze(0), dch.unsqueeze(0)


def test_config_gate() -> None:
    with pytest.raises(AssertionError):
        ModelConfig(audio_delay_mode="flat", use_depth=False)
    with pytest.raises(AssertionError):
        ModelConfig(audio_delay_mode="stagger", use_depth=True)
    cfg = ModelConfig(audio_delay_mode="flat", use_depth=True)
    assert cfg.max_delay == 1


def test_load_config_overrides_keep_depth_lock() -> None:
    """Dotted overrides / YAML dicts mutate the dataclass with setattr; load_config
    must re-run the ModelConfig invariants so the flat<->use_depth lock cannot be
    silently broken on the primary n_codebooks=32 presets (quality/small)."""
    from omni.config import load_config

    with pytest.raises(AssertionError, match="use_depth"):
        load_config("quality", ["model.use_depth=false"])
    with pytest.raises(AssertionError, match="use_depth"):
        load_config(
            "quality",
            ["model.audio_delay_mode=stagger", "data.max_sample_frames=2000"],
        )
    # the depth-lock message must win over the unrelated max_frames assert
    with pytest.raises(AssertionError, match="use_depth"):
        load_config("quality", ["model.audio_delay_mode=stagger"])
    # coherent overrides still load
    cfg = load_config("quality", ["model.n_layers=2"])
    assert cfg.model.use_depth and cfg.model.n_codebooks == 32


@pytest.mark.parametrize("n_q", [2, 8])
def test_flat_delay_roundtrip(n_q: int) -> None:
    g = torch.Generator().manual_seed(n_q)
    grid = torch.randint(0, V, (1 + n_q, 23), generator=g)
    mask = torch.rand((1 + n_q, 23), generator=g) > 0.5
    chan = torch.zeros(23, dtype=torch.long)
    dg, dm, _ = streams.apply_delay(grid, mask, chan, V, mode="flat")
    assert dg.shape == (1 + n_q, 24)
    # every audio row shifted by exactly one; single leading APAD filler
    apad = streams.audio_pad_id(V)
    assert (dg[1:, 0] == apad).all() and not dm[1:, 0].any()
    assert torch.equal(streams.undelay(dg, mode="flat"), grid)


def test_forward_shapes_and_loss(depth_cfg) -> None:
    torch.manual_seed(0)
    model = OmniModel(depth_cfg.model)
    model.init_weights()
    model.eval()
    assert model.depth is not None
    grid, mask, chan = _tts_batch(depth_cfg)
    out = model(grid, chan)
    n_q = depth_cfg.model.n_codebooks
    assert out.text_logits.shape == (1, grid.shape[2], depth_cfg.model.text_vocab_size)
    assert out.audio_logits.shape == (1, n_q, grid.shape[2], depth_cfg.model.audio_vocab_size)
    assert out.audio_logits.dtype == torch.float32
    loss, metrics = model.loss(out, grid, mask)
    assert torch.isfinite(loss)
    assert "loss/text" in metrics and f"loss/audio_{n_q - 1}" in metrics


def test_depth_kv_parity(depth_cfg) -> None:
    """forward() == prefill_hidden/step_hidden + depth.forward at every position."""
    torch.manual_seed(1)
    model = OmniModel(depth_cfg.model)
    model.init_weights()
    model.eval()
    grid, _, chan = _tts_batch(depth_cfg, seed=1)
    T = grid.shape[2]
    n_q = depth_cfg.model.n_codebooks
    with torch.no_grad():
        out = model(grid, chan)
        cache = KVCache.allocate(depth_cfg.model, 1, "cpu", torch.float32)
        tl, h = model.prefill_hidden(grid[:, :, :1], chan[:, :1], cache)
        for p in range(1, T):
            teacher = grid[:, 1 : 1 + n_q, p]
            da = model.depth(h, teacher)
            assert torch.allclose(da, out.audio_logits[:, :, p - 1], atol=1e-4, rtol=1e-5)
            assert torch.allclose(tl, out.text_logits[:, p - 1], atol=1e-4, rtol=1e-5)
            tl, h = model.step_hidden(grid[:, :, p], chan[:, p], cache)
        assert torch.allclose(tl, out.text_logits[:, T - 1], atol=1e-4, rtol=1e-5)
        assert cache.pos == T


def test_step_and_prefill_raise_on_depth(depth_cfg) -> None:
    model = OmniModel(depth_cfg.model)
    model.eval()
    grid, _, chan = _tts_batch(depth_cfg, frames=8)
    cache = KVCache.allocate(depth_cfg.model, 1, "cpu", torch.float32)
    with pytest.raises(RuntimeError):
        model.prefill(grid, chan, cache)
    with pytest.raises(RuntimeError):
        model.step(grid[:, :, 0], chan[:, 0], cache)


def test_depth_overfit(depth_cfg) -> None:
    torch.manual_seed(2)
    model = OmniModel(depth_cfg.model)
    model.init_weights()
    grid, mask, chan = _tts_batch(depth_cfg, frames=24, seed=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        out = model(grid, chan)
        loss, _ = model.loss(out, grid, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.5 * first, (first, loss.item())


def test_depth_save_load_roundtrip(depth_cfg, tmp_path) -> None:
    torch.manual_seed(3)
    model = OmniModel(depth_cfg.model)
    model.init_weights()
    model.save_pretrained(tmp_path / "export")
    loaded = OmniModel.from_pretrained(tmp_path / "export")
    assert loaded.cfg.use_depth and loaded.cfg.audio_delay_mode == "flat"
    assert loaded.depth is not None
    sd, sd2 = model.state_dict(), loaded.state_dict()
    assert sd.keys() == sd2.keys()
    for k in sd:
        assert torch.equal(sd[k], sd2[k]), k
    grid, _, chan = _tts_batch(depth_cfg, frames=10)
    model.eval(), loaded.eval()
    with torch.no_grad():
        a, b = model(grid, chan), loaded(grid, chan)
    assert torch.equal(a.audio_logits, b.audio_logits)


def test_depth_generator_tts_seeded(depth_cfg, byte_tok, fake_codec) -> None:
    from omni.infer.generate import OmniGenerator

    torch.manual_seed(4)
    model = OmniModel(depth_cfg.model)
    model.init_weights()
    model.eval()
    gen = OmniGenerator(model, depth_cfg, device="cpu", tokenizer=byte_tok)
    r1 = gen.tts("hi there", fake_codec, max_frames=10, seed=5)
    r2 = gen.tts("hi there", fake_codec, max_frames=10, seed=5)
    assert r1.frames <= 10 and r1.audio_codes.shape[0] == depth_cfg.model.n_codebooks
    assert torch.equal(r1.audio_codes, r2.audio_codes)
    assert torch.isfinite(fake_codec.decode(r1.audio_codes)).all()


def test_benchmark_decode_depth_32_codebooks() -> None:
    """perf.benchmark_decode must route depth models through prefill_hidden/
    step_hidden + depth.sample (INTERFACES.md v2: 'works for all configs');
    mirrors the quality/small presets' n_codebooks=32 flat+depth shape."""
    from omni.config import load_config
    from omni.optim.perf import benchmark_decode

    cfg = load_config(
        "tiny",
        [
            "model.n_codebooks=32", "model.audio_delay_mode=flat",
            "model.use_depth=true", "model.d_model=64", "model.n_layers=1",
            "model.n_heads=2", "model.n_kv_heads=1", "model.d_ff=128",
            "model.depth_d_model=32", "model.depth_n_layers=1",
            "model.depth_n_heads=2", "model.audio_codec_vocab=128",
            "model.text_vocab_size=320", "model.max_frames=32",
            "data.max_sample_frames=16",
        ],
    )
    torch.manual_seed(0)
    model = OmniModel(cfg.model)
    res = benchmark_decode(model, cfg, "cpu", n_frames=2)
    assert set(res) == {
        "steps_per_s", "rtf", "ms_per_step", "prefill_ms", "n_frames", "batch",
    }
    assert res["n_frames"] == 2.0 and res["steps_per_s"] > 0


def test_trainer_flat_mode(depth_cfg, fake_shards, tmp_path) -> None:
    from omni.data.dataset import build_dataloader
    from omni.train.loop import Trainer

    cfg = copy.deepcopy(depth_cfg)
    cfg.train.max_steps = 5
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_dir = str(tmp_path / "ckpt")
    cfg.train.save_every = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.log_every = 1
    torch.manual_seed(5)
    model = OmniModel(cfg.model)
    model.init_weights()
    loader = build_dataloader(cfg, [str(fake_shards)])
    batch = next(iter(loader))
    # flat mode: delayed length = undelayed + 1
    assert batch["grid"].shape[1] == cfg.model.n_streams
    metrics = Trainer(cfg, model, loader).fit()
    assert any("loss" in k for k in metrics)
    assert all(
        torch.isfinite(torch.as_tensor(float(v))).item()
        for v in metrics.values()
        if isinstance(v, (int, float, torch.Tensor))
    )
