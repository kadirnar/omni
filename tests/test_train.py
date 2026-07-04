"""Tests for omni.train.loop: fit metrics, checkpoint/resume, export, lr schedule."""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from omni.config import TrainConfig
from omni.data.dataset import build_dataloader
from omni.model.omni import OmniModel
from omni.train.loop import Trainer, build_lr_lambda

STEPS = 6


def _train_cfg(test_cfg, ckpt_dir, max_steps: int = STEPS):
    cfg = copy.deepcopy(test_cfg)
    cfg.train.max_steps = max_steps
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_dir = str(ckpt_dir)
    cfg.train.save_every = 10_000  # only the end-of-fit save
    cfg.train.eval_every = 10_000
    cfg.train.warmup_steps = 2
    cfg.train.lr = 1e-3
    cfg.train.log_every = 1
    cfg.train.resume = True
    cfg.train.seed = 0
    return cfg


@pytest.fixture(scope="module")
def trained(test_cfg, fake_shards, tmp_path_factory):
    """One 6-step CPU fit shared by the metric / checkpoint / export tests."""
    cfg = _train_cfg(test_cfg, tmp_path_factory.mktemp("ckpt"))
    torch.manual_seed(0)
    model = OmniModel(cfg.model)
    model.init_weights()
    loader = build_dataloader(cfg, [str(fake_shards)])
    trainer = Trainer(cfg, model, loader)
    metrics = trainer.fit()
    return SimpleNamespace(
        cfg=cfg, model=model, loader=loader, trainer=trainer, metrics=metrics
    )


def test_fit_returns_finite_metrics(trained) -> None:
    m = trained.metrics
    assert isinstance(m, dict) and m
    assert any("loss" in k for k in m), sorted(m)
    for k, v in m.items():
        if isinstance(v, torch.Tensor):
            v = float(v)
        if isinstance(v, (int, float)):
            assert math.isfinite(v), f"metric {k}={v} not finite"


def test_checkpoint_layout(trained) -> None:
    ck = Path(trained.cfg.train.ckpt_dir)
    latest = (ck / "latest.txt").read_text().strip()
    step_dir = ck / Path(latest).name
    assert step_dir.name == f"step_{STEPS:08d}"
    assert step_dir.is_dir()
    assert (step_dir / "trainer_state.pt").exists(), "single-process save is torch.save"


def _float_leaves(obj, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten any nested checkpoint structure to {path: float tensor}."""
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            out[prefix] = obj
    elif isinstance(obj, dict):
        for k in obj:
            out.update(_float_leaves(obj[k], f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_float_leaves(v, f"{prefix}[{i}]"))
    return out


def test_resume_restores_step_model_and_optimizer(trained, fake_shards, tmp_path) -> None:
    ck_a = Path(trained.cfg.train.ckpt_dir)
    step_name = Path((ck_a / "latest.txt").read_text().strip()).name
    ck_b = tmp_path / "ck_b"
    shutil.copytree(ck_a, ck_b)

    cfg_b = copy.deepcopy(trained.cfg)
    cfg_b.train.ckpt_dir = str(ck_b)
    torch.manual_seed(1234)  # deliberately different init
    model_b = OmniModel(cfg_b.model)
    model_b.init_weights()
    sd_a = {k: v.detach().clone() for k, v in trained.model.state_dict().items()}
    assert any(
        not torch.equal(model_b.state_dict()[k], sd_a[k]) for k in sd_a
    ), "fresh model should differ before resume"

    loader_b = build_dataloader(cfg_b, [str(fake_shards)])
    trainer_b = Trainer(cfg_b, model_b, loader_b)
    step = trainer_b.maybe_resume()
    assert int(step) == STEPS, "maybe_resume must return the checkpointed step"

    sd_b = model_b.state_dict()
    assert set(sd_b) == set(sd_a)
    for k in sd_a:
        torch.testing.assert_close(sd_b[k], sd_a[k], rtol=0, atol=0)

    # Re-saving right after resume must reproduce the same model AND optimizer
    # tensors as the original checkpoint (proves optimizer state was restored).
    trainer_b.save_checkpoint(STEPS)
    fa = _float_leaves(
        torch.load(ck_a / step_name / "trainer_state.pt", map_location="cpu", weights_only=False)
    )
    fb = _float_leaves(
        torch.load(ck_b / step_name / "trainer_state.pt", map_location="cpu", weights_only=False)
    )
    assert set(fa) == set(fb)
    assert any("exp_avg" in k for k in fa), "checkpoint must contain AdamW moments"
    for k in fa:
        assert torch.equal(fa[k], fb[k]), f"checkpoint tensor differs after resume: {k}"


def test_export_model_roundtrip(trained, tmp_path) -> None:
    out = tmp_path / "export"
    trained.trainer.export_model(out)
    loaded = OmniModel.from_pretrained(out)
    loaded.eval()
    trained.model.eval()
    batch = next(iter(trained.loader))
    with torch.no_grad():
        a = trained.model(batch["grid"], batch["channel"])
        b = loaded(batch["grid"], batch["channel"])
    torch.testing.assert_close(b.text_logits, a.text_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b.audio_logits, a.audio_logits, rtol=1e-5, atol=1e-6)


def test_build_lr_lambda_schedules() -> None:
    base = dict(warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    for sched, end in (("cosine", 0.1), ("wsd", 0.1), ("constant", 1.0)):
        f = build_lr_lambda(TrainConfig(schedule=sched, **base))
        vals = [f(s) for s in range(101)]
        assert all(0.0 <= v <= 1.0 + 1e-6 for v in vals), sched
        assert f(5) < f(10), f"{sched}: warmup must ramp up"
        assert f(10) == pytest.approx(1.0, abs=0.1), f"{sched}: ~1.0 after warmup"
        assert f(100) == pytest.approx(end, abs=0.05), f"{sched}: wrong final ratio"
    wsd = build_lr_lambda(TrainConfig(schedule="wsd", **base))
    assert wsd(79) > 0.9, "wsd holds ~1.0 until 0.8 * max_steps"
    assert wsd(95) < 0.9, "wsd decays linearly after 0.8 * max_steps"
