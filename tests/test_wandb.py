"""W&B integration: rank0 init/log/finish, run-id persistence, resume, errors.

Uses a fake in-process ``wandb`` module so the suite never needs the real
package (or a network): the contract under test is OUR call pattern.
"""

from __future__ import annotations

import copy
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from omni.data.dataset import build_dataloader
from omni.model.omni import OmniModel
from omni.train.loop import Trainer


class FakeWandb(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int]] = []
        self.finished = 0
        self._next_id = 0

    def init(self, **kw):
        self.init_calls.append(kw)
        rid = kw.get("id")
        if rid is None:
            self._next_id += 1
            rid = f"fake-run-{self._next_id}"
        return SimpleNamespace(id=rid)

    def log(self, data: dict, step: int | None = None) -> None:
        self.log_calls.append((dict(data), step))

    def finish(self) -> None:
        self.finished += 1


def _cfg(test_cfg, ckpt_dir, steps=4):
    cfg = copy.deepcopy(test_cfg)
    cfg.train.max_steps = steps
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_dir = str(ckpt_dir)
    cfg.train.log_every = 1
    cfg.train.save_every = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.wandb = True
    cfg.train.wandb_project = "omni-test"
    cfg.train.wandb_run_name = "unit"
    cfg.train.wandb_tags = ("ci", "tiny")
    cfg.train.wandb_mode = "offline"
    return cfg


@pytest.fixture()
def fake_wandb(monkeypatch):
    mod = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


def _fit(cfg, shards):
    torch.manual_seed(0)
    model = OmniModel(cfg.model)
    model.init_weights()
    Trainer(cfg, model, build_dataloader(cfg, [str(shards)])).fit()


def test_wandb_init_log_finish(fake_wandb, test_cfg, fake_shards, tmp_path) -> None:
    cfg = _cfg(test_cfg, tmp_path / "ckpt")
    _fit(cfg, fake_shards)

    assert len(fake_wandb.init_calls) == 1
    init = fake_wandb.init_calls[0]
    assert init["project"] == "omni-test" and init["name"] == "unit"
    assert init["mode"] == "offline" and init["tags"] == ["ci", "tiny"]
    assert init["id"] is None and init["resume"] is None
    assert init["config"]["train"]["max_steps"] == 4  # full config captured
    assert init["dir"] == str(tmp_path / "ckpt")

    assert len(fake_wandb.log_calls) == 4  # log_every=1
    data, step = fake_wandb.log_calls[-1]
    assert step == 4
    for key in ("loss", "lr", "grad_norm", "steps_per_s", "frames_per_s", "epoch"):
        assert key in data, key
    # per-head keys appear whenever a step's batch had targets for that head;
    # check across the run (a single step can legitimately be text-only)
    logged = {k for d, _s in fake_wandb.log_calls for k in d}
    assert any(k.startswith("loss/audio_") for k in logged)
    assert "loss/text" in logged
    assert fake_wandb.finished == 1
    # run id persisted next to the checkpoints for resume
    assert (tmp_path / "ckpt" / "wandb_run_id.txt").read_text() == "fake-run-1"


def test_wandb_resume_reuses_run_id(fake_wandb, test_cfg, fake_shards, tmp_path) -> None:
    cfg = _cfg(test_cfg, tmp_path / "ckpt", steps=3)
    cfg.train.save_every = 3
    _fit(cfg, fake_shards)
    cfg2 = _cfg(test_cfg, tmp_path / "ckpt", steps=6)
    cfg2.train.save_every = 6
    _fit(cfg2, fake_shards)  # resumes from step 3

    assert len(fake_wandb.init_calls) == 2
    second = fake_wandb.init_calls[1]
    assert second["id"] == "fake-run-1" and second["resume"] == "allow"
    # id file untouched by the resumed run
    assert (tmp_path / "ckpt" / "wandb_run_id.txt").read_text() == "fake-run-1"
    # resumed steps continue the history: 4, 5, 6
    steps = [s for _, s in fake_wandb.log_calls[3:]]
    assert steps == [4, 5, 6]


def test_wandb_missing_package_message(monkeypatch, test_cfg, fake_shards, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` fail
    cfg = _cfg(test_cfg, tmp_path / "ckpt")
    torch.manual_seed(0)
    model = OmniModel(cfg.model)
    model.init_weights()
    trainer = Trainer(cfg, model, build_dataloader(cfg, [str(fake_shards)]))
    with pytest.raises(ImportError, match=r"omni\[wandb\]"):
        trainer.fit()


def test_wandb_off_touches_nothing(fake_wandb, test_cfg, fake_shards, tmp_path) -> None:
    cfg = _cfg(test_cfg, tmp_path / "ckpt")
    cfg.train.wandb = False
    _fit(cfg, fake_shards)
    assert fake_wandb.init_calls == [] and fake_wandb.log_calls == []
    assert not (tmp_path / "ckpt" / "wandb_run_id.txt").exists()
