"""Config loading/override behavior: strict coercion, delay-mode invariants."""

from __future__ import annotations

import pytest

from omni.config import ModelConfig, load_config


def test_bool_override_typo_raises() -> None:
    with pytest.raises(ValueError, match="invalid boolean"):
        load_config("tiny", ["model.duplex=frue"])
    assert load_config("tiny", ["model.duplex=true"]).model.duplex is True
    assert load_config("tiny", ["model.duplex=false"]).model.duplex is False
    assert load_config("tiny", ["train.wandb=on"]).train.wandb is True


def test_tuple_override_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="2 elements"):
        load_config("tiny", ["train.betas=[0.9,0.95,0.99]"])
    with pytest.raises(ValueError, match="2 elements"):
        load_config("tiny", ["train.betas=[0.9]"])
    cfg = load_config("tiny", ["train.betas=[0.5,0.9]"])
    assert cfg.train.betas == (0.5, 0.9)


def test_string_tuple_stays_variable_length() -> None:
    cfg = load_config("tiny", ["train.wandb_tags=[a,b,c]"])
    assert cfg.train.wandb_tags == ("a", "b", "c")


def test_lead_mode_config() -> None:
    cfg = load_config(
        "tiny", ["model.audio_delay_mode=lead", "model.use_depth=true"]
    )
    assert cfg.model.max_delay == 2
    # lead without depth violates the lock, both ways
    with pytest.raises(AssertionError):
        load_config("tiny", ["model.audio_delay_mode=lead"])
    with pytest.raises(AssertionError):
        ModelConfig(audio_delay_mode="lead", use_depth=False)
    # single-codebook lead degenerates to delay 1
    assert ModelConfig(
        audio_delay_mode="lead", use_depth=True, n_codebooks=1
    ).max_delay == 1


def test_depth_loss_ratio_validation() -> None:
    with pytest.raises(AssertionError):
        ModelConfig(depth_loss_ratio=0.0)
    with pytest.raises(AssertionError):
        ModelConfig(depth_loss_ratio=1.5)
    assert ModelConfig(depth_loss_ratio=1.0).depth_loss_ratio == 1.0
    # depth presets adopt the CSM 1/16 amortization
    for preset in ("quality", "small", "qwen3-1.7b"):
        assert load_config(preset).model.depth_loss_ratio == pytest.approx(0.0625)
    assert load_config("tiny").model.depth_loss_ratio == 1.0
