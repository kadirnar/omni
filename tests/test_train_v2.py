"""Tests for the 2026-07 review fixes in model/training: FSDP2 block dispatch,
trainable-param strategy, resume config guard, lean checkpoints, retention,
depth-loss amortization, constant-sum loss, --init-from warm starts, wsd."""

from __future__ import annotations

import copy

import pytest
import torch

from omni.config import ModelConfig, load_config
from omni.data.dataset import build_dataloader
from omni.model import build_model, load_weights, structural_diff
from omni.model.omni import ModelOutput, OmniModel, multistream_loss
from omni.train.distributed import DistContext, _shardable_blocks, pick_strategy
from omni.train.loop import Trainer, build_lr_lambda
from test_hf_backbone import hf_cfg, tiny_backbone


def _ctx(world_size: int = 2, device: str = "cuda") -> DistContext:
    # a device OBJECT only — nothing here allocates or touches a real GPU
    return DistContext(
        rank=0, local_rank=0, world_size=world_size,
        device=torch.device(device), is_main=True, initialized=False,
    )


# ---------------------------------------------------------------- distributed
def test_shardable_blocks_dispatch() -> None:
    scratch = OmniModel(ModelConfig(text_vocab_size=320, n_codebooks=2))
    assert _shardable_blocks(scratch) == list(scratch.blocks)

    hf = build_model(hf_cfg().model, backbone=tiny_backbone())
    layers = _shardable_blocks(hf)
    assert layers and layers == list(hf.backbone.get_decoder().layers)

    with pytest.raises(RuntimeError, match="ddp"):
        _shardable_blocks(torch.nn.Linear(4, 4))


def test_pick_strategy_counts_trainable_params() -> None:
    cfg = load_config("tiny", ["train.fsdp_threshold_params=1000"])
    frozen_hf = build_model(hf_cfg().model, backbone=tiny_backbone())
    trainable = sum(p.numel() for p in frozen_hf.parameters() if p.requires_grad)
    total = sum(p.numel() for p in frozen_hf.parameters())
    assert trainable < total  # the backbone really is frozen
    # tiny threshold: trainable alone crosses it -> fsdp2 on cuda
    assert pick_strategy(cfg, _ctx(device="cuda"), trainable) == "fsdp2"
    # a frozen backbone under the threshold stays on ddp even on cuda
    cfg_big = load_config("tiny")  # default 300M threshold
    assert pick_strategy(cfg_big, _ctx(device="cuda"), trainable) == "ddp"
    assert pick_strategy(cfg_big, _ctx(device="cpu"), trainable) == "ddp"
    assert pick_strategy(cfg_big, _ctx(world_size=1), trainable) == "none"


# ------------------------------------------------------------------- trainer
def _mini_train_cfg(test_cfg, ckpt_dir, **over):
    cfg = copy.deepcopy(test_cfg)
    cfg.train.max_steps = 2
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_dir = str(ckpt_dir)
    cfg.train.save_every = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.log_every = 0
    cfg.train.warmup_steps = 1
    for k, v in over.items():
        setattr(cfg.train, k, v)
    return cfg


def test_resume_rejects_incompatible_config(test_cfg, fake_shards, tmp_path) -> None:
    cfg = _mini_train_cfg(test_cfg, tmp_path / "ckpt")
    torch.manual_seed(0)
    model = OmniModel(cfg.model)
    loader = build_dataloader(cfg, [str(fake_shards)])
    Trainer(cfg, model, loader).fit()  # leaves a checkpoint behind

    cfg2 = copy.deepcopy(cfg)
    cfg2.model.d_model = cfg.model.d_model * 2  # same shards, different model
    cfg2.model.n_heads = cfg.model.n_heads
    torch.manual_seed(0)
    model2 = OmniModel(cfg2.model)
    trainer2 = Trainer(cfg2, model2, loader)
    with pytest.raises(ValueError, match="different config"):
        trainer2.maybe_resume()


def test_save_keep_prunes_old_checkpoints(test_cfg, fake_shards, tmp_path) -> None:
    cfg = _mini_train_cfg(
        test_cfg, tmp_path / "ckpt", max_steps=4, save_every=1, save_keep=2
    )
    cfg.train.max_steps = 4
    torch.manual_seed(0)
    loader = build_dataloader(cfg, [str(fake_shards)])
    Trainer(cfg, OmniModel(cfg.model), loader).fit()
    dirs = sorted(d.name for d in (tmp_path / "ckpt").glob("step_*"))
    assert dirs == ["step_00000003", "step_00000004"]
    latest = (tmp_path / "ckpt" / "latest.txt").read_text().strip()
    assert latest == "step_00000004"


def test_lean_checkpoint_skips_frozen_backbone(tmp_path) -> None:
    """A frozen v6 backbone must not be written into training checkpoints,
    and resume must restore trainable state from the lean checkpoint."""
    cfg = hf_cfg()
    cfg.train.max_steps = 2
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_dir = str(tmp_path / "ckpt")
    cfg.train.save_every = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.log_every = 0
    cfg.train.warmup_steps = 1
    from omni.data.prepare import prepare_fake

    prepare_fake(tmp_path / "shards", n_samples=16, cfg=cfg, seed=0)
    torch.manual_seed(0)
    bb = tiny_backbone()
    model = build_model(cfg.model, backbone=bb)
    cfg.model = model.cfg
    loader = build_dataloader(cfg, [str(tmp_path / "shards")])
    trainer = Trainer(cfg, model, loader)
    trainer.fit()

    ckpt = sorted((tmp_path / "ckpt").glob("step_*/trainer_state.pt"))[-1]
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    bb_weight_keys = [k for k in saved if k.startswith("backbone.") and "lora_" not in k]
    frozen_names = {
        n for n, p in model.named_parameters() if not p.requires_grad
    }
    assert not (set(bb_weight_keys) & frozen_names), "frozen backbone params leaked"
    # every trainable param IS in the checkpoint
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names <= set(saved)

    # resume round-trip on a fresh trainer (same backbone object = hub reload)
    torch.manual_seed(0)
    model2 = build_model(cfg.model, backbone=bb)
    trainer2 = Trainer(cfg, model2, loader)
    assert trainer2.maybe_resume() == 2
    for n in sorted(trainable_names):
        a = dict(model.named_parameters())[n]
        b = dict(model2.named_parameters())[n]
        assert torch.equal(a, b), f"trainable param {n} not restored"


# --------------------------------------------------------- depth amortization
def _depth_cfg(ratio: float) -> ModelConfig:
    return ModelConfig(
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128,
        text_vocab_size=320, n_codebooks=4, max_frames=64,
        audio_delay_mode="flat", use_depth=True, depth_loss_ratio=ratio,
        depth_d_model=32, depth_n_layers=1, depth_n_heads=2,
    )


def test_depth_amortization_subsamples_positions() -> None:
    torch.manual_seed(0)
    model = OmniModel(_depth_cfg(0.25))
    B, S, T = 2, 5, 33
    grid = torch.randint(64, 320, (B, S, T))
    grid[:, 1:] = torch.randint(0, 2048, (B, S - 1, T))
    channel = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, S, T, dtype=torch.bool)

    model.train()
    out = model(grid, channel)
    assert out.audio_positions is not None
    P = int(out.audio_positions.shape[0])
    assert P == max(1, int(round(0.25 * (T - 1))))
    assert out.audio_logits.shape == (B, 4, P, 2051)
    assert bool((out.audio_positions[1:] > out.audio_positions[:-1]).all())
    total, metrics = model.loss(out, grid, mask)
    assert torch.isfinite(total)
    total.backward()  # graph reaches the depth transformer
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.depth.parameters()
    )

    model.eval()
    out_eval = model(grid, channel)
    assert out_eval.audio_positions is None
    assert out_eval.audio_logits.shape == (B, 4, T, 2051)


def test_depth_ratio_one_is_exact_old_behavior() -> None:
    torch.manual_seed(0)
    model = OmniModel(_depth_cfg(1.0))
    B, S, T = 1, 5, 9
    grid = torch.randint(64, 320, (B, S, T))
    grid[:, 1:] = torch.randint(0, 2048, (B, S - 1, T))
    channel = torch.zeros(B, T, dtype=torch.long)
    model.train()
    out = model(grid, channel)
    assert out.audio_positions is None
    assert out.audio_logits.shape[2] == T


# ------------------------------------------------------------- loss semantics
def test_loss_constant_sum_normalization() -> None:
    """Head gradient scale must not depend on which heads have targets."""
    g = torch.Generator().manual_seed(0)
    B, T, n_q, vt, va = 1, 6, 2, 320, 2051
    text_logits = torch.randn(B, T, vt, generator=g)
    audio_logits = torch.randn(B, n_q, T, va, generator=g)
    grid = torch.randint(64, 320, (B, 1 + n_q, T), generator=g)
    grid[:, 1:] = torch.randint(0, 2048, (B, n_q, T), generator=g)

    out = ModelOutput(text_logits=text_logits, audio_logits=audio_logits)
    mask_text_only = torch.zeros(B, 1 + n_q, T, dtype=torch.bool)
    mask_text_only[:, 0] = True
    total, metrics = multistream_loss(out, grid, mask_text_only, n_q, (1.0, 1.0, 1.0))
    # denom = text_w + audio_w * (sem_w + n_q - 1) = 3, only text contributes
    manual = torch.nn.functional.cross_entropy(
        text_logits[:, :-1].reshape(-1, vt), grid[:, 0, 1:].reshape(-1)
    )
    assert torch.allclose(total, manual / 3.0, atol=1e-5)
    assert "loss/text" in metrics
    assert not any(k.startswith("loss/audio") for k in metrics)

    # semantic weighting enters the constant denominator
    total_sem, _ = multistream_loss(out, grid, mask_text_only, n_q, (1.0, 1.0, 100.0))
    assert torch.allclose(total_sem, manual / 102.0, atol=1e-5)


def test_loss_amortized_positions_alignment() -> None:
    """With audio_positions set, subset logits at p must score column p+1."""
    B, T, n_q, va = 1, 8, 1, 2051
    sel = torch.tensor([2, 5])
    grid = torch.zeros(B, 2, T, dtype=torch.long)
    grid[:, 1] = torch.arange(T)  # audio target at column c is c
    mask = torch.zeros(B, 2, T, dtype=torch.bool)
    mask[:, 1] = True
    # logits that put all mass on the CORRECT next column value
    audio_logits = torch.full((B, n_q, 2, va), -20.0)
    audio_logits[0, 0, 0, 3] = 20.0  # position 2 predicts column 3
    audio_logits[0, 0, 1, 6] = 20.0  # position 5 predicts column 6
    out = ModelOutput(
        text_logits=torch.zeros(B, T, 320),
        audio_logits=audio_logits,
        audio_positions=sel,
    )
    total, metrics = multistream_loss(out, grid, mask, n_q, (0.0, 1.0, 1.0))
    assert float(metrics["loss/audio_0"]) < 1e-3  # correct alignment -> ~0 CE


# ------------------------------------------------------------------ init-from
def test_load_weights_scratch_roundtrip(tmp_path) -> None:
    cfg = ModelConfig(text_vocab_size=320, n_codebooks=2, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, d_ff=128)
    torch.manual_seed(0)
    src = OmniModel(cfg)
    src.save_pretrained(tmp_path / "export")
    torch.manual_seed(1)
    dst = OmniModel(cfg)
    assert not torch.equal(dst.text_emb.weight, src.text_emb.weight)
    load_weights(dst, tmp_path / "export")
    for (n, a), (_, b) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert torch.equal(a, b), n


def test_load_weights_rejects_structural_mismatch(tmp_path) -> None:
    cfg = ModelConfig(text_vocab_size=320, n_codebooks=2, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, d_ff=128)
    OmniModel(cfg).save_pretrained(tmp_path / "export")
    other = ModelConfig(text_vocab_size=320, n_codebooks=4, d_model=64, n_layers=2,
                        n_heads=4, n_kv_heads=2, d_ff=128)
    assert structural_diff({"n_codebooks": 2}, other) == [
        "n_codebooks: saved=2 vs current=4"
    ]
    with pytest.raises(ValueError, match="structurally incompatible"):
        load_weights(OmniModel(other), tmp_path / "export")


def test_load_weights_hf_stage_transition(tmp_path) -> None:
    """Stage 1 (frozen) export warm-starts a stage 2 (unfrozen) model."""
    bb = tiny_backbone()
    stage1 = build_model(hf_cfg().model, backbone=bb)
    stage1.save_pretrained(tmp_path / "stage1")

    stage2_cfg = hf_cfg("model.freeze_backbone=false").model
    torch.manual_seed(123)
    stage2 = build_model(stage2_cfg, backbone=tiny_backbone())
    load_weights(stage2, tmp_path / "stage1")
    assert torch.equal(stage2.special_emb.weight, stage1.special_emb.weight)
    assert torch.equal(
        stage2.audio_embs[0].weight.float(), stage1.audio_embs[0].weight.float()
    )


# ------------------------------------------------------------------- schedule
def test_wsd_never_decays_before_warmup_ends() -> None:
    from omni.config import TrainConfig

    tc = TrainConfig(schedule="wsd", warmup_steps=90, max_steps=100, min_lr_ratio=0.1)
    lam = build_lr_lambda(tc)
    assert lam(89) == pytest.approx(90 / 91)  # still warming up
    assert lam(90) == pytest.approx(1.0)  # decay starts AT warmup end, from 1.0
    assert lam(100) == pytest.approx(0.1)
