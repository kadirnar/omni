"""End-to-end CPU pipeline: prepare_fake -> train -> export -> TTS -> wav.

Everything runs in a tmp dir with the tiny test config; budget < 2 min on CPU.
"""

from __future__ import annotations

import copy

import torch

from omni.data.dataset import build_dataloader
from omni.data.prepare import prepare_fake
from omni.infer.generate import OmniGenerator
from omni.model.omni import OmniModel
from omni.train.loop import Trainer

SPF = 1920


def test_e2e_fake_train_export_tts(tmp_path, test_cfg, byte_tok, fake_codec) -> None:
    cfg = copy.deepcopy(test_cfg)
    cfg.train.max_steps = 30
    cfg.train.strategy = "none"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_dir = str(tmp_path / "ckpt")
    cfg.train.save_every = 10_000
    cfg.train.eval_every = 10_000
    cfg.train.warmup_steps = 3
    cfg.train.lr = 1e-3
    cfg.train.log_every = 10

    # 1. data
    shards = tmp_path / "shards"
    shards.mkdir()
    prepare_fake(shards, n_samples=24, cfg=cfg, seed=0)

    # 2. train 30 steps; loss must drop vs step 0 on a fixed probe batch
    torch.manual_seed(0)
    model = OmniModel(cfg.model)
    model.init_weights()
    loader = build_dataloader(cfg, [str(shards)])
    probe = next(iter(loader))

    def probe_loss() -> float:
        model.eval()
        with torch.no_grad():
            out = model(probe["grid"], probe["channel"])
            loss, _ = model.loss(out, probe["grid"], probe["loss_mask"])
        return float(loss)

    loss_before = probe_loss()
    assert loss_before == loss_before  # finite
    trainer = Trainer(cfg, model, loader)
    trainer.fit()
    loss_after = probe_loss()
    assert loss_after < 0.98 * loss_before, (
        f"training did not reduce loss: {loss_before:.4f} -> {loss_after:.4f}"
    )

    # 3. export consolidated weights and reload
    export_dir = tmp_path / "export"
    trainer.export_model(export_dir)
    loaded = OmniModel.from_pretrained(export_dir)
    loaded.eval()

    # 4. generate speech and decode it
    gen = OmniGenerator(loaded, cfg, device="cpu", tokenizer=byte_tok)
    r = gen.tts("hello omni", fake_codec, max_frames=24, seed=0)
    assert r.frames > 0
    assert r.audio_codes.shape == (cfg.model.n_codebooks, r.frames)
    assert int(r.audio_codes.min()) >= 0
    assert int(r.audio_codes.max()) < cfg.model.audio_codec_vocab

    wav = fake_codec.decode(r.audio_codes)
    assert wav.shape == (r.frames * SPF,)
    assert torch.isfinite(wav).all()
    assert float(wav.abs().max()) > 0.0, "decoded wav is silent"
