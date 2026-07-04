"""Tests for omni.model: shapes, loss, overfit, KV-cache parity, checkpoint io."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from omni.grids import build_s2s
from omni.model.layers import KVCache
from omni.model.omni import ModelOutput, OmniModel
from omni.streams import apply_delay


def _model(cfg, seed: int = 0) -> OmniModel:
    torch.manual_seed(seed)
    m = OmniModel(cfg.model)
    m.init_weights()
    return m


def _random_delayed_batch(mcfg, B: int, T: int, seed: int = 0):
    """Random valid UNDELAYED samples of equal length -> delayed batch.

    Returns grid [B, S, T + max_delay], loss_mask (same), channel [B, T + max_delay].
    """
    g = torch.Generator().manual_seed(seed)
    S = mcfg.n_streams
    grids, masks, chans = [], [], []
    for _ in range(B):
        grid = torch.zeros((S, T), dtype=torch.long)
        grid[0] = torch.randint(64, mcfg.text_vocab_size, (T,), generator=g)
        grid[1:] = torch.randint(0, mcfg.audio_codec_vocab, (S - 1, T), generator=g)
        mask = torch.ones((S, T), dtype=torch.bool)
        channel = torch.zeros((T,), dtype=torch.long)
        channel[T // 2 :] = 1
        dg, dm, dc = apply_delay(grid, mask, channel, mcfg.audio_codec_vocab)
        grids.append(dg)
        masks.append(dm)
        chans.append(dc)
    return torch.stack(grids), torch.stack(masks), torch.stack(chans)


def _periodic_delayed_batch(mcfg, B: int = 2, T: int = 32):
    """Highly learnable periodic batch (identical rows) for the overfit test."""
    S = mcfg.n_streams
    t = torch.arange(T)
    grid = torch.zeros((S, T), dtype=torch.long)
    grid[0] = 64 + (t % 7)
    for k in range(S - 1):
        grid[1 + k] = (t % 5) + 11 * k
    mask = torch.ones((S, T), dtype=torch.bool)
    channel = torch.zeros((T,), dtype=torch.long)
    dg, dm, dc = apply_delay(grid, mask, channel, mcfg.audio_codec_vocab)
    return (
        dg.unsqueeze(0).repeat(B, 1, 1),
        dm.unsqueeze(0).repeat(B, 1, 1),
        dc.unsqueeze(0).repeat(B, 1),
    )


def test_forward_shapes(test_cfg) -> None:
    mcfg = test_cfg.model
    model = _model(test_cfg)
    grid, _, channel = _random_delayed_batch(mcfg, B=2, T=17)
    model.eval()
    with torch.no_grad():
        out = model(grid, channel)
    assert isinstance(out, ModelOutput)
    Tp = 17 + mcfg.max_delay
    assert out.text_logits.shape == (2, Tp, mcfg.text_vocab_size)
    assert out.audio_logits.shape == (2, mcfg.n_codebooks, Tp, mcfg.audio_vocab_size)
    assert out.text_logits.dtype == torch.float32
    assert out.audio_logits.dtype == torch.float32
    assert torch.isfinite(out.text_logits).all()
    assert torch.isfinite(out.audio_logits).all()


def test_loss_finite_and_metric_keys(test_cfg) -> None:
    model = _model(test_cfg, seed=1)
    grid, mask, channel = _random_delayed_batch(test_cfg.model, B=2, T=15, seed=1)
    model.train()
    out = model(grid, channel)
    loss, metrics = model.loss(out, grid, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    for key in ("loss", "loss/text", "loss/audio_0", "loss/audio_1"):
        assert key in metrics, f"missing {key} in {sorted(metrics)}"
        assert torch.isfinite(torch.as_tensor(metrics[key])).all()
    assert float(metrics["loss"]) == pytest.approx(float(loss.detach()), rel=1e-4)


def test_overfit_one_batch(test_cfg) -> None:
    model = _model(test_cfg, seed=1)
    grid, mask, channel = _periodic_delayed_batch(test_cfg.model)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    first = last = None
    for _ in range(80):
        out = model(grid, channel)
        loss, _ = model.loss(out, grid, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert last < 0.4 * first, f"80 Adam steps: loss {first:.3f} -> {last:.3f} (<60% drop)"


def test_kv_cache_parity_forward_vs_prefill_step(test_cfg) -> None:
    """Load-bearing: full forward logits == prefill + step over the same delayed grid."""
    mcfg = test_cfg.model
    model = _model(test_cfg, seed=2)
    model.eval()

    sample = build_s2s(
        [(
            torch.randint(0, mcfg.audio_codec_vocab, (mcfg.n_codebooks, 5)),
            [70, 71, 72, 73],
            torch.randint(0, mcfg.audio_codec_vocab, (mcfg.n_codebooks, 6)),
        )],
        mcfg.n_codebooks,
        mcfg.audio_codec_vocab,
    )
    dg, _, dc = apply_delay(
        sample.grid, sample.loss_mask, sample.channel, mcfg.audio_codec_vocab
    )
    grid = dg.unsqueeze(0)      # [1, S, T']
    channel = dc.unsqueeze(0)   # [1, T']
    Tp = grid.shape[-1]

    with torch.no_grad():
        full = model(grid, channel)

        cache = KVCache.allocate(mcfg, 1, torch.device("cpu"), torch.float32)
        T0 = Tp // 2
        tl, al = model.prefill(grid[:, :, :T0], channel[:, :T0], cache)
        assert cache.pos == T0
        assert tl.shape == (1, mcfg.text_vocab_size)
        assert al.shape == (1, mcfg.n_codebooks, mcfg.audio_vocab_size)
        torch.testing.assert_close(
            tl, full.text_logits[:, T0 - 1], atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            al, full.audio_logits[:, :, T0 - 1], atol=1e-4, rtol=1e-5
        )

        for p in range(T0, Tp):
            tl, al = model.step(grid[:, :, p], channel[:, p], cache)
            assert cache.pos == p + 1
            torch.testing.assert_close(
                tl, full.text_logits[:, p], atol=1e-4, rtol=1e-5,
                msg=lambda m: f"text logits diverge at step {p}: {m}",
            )
            torch.testing.assert_close(
                al, full.audio_logits[:, :, p], atol=1e-4, rtol=1e-5,
                msg=lambda m: f"audio logits diverge at step {p}: {m}",
            )


def test_grad_checkpoint_matches(test_cfg) -> None:
    base = _model(test_cfg, seed=3)
    ck_cfg = dataclasses.replace(test_cfg.model, grad_checkpoint=True)
    torch.manual_seed(3)
    ck = OmniModel(ck_cfg)
    ck.init_weights()
    ck.load_state_dict(base.state_dict())

    grid, mask, channel = _random_delayed_batch(test_cfg.model, B=2, T=12, seed=4)
    base.train()
    ck.train()
    out_a = base(grid, channel)
    out_b = ck(grid, channel)
    torch.testing.assert_close(out_b.text_logits, out_a.text_logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_b.audio_logits, out_a.audio_logits, atol=1e-6, rtol=1e-6)

    loss_a, _ = base.loss(out_a, grid, mask)
    loss_b, _ = ck.loss(out_b, grid, mask)
    loss_a.backward()
    loss_b.backward()
    grads_a = {n: p.grad for n, p in base.named_parameters()}
    for n, p in ck.named_parameters():
        assert p.grad is not None, f"missing grad through checkpoint for {n}"
        torch.testing.assert_close(p.grad, grads_a[n], atol=1e-6, rtol=1e-6)


def test_param_counts_and_embed(test_cfg) -> None:
    mcfg = test_cfg.model
    model = _model(test_cfg, seed=5)
    counts = model.param_counts()
    assert counts["total"] > counts["non_embedding"] > 0
    grid, _, channel = _random_delayed_batch(mcfg, B=2, T=6, seed=6)
    emb = model.embed(grid, channel)
    assert emb.shape == (2, 6 + mcfg.max_delay, mcfg.d_model)
    assert torch.isfinite(emb).all()


def test_save_load_pretrained(tmp_path, test_cfg) -> None:
    model = _model(test_cfg, seed=7)
    save_dir = tmp_path / "m"
    model.save_pretrained(save_dir)
    assert (save_dir / "model.safetensors").exists()
    assert (save_dir / "config.yaml").exists()

    loaded = OmniModel.from_pretrained(save_dir)
    grid, _, channel = _random_delayed_batch(test_cfg.model, B=1, T=9, seed=8)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        a = model(grid, channel)
        b = loaded(grid, channel)
    torch.testing.assert_close(b.text_logits, a.text_logits)
    torch.testing.assert_close(b.audio_logits, a.audio_logits)
