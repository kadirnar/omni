"""Tests for omni.streams (frozen format core): delay pattern, specials, helpers."""

from __future__ import annotations

import pytest
import torch

from omni import streams
from omni.streams import (
    PAD,
    TEXT_STREAM,
    Sample,
    apply_delay,
    audio_bos_id,
    audio_eos_id,
    audio_pad_id,
    audio_vocab_size,
    delays,
    max_delay,
    sanitize_codes,
    trim_audio_at_eos,
    undelay,
)

CV = 2048  # codec vocab used throughout


def _rand(n_q: int, T: int, seed: int = 0):
    """Random undelayed grid [S, T] with valid text/audio ids, mask, channel."""
    g = torch.Generator().manual_seed(seed)
    S = 1 + n_q
    grid = torch.zeros((S, T), dtype=torch.long)
    grid[TEXT_STREAM] = torch.randint(64, 320, (T,), generator=g)
    grid[1:] = torch.randint(0, CV, (n_q, T), generator=g)
    mask = torch.randint(0, 2, (S, T), generator=g).bool()
    channel = torch.randint(0, 2, (T,), generator=g)
    return grid, mask, channel


@pytest.mark.parametrize("n_q", [2, 8])
def test_delay_roundtrip(n_q: int) -> None:
    grid, mask, channel = _rand(n_q, T=13, seed=n_q)
    d, m, c = apply_delay(grid, mask, channel, CV)
    S, T = grid.shape
    assert d.shape == (S, T + n_q)
    assert m.shape == (S, T + n_q)
    assert c.shape == (T + n_q,)
    assert torch.equal(undelay(d), grid)
    # the mask shifts by exactly the same per-stream delays
    assert torch.equal(undelay(m), mask)


@pytest.mark.parametrize("n_q", [2, 8])
def test_delay_fillers_and_mask_shift(n_q: int) -> None:
    grid, mask, channel = _rand(n_q, T=11, seed=n_q + 10)
    d, m, c = apply_delay(grid, mask, channel, CV)
    T = grid.shape[1]
    apad = audio_pad_id(CV)
    dl = delays(n_q)
    assert dl == [0] + [k + 1 for k in range(n_q)]
    assert max_delay(n_q) == n_q

    # text row: delay 0, PAD tail, no loss on filler
    assert torch.equal(d[TEXT_STREAM, :T], grid[TEXT_STREAM])
    assert (d[TEXT_STREAM, T:] == PAD).all()
    assert not m[TEXT_STREAM, T:].any()

    for k in range(n_q):
        s, dk = 1 + k, dl[1 + k]
        assert dk == k + 1
        assert (d[s, :dk] == apad).all(), "head filler must be AUDIO_PAD"
        assert (d[s, dk + T :] == apad).all(), "tail filler must be AUDIO_PAD"
        assert torch.equal(d[s, dk : dk + T], grid[s])
        assert torch.equal(m[s, dk : dk + T], mask[s])
        assert not m[s, :dk].any() and not m[s, dk + T :].any()

    # channel extends with its last value
    assert torch.equal(c[:T], channel)
    assert (c[T:] == channel[-1]).all()


@pytest.mark.parametrize("n_q", [1, 2, 8, 32])
def test_lead_mode_delays(n_q: int) -> None:
    dl = delays(n_q, "lead")
    assert dl == [0] + [1] + [2] * (n_q - 1)
    assert max_delay(n_q, "lead") == max(dl)
    # semantic codebook leads every acoustic codebook by exactly one frame
    if n_q > 1:
        assert all(d - dl[1] == 1 for d in dl[2:])


@pytest.mark.parametrize("mode", ["stagger", "flat", "lead"])
@pytest.mark.parametrize("duplex", [False, True])
def test_delay_roundtrip_all_modes(mode: str, duplex: bool) -> None:
    n_q = 4
    S = 1 + (2 * n_q if duplex else n_q)
    g = torch.Generator().manual_seed(7)
    grid = torch.zeros((S, 15), dtype=torch.long)
    grid[TEXT_STREAM] = torch.randint(64, 320, (15,), generator=g)
    grid[1:] = torch.randint(0, CV, (S - 1, 15), generator=g)
    mask = torch.randint(0, 2, (S, 15), generator=g).bool()
    channel = torch.randint(0, 2, (15,), generator=g)
    d, m, c = apply_delay(grid, mask, channel, CV, mode=mode, duplex=duplex)
    D = max(streams.stream_delays(n_q, mode, duplex))
    assert d.shape == (S, 15 + D)
    assert torch.equal(undelay(d, mode=mode, duplex=duplex), grid)
    assert torch.equal(undelay(m, mode=mode, duplex=duplex), mask)


def test_turn_prefix_intensity_requires_style() -> None:
    with pytest.raises(ValueError, match="response style"):
        streams.turn_prefix(intensity="hi")
    ids = streams.turn_prefix(response_style="happy", intensity="hi")
    assert ids == [streams.EMO_RSP, streams.EMOTION_CLASSES["happy"], streams.INTENSITY_HI]


def test_apply_delay_empty_grid() -> None:
    n_q = 2
    grid = torch.zeros((1 + n_q, 0), dtype=torch.long)
    mask = torch.zeros((1 + n_q, 0), dtype=torch.bool)
    channel = torch.zeros((0,), dtype=torch.long)
    d, m, c = apply_delay(grid, mask, channel, CV)
    assert d.shape == (3, n_q) and m.shape == (3, n_q) and c.shape == (n_q,)
    assert torch.equal(undelay(d), grid)


def test_special_ids_pinned() -> None:
    assert audio_pad_id(CV) == CV
    assert audio_bos_id(CV) == CV + 1
    assert audio_eos_id(CV) == CV + 2
    assert audio_vocab_size(CV) == CV + 3
    assert streams.PAD == 0
    assert streams.BOS == 1
    assert streams.EOS == 2
    assert streams.TEXT_PAD == 3
    assert streams.USER == 4
    assert streams.ASSISTANT == 5
    assert streams.END_OF_TURN == 6
    assert streams.N_RESERVED_SPECIALS == 64
    assert streams.TEXT_STREAM == 0
    assert streams.SPECIAL_TOKENS["<pad>"] == 0
    assert streams.SPECIAL_TOKENS["<s2s>"] == streams.TASK_S2S
    assert streams.TASK_TAGS == {
        "asr": streams.TASK_ASR,
        "tts": streams.TASK_TTS,
        "s2s": streams.TASK_S2S,
        "alm": streams.TASK_ALM,
        "ser": streams.TASK_SER,  # v4 (DESIGN_V4_EMOTION_I18N.md)
    }


def test_sanitize_codes() -> None:
    codes = torch.tensor([[5, CV, CV + 2], [0, CV + 1, 7]])
    out = sanitize_codes(codes, CV)
    assert torch.equal(out, torch.tensor([[5, 0, 0], [0, 0, 7]]))
    raw = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(sanitize_codes(raw, CV), raw)


def test_trim_audio_at_eos() -> None:
    eos = audio_eos_id(CV)
    codes = torch.tensor([[3, eos, 4, 5], [7, 8, 9, 10]])
    assert torch.equal(trim_audio_at_eos(codes, CV), codes[:, :1])
    no_eos = torch.tensor([[3, 4], [5, 6]])
    assert torch.equal(trim_audio_at_eos(no_eos, CV), no_eos)
    # AUDIO_EOS on a non-semantic row does not trim
    other_row = torch.tensor([[3, 4], [eos, 6]])
    assert torch.equal(trim_audio_at_eos(other_row, CV), other_row)


def test_sample_validate() -> None:
    grid, mask, channel = _rand(2, 9, seed=3)
    s = Sample(grid=grid, loss_mask=mask, channel=channel, task="tts")
    s.validate(CV, 320)
    assert s.n_streams == 3
    assert s.n_frames == 9

    bad = Sample(grid=grid.clone(), loss_mask=mask, channel=channel)
    bad.grid[TEXT_STREAM, 0] = 320  # text id out of vocab
    with pytest.raises(AssertionError):
        bad.validate(CV, 320)

    wrong_shape = Sample(grid=grid, loss_mask=mask[:, :-1], channel=channel)
    with pytest.raises(AssertionError):
        wrong_shape.validate(CV, 320)
