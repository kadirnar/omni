"""Tests for omni.grids builders: layout and loss-mask per the docstring spec."""

from __future__ import annotations

import pytest
import torch

from omni.grids import (
    build_asr,
    build_asr_prompt,
    build_audiolm,
    build_s2s,
    build_s2s_prompt,
    build_textlm,
    build_tts,
    build_tts_prompt,
    prompt_forced_text,
)
from omni.streams import (
    ASSISTANT,
    BOS,
    CHANNEL_ASSISTANT,
    CHANNEL_USER,
    END_OF_TURN,
    EOS,
    TASK_TAGS,
    TEXT_PAD,
    USER,
    audio_eos_id,
    audio_pad_id,
)

NQ = 2
CV = 2048
TV = 320
APAD = audio_pad_id(CV)
AEOS = audio_eos_id(CV)


def _codes(T: int, base: int = 0) -> torch.Tensor:
    """Deterministic raw codes [NQ, T] (no specials)."""
    return (torch.arange(NQ * T, dtype=torch.long).reshape(NQ, T) + base) % CV


def test_textlm_layout() -> None:
    ids = [100, 101, 102]
    s = build_textlm(ids, NQ, CV)
    s.validate(CV, TV)
    assert s.task == "textlm"
    assert s.grid[0].tolist() == [BOS, *ids, EOS]
    assert (s.grid[1:] == APAD).all(), "textlm audio rows are all AUDIO_PAD"
    assert s.loss_mask[0].all()
    assert not s.loss_mask[1:].any(), "no audio loss in textlm"
    assert (s.channel == CHANNEL_ASSISTANT).all()


def test_audiolm_layout() -> None:
    T = 6
    codes = _codes(T, base=9)
    s = build_audiolm(codes, NQ, CV)
    s.validate(CV, TV)
    assert s.task == "alm"
    # <bos> <alm> [T code cols] [eos-frame] <eos>
    assert s.n_frames == T + 4
    assert s.grid[0].tolist() == [BOS, TASK_TAGS["alm"], *([TEXT_PAD] * (T + 1)), EOS]
    assert torch.equal(s.grid[1:, 2 : 2 + T], codes)
    assert (s.grid[1:, :2] == APAD).all()
    # EOS-frame: codebook 0 = AUDIO_EOS, higher codebooks = AUDIO_PAD
    assert int(s.grid[1, 2 + T]) == AEOS
    assert (s.grid[2:, 2 + T] == APAD).all()
    assert s.loss_mask[0].tolist() == [True, False] + [True] * (T + 2)
    exp_amask = [False, False] + [True] * (T + 1) + [False]
    assert s.loss_mask[1].tolist() == exp_amask
    assert s.loss_mask[2].tolist() == exp_amask
    assert (s.channel == CHANNEL_ASSISTANT).all()


def test_asr_layout() -> None:
    Tu, ids = 4, [200, 201, 202]
    u = _codes(Tu, base=5)
    s = build_asr(u, ids, NQ, CV)
    s.validate(CV, TV)
    assert s.task == "asr"
    # <bos> <asr> [user seg] <end_of_turn> <assistant> w.. <eos>
    assert s.n_frames == Tu + len(ids) + 5
    assert s.grid[0].tolist() == [
        BOS, TASK_TAGS["asr"], USER, TEXT_PAD, TEXT_PAD, TEXT_PAD,
        END_OF_TURN, ASSISTANT, *ids, EOS,
    ]
    assert torch.equal(s.grid[1:, 2 : 2 + Tu], u)
    assert (s.grid[1:, :2] == APAD).all()
    assert (s.grid[1:, 2 + Tu :] == APAD).all()
    # bos/task cols, user seg, <end_of_turn>, injected <assistant>: never targets
    assert not s.loss_mask[:, : 2 + Tu + 2].any()
    # transcript ids + <eos> are text targets
    assert s.loss_mask[0, 2 + Tu + 2 :].all()
    assert not s.loss_mask[1:].any(), "asr has no audio loss anywhere"
    exp_ch = (
        [CHANNEL_ASSISTANT] * 2
        + [CHANNEL_USER] * Tu
        + [CHANNEL_ASSISTANT] * (len(ids) + 3)
    )
    assert s.channel.tolist() == exp_ch


def test_tts_layout_audio_longer_than_text() -> None:
    ids, Ta = [110, 111], 5
    codes = _codes(Ta, base=30)
    s = build_tts(ids, codes, NQ, CV)
    s.validate(CV, TV)
    assert s.task == "tts"
    # segment length max(Ta + 1, k + 2) = 6 -> total 2 + 6 + 1
    assert s.n_frames == 9
    assert s.grid[0].tolist() == [
        BOS, TASK_TAGS["tts"], ASSISTANT, 110, 111, END_OF_TURN,
        TEXT_PAD, TEXT_PAD, EOS,
    ]
    assert torch.equal(s.grid[1:, 2:7], codes)
    assert int(s.grid[1, 7]) == AEOS and (s.grid[2:, 7] == APAD).all()
    # <bos>, <tts>, injected <assistant> unmasked; everything after masked
    assert s.loss_mask[0].tolist() == [False, False, False] + [True] * 6
    exp_amask = [False, False] + [True] * 6 + [False]
    assert s.loss_mask[1].tolist() == exp_amask
    assert s.loss_mask[2].tolist() == exp_amask
    assert (s.channel == CHANNEL_ASSISTANT).all()


def test_tts_layout_text_longer_than_audio() -> None:
    ids, Ta = [120, 121, 122, 123, 124, 125], 3
    codes = _codes(Ta, base=40)
    s = build_tts(ids, codes, NQ, CV)
    s.validate(CV, TV)
    # text layout (k + 2 = 8) > Ta + 1 = 4 -> segment stretches to 8, total 11
    assert s.n_frames == 11
    assert s.grid[0].tolist() == [BOS, TASK_TAGS["tts"], ASSISTANT, *ids, END_OF_TURN, EOS]
    assert torch.equal(s.grid[1:, 2:5], codes)
    # EOS-frame right after the audio while the text keeps going
    assert int(s.grid[1, 5]) == AEOS and (s.grid[2:, 5] == APAD).all()
    # audio exhausted: APAD columns with audio loss off, text still a target
    assert (s.grid[1:, 6:] == APAD).all()
    assert not s.loss_mask[1:, 6:].any()
    assert s.loss_mask[0, 3:].all()
    assert not s.loss_mask[0, :3].any()
    exp_amask = [False, False] + [True] * 4 + [False] * 5
    assert s.loss_mask[1].tolist() == exp_amask
    assert s.loss_mask[2].tolist() == exp_amask


def test_s2s_layout_two_turns() -> None:
    u1, t1, a1 = _codes(4, 1), [130, 131], _codes(5, 2)
    u2, t2, a2 = _codes(3, 3), [140], _codes(4, 4)
    s = build_s2s([(u1, t1, a1), (u2, t2, a2)], NQ, CV)
    s.validate(CV, TV)
    assert s.task == "s2s"
    # cols: 0 bos | 1 tag | 2-5 user1 | 6 eot | 7-12 seg1 (len 6)
    #       13-15 user2 | 16 eot | 17-21 seg2 (len 5) | 22 eos
    assert s.n_frames == 23

    ch = s.channel.tolist()
    assert ch[2:6] == [CHANNEL_USER] * 4
    assert ch[13:16] == [CHANNEL_USER] * 3
    assert ch[:2] == [CHANNEL_ASSISTANT] * 2
    assert ch[6:13] == [CHANNEL_ASSISTANT] * 7
    assert ch[16:] == [CHANNEL_ASSISTANT] * 7

    # user segments fully unmasked (incl. the <end_of_turn> after them)
    assert not s.loss_mask[:, 2:7].any()
    assert not s.loss_mask[:, 13:17].any()
    assert torch.equal(s.grid[1:, 2:6], u1)
    assert torch.equal(s.grid[1:, 13:16], u2)
    assert s.grid[0, 2] == USER and s.grid[0, 13] == USER

    # first assistant segment: <assistant> unmasked, rest of text masked on
    assert s.grid[0, 7] == ASSISTANT and not s.loss_mask[0, 7]
    assert s.grid[0, 8:10].tolist() == t1
    assert s.grid[0, 10] == END_OF_TURN
    assert s.loss_mask[0, 8:13].all()
    assert torch.equal(s.grid[1:, 7:12], a1)
    assert int(s.grid[1, 12]) == AEOS and (s.grid[2:, 12] == APAD).all()
    assert s.loss_mask[1, 7:13].all() and s.loss_mask[2, 7:13].all()

    # second assistant segment
    assert s.grid[0, 17] == ASSISTANT and not s.loss_mask[0, 17]
    assert int(s.grid[0, 18]) == t2[0]
    assert torch.equal(s.grid[1:, 17:21], a2)
    assert int(s.grid[1, 21]) == AEOS

    # trailing document <eos>
    assert s.grid[0, 22] == EOS and s.loss_mask[0, 22]


def test_inference_prompts_have_no_loss() -> None:
    u = _codes(3, 7)
    s2s_p = build_s2s_prompt(u, NQ, CV)
    asr_p = build_asr_prompt(u, NQ, CV)
    for p in (s2s_p, asr_p):
        assert not p.loss_mask.any()
        assert p.n_frames == 2 + 3 + 1
        assert p.grid[0, 0] == BOS
        assert p.grid[0, 2] == USER
        assert p.grid[0, -1] == END_OF_TURN
        assert torch.equal(p.grid[1:, 2:5], u)
        assert p.channel[2:5].tolist() == [CHANNEL_USER] * 3
    assert s2s_p.grid[0, 1] == TASK_TAGS["s2s"]
    assert asr_p.grid[0, 1] == TASK_TAGS["asr"]

    tts_p = build_tts_prompt(NQ, CV)
    assert tts_p.grid[0].tolist() == [BOS, TASK_TAGS["tts"]]
    assert (tts_p.grid[1:] == APAD).all()
    assert not tts_p.loss_mask.any()


def test_prompt_forced_text() -> None:
    assert prompt_forced_text("s2s") == [ASSISTANT]
    assert prompt_forced_text("asr") == [ASSISTANT]
    assert prompt_forced_text("tts", [8, 9]) == [ASSISTANT, 8, 9, END_OF_TURN]
    assert prompt_forced_text("alm") == []
    with pytest.raises(ValueError):
        prompt_forced_text("nope")


def test_builders_reject_special_codes() -> None:
    bad = torch.full((NQ, 3), CV, dtype=torch.long)  # AUDIO_PAD is not a raw code
    with pytest.raises(AssertionError):
        build_tts([100], bad, NQ, CV)
    with pytest.raises(AssertionError):
        build_audiolm(bad, NQ, CV)
