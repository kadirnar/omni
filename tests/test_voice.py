"""Extensions v5: reference-voice cloning (docs/DESIGN_V5_VOICE.md "Grid layout
(binding)", INTERFACES.md "Extensions v5").

Coverage map:
- frozen-core voice segment (streams ids 49..51 + grids voice_segment /
  voice_codes kwargs): layout, masking, delay roundtrips, rejects — these run
  against already-landed code;
- peer APIs that land concurrently (prepare voice_p / sample_voice_ref,
  OmniGenerator.set_voice / voice_wav / voice_codes, DuplexGenerator
  voice_codes, benchmark_decode voice_frames, chat --voice): their names are
  imported/exercised lazily inside each test, so a missing peer fails that test
  in isolation, never collection.

Backward compatibility is pinned with GOLDEN values captured by running the
PRE-v5 code of this repo (torch 2.12.1, CPU — scratch script equivalent to the
`_duplex_*` / `_dataset_sha` helpers below, same seeds): with no voice, the
duplex tick loop and prepare_s2s must stay bit-identical.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest
import torch
import yaml

from omni.config import SamplingConfig
from omni.grids import (
    build_asr,
    build_asr_prompt,
    build_duplex,
    build_s2s,
    build_s2s_prompt,
    build_tts,
    build_tts_prompt,
)
from omni.infer.duplex import DuplexGenerator
from omni.infer.generate import OmniGenerator
from omni.model.omni import OmniModel
from omni.streams import (
    ASSISTANT,
    BOS,
    CHANNEL_ASSISTANT,
    TASK_TAGS,
    TEXT_PAD,
    VOICE,
    VOICE_END,
    apply_delay,
    audio_eos_id,
    audio_pad_id,
    max_delay,
    undelay,
)

NQ = 2  # matches test_cfg.model.n_codebooks
CV = 2048  # matches test_cfg.model.audio_codec_vocab
TV = 320  # ByteTokenizer vocab (test_cfg.model.text_vocab_size)
APAD = audio_pad_id(CV)
AEOS = audio_eos_id(CV)

# ---------------------------------------------------------------------------
# Pre-v5 goldens (bit-stability contract for the default / no-voice paths).
# Captured from the pre-voice code with the exact recipes reproduced by
# `_duplex_user_frames` / `_build_model` / `_dataset_sha` below.
# ---------------------------------------------------------------------------
_PRE_V5_SAMPLED_TEXT = [139, 295, 159, 217, 230, 57, 221, 221, 214, 255, 299, 52, 165, 37]
_PRE_V5_SAMPLED_FRAMES = [
    [-1, -1], [-1, -1], [1759, 609], [223, 1737], [723, 32], [804, 1988],
    [1349, 609], [223, 227], [986, 1711], [401, 585], [1912, 540], [390, 57],
    [1319, 1014], [62, 117],
]
_PRE_V5_GREEDY_TEXT = [1, 1, 1, 314, 314, 314, 314, 314, 314, 314, 314, 314, 314, 314]
_PRE_V5_GREEDY_FRAMES = [
    [-1, -1], [-1, -1], [1213, 1466], [1213, 1869], [386, 1305], [1962, 1815],
    [1060, 1376], [1802, 1393], [246, 348], [658, 348], [1275, 417],
    [1422, 787], [1281, 1104], [386, 1554],
]
_PRE_V5_S2S_N_SAMPLES = 6
# Recaptured 2026-07-03: fake_dialogues now rotates languages with per-language
# word pools and SineTTS applies a deterministic lang shift (multilingual pass)
# — an intentional content change. The invariant this golden protects is
# unchanged: the default/no-voice prepare_s2s path must stay bit-stable.
_PRE_V5_S2S_SHA = "d3764116be75ea281855d6edb5ae3091328d8aa80751cb775f8abd7eddee5491"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ref(R: int, seed: int = 0) -> torch.Tensor:
    """Deterministic raw reference codes [NQ, R]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, CV, (NQ, R), generator=g)


def _codes(T: int, base: int = 0) -> torch.Tensor:
    """Deterministic raw codes [NQ, T] (no specials)."""
    return (torch.arange(NQ * T, dtype=torch.long).reshape(NQ, T) + base) % CV


def _apad_tracks(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(user, assistant) duplex tracks [NQ, T]: APAD with short code regions."""
    g = torch.Generator().manual_seed(seed)
    user = torch.full((NQ, T), APAD, dtype=torch.long)
    asst = torch.full((NQ, T), APAD, dtype=torch.long)
    user[:, 1:5] = torch.randint(0, CV, (NQ, 4), generator=g)
    asst[:, 5:9] = torch.randint(0, CV, (NQ, 4), generator=g)
    return user, asst


def _build_model(model_cfg) -> OmniModel:
    """Deterministic random-init model (same recipe as the golden capture)."""
    torch.manual_seed(0)
    model = OmniModel(model_cfg)
    model.init_weights()
    return model.eval()


def _same_result(a, b) -> bool:
    return (
        a.text_ids == b.text_ids
        and a.frames == b.frames
        and torch.equal(a.audio_codes, b.audio_codes)
    )


def _assert_voice_prefix(s, plain, ref: torch.Tensor) -> None:
    """Binding segment layout: <bos> [V] then byte-identical to the no-voice grid."""
    R = ref.shape[1]
    assert s.n_frames == plain.n_frames + R + 1
    assert s.task == plain.task
    assert int(s.grid[0, 0]) == BOS
    # text row over the segment: <voice> TEXT_PAD.. <voice_end>
    assert s.grid[0, 1 : R + 2].tolist() == [VOICE] + [TEXT_PAD] * (R - 1) + [VOICE_END]
    # audio rows: reference codes under cols 1..R, APAD under both markers' edges
    assert torch.equal(s.grid[1 : 1 + NQ, 1 : R + 1], ref)
    assert (s.grid[1 : 1 + NQ, 0] == APAD).all()
    assert (s.grid[1 : 1 + NQ, R + 1] == APAD).all()
    # loss masked on EVERY row over the whole segment (incl. <bos>)
    assert not s.loss_mask[:, : R + 2].any()
    # channel ASSISTANT throughout the segment
    assert (s.channel[: R + 2] == CHANNEL_ASSISTANT).all()
    # after the segment the grid is byte-identical to today's layout
    assert torch.equal(s.grid[:, R + 2 :], plain.grid[:, 1:])
    assert torch.equal(s.loss_mask[:, R + 2 :], plain.loss_mask[:, 1:])
    assert torch.equal(s.channel[R + 2 :], plain.channel[1:])


# ---------------------------------------------------------------------------
# Frozen-core: voice segment invariants (beyond the landed sanity checks)
# ---------------------------------------------------------------------------
def test_voice_segment_layout_tts() -> None:
    ids, Ta, R = [110, 111, 112], 5, 4
    codes, ref = _codes(Ta, base=30), _ref(R, seed=1)
    plain = build_tts(ids, codes, NQ, CV)
    s = build_tts(ids, codes, NQ, CV, voice_codes=ref)
    s.validate(CV, TV)
    _assert_voice_prefix(s, plain, ref)
    # task tag directly after <voice_end>
    assert int(s.grid[0, R + 2]) == TASK_TAGS["tts"]


def test_voice_prefix_all_builders_and_prompts() -> None:
    u, ids, a, ref = _codes(4, base=5), [200, 201], _codes(5, base=9), _ref(3, seed=2)
    cases = [
        (build_asr(u, ids, NQ, CV), build_asr(u, ids, NQ, CV, voice_codes=ref)),
        (
            build_s2s([(u, ids, a)], NQ, CV),
            build_s2s([(u, ids, a)], NQ, CV, voice_codes=ref),
        ),
        (build_s2s_prompt(u, NQ, CV), build_s2s_prompt(u, NQ, CV, voice_codes=ref)),
        (build_asr_prompt(u, NQ, CV), build_asr_prompt(u, NQ, CV, voice_codes=ref)),
        (build_tts_prompt(NQ, CV), build_tts_prompt(NQ, CV, voice_codes=ref)),
    ]
    for plain, s in cases:
        _assert_voice_prefix(s, plain, ref)
        # ONE [<bos>+voice] prefix, identical across tasks (KV-prefix reuse)
        assert torch.equal(s.grid[:, : ref.shape[1] + 2], cases[0][1].grid[:, : ref.shape[1] + 2])
    # inference prompts stay loss-free everywhere
    for _, s in cases[2:]:
        assert not s.loss_mask.any()


def test_voice_segment_delay_roundtrip_stagger_and_flat() -> None:
    ref = _ref(4, seed=3)
    s = build_tts([100, 101], _codes(5, base=7), NQ, CV, voice_codes=ref)
    for mode in ("stagger", "flat"):
        d, m, c = apply_delay(s.grid, s.loss_mask, s.channel, CV, mode=mode)
        D = max_delay(NQ, mode)
        assert d.shape == (1 + NQ, s.n_frames + D)
        assert c.shape == (s.n_frames + D,)
        assert torch.equal(undelay(d, mode=mode), s.grid)
        assert torch.equal(undelay(m, mode=mode), s.loss_mask)
    user, asst = _apad_tracks(10)
    dup = build_duplex(user, asst, [(2, [70])], NQ, CV, voice_codes=ref)
    for mode in ("stagger", "flat"):
        d, m, _ = apply_delay(
            dup.grid, dup.loss_mask, dup.channel, CV, mode=mode, duplex=True
        )
        assert torch.equal(undelay(d, mode=mode, duplex=True), dup.grid)
        assert torch.equal(undelay(m, mode=mode, duplex=True), dup.loss_mask)


def test_voice_segment_budget_edge_r1() -> None:
    ref1 = _ref(1, seed=4)
    p = build_tts_prompt(NQ, CV, voice_codes=ref1)
    assert p.grid[0].tolist() == [BOS, VOICE, VOICE_END, TASK_TAGS["tts"]]
    assert torch.equal(p.grid[1:, 1:2], ref1)
    assert (p.grid[1:, 0] == APAD).all() and (p.grid[1:, 2] == APAD).all()
    assert not p.loss_mask.any()
    plain = build_tts([100], _codes(3), NQ, CV)
    s = build_tts([100], _codes(3), NQ, CV, voice_codes=ref1)
    assert s.n_frames == plain.n_frames + 2  # R + 1 = 2 extra cols


def test_voice_segment_rejects_bad_refs() -> None:
    codes = _codes(4)
    empty = torch.zeros((NQ, 0), dtype=torch.long)  # R = 0
    specials = torch.full((NQ, 3), APAD, dtype=torch.long)  # not raw codes
    wrong_rows = torch.zeros((NQ + 1, 3), dtype=torch.long)
    floaty = torch.zeros((NQ, 3))  # wrong dtype
    for bad in (empty, specials, wrong_rows, floaty):
        with pytest.raises(AssertionError):
            build_tts([100], codes, NQ, CV, voice_codes=bad)
    user, asst = _apad_tracks(10)
    for bad in (empty, specials):
        with pytest.raises(AssertionError):
            build_duplex(user, asst, [], NQ, CV, voice_codes=bad)


def test_build_duplex_voice_layout() -> None:
    T, R = 10, 3
    ref = _ref(R, seed=5)
    user, asst = _apad_tracks(T, seed=6)
    words = [(0, [70]), (4, [71, 72])]
    plain = build_duplex(user, asst, words, NQ, CV)
    s = build_duplex(user, asst, words, NQ, CV, voice_codes=ref)
    S, L = s.grid.shape
    f0 = R + 2  # conversation frame f sits at col f + R + 2
    assert (S, L) == (1 + 2 * NQ, 1 + (R + 1) + T + 1)
    # text row: <bos> <voice> TEXT_PAD.. <voice_end>, then the monologue
    assert s.grid[0, : f0].tolist() == [BOS, VOICE] + [TEXT_PAD] * (R - 1) + [VOICE_END]
    assert int(s.grid[0, f0]) == 70  # word at frame 0 -> col f0
    assert s.grid[0, f0 + 4 : f0 + 6].tolist() == [71, 72]  # frame 4 -> col f0+4
    # assistant rows carry the reference; user rows are APAD over the segment
    assert torch.equal(s.grid[1 : 1 + NQ, 1 : 1 + R], ref)
    assert (s.grid[1 : 1 + NQ, R + 1] == APAD).all()
    assert (s.grid[1 + NQ :, : f0] == APAD).all()
    # both tracks shifted to cols f0..f0+T-1
    assert torch.equal(s.grid[1 : 1 + NQ, f0 : f0 + T], asst)
    assert torch.equal(s.grid[1 + NQ :, f0 : f0 + T], user)
    # loss: text + assistant rows FALSE over cols 0..R+1, resuming at col R+2
    assert not s.loss_mask[:, : f0].any()
    assert s.loss_mask[0, f0:].all()
    assert s.loss_mask[1 : 1 + NQ, f0 : f0 + T].all()
    assert not s.loss_mask[1 : 1 + NQ, f0 + T :].any()
    assert not s.loss_mask[1 + NQ :].any()  # user rows never contribute
    assert (s.channel == CHANNEL_ASSISTANT).all()
    # after the segment the grid is byte-identical to the no-voice layout
    assert torch.equal(s.grid[:, f0:], plain.grid[:, 1:])
    assert torch.equal(s.loss_mask[:, f0:], plain.loss_mask[:, 1:])


# ---------------------------------------------------------------------------
# Data preparation (agent D): sample_voice_ref + voice_p plumbing
# ---------------------------------------------------------------------------
def _contains_window(big: torch.Tensor, small: torch.Tensor) -> bool:
    w = small.shape[1]
    if w == 0 or w > big.shape[1]:
        return False
    return any(
        torch.equal(big[:, i : i + w], small) for i in range(big.shape[1] - w + 1)
    )


def test_sample_voice_ref_contract() -> None:
    from omni.data.prepare import sample_voice_ref

    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, CV, (NQ, 300), generator=g)
    ref = sample_voice_ref(codes, rng=random.Random(1))  # default 37..250 frames
    assert ref.ndim == 2 and ref.shape[0] == NQ
    assert 37 <= ref.shape[1] <= 250
    assert _contains_window(codes, ref), "reference must be a contiguous chunk"
    # deterministic for a fixed rng
    assert torch.equal(sample_voice_ref(codes, rng=random.Random(1)), ref)
    # explicit bounds respected
    r = sample_voice_ref(codes, min_frames=10, max_frames=20, rng=random.Random(2))
    assert 10 <= r.shape[1] <= 20 and _contains_window(codes, r)
    # clamps to the utterance length, never returns 0 frames
    short = codes[:, :5]
    r5 = sample_voice_ref(short, rng=random.Random(3))
    assert r5.shape == (NQ, 5) and torch.equal(r5, short)
    one = codes[:, :1]
    r1 = sample_voice_ref(one, rng=random.Random(4))
    assert r1.shape == (NQ, 1) and torch.equal(r1, one)


def test_sample_voice_ref_rejects_degenerate_bounds() -> None:
    """Inverted/degenerate bounds must raise, never exceed the requested max
    (regression: min_frames=50, max_frames=10 used to return a 50-frame chunk)."""
    from omni.data.prepare import sample_voice_ref

    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, CV, (NQ, 100), generator=g)
    with pytest.raises(ValueError):
        sample_voice_ref(codes, min_frames=50, max_frames=10, rng=random.Random(0))
    with pytest.raises(ValueError):
        sample_voice_ref(codes, min_frames=0, max_frames=0, rng=random.Random(0))
    # permissive lower bound stays allowed (result still never empty)
    r = sample_voice_ref(codes, min_frames=0, max_frames=3, rng=random.Random(0))
    assert 1 <= r.shape[1] <= 3


def test_distinct_ref_text_no_containment_either_direction(monkeypatch) -> None:
    """The reference text may not contain any target word-aligned, nor be
    contained in one (regression: the old append-on-collision repair could
    assemble a target across the join, leaking its exact frames — SineTTS
    is one frame per word — into the loss-masked voice segment)."""
    from omni.data import prepare
    from omni.data.synthesize import _fake_sentence

    # scripted draws pin the exact leak shape: draw1 is a prefix of the
    # target; appending draw2 would have produced 'hello world extra tail'
    # which word-contains the target 'hello world'. A whole-sentence redraw
    # must return draw2 untouched.
    draws = iter(["hello", "world extra tail", "clean fresh words"])
    monkeypatch.setattr(prepare, "_fake_sentence", lambda rng: next(draws))
    out = prepare._distinct_ref_text(random.Random(0), ["hello world"])
    assert out == "world extra tail"
    monkeypatch.undo()

    # property over the real pool, targets built adversarially from the very
    # draws the function will consume (the shape that leaked before the fix)
    for seed in range(50):
        obs = random.Random(seed)
        s1, s2 = _fake_sentence(obs), _fake_sentence(obs)
        targets = [s1 + " " + s2.split()[0], _fake_sentence(obs)]
        ref = prepare._distinct_ref_text(random.Random(seed), targets)
        padded_ref = f" {ref} "
        for t in targets:
            padded_t = f" {t} "
            assert padded_ref not in padded_t, "reference contained in a target"
            assert padded_t not in padded_ref, "target contained in the reference"


def _s2s_kwargs(test_cfg, byte_tok) -> dict:
    """The exact prepare_s2s inputs the pre-v5 golden was captured with."""
    from omni.audio.codec import FakeCodec
    from omni.data.synthesize import SineTTS, fake_dialogues

    return dict(
        dialogues=list(fake_dialogues(6, seed=0)),
        tts=SineTTS(),
        codec=FakeCodec(n_codebooks=NQ, codec_vocab=CV),
        tokenizer=byte_tok,
        cfg=test_cfg,
        max_samples=6,
        seed=0,
    )


def _dataset_sha(shard_dir) -> str:
    """Order- and content-exact hash over every sample of a shard dir."""
    from omni.data.dataset import ShardDataset

    ds = ShardDataset(shard_dir)
    h = hashlib.sha256()
    for i in range(len(ds)):
        s = ds[i]
        h.update(s.task.encode())
        h.update(s.grid.numpy().astype("<i8").tobytes())
        h.update(s.loss_mask.numpy().astype("u1").tobytes())
        h.update(s.channel.numpy().astype("<i8").tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def s2s_default_shards(tmp_path_factory: pytest.TempPathFactory, test_cfg, byte_tok):
    """prepare_s2s with pure defaults (the v4 call shape: no voice kwargs)."""
    from omni.data.prepare import prepare_s2s

    out = tmp_path_factory.mktemp("s2s_v4_default") / "shards"
    prepare_s2s(out, **_s2s_kwargs(test_cfg, byte_tok))
    return out


def test_prepare_s2s_default_path_matches_pre_v5_golden(s2s_default_shards) -> None:
    """No-voice prepare_s2s must stay BIT-IDENTICAL to the pre-v5 code."""
    from omni.data.dataset import ShardDataset

    ds = ShardDataset(s2s_default_shards)
    assert len(ds) == _PRE_V5_S2S_N_SAMPLES
    for i in range(len(ds)):
        assert not (ds[i].grid[0] == VOICE).any(), "default samples carry no segment"
    assert _dataset_sha(s2s_default_shards) == _PRE_V5_S2S_SHA


def test_prepare_s2s_voice_p0_bit_identical_to_default(
    tmp_path, test_cfg, byte_tok, s2s_default_shards
) -> None:
    """voice_p=0.0 must not perturb anything (rng draws included)."""
    from omni.data.prepare import prepare_s2s

    out = tmp_path / "p0"
    prepare_s2s(out, **_s2s_kwargs(test_cfg, byte_tok), voice_p=0.0)
    assert _dataset_sha(out) == _dataset_sha(s2s_default_shards) == _PRE_V5_S2S_SHA


def _voice_span(s) -> tuple[int, int]:
    row = s.grid[0]
    v = (row == VOICE).nonzero().flatten().tolist()
    e = (row == VOICE_END).nonzero().flatten().tolist()
    assert len(v) == 1 and len(e) == 1, "exactly one voice segment per sample"
    return int(v[0]), int(e[0])


def _assistant_audio_segments(s) -> list[torch.Tensor]:
    """Raw-code block of every assistant speech segment (up to its EOS-frame)."""
    row = s.grid[0]
    segs = []
    for c0 in (row == ASSISTANT).nonzero().flatten().tolist():
        c = c0
        while c < s.n_frames and int(s.grid[1, c]) != AEOS:
            c += 1
        segs.append(s.grid[1 : 1 + NQ, c0:c])
    return segs


def test_prepare_s2s_voice_p1_every_sample_conditioned(
    tmp_path, test_cfg, byte_tok
) -> None:
    """voice_p=1.0: every sample carries the segment; the reference is a
    DIFFERENT utterance than every target; the R+2 budget is respected."""
    from omni.audio.codec import FakeCodec
    from omni.data.dataset import ShardDataset
    from omni.data.prepare import prepare_s2s
    from omni.data.synthesize import SineTTS, fake_dialogues

    out = tmp_path / "voiced"
    prepare_s2s(
        out,
        dialogues=list(fake_dialogues(8, seed=1)),
        tts=SineTTS(),
        codec=FakeCodec(n_codebooks=NQ, codec_vocab=CV),
        tokenizer=byte_tok,
        cfg=test_cfg,
        max_samples=8,
        seed=0,
        voice_p=1.0,
    )
    ds = ShardDataset(out)
    assert len(ds) > 0
    msf = test_cfg.data.max_sample_frames
    for i in range(len(ds)):
        s = ds[i]
        assert s.task == "s2s"
        assert s.n_frames <= msf, "R + 2 budget must keep samples under the cap"
        v0, v1 = _voice_span(s)
        assert v0 == 1, "segment sits directly after <bos>"
        R = v1 - v0
        assert R >= 1
        assert int(s.grid[0, 0]) == BOS
        assert int(s.grid[0, v1 + 1]) == TASK_TAGS["s2s"], "task tag follows <voice_end>"
        assert (s.grid[0, v0 + 1 : v1] == TEXT_PAD).all(), "transcript-free segment"
        ref = s.grid[1 : 1 + NQ, v0:v1]
        assert int(ref.max()) < CV, "reference frames are raw codes"
        assert (s.grid[1 : 1 + NQ, v1] == APAD).all()
        assert not s.loss_mask[:, : v1 + 1].any(), "whole segment loss-masked"
        assert (s.channel[: v1 + 1] == CHANNEL_ASSISTANT).all()
        segs = _assistant_audio_segments(s)
        assert segs, "an s2s sample must contain assistant speech"
        for seg in segs:
            if seg.shape == ref.shape:
                assert not torch.equal(seg, ref), (
                    "reference must be a different utterance than every target"
                )


def test_prepare_asr_tts_speaker_column_guard(
    tmp_path, test_cfg, byte_tok, fake_codec
) -> None:
    """voice_p > 0 without a speaker column must raise before any dataset I/O."""
    from omni.data.prepare import prepare_asr_tts

    with pytest.raises(ValueError):
        prepare_asr_tts(
            tmp_path / "asr",
            dataset_id="local/does-not-exist",
            name=None,
            split="train",
            codec=fake_codec,
            tokenizer=byte_tok,
            cfg=test_cfg,
            max_samples=2,
            voice_p=0.5,
        )


# ---------------------------------------------------------------------------
# OmniGenerator (agent I): set_voice + per-call voice_wav / voice_codes
# ---------------------------------------------------------------------------
def test_generator_set_voice_seeded_and_clearable(test_cfg, byte_tok, fake_codec) -> None:
    model = _build_model(test_cfg.model)
    gen = OmniGenerator(model, test_cfg, device="cpu", tokenizer=byte_tok)
    spf = fake_codec.samples_per_frame
    ref_wav = torch.sin(torch.linspace(0, 500, spf * 8))

    r_plain = gen.tts("hi omni", fake_codec, max_frames=8, seed=3)
    gen.set_voice(ref_wav, fake_codec)
    r_v1 = gen.tts("hi omni", fake_codec, max_frames=8, seed=3)
    r_v2 = gen.tts("hi omni", fake_codec, max_frames=8, seed=3)
    assert _same_result(r_v1, r_v2), "voiced generation must be seed-reproducible"
    assert 0 <= r_v1.frames <= 8, "frames cap respected with a voice prompt"
    assert not _same_result(r_v1, r_plain), "voice prompt must change the run"
    gen.set_voice(None)  # clearing restores the bit-identical no-voice path
    assert _same_result(gen.tts("hi omni", fake_codec, max_frames=8, seed=3), r_plain)


def test_generator_voice_oneshot_paths(test_cfg, byte_tok, fake_codec) -> None:
    model = _build_model(test_cfg.model)
    gen = OmniGenerator(model, test_cfg, device="cpu", tokenizer=byte_tok)
    spf = fake_codec.samples_per_frame
    ref_wav = torch.sin(torch.linspace(0, 500, spf * 8))
    other_wav = torch.sin(torch.linspace(0, 900, spf * 8))
    ref_codes = fake_codec.encode(ref_wav)

    # one-shot voice_wav == one-shot voice_codes == set_voice session state
    r_wav = gen.tts("hello", fake_codec, max_frames=6, seed=5, voice_wav=ref_wav)
    r_codes = gen.tts("hello", fake_codec, max_frames=6, seed=5, voice_codes=ref_codes)
    assert _same_result(r_wav, r_codes)
    gen.set_voice(ref_wav, fake_codec)
    r_sess = gen.tts("hello", fake_codec, max_frames=6, seed=5)
    assert _same_result(r_sess, r_wav)
    # a per-call reference overrides the session state
    gen.set_voice(other_wav, fake_codec)
    r_over = gen.tts("hello", fake_codec, max_frames=6, seed=5, voice_codes=ref_codes)
    assert _same_result(r_over, r_wav)
    gen.set_voice(None)

    # s2s threads the reference; asr accepts it too (invariance path)
    user = torch.sin(torch.linspace(0, 300, spf * 5))
    r_s2s_v = gen.s2s(user, fake_codec, max_frames=6, seed=2, voice_wav=ref_wav)
    assert _same_result(
        r_s2s_v, gen.s2s(user, fake_codec, max_frames=6, seed=2, voice_wav=ref_wav)
    )
    assert not _same_result(r_s2s_v, gen.s2s(user, fake_codec, max_frames=6, seed=2))
    r_asr_v = gen.asr(user, fake_codec, max_frames=6, seed=2, voice_wav=ref_wav)
    assert _same_result(
        r_asr_v, gen.asr(user, fake_codec, max_frames=6, seed=2, voice_wav=ref_wav)
    )


def test_generator_voice_capacity_and_trim(test_cfg, byte_tok, fake_codec) -> None:
    """The max_frames clamp must account for the longer prompt; set_voice trims."""
    model = _build_model(test_cfg.model)
    gen = OmniGenerator(model, test_cfg, device="cpu", tokenizer=byte_tok)
    spf = fake_codec.samples_per_frame
    long_wav = torch.sin(torch.linspace(0, 4000, spf * 130))

    # default trim (125 frames) -> prompt 128 cols: no room in max_frames=128.
    # set_voice must stay OUTSIDE the raises block: it does not raise (it
    # trims), and the ValueError being pinned is generate()'s capacity clamp.
    gen.set_voice(long_wav, fake_codec)
    with pytest.raises(ValueError):
        gen.tts("hi", fake_codec, max_frames=8, seed=0)
    # explicit trim leaves room; the generated frames respect the shrunk budget
    gen.set_voice(long_wav, fake_codec, max_frames=100)
    r = gen.tts("hi", fake_codec, max_frames=50, seed=0)
    # prompt = <bos> + 101 segment cols + <tts> = 103; 128 - 103 - max_delay = 23
    assert 0 <= r.frames <= 23
    gen.set_voice(None)


# ---------------------------------------------------------------------------
# DuplexGenerator (agent I): voice prefix + no-voice bit-stability
# ---------------------------------------------------------------------------
def _duplex_user_frames(n_ticks: int) -> list[torch.Tensor | None]:
    """Deterministic user frames; the generator is consumed on EVERY tick so the
    None pattern cannot shift the draw sequence (matches the golden capture)."""
    g = torch.Generator().manual_seed(9)
    frames: list[torch.Tensor | None] = []
    for t in range(n_ticks):
        fr = torch.randint(0, CV, (NQ,), generator=g)
        frames.append(None if t % 4 == 3 else fr)
    return frames


def _tick_trace(gen: DuplexGenerator, n_ticks: int) -> tuple[list[int], list[list[int]]]:
    text: list[int] = []
    frames: list[list[int]] = []
    for fr in _duplex_user_frames(n_ticks):
        st = gen.step(fr)
        text.append(int(st.text_id))
        frames.append(
            [-1] * NQ if st.assistant_frame is None else [int(x) for x in st.assistant_frame]
        )
    return text, frames


def test_duplex_no_voice_bit_stable_vs_pre_v5(duplex_cfg) -> None:
    """Seeded no-voice ticks must reproduce the pre-v5 goldens exactly."""
    sampled_sc = SamplingConfig(
        text_temperature=0.7, text_top_k=25,
        audio_temperature=0.8, audio_top_k=250, max_frames=375,
    )
    gen = DuplexGenerator(
        _build_model(duplex_cfg.model), duplex_cfg, device="cpu",
        sampling=sampled_sc, seed=11,
    )
    text, frames = _tick_trace(gen, 14)
    assert text == _PRE_V5_SAMPLED_TEXT
    assert frames == _PRE_V5_SAMPLED_FRAMES

    greedy_sc = SamplingConfig(
        text_temperature=0.0, text_top_k=0,
        audio_temperature=0.0, audio_top_k=0, max_frames=375,
    )
    gen = DuplexGenerator(
        _build_model(duplex_cfg.model), duplex_cfg, device="cpu",
        sampling=greedy_sc, seed=11,
    )
    text, frames = _tick_trace(gen, 14)
    assert text == _PRE_V5_GREEDY_TEXT
    assert frames == _PRE_V5_GREEDY_FRAMES


def test_duplex_voice_reset_paths(duplex_cfg) -> None:
    model = _build_model(duplex_cfg.model)
    sc = SamplingConfig(
        text_temperature=0.7, text_top_k=25,
        audio_temperature=0.8, audio_top_k=250, max_frames=375,
    )
    ref = _ref(6, seed=3)

    gen_v = DuplexGenerator(
        model, duplex_cfg, device="cpu", sampling=sc, seed=5, voice_codes=ref
    )
    rec_v = _tick_trace(gen_v, 12)
    gen_v.reset()  # no-arg reset keeps the stored voice
    assert _tick_trace(gen_v, 12) == rec_v

    gen_p = DuplexGenerator(model, duplex_cfg, device="cpu", sampling=sc, seed=5)
    rec_p = _tick_trace(gen_p, 12)
    assert rec_p != rec_v, "the voice prefix must change the seeded run"
    gen_p.reset(voice_codes=ref)  # per-reset override == constructor voice
    assert _tick_trace(gen_p, 12) == rec_v


def test_duplex_voice_run_file_invariants(duplex_cfg, fake_codec) -> None:
    model = _build_model(duplex_cfg.model)
    spf = fake_codec.samples_per_frame
    wav = torch.sin(torch.linspace(0, 800, spf * 10))
    n_in = fake_codec.encode(wav).shape[1]

    gen = DuplexGenerator(model, duplex_cfg, device="cpu", seed=3, voice_codes=_ref(8, seed=4))
    text, out = gen.run_file(wav, fake_codec)
    assert isinstance(text, str)
    assert out.shape[0] == n_in * spf, "assistant track length invariant unchanged"
    assert torch.isfinite(out).all()

    # capacity: t_user + D + R + 2 must fit max_frames (adds R+2 vs today)
    long_wav = torch.sin(torch.linspace(0, 3000, spf * 100))
    plain = DuplexGenerator(model, duplex_cfg, device="cpu", seed=3)
    _, out_plain = plain.run_file(long_wav, fake_codec)  # 100 + 2 + 1 <= 128: fits
    assert out_plain.shape[0] == 100 * spf
    gen30 = DuplexGenerator(
        model, duplex_cfg, device="cpu", seed=3, voice_codes=_ref(30, seed=5)
    )
    with pytest.raises(ValueError):
        gen30.run_file(long_wav, fake_codec)  # 100 + 2 + 30 + 2 > 128


# ---------------------------------------------------------------------------
# perf + chat (agent I)
# ---------------------------------------------------------------------------
def test_benchmark_decode_voice_frames(test_cfg) -> None:
    from omni.optim.perf import benchmark_decode

    torch.manual_seed(0)
    model = OmniModel(test_cfg.model)
    res = benchmark_decode(model, test_cfg, "cpu", n_frames=2, voice_frames=32)
    for key in ("steps_per_s", "rtf", "ms_per_step", "prefill_ms", "n_frames", "batch"):
        assert key in res, f"standard benchmark key {key!r} missing"
    assert res["voice_prefill_ms"] > 0
    assert res["steps_per_s"] > 0 and res["rtf"] > 0


def _tiny_cfg_yaml(tmp_path) -> Path:
    """Shrunken tiny preset (mirrors the test_cfg dims) for CLI smoke runs."""
    cfg = {
        "preset": "tiny",
        "model": {
            "n_codebooks": NQ, "d_model": 64, "n_layers": 2, "n_heads": 2,
            "n_kv_heads": 1, "d_ff": 128, "max_frames": 128, "text_vocab_size": TV,
        },
        "data": {"max_sample_frames": 120},
        "codec": "fake",
        "tokenizer_path": "byte",
    }
    p = tmp_path / "tiny_voice.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_chat_voice_cli_tts_smoke(tmp_path) -> None:
    from omni.audio.codec import save_wav
    from omni.infer.chat import main as chat_main

    spf = 1920
    ref = tmp_path / "ref.wav"
    save_wav(ref, torch.sin(torch.linspace(0, 440, spf * 8)), 24_000)
    out = tmp_path / "out.wav"
    rc = chat_main(
        [
            "--task", "tts", "--text", "hi omni", "--out", str(out),
            "--codec", "fake", "--tokenizer", "byte",
            "--config", str(_tiny_cfg_yaml(tmp_path)),
            # 9 forced ids (7 bytes + <assistant> + <end_of_turn>) must fit:
            # short budgets now raise instead of truncating the monologue
            "--max-frames", "10", "--seed", "0", "--voice", str(ref),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_chat_voice_asr_warns_and_continues(tmp_path, capsys, recwarn) -> None:
    from omni.audio.codec import save_wav
    from omni.infer.chat import main as chat_main

    spf = 1920
    ref = tmp_path / "ref.wav"
    save_wav(ref, torch.sin(torch.linspace(0, 440, spf * 8)), 24_000)
    usr = tmp_path / "user.wav"
    save_wav(usr, torch.sin(torch.linspace(0, 200, spf * 4)), 24_000)
    rc = chat_main(
        [
            "--task", "asr", "--in", str(usr),
            "--codec", "fake", "--tokenizer", "byte",
            "--config", str(_tiny_cfg_yaml(tmp_path)),
            "--max-frames", "6", "--seed", "0", "--voice", str(ref),
        ]
    )
    assert rc == 0, "--voice with asr must warn and continue, not fail"
    err = capsys.readouterr().err.lower()
    warned = "voice" in err or any(
        "voice" in str(w.message).lower() for w in recwarn.list
    )
    assert warned, "asr + --voice should emit a warning mentioning the voice flag"
