"""Token-stream layout: special ids, delay pattern, and the Sample container.

This module is the format contract shared by data prep, the model, and inference.
See docs/DESIGN.md §1 for the rationale.

Grid convention
---------------
A sample is a grid of S = 1 + n_codebooks integer rows over T frame steps
(12.5 Hz, one step = 80 ms):

    row 0                : text stream (BPE ids + specials below)
    rows 1 .. n_codebooks: audio codebook streams (codec codes + audio specials)

Grids are stored and built UNDELAYED (audio frame t of every codebook sits in
column t, time-aligned with the text token of that step). `apply_delay` shifts
audio row k right by k+1 steps (text delay 0) right before batching, so the
delay pattern can change without re-tokenizing data. The model always consumes
DELAYED grids; generation writes sampled tokens back at their per-stream delays.

`loss_mask[s, t]` means: grid[s, t], when used as a *target*, contributes loss.
`channel[t]` is the turn owner of step t (CHANNEL_USER while the user speaks,
CHANNEL_ASSISTANT otherwise). After delay-shifting, trailing audio from a user
segment can overlap assistant-owned steps; channel keeps the turn owner there
(documented approximation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Special tokens — text stream (ids 0..63 reserved; BPE ids start at 64)
# ---------------------------------------------------------------------------
PAD = 0  # batch padding, never a loss target
BOS = 1  # document start
EOS = 2  # document end
TEXT_PAD = 3  # inner-monologue filler while audio continues
USER = 4  # user turn marker
ASSISTANT = 5  # assistant turn marker
END_OF_TURN = 6
TASK_ASR = 7
TASK_TTS = 8
TASK_S2S = 9
TASK_ALM = 10  # audio language modelling (speech continuation)

# --- v4 extensions (docs/DESIGN_V4_EMOTION_I18N.md): emotion, paralinguistics,
# languages. All ride the text stream as ordinary ids; grids are unchanged.
TASK_SER = 11  # speech-emotion understanding (v1: SER-tagged ASR)
EMO_PCV = 12  # marker: perceived user emotion follows (auto-predicted)
EMO_RSP = 13  # marker: chosen response style follows (auto, user-forceable)
INTENSITY_LO, INTENSITY_MD, INTENSITY_HI = 14, 15, 16

#: shared emotion/style inventory; SER perception is trained/evaluated on the
#: first 8, response styles may use all 12.
EMOTION_CLASSES: dict[str, int] = {
    "neutral": 17, "happy": 18, "sad": 19, "angry": 20, "surprised": 21,
    "fearful": 22, "disgusted": 23, "sarcastic": 24,
    "calm": 25, "empathetic": 26, "excited": 27, "serious": 28,
}
N_PERCEIVED_EMOTIONS = 8  # neutral..sarcastic

PARALING_TAGS: dict[str, int] = {
    "laugh": 29, "sigh": 30, "chuckle": 31, "gasp": 32,
    "cough": 33, "breath": 34, "sniffle": 35, "yawn": 36,
}

LANG_TAGS: dict[str, int] = {
    "en": 37, "zh": 38, "fr": 39, "de": 40, "es": 41, "ja": 42,
    "ko": 43, "tr": 44, "ru": 45, "it": 46, "pt": 47, "nl": 48,
}

# --- v5 extensions (docs/DESIGN_V5_VOICE.md): reference-voice prompt segment.
VOICE = 49  # voice-reference segment start (text stream)
VOICE_END = 50  # voice-reference segment end
ACCENT_KEEP = 51  # reserved keep-reference-accent switch; UNTRAINED in v1

INTENSITY_TAGS: dict[str, int] = {
    "lo": INTENSITY_LO, "md": INTENSITY_MD, "hi": INTENSITY_HI,
}

N_RESERVED_SPECIALS = 64

#: name -> id, in id order; used when training the BPE (specials claim ids 0..)
SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": PAD,
    "<bos>": BOS,
    "<eos>": EOS,
    "<text_pad>": TEXT_PAD,
    "<user>": USER,
    "<assistant>": ASSISTANT,
    "<end_of_turn>": END_OF_TURN,
    "<asr>": TASK_ASR,
    "<tts>": TASK_TTS,
    "<s2s>": TASK_S2S,
    "<alm>": TASK_ALM,
    "<ser>": TASK_SER,
    "<emo_pcv>": EMO_PCV,
    "<emo_rsp>": EMO_RSP,
    "<intensity_lo>": INTENSITY_LO,
    "<intensity_md>": INTENSITY_MD,
    "<intensity_hi>": INTENSITY_HI,
    **{f"<{name}>": i for name, i in EMOTION_CLASSES.items()},
    **{f"<{name}>": i for name, i in PARALING_TAGS.items()},
    **{f"<lang_{name}>": i for name, i in LANG_TAGS.items()},
    "<voice>": VOICE,
    "<voice_end>": VOICE_END,
    "<accent_keep>": ACCENT_KEEP,
}
RESERVED_SPECIAL_FORMAT = "<reserved_{i}>"  # fills the unnamed ids up to 63

TASK_TAGS = {
    "asr": TASK_ASR, "tts": TASK_TTS, "s2s": TASK_S2S, "alm": TASK_ALM,
    "ser": TASK_SER,
}


def turn_prefix(
    lang: str | None = None,
    perceived: str | None = None,
    response_style: str | None = None,
    intensity: str | None = None,
) -> list[int]:
    """Monologue prefix ids for an assistant turn (DESIGN_V4 §1):
    ``<lang_XX> <emo_pcv> <PCV> <emo_rsp> <RSP> [<intensity_*>]``.

    Prepend to the assistant ``text_ids`` handed to the grid builders; every
    field is optional (None omits it, so untagged v1 data is unchanged).
    """
    if intensity is not None and response_style is None:
        raise ValueError(
            "intensity modifies a response style; pass response_style with it"
        )
    ids: list[int] = []
    if lang is not None:
        ids.append(LANG_TAGS[lang])
    if perceived is not None:
        ids += [EMO_PCV, EMOTION_CLASSES[perceived]]
    if response_style is not None:
        ids += [EMO_RSP, EMOTION_CLASSES[response_style]]
        if intensity is not None:
            ids.append(INTENSITY_TAGS[intensity])
    return ids

# Channel (turn owner) ids
CHANNEL_USER = 0
CHANNEL_ASSISTANT = 1

# Stream index of the text row
TEXT_STREAM = 0


# ---------------------------------------------------------------------------
# Audio specials — appended after the codec's real codes in each codebook row
# ---------------------------------------------------------------------------
def audio_pad_id(codec_vocab: int) -> int:
    """Filler: delay slack, non-speech steps, 'not generated yet' at inference."""
    return codec_vocab


def audio_bos_id(codec_vocab: int) -> int:  # reserved, unused in v1
    return codec_vocab + 1


def audio_eos_id(codec_vocab: int) -> int:
    """End of assistant speech; emitted/checked on codebook 0 (row 1)."""
    return codec_vocab + 2


def audio_vocab_size(codec_vocab: int) -> int:
    return codec_vocab + 3


# ---------------------------------------------------------------------------
# Delay pattern
# ---------------------------------------------------------------------------
def delays(n_codebooks: int, mode: str = "stagger") -> list[int]:
    """Per-stream delays in frames for text + one audio group.

    "stagger": text 0, audio codebook k -> k+1 (MusicGen-style).
    "flat":    text 0, every audio codebook -> 1 (requires the depth transformer).
    "lead":    text 0, semantic codebook 0 -> 1, acoustic codebooks -> 2
               (Moshi's ablated winner: acoustics one frame behind the semantic
               stream, ppl 42.2 -> 36.8 in the Moshi paper; requires the depth
               transformer, which factorizes each column sequentially).
    """
    if mode == "stagger":
        return [0] + [k + 1 for k in range(n_codebooks)]
    if mode == "flat":
        return [0] + [1] * n_codebooks
    if mode == "lead":
        return [0] + [1] + [2] * (n_codebooks - 1)
    raise ValueError(f"unknown delay mode {mode!r}")


def max_delay(n_codebooks: int, mode: str = "stagger") -> int:
    return max(delays(n_codebooks, mode))


def stream_delays(n_codebooks: int, mode: str = "stagger", duplex: bool = False) -> list[int]:
    """Per-row delays for the full grid. Duplex grids carry a second, identically
    delayed audio group (user, input-only) after the assistant group."""
    d = delays(n_codebooks, mode)
    return d + d[1:] if duplex else d


def n_streams(n_codebooks: int, duplex: bool = False) -> int:
    return 1 + (2 * n_codebooks if duplex else n_codebooks)


def infer_n_codebooks(S: int, duplex: bool = False) -> int:
    """Recover n_codebooks from a grid's row count."""
    n_q = (S - 1) // 2 if duplex else S - 1
    assert n_streams(n_q, duplex) == S, f"S={S} inconsistent with duplex={duplex}"
    return n_q


@dataclass
class Sample:
    """One training document, UNDELAYED. grid/loss_mask: [S, T]; channel: [T]."""

    grid: torch.Tensor  # long [S, T]
    loss_mask: torch.Tensor  # bool [S, T]
    channel: torch.Tensor  # long [T]
    task: str = ""

    @property
    def n_streams(self) -> int:
        return int(self.grid.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.grid.shape[1])

    def validate(self, codec_vocab: int, text_vocab_size: int) -> None:
        S, T = self.grid.shape
        assert self.loss_mask.shape == (S, T), "loss_mask must match grid"
        assert self.channel.shape == (T,), "channel must be [T]"
        assert self.grid.dtype == torch.long and self.loss_mask.dtype == torch.bool
        assert int(self.grid[TEXT_STREAM].max()) < text_vocab_size
        if S > 1:
            assert int(self.grid[1:].max()) < audio_vocab_size(codec_vocab)
        assert int(self.grid.min()) >= 0


def apply_delay(
    grid: torch.Tensor,
    loss_mask: torch.Tensor,
    channel: torch.Tensor,
    codec_vocab: int,
    *,
    mode: str = "stagger",
    duplex: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift each stream right by its delay. [S, T] -> [S, T + max_delay].

    Fillers: text row tail = PAD (mask off); audio row heads/tails = AUDIO_PAD
    (mask off). Channel extends with its last value.
    """
    S, T = grid.shape
    n_q = infer_n_codebooks(S, duplex)
    dl = stream_delays(n_q, mode, duplex)
    D = max(dl)
    apad = audio_pad_id(codec_vocab)

    out = torch.full((S, T + D), apad, dtype=grid.dtype, device=grid.device)
    out[TEXT_STREAM] = PAD
    mask = torch.zeros((S, T + D), dtype=torch.bool, device=grid.device)
    for s in range(S):
        d = dl[s]
        out[s, d : d + T] = grid[s]
        mask[s, d : d + T] = loss_mask[s]
    tail = channel[-1:].expand(D) if T > 0 else channel.new_zeros(D)
    ch = torch.cat([channel, tail])
    return out, mask, ch


def undelay(
    grid: torch.Tensor,
    *,
    mode: str = "stagger",
    duplex: bool = False,
) -> torch.Tensor:
    """Inverse of `apply_delay` on the grid: [S, T'] -> [S, T' - max_delay]."""
    S, Tp = grid.shape
    n_q = infer_n_codebooks(S, duplex)
    dl = stream_delays(n_q, mode, duplex)
    D = max(dl)
    T = Tp - D
    assert T >= 0, "grid shorter than its delay slack"
    rows = [grid[s, dl[s] : dl[s] + T] for s in range(S)]
    return torch.stack(rows)


def sanitize_codes(codes: torch.Tensor, codec_vocab: int) -> torch.Tensor:
    """Replace audio special ids with code 0 so a codec can decode. [n_q, T]."""
    return torch.where(codes < codec_vocab, codes, torch.zeros_like(codes))


def trim_audio_at_eos(codes: torch.Tensor, codec_vocab: int) -> torch.Tensor:
    """Cut an UNDELAYED audio block [n_q, T] at the first AUDIO_EOS on row 0."""
    eos = audio_eos_id(codec_vocab)
    hits = (codes[0] == eos).nonzero()
    if hits.numel():
        codes = codes[:, : int(hits[0, 0])]
    return codes
