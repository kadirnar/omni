"""Task grid builders: turn text/audio pairs into UNDELAYED Sample grids.

Layouts (docs/DESIGN.md §1). One column = one 80 ms frame step. `w1..wk` are
BPE ids of the assistant text (inner monologue, packed one-per-frame from turn
start, then <text_pad> while audio continues).

  textlm : <bos> w1 .. wk <eos>                                (audio: APAD, no loss)
  audiolm: <bos> <alm> [TEXT_PAD over audio.. +eos-frame] <eos>
  asr    : <bos> <asr> [user seg] <end_of_turn> <assistant> w1..wk <eos>
  tts    : <bos> <tts> [assistant speech seg] <eos>
  s2s    : <bos> <s2s> ([user seg] <end_of_turn> [assistant speech seg])xN <eos>

  user seg           : text = <user> TEXT_PAD..; audio = user codes; loss-masked; channel USER
  assistant speech
  seg (audio T fr.)  : text  = <assistant> w1 .. wk <end_of_turn> TEXT_PAD ..
                       audio = f0 f1 .. f(T-1) EOS-frame [APAD ..]
                       length = max(T + 1, k + 2); loss on except the injected <assistant>
  EOS-frame          : codebook-0 = AUDIO_EOS, other codebooks = AUDIO_PAD

Tokens the runtime injects at inference (<bos>, task tags, <user>,
<end_of_turn> after user speech, <assistant>) are never loss targets.
"""

from __future__ import annotations

import torch

from .streams import (
    ASSISTANT,
    BOS,
    CHANNEL_ASSISTANT,
    CHANNEL_USER,
    END_OF_TURN,
    EOS,
    TASK_TAGS,
    TEXT_PAD,
    TEXT_STREAM,
    USER,
    VOICE,
    VOICE_END,
    Sample,
    audio_eos_id,
    audio_pad_id,
)


class _GridBuilder:
    def __init__(self, n_codebooks: int, codec_vocab: int):
        self.n_q = n_codebooks
        self.codec_vocab = codec_vocab
        self._apad = audio_pad_id(codec_vocab)
        self.text: list[int] = []
        self.audio: list[torch.Tensor | None] = []  # None -> all AUDIO_PAD
        self.tmask: list[bool] = []
        self.amask: list[bool] = []
        self.chan: list[int] = []

    def col(
        self,
        text_id: int,
        audio: torch.Tensor | None = None,
        *,
        tmask: bool = False,
        amask: bool = False,
        chan: int = CHANNEL_ASSISTANT,
    ) -> None:
        self.text.append(int(text_id))
        self.audio.append(audio)
        self.tmask.append(tmask)
        self.amask.append(amask)
        self.chan.append(chan)

    def eos_frame(self, *, chan: int = CHANNEL_ASSISTANT, text_id: int = TEXT_PAD, tmask: bool = True) -> None:
        v = torch.full((self.n_q,), self._apad, dtype=torch.long)
        v[0] = audio_eos_id(self.codec_vocab)
        self.col(text_id, v, tmask=tmask, amask=True, chan=chan)

    def voice_segment(self, ref_codes: torch.Tensor) -> None:
        """Reference-voice prompt (DESIGN_V5): pins the speaker identity.

        Layout: ``<voice>`` + first reference frame, then TEXT_PAD columns over
        the remaining frames, then a ``<voice_end>`` column (audio APAD) —
        R + 1 columns for R reference frames. Transcript-free, channel
        ASSISTANT, loss masked on EVERY row (PersonaPlex whole-prompt masking).
        Callers place it directly after ``<bos>``, before the task tag, so the
        ``[<bos>+segment]`` KV prefix is identical across tasks/turns/sessions.
        """
        _check_codes(ref_codes, self.n_q, self.codec_vocab)
        R = ref_codes.shape[1]
        assert R > 0, "voice reference must contain at least one frame"
        for t in range(R):
            self.col(VOICE if t == 0 else TEXT_PAD, ref_codes[:, t], tmask=False, amask=False)
        self.col(VOICE_END, None, tmask=False, amask=False)

    def user_segment(self, user_codes: torch.Tensor) -> None:
        """Input-only user speech: no loss anywhere; channel USER."""
        _check_codes(user_codes, self.n_q, self.codec_vocab)
        for t in range(user_codes.shape[1]):
            self.col(
                USER if t == 0 else TEXT_PAD,
                user_codes[:, t],
                tmask=False,
                amask=False,
                chan=CHANNEL_USER,
            )

    def assistant_speech_segment(
        self,
        text_ids: list[int],
        codes: torch.Tensor,
        word_frames: list[tuple[int, list[int]]] | None = None,
    ) -> None:
        """Parallel text monologue + audio; loss on both (not on <assistant>).

        word_frames: optional word-aligned monologue — [(frame, token_ids), ...]
        with frames relative to the segment's first audio frame. Each word's
        tokens are placed no earlier than its frame column (col 0 holds
        <assistant>, so frame f maps to col >= max(1, f)); overlapping words
        shift right. None keeps the packed one-token-per-frame layout, in which
        case `text_ids` supplies the tokens (ignored when word_frames is given).
        """
        _check_codes(codes, self.n_q, self.codec_vocab)
        assert codes.shape[1] > 0, (
            "assistant speech segment needs at least one audio frame "
            "(empty TTS/synthesis output must be filtered upstream)"
        )
        T = codes.shape[1]
        seq = word_frames
        if seq is None:
            seq = [(0, [int(i) for i in text_ids])] if len(text_ids) else []
        text_at: dict[int, int] = {}
        next_col = 1  # col 0 is <assistant>
        for frame, ids in seq:
            c = max(next_col, int(frame))
            for tid in ids:
                text_at[c] = int(tid)
                c += 1
            next_col = c
        eot_col = next_col
        length = max(T + 1, eot_col + 1)  # +1 for the EOS-frame
        for c in range(length):
            tid = ASSISTANT if c == 0 else text_at.get(c, END_OF_TURN if c == eot_col else TEXT_PAD)
            tm = c > 0  # the injected <assistant> is not a target
            if c < T:
                self.col(tid, codes[:, c], tmask=tm, amask=True)
            elif c == T:
                self.eos_frame(text_id=tid, tmask=tm)
            else:  # text longer than audio: APAD columns, audio loss off
                self.col(tid, None, tmask=tm, amask=False)

    def build(self, task: str) -> Sample:
        T = len(self.text)
        S = 1 + self.n_q
        grid = torch.full((S, T), self._apad, dtype=torch.long)
        grid[0] = torch.tensor(self.text, dtype=torch.long)
        for t, a in enumerate(self.audio):
            if a is not None:
                grid[1:, t] = a
        mask = torch.zeros((S, T), dtype=torch.bool)
        mask[0] = torch.tensor(self.tmask)
        mask[1:] = torch.tensor(self.amask).unsqueeze(0)
        channel = torch.tensor(self.chan, dtype=torch.long)
        return Sample(grid=grid, loss_mask=mask, channel=channel, task=task)


def _check_codes(codes: torch.Tensor, n_q: int, codec_vocab: int) -> None:
    assert codes.ndim == 2 and codes.shape[0] == n_q, (
        f"audio codes must be [n_codebooks={n_q}, T], got {tuple(codes.shape)}"
    )
    assert codes.dtype == torch.long, "audio codes must be long"
    if codes.numel():
        assert 0 <= int(codes.min()) and int(codes.max()) < codec_vocab, (
            "builders take raw codec codes only (no specials)"
        )


def build_textlm(text_ids: list[int], n_codebooks: int, codec_vocab: int) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS, tmask=True)
    for i in text_ids:
        b.col(int(i), tmask=True)
    b.col(EOS, tmask=True)
    return b.build("textlm")


def build_audiolm(codes: torch.Tensor, n_codebooks: int, codec_vocab: int) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    _check_codes(codes, n_codebooks, codec_vocab)
    b.col(BOS, tmask=True)
    b.col(TASK_TAGS["alm"], tmask=False)
    for t in range(codes.shape[1]):
        b.col(TEXT_PAD, codes[:, t], tmask=True, amask=True)
    b.eos_frame()
    b.col(EOS, tmask=True)
    return b.build("alm")


def build_asr(
    user_codes: torch.Tensor,
    text_ids: list[int],
    n_codebooks: int,
    codec_vocab: int,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS, tmask=False)
    if voice_codes is not None:
        b.voice_segment(voice_codes)
    b.col(TASK_TAGS["asr"], tmask=False)
    b.user_segment(user_codes)
    b.col(END_OF_TURN, tmask=False)
    b.col(ASSISTANT, tmask=False)
    for i in text_ids:
        b.col(int(i), tmask=True)
    b.col(EOS, tmask=True)
    return b.build("asr")


def build_tts(
    text_ids: list[int],
    codes: torch.Tensor,
    n_codebooks: int,
    codec_vocab: int,
    word_frames: list[tuple[int, list[int]]] | None = None,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS, tmask=False)
    if voice_codes is not None:
        b.voice_segment(voice_codes)
    b.col(TASK_TAGS["tts"], tmask=False)
    b.assistant_speech_segment(text_ids, codes, word_frames=word_frames)
    b.col(EOS, tmask=True)
    return b.build("tts")


def build_s2s(
    turns: list[tuple[torch.Tensor, list[int], torch.Tensor]],
    n_codebooks: int,
    codec_vocab: int,
    word_frames_per_turn: list[list[tuple[int, list[int]]] | None] | None = None,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    """turns: [(user_codes [n_q,Tu], assistant_text_ids, assistant_codes [n_q,Ta]), ...]

    word_frames_per_turn: optional per-turn word-aligned monologue (see
    `assistant_speech_segment`); None or a None entry keeps the packed layout.
    voice_codes: optional reference-voice frames — ONE segment pins all turns.
    """
    assert turns, "need at least one turn"
    if word_frames_per_turn is not None:
        assert len(word_frames_per_turn) == len(turns)
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS, tmask=False)
    if voice_codes is not None:
        b.voice_segment(voice_codes)
    b.col(TASK_TAGS["s2s"], tmask=False)
    for i, (user_codes, asst_text, asst_codes) in enumerate(turns):
        wf = word_frames_per_turn[i] if word_frames_per_turn is not None else None
        b.user_segment(user_codes)
        b.col(END_OF_TURN, tmask=False)
        b.assistant_speech_segment(asst_text, asst_codes, word_frames=wf)
    b.col(EOS, tmask=True)
    return b.build("s2s")


# ---------------------------------------------------------------------------
# Full duplex (S = 1 + 2*n_codebooks rows: text, assistant group, user group)
# ---------------------------------------------------------------------------
def _check_track(track: torch.Tensor, n_q: int, codec_vocab: int) -> None:
    assert track.ndim == 2 and track.shape[0] == n_q, (
        f"track must be [n_codebooks={n_q}, T], got {tuple(track.shape)}"
    )
    assert track.dtype == torch.long, "tracks must be long"
    if track.numel():
        apad = audio_pad_id(codec_vocab)
        ok = ((track >= 0) & (track < codec_vocab)) | (track == apad)
        assert bool(ok.all()), "duplex tracks may contain raw codes or AUDIO_PAD only"


def build_duplex(
    user_track: torch.Tensor,
    assistant_track: torch.Tensor,
    text_word_frames: list[tuple[int, list[int]]],
    n_codebooks: int,
    codec_vocab: int,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    """Full-duplex sample over one shared timeline (no turn-taking).

    Both tracks are [n_q, T] over the SAME T frames with AUDIO_PAD in non-speech
    regions. text_word_frames is the assistant's word-aligned monologue,
    [(frame, token_ids), ...] with absolute frames in [0, T).

    Layout: col 0 = <bos> (audio APAD); optional DESIGN_V5 voice segment at
    cols 1..R+1 (<voice> + R reference frames on the ASSISTANT rows +
    <voice_end>; user rows APAD; loss masked on every row); then the two tracks
    (frame f at col f+1+V where V = R+1 when a segment is present, else 0);
    last col = <eos>. Rows: 0 text, 1..n_q assistant audio (loss ON outside the
    segment — emitting AUDIO_PAD while listening is what's being learned),
    n_q+1..2n_q user audio (input-only, loss OFF). Text row: monologue tokens
    at max(frame+1+V, prev+1), TEXT_PAD elsewhere, loss ON after the segment.
    Channel is CHANNEL_ASSISTANT throughout (groups, not channel, distinguish
    speakers).
    """
    _check_track(user_track, n_codebooks, codec_vocab)
    _check_track(assistant_track, n_codebooks, codec_vocab)
    assert user_track.shape == assistant_track.shape, "tracks must share one timeline"
    n_q = n_codebooks
    T = user_track.shape[1]
    S = 1 + 2 * n_q
    R = 0
    if voice_codes is not None:
        _check_codes(voice_codes, n_q, codec_vocab)
        R = voice_codes.shape[1]
        assert R > 0, "voice reference must contain at least one frame"
    V = R + 1 if voice_codes is not None else 0  # segment cols (<voice>..<voice_end>)
    L = 1 + V + T + 1  # <bos> + segment + frames + <eos>
    apad = audio_pad_id(codec_vocab)

    grid = torch.full((S, L), apad, dtype=torch.long)
    grid[TEXT_STREAM] = TEXT_PAD
    grid[TEXT_STREAM, 0] = BOS
    grid[TEXT_STREAM, L - 1] = EOS
    if voice_codes is not None:
        grid[TEXT_STREAM, 1] = VOICE
        grid[TEXT_STREAM, V] = VOICE_END
        grid[1 : 1 + n_q, 1 : 1 + R] = voice_codes  # assistant rows only
    f0 = 1 + V  # first conversation-frame column
    next_col = f0
    for frame, ids in text_word_frames:
        assert 0 <= int(frame) < T, f"word frame {frame} outside [0, {T})"
        c = max(next_col, int(frame) + f0)
        for tid in ids:
            assert c < L - 1, "monologue overruns the duplex timeline"
            grid[TEXT_STREAM, c] = int(tid)
            c += 1
        next_col = c
    grid[1 : 1 + n_q, f0 : f0 + T] = assistant_track
    grid[1 + n_q :, f0 : f0 + T] = user_track

    mask = torch.zeros((S, L), dtype=torch.bool)
    mask[TEXT_STREAM, f0:] = True
    mask[1 : 1 + n_q, f0 : f0 + T] = True
    channel = torch.full((L,), CHANNEL_ASSISTANT, dtype=torch.long)
    return Sample(grid=grid, loss_mask=mask, channel=channel, task="duplex")


# ---------------------------------------------------------------------------
# Inference prompts. The generator then feeds `prompt_forced_text(...)` ids as
# forced text-stream inputs (one per generated frame) before sampling freely.
# ---------------------------------------------------------------------------
def build_s2s_prompt(
    user_codes: torch.Tensor,
    n_codebooks: int,
    codec_vocab: int,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS)
    if voice_codes is not None:
        b.voice_segment(voice_codes)
    b.col(TASK_TAGS["s2s"])
    b.user_segment(user_codes)
    b.col(END_OF_TURN)
    return b.build("s2s")


def build_asr_prompt(
    user_codes: torch.Tensor,
    n_codebooks: int,
    codec_vocab: int,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS)
    if voice_codes is not None:
        b.voice_segment(voice_codes)
    b.col(TASK_TAGS["asr"])
    b.user_segment(user_codes)
    b.col(END_OF_TURN)
    return b.build("asr")


def build_tts_prompt(
    n_codebooks: int,
    codec_vocab: int,
    voice_codes: torch.Tensor | None = None,
) -> Sample:
    b = _GridBuilder(n_codebooks, codec_vocab)
    b.col(BOS)
    if voice_codes is not None:
        b.voice_segment(voice_codes)
    b.col(TASK_TAGS["tts"])
    return b.build("tts")


def prompt_forced_text(task: str, text_ids: list[int] | None = None) -> list[int]:
    """Forced text-stream inputs consumed one per frame at generation start."""
    if task in ("s2s", "asr"):
        return [ASSISTANT]
    if task == "tts":
        assert text_ids is not None, "tts needs the text to speak"
        return [ASSISTANT] + [int(i) for i in text_ids] + [END_OF_TURN]
    if task == "alm":
        return []
    raise ValueError(f"unknown task {task!r}")
