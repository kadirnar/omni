"""Extensions v2: word-aligned inner monologue (INTERFACES.md "Extensions v2")."""

from __future__ import annotations

import torch

from omni import grids, streams
from omni.data.synthesize import (
    SineTTS,
    UniformAligner,
    build_aligner,
    fake_dialogues,
    word_frames_from_alignment,
)

V = 2048


def test_word_frames_placement() -> None:
    n_q = 2
    codes = torch.randint(0, V, (n_q, 25))
    wf = [(0, [101, 102]), (6, [103]), (6, [104, 105]), (15, [106])]
    s = grids.build_tts([], codes, n_q, V, word_frames=wf)
    seg = 2  # <bos> <tts>
    tr = s.grid[0]
    assert tr[seg + 1] == 101 and tr[seg + 2] == 102  # frame 0 -> col 1
    assert tr[seg + 3] == streams.TEXT_PAD  # gap
    assert tr[seg + 6] == 103
    assert tr[seg + 7] == 104 and tr[seg + 8] == 105  # collision shifts right
    assert tr[seg + 15] == 106 and tr[seg + 16] == streams.END_OF_TURN


def test_word_beyond_audio_extends_segment() -> None:
    n_q = 2
    T = 10
    codes = torch.randint(0, V, (n_q, T))
    s = grids.build_tts([], codes, n_q, V, word_frames=[(T + 4, [107, 108])])
    seg = 2
    tr = s.grid[0]
    assert tr[seg + T + 4] == 107 and tr[seg + T + 5] == 108
    assert tr[seg + T + 6] == streams.END_OF_TURN
    # audio side: eos-frame at col T, APAD (loss off) beyond
    assert s.grid[1, seg + T] == streams.audio_eos_id(V)
    assert not s.loss_mask[1:, seg + T + 1 :].any()


def test_packed_equals_pseudo_word() -> None:
    """word_frames=[(0, ids)] must reproduce the packed layout exactly."""
    n_q = 2
    codes = torch.randint(0, V, (n_q, 20))
    ids = [64, 65, 66, 67]
    packed = grids.build_tts(ids, codes, n_q, V)
    pseudo = grids.build_tts([], codes, n_q, V, word_frames=[(0, ids)])
    assert torch.equal(packed.grid, pseudo.grid)
    assert torch.equal(packed.loss_mask, pseudo.loss_mask)


def test_uniform_aligner() -> None:
    al = UniformAligner()
    wav = torch.zeros(24000 * 2)
    a1 = al.align(wav, 24000, "hello brave new world")
    a2 = al.align(wav, 24000, "hello brave new world")
    assert a1 == a2, "UniformAligner must be deterministic"
    assert [w for _, _, w in a1] == ["hello", "brave", "new", "world"]
    starts = [s for s, _, _ in a1]
    ends = [e for _, e, _ in a1]
    assert all(0.0 <= s < e <= 2.0 + 1e-6 for s, e in zip(starts, ends))
    assert starts == sorted(starts)
    assert all(a < b for a, b in zip(starts, starts[1:])), "monotonic starts"


def test_word_frames_from_alignment(byte_tok) -> None:
    ali = [(0.0, 0.4, "hi"), (0.8, 1.2, "there"), (1.6, 2.0, "friend")]
    wf = word_frames_from_alignment(ali, byte_tok)
    assert [f for f, _ in wf] == [0, 10, 20]  # int(start * 12.5)
    assert all(ids for _, ids in wf)
    frames = [f for f, _ in wf]
    assert frames == sorted(frames)


def test_build_aligner_dispatch() -> None:
    assert build_aligner(None) is None
    assert build_aligner("none") is None
    assert isinstance(build_aligner("uniform"), UniformAligner)


class _TailAligner(UniformAligner):
    """Test stub: crams every word into the last 20% of the wav, so the aligned
    monologue must start with a visible <text_pad> gap (unlike packed). Byte
    tokenization is denser than 12.5 fps frames, so UniformAligner itself
    legitimately degenerates to the packed layout on short SineTTS clips —
    this stub proves prepare_s2s threads the aligner through regardless."""

    def align(self, wav: torch.Tensor, sr: int, text: str):
        words = text.split()
        dur = wav.shape[0] / sr
        start = 0.8 * dur
        step = (dur - start) / max(len(words), 1)
        return [
            (start + i * step, start + (i + 1) * step, w) for i, w in enumerate(words)
        ]


def test_prepare_s2s_aligner_threaded_through(tmp_path, test_cfg) -> None:
    from omni.audio.codec import FakeCodec
    from omni.data.dataset import ShardDataset
    from omni.data.prepare import prepare_s2s
    from omni.text.tokenizer import ByteTokenizer

    def prep(out, aligner):
        prepare_s2s(
            out,
            dialogues=list(fake_dialogues(4, seed=0)),
            tts=SineTTS(),
            codec=FakeCodec(
                n_codebooks=test_cfg.model.n_codebooks,
                codec_vocab=test_cfg.model.audio_codec_vocab,
            ),
            tokenizer=ByteTokenizer(),
            cfg=test_cfg,
            max_samples=4,
            seed=0,
            aligner=aligner,
        )
        return ShardDataset(out)

    packed = prep(tmp_path / "packed", None)
    aligned = prep(tmp_path / "aligned", _TailAligner())
    assert len(packed) > 0 and len(aligned) > 0
    for ds in (packed, aligned):
        for i in range(len(ds)):
            ds[i].validate(test_cfg.model.audio_codec_vocab, test_cfg.model.text_vocab_size)
    diff = any(
        i >= len(aligned)
        or packed[i].grid.shape != aligned[i].grid.shape
        or not torch.equal(packed[i].grid[0], aligned[i].grid[0])
        for i in range(len(packed))
    )
    assert diff, "tail alignment must move monologue tokens vs packed layout"
