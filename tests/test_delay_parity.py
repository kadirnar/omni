"""Cross-implementation delay parity: the generator's hand-derived index math
(`_delayed_prompt`, `_input_column`) must equal `streams.apply_delay` exactly,
for every delay mode and duplex layout. A future off-by-one in either
implementation would otherwise pass the whole suite and only surface as
degraded audio after a paid training run (2026-07 review, top finding)."""

from __future__ import annotations

import pytest
import torch

from omni.infer.generate import _delayed_prompt, _input_column
from omni.streams import PAD, TEXT_STREAM, apply_delay, audio_pad_id, stream_delays

CV = 2048
APAD = audio_pad_id(CV)


def _grid(S: int, T: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    grid = torch.zeros((S, T), dtype=torch.long)
    grid[TEXT_STREAM] = torch.randint(64, 320, (T,), generator=g)
    grid[1:] = torch.randint(0, CV, (S - 1, T), generator=g)
    mask = torch.randint(0, 2, (S, T), generator=g).bool()
    channel = torch.randint(0, 2, (T,), generator=g)
    return grid, mask, channel


@pytest.mark.parametrize("mode", ["stagger", "flat", "lead"])
@pytest.mark.parametrize("duplex", [False, True])
@pytest.mark.parametrize("t0", [1, 2, 5, 13])
def test_delayed_prompt_equals_apply_delay(mode: str, duplex: bool, t0: int) -> None:
    """The prefill grid built by the generator IS apply_delay's prefix."""
    n_q = 3
    S = 1 + (2 * n_q if duplex else n_q)
    dl = stream_delays(n_q, mode, duplex)
    grid, mask, channel = _grid(S, t0, seed=t0 * 7 + len(mode))
    want, _, _ = apply_delay(grid, mask, channel, CV, mode=mode, duplex=duplex)

    u = torch.full((S, t0 + 8), APAD, dtype=torch.long)
    u[TEXT_STREAM] = PAD
    u[:, :t0] = grid
    got = _delayed_prompt(u, t0, dl, APAD)
    assert torch.equal(got, want[:, :t0])


@pytest.mark.parametrize("mode", ["stagger", "flat", "lead"])
@pytest.mark.parametrize("duplex", [False, True])
def test_input_column_equals_apply_delay(mode: str, duplex: bool) -> None:
    """Every step's input column equals the corresponding apply_delay column."""
    n_q = 3
    S = 1 + (2 * n_q if duplex else n_q)
    dl = stream_delays(n_q, mode, duplex)
    T = 17
    grid, mask, channel = _grid(S, T, seed=len(mode) * 31 + int(duplex))
    want, _, _ = apply_delay(grid, mask, channel, CV, mode=mode, duplex=duplex)

    u = torch.full((S, T + 8), APAD, dtype=torch.long)
    u[TEXT_STREAM] = PAD
    u[:, :T] = grid
    for p in range(T):  # inside the content region, incl. the delay head
        col = _input_column(u, p, dl, APAD)
        assert torch.equal(col, want[:, p]), f"mode={mode} duplex={duplex} p={p}"


def test_forced_text_overflow_raises() -> None:
    """A forced monologue longer than the frame budget must error loudly, not
    silently speak a truncated sentence."""
    from omni.audio.codec import FakeCodec
    from omni.config import load_config
    from omni.infer.generate import OmniGenerator
    from omni.model.omni import OmniModel
    from omni.text.tokenizer import ByteTokenizer

    cfg = load_config(
        "tiny",
        [
            "model.n_codebooks=2", "model.text_vocab_size=320",
            "model.max_frames=64", "data.max_sample_frames=56",
        ],
    )
    torch.manual_seed(0)
    gen = OmniGenerator(OmniModel(cfg.model), cfg, device="cpu", tokenizer=ByteTokenizer())
    with pytest.raises(ValueError, match="forced_text"):
        gen.tts("x" * 300, FakeCodec(n_codebooks=2), max_frames=8)
