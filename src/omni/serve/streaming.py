"""Progressive PCM <-> codes streaming for the test console.

`ChunkDecoder` turns per-frame codec codes into playable PCM as they arrive:
it re-decodes a small rolling window of recent frames and emits only the new
samples. With ``crossfade_ms > 0`` it holds back a few milliseconds and
linearly crossfades each boundary — masking the chunk-boundary artifacts of
stateless codec decoders (transformers Mimi; docs/DESIGN.md risk list).
`FakeCodec` decodes frames independently, so ``crossfade_ms=0`` is exact
there (pinned by tests).

`StreamingEncoder` is the mic-side mirror: rolling-context encoding so live
duplex codes approximate training-time whole-utterance codes.
"""

from __future__ import annotations

import torch

from ..audio.codec import AudioCodec

__all__ = ["ChunkDecoder", "StreamingEncoder", "pcm16_bytes"]


def pcm16_bytes(wav: torch.Tensor) -> bytes:
    """float32 [-1, 1] -> little-endian int16 PCM bytes."""
    x = (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    return x.numpy().tobytes()


class StreamingEncoder:
    """Mic-side incremental encoder with rolling left context.

    The transformers Mimi encoder exposes no streaming state, but its
    receptive field spans seconds — encoding each 80 ms chunk independently
    yields codes materially different from the whole-utterance codes training
    saw (the review's duplex train/serve mismatch). This keeps the last
    ``context_s`` seconds of raw PCM, encodes that window per chunk, and
    returns only the NEWEST frame's codes: every frame is encoded with real
    left context at bounded cost. FakeCodec hashes frames independently, so
    the window changes nothing there (one code path for both codecs).

    feed() takes exactly one frame of samples (``codec.samples_per_frame``)
    and returns its codes ``[n_q]``.
    """

    def __init__(self, codec: AudioCodec, *, context_s: float = 2.0) -> None:
        self._codec = codec
        self._spf = codec.samples_per_frame
        self._max_frames = max(1, int(round(context_s * codec.frame_rate)))
        self._buf = torch.zeros(0, dtype=torch.float32)

    def feed(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.numel() != self._spf:
            raise ValueError(
                f"StreamingEncoder.feed wants exactly {self._spf} samples, got {chunk.numel()}"
            )
        self._buf = torch.cat([self._buf, chunk.reshape(-1).float()])
        self._buf = self._buf[-self._max_frames * self._spf :]
        with torch.no_grad():
            codes = self._codec.encode(self._buf)
        return codes[:, -1].to("cpu")

    def reset(self) -> None:
        self._buf = torch.zeros(0, dtype=torch.float32)


class ChunkDecoder:
    """Incremental decoder over one growing code stream.

    feed() accepts one frame ``[n_q]`` (or a block ``[n_q, k]``) of RAW codec
    codes and returns the newly available float32 samples ``[n]`` (empty while
    samples are held back for crossfading). flush() drains the held tail.
    """

    def __init__(
        self,
        codec: AudioCodec,
        *,
        context_frames: int = 16,
        crossfade_ms: float = 0.0,
    ) -> None:
        assert context_frames >= 1
        self._codec = codec
        self._ctx = int(context_frames)
        self._spf = int(round(codec.sample_rate / codec.frame_rate))
        self._xf = int(crossfade_ms / 1000.0 * codec.sample_rate)
        assert self._xf < self._spf, "crossfade must be shorter than one frame"
        self._codes: torch.Tensor | None = None  # [n_q, T] all codes so far
        self._held: torch.Tensor | None = None  # last _xf samples, not yet emitted

    @property
    def frames(self) -> int:
        return 0 if self._codes is None else int(self._codes.shape[1])

    def feed(self, frame: torch.Tensor) -> torch.Tensor:
        """One new frame [n_q] or block [n_q, k] -> new float32 samples [n]."""
        if frame.ndim == 1:
            frame = frame[:, None]
        assert frame.ndim == 2 and frame.shape[1] >= 1
        frame = frame.to(device="cpu", dtype=torch.long)
        self._codes = frame if self._codes is None else torch.cat([self._codes, frame], dim=1)
        n_new = int(frame.shape[1])

        window = self._codes[:, -min(self.frames, self._ctx + n_new) :]
        with torch.no_grad():
            wav = self._codec.decode(window).reshape(-1).float().cpu()
        new = wav[-n_new * self._spf :]
        if self._xf == 0:
            return new

        xf = self._xf
        if self._held is None:  # first chunk: nothing to blend yet
            out = new[:-xf]
        else:
            # fresh rendering of the held-back region, from the larger context
            fresh_held = wav[-n_new * self._spf - xf : -n_new * self._spf]
            ramp = torch.linspace(0.0, 1.0, xf)
            blended = self._held * (1.0 - ramp) + fresh_held * ramp
            out = torch.cat([blended, new[:-xf]])
        self._held = new[-xf:].clone()
        return out

    def flush(self) -> torch.Tensor:
        """Emit the held-back tail (empty when crossfade is off)."""
        out = self._held if self._held is not None else torch.zeros(0)
        self._held = None
        return out
