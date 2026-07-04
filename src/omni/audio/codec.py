"""Audio codecs: the frozen Mimi tokenizer and an offline deterministic stand-in.

All codecs share one contract (see docs/INTERFACES.md):

- ``encode``: float32 wav ``[T_samples]`` or ``[B, T_samples]`` (values ~[-1, 1],
  at ``self.sample_rate``) -> long codes ``[n_q, T_frames]`` / ``[B, n_q, T_frames]``
  with values in ``[0, codec_vocab)``.
- ``decode``: long RAW codes (caller sanitizes specials first, see
  ``streams.sanitize_codes``) ``[n_q, T]`` / ``[B, n_q, T]`` -> float32 wav
  ``[T_samples]`` / ``[B, T_samples]``.

Nothing here touches the network at import time; only ``MimiCodec.__init__``
(the explicit ``--codec mimi`` path) may download weights from the HF hub.
"""

from __future__ import annotations

import abc
import hashlib
import math
import struct
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F


class AudioCodec(abc.ABC):
    """Abstract frame-level audio codec (12.5 Hz frames for everything in v1)."""

    sample_rate: int  # Hz of encode input / decode output
    frame_rate: float  # frames per second (12.5 for Mimi)
    n_codebooks: int
    codec_vocab: int  # real codes per codebook (2048 for Mimi); specials live above

    @abc.abstractmethod
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav float32 [T_samples] or [B, T_samples] -> long codes [n_q, T_frames] or [B, n_q, T_frames]."""

    @abc.abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """RAW long codes [n_q, T] or [B, n_q, T] -> float32 wav [T_samples] or [B, T_samples]."""

    def to(self, device: str | torch.device) -> "AudioCodec":
        """Move any underlying weights to `device`; returns self. Default: no-op."""
        return self

    @property
    def samples_per_frame(self) -> int:
        """Waveform samples per 1/frame_rate frame (1920 for 24 kHz / 12.5 Hz)."""
        return int(round(self.sample_rate / self.frame_rate))


class MimiCodec(AudioCodec):
    """Frozen Mimi RVQ codec via ``transformers.MimiModel`` (kyutai/mimi).

    24 kHz in/out, 12.5 Hz frames, ``codec_vocab`` 2048 per codebook; codebook 0
    is WavLM-distilled (semantic). Construction downloads weights from the HF
    hub on first use — never build one on offline/test paths (use FakeCodec).
    """

    def __init__(
        self,
        model_id: str = "kyutai/mimi",
        n_codebooks: int = 8,
        device: str | torch.device = "cpu",
    ):
        # Local import keeps `import omni.audio.codec` light and offline-safe.
        from transformers import MimiModel

        try:
            model = MimiModel.from_pretrained(model_id)
        except Exception as e:  # noqa: BLE001 - annotate the likely cause
            raise RuntimeError(
                f"could not load Mimi weights {model_id!r}. This path fetches from "
                "the Hugging Face hub on first use; for offline runs use the fake "
                "codec (--codec fake)."
            ) from e
        if not (1 <= n_codebooks <= int(model.config.num_quantizers)):
            raise ValueError(
                f"n_codebooks must be in [1, {model.config.num_quantizers}], got {n_codebooks}"
            )
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._model.requires_grad_(False)
        self.sample_rate = int(model.config.sampling_rate)  # 24_000
        self.frame_rate = float(model.config.frame_rate)  # 12.5
        self.n_codebooks = int(n_codebooks)
        self.codec_vocab = int(model.config.codebook_size)  # 2048

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav float32 [T] or [B, T] at 24 kHz -> long codes [n_q, T_fr] or [B, n_q, T_fr].

        Output lives on the codec's device.
        """
        if wav.ndim not in (1, 2):
            raise ValueError(f"wav must be [T] or [B, T], got shape {tuple(wav.shape)}")
        unbatched = wav.ndim == 1
        x = wav[None] if unbatched else wav
        x = x.to(self._device, dtype=torch.float32)[:, None, :]  # [B, 1, T]
        out = self._model.encode(x, num_quantizers=self.n_codebooks, return_dict=True)
        codes = out.audio_codes.to(torch.long)  # [B, n_q, T_fr]
        return codes[0] if unbatched else codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """RAW long codes [n_q, T] or [B, n_q, T] -> float32 wav [T_samples] or [B, T_samples]."""
        if codes.ndim not in (2, 3):
            raise ValueError(f"codes must be [n_q, T] or [B, n_q, T], got {tuple(codes.shape)}")
        if codes.shape[-2] != self.n_codebooks:
            raise ValueError(f"expected {self.n_codebooks} codebooks, got {codes.shape[-2]}")
        unbatched = codes.ndim == 2
        c = codes[None] if unbatched else codes
        c = c.to(self._device, dtype=torch.long)
        out = self._model.decode(c, return_dict=True)
        wav = out.audio_values  # [B, 1, T_samples] (or [B, T_samples] per docstring)
        if wav.ndim == 3:
            wav = wav[:, 0]
        wav = wav.to(torch.float32)
        return wav[0] if unbatched else wav

    def to(self, device: str | torch.device) -> "MimiCodec":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self


class FakeCodec(AudioCodec):
    """Fully offline, deterministic codec stand-in for tests and fake data.

    encode: each 80 ms frame (1920 samples at 24 kHz) is quantized to int16 and
    hashed (blake2b); the digest yields one code per codebook. Same wav -> same
    codes; partial trailing frames are zero-padded, so T_frames = ceil(T / 1920).

    decode: each (codebook, code) pair maps to a fixed small sinusoid; the n_q
    sinusoids of a frame are summed and clamped to [-1, 1] (always nonzero,
    1920 samples per frame). encode(decode(x)) does NOT roundtrip.

    Everything runs on CPU; inputs on other devices are copied, outputs are CPU.
    """

    def __init__(self, n_codebooks: int = 8, codec_vocab: int = 2048):
        assert 1 <= n_codebooks <= 32, "FakeCodec supports 1..32 codebooks"
        assert codec_vocab >= 2
        self.sample_rate = 24_000
        self.frame_rate = 12.5
        self.n_codebooks = int(n_codebooks)
        self.codec_vocab = int(codec_vocab)

    # -- encode ------------------------------------------------------------
    def _encode_one(self, wav: torch.Tensor) -> torch.Tensor:
        """wav float32 [T] (cpu) -> long [n_q, ceil(T / samples_per_frame)]."""
        spf = self.samples_per_frame
        n_frames = math.ceil(wav.shape[0] / spf)
        if n_frames == 0:
            return torch.zeros((self.n_codebooks, 0), dtype=torch.long)
        pad = n_frames * spf - wav.shape[0]
        if pad:
            wav = torch.cat([wav, wav.new_zeros(pad)])
        # int16 quantization makes the hash robust to float32 storage roundtrips
        i16 = (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
        frames = i16.view(n_frames, spf).numpy().astype("<i2")
        codes = torch.empty((self.n_codebooks, n_frames), dtype=torch.long)
        for t in range(n_frames):
            digest = hashlib.blake2b(
                frames[t].tobytes(), digest_size=2 * self.n_codebooks
            ).digest()
            for k in range(self.n_codebooks):
                v = digest[2 * k] | (digest[2 * k + 1] << 8)
                codes[k, t] = v % self.codec_vocab
        return codes

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav float32 [T] or [B, T] -> long codes [n_q, T_fr] or [B, n_q, T_fr]."""
        if wav.ndim not in (1, 2):
            raise ValueError(f"wav must be [T] or [B, T], got shape {tuple(wav.shape)}")
        x = wav.detach().to("cpu", torch.float32)
        if x.ndim == 1:
            return self._encode_one(x)
        return torch.stack([self._encode_one(row) for row in x])

    # -- decode ------------------------------------------------------------
    def _decode_one(self, codes: torch.Tensor) -> torch.Tensor:
        """codes long [n_q, T] (cpu) -> wav float32 [T * samples_per_frame]."""
        n_q, T = codes.shape
        spf = self.samples_per_frame
        out = torch.empty(T * spf, dtype=torch.float32)
        t_axis = torch.arange(spf, dtype=torch.float32) / self.sample_rate  # [spf]
        # code -> frequency in ~[60, 3060) Hz, plus a per-codebook offset (< Nyquist)
        step = 3000.0 / self.codec_vocab
        k_off = 17.0 * torch.arange(n_q, dtype=torch.float32).unsqueeze(1)  # [n_q, 1]
        amp = 0.8 / n_q
        chunk = 1024  # frames per chunk: bounds peak memory for long decodes
        for s in range(0, T, chunk):
            c = codes[:, s : s + chunk].to(torch.float32)
            freq = 60.0 + c * step + k_off  # [n_q, Tc]
            ang = (2.0 * math.pi) * freq.unsqueeze(-1) * t_axis  # [n_q, Tc, spf]
            wav = torch.sin(ang).sum(dim=0) * amp  # [Tc, spf]
            out[s * spf : (s + wav.shape[0]) * spf] = wav.reshape(-1)
        return out.clamp_(-1.0, 1.0)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """RAW long codes [n_q, T] or [B, n_q, T] -> float32 wav [T*1920] or [B, T*1920]."""
        if codes.ndim not in (2, 3):
            raise ValueError(f"codes must be [n_q, T] or [B, n_q, T], got {tuple(codes.shape)}")
        c = codes.detach().to("cpu", torch.long)
        if c.numel() and (int(c.min()) < 0 or int(c.max()) >= self.codec_vocab):
            raise ValueError(
                "decode takes RAW codes in [0, codec_vocab) only — special ids "
                "(AUDIO_PAD/BOS/EOS) must be removed with streams.sanitize_codes first"
            )
        if c.ndim == 2:
            if c.shape[0] != self.n_codebooks:
                raise ValueError(f"expected {self.n_codebooks} codebooks, got {c.shape[0]}")
            return self._decode_one(c)
        if c.shape[1] != self.n_codebooks:
            raise ValueError(f"expected {self.n_codebooks} codebooks, got {c.shape[1]}")
        return torch.stack([self._decode_one(row) for row in c])


def build_codec(
    name: str,
    *,
    model_id: str = "kyutai/mimi",
    n_codebooks: int = 8,
    device: str | torch.device = "cpu",
) -> AudioCodec:
    """Build a codec by name: "mimi" (downloads weights on first use) or "fake"."""
    if name == "mimi":
        return MimiCodec(model_id=model_id, n_codebooks=n_codebooks, device=device)
    if name == "fake":
        return FakeCodec(n_codebooks=n_codebooks)
    raise ValueError(f"unknown codec {name!r}; expected 'mimi' or 'fake'")


# ---------------------------------------------------------------------------
# Resampling + wav file I/O
# ---------------------------------------------------------------------------
# Kaiser-windowed sinc parameters (torchaudio "kaiser_best" quality).
_RESAMPLE_WIDTH = 64
_RESAMPLE_ROLLOFF = 0.9475937167399596
_RESAMPLE_BETA = 14.769656459379492


def resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Anti-aliased polyphase windowed-sinc resampling of mono audio.

    wav float32 ``[T]`` at ``sr`` -> float32 ``[ceil(T * target_sr / sr)]``.
    Kaiser-windowed sinc low-pass at ``rolloff * Nyquist`` of the slower rate
    (the torchaudio ``sinc_interp_kaiser`` / "kaiser_best" formulation), so
    downsampling real 44.1/48 kHz corpora to a codec's 24 kHz no longer folds
    >Nyquist energy into band. Pure torch, deterministic, CPU-safe. This is
    the ONE resampler in the repo — data prep and inference share it.
    """
    if sr == target_sr or wav.numel() == 0:
        return wav.to(torch.float32)
    if wav.ndim != 1:
        raise ValueError(f"resample takes mono [T], got shape {tuple(wav.shape)}")
    sr, target_sr = int(sr), int(target_sr)
    g = math.gcd(sr, target_sr)
    orig, new = sr // g, target_sr // g  # M (down), L (up) in gcd units

    base_freq = min(orig, new) * _RESAMPLE_ROLLOFF
    width = math.ceil(_RESAMPLE_WIDTH * orig / base_freq)
    # kernel time grid: one row per output phase, taps over [-width, width+orig)
    idx = torch.arange(-width, width + orig, dtype=torch.float64)[None, :] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None] / new + idx
    t = (t * base_freq).clamp_(-_RESAMPLE_WIDTH, _RESAMPLE_WIDTH)
    window = torch.special.i0(
        _RESAMPLE_BETA * torch.sqrt(1 - (t / _RESAMPLE_WIDTH) ** 2)
    ) / torch.special.i0(torch.tensor(_RESAMPLE_BETA, dtype=torch.float64))
    t = t * math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0, dtype=torch.float64), t.sin() / t)
    kernel = (kernel * window * (base_freq / orig)).to(torch.float32)  # [new, K]

    x = wav.detach().to("cpu", torch.float32)
    n_in = x.shape[0]
    padded = F.pad(x[None, None], (width, width + orig))
    y = F.conv1d(padded, kernel[:, None, :], stride=orig)  # [1, new, ceil(T/orig)]
    y = y.transpose(1, 2).reshape(-1)  # interleave phases -> output stream
    n_out = math.ceil(new * n_in / orig)
    return y[:n_out]


def load_wav(path: str | Path, target_sr: int) -> torch.Tensor:
    """Load an audio file as mono float32 [T] at ``target_sr``.

    Multi-channel input is averaged to mono; rate conversion goes through the
    anti-aliased :func:`resample`.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(data).mean(dim=1)  # [T]
    return resample(wav, int(sr), int(target_sr))


def save_wav(path: str | Path, wav: torch.Tensor, sr: int) -> None:
    """Write float32 wav [T] (or [1, T]) to `path` (WAV, float32 subtype).

    The RIFF file is written by hand rather than through libsndfile: for
    floating-point WAVs libsndfile appends a PEAK chunk stamped with the
    wall-clock time, so two identical renders saved a second apart differ by
    one header byte. This writer is a pure function of ``(samples, sr)`` —
    byte-deterministic outputs (seeded chat runs, golden hashing) — and still
    roundtrips exactly through :func:`load_wav` at the same sample rate.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = wav.detach().to("cpu", torch.float32)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 1:
        raise ValueError(f"wav must be [T] or [1, T], got shape {tuple(wav.shape)}")
    data = x.clamp(-1.0, 1.0).numpy().astype("<f4", copy=False).tobytes()
    sr = int(sr)
    header = b"".join(
        (
            b"RIFF",
            struct.pack("<I", 4 + 24 + 12 + 8 + len(data)),  # riff size = file - 8
            b"WAVE",
            b"fmt ",
            #                    fmt=3 (IEEE float), mono, 32-bit
            struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32),
            b"fact",  # required for non-PCM: total frame count
            struct.pack("<II", 4, x.shape[0]),
            b"data",
            struct.pack("<I", len(data)),
        )
    )
    p.write_bytes(header + data)
