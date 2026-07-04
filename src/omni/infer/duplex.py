"""Full-duplex streaming inference: one model step per 80 ms tick, forever.

A duplex model (cfg.model.duplex) sees S = 1 + 2*n_q rows: text (row 0), the
assistant audio group (rows 1..n_q, predicted) and the user audio group (rows
n_q+1..2n_q, input-only). Both audio groups share one timeline and one delay
pattern; there is no turn-taking and no EOS stop — the conversation runs until
the caller stops ticking (or the KV cache fills).

Undelayed-buffer scheme, re-derived from ``streams.apply_delay`` over
``dl = streams.stream_delays(n_q, mode, duplex=True)`` (= [0, d_1..d_n_q,
d_1..d_n_q]; stagger d_k = k, i.e. audio delays 1..n_q twice, flat all 1) with
``D = max(dl)`` and PREFIX length ``P``:

- The undelayed buffer ``u`` [S, cap] mirrors `grids.build_duplex`: cols
  0..P-1 hold the prompt prefix — col 0 the <bos> column (text BOS, audio
  rows AUDIO_PAD) and, when a DESIGN_V5 reference voice of R frames is set,
  cols 1..R+1 the voice segment (text <voice>, TEXT_PAD.., <voice_end>;
  reference codes on the ASSISTANT rows at cols 1..R; user rows AUDIO_PAD
  throughout). P = R + 2 with a voice, P = 1 without. Conversation frame f
  sits at col f + P: every index below is the v2 (P = 1) derivation with the
  R + 1 extra segment cols shifting it right.
- Delayed position p of row s shows undelayed col p - dl[s] (filler
  PAD/AUDIO_PAD where negative), so the input column fed to the model at
  delayed step p is ``u[s, p - dl[s]]`` — exactly `_input_column`. `reset`
  prefills delayed positions 0..P-1 in ONE `model.prefill` call over
  `_delayed_prompt(u, P, dl, apad)`; for P = 1 that is the bare <bos> column
  of v2 (every audio delay >= 1). Reference frames whose DELAYED positions
  spill past P-1 (row k's col c surfaces at c + dl[1+k] > R+1) are NOT lost:
  they stay in ``u`` and enter as the first ticks' step inputs, exactly as
  apply_delay lays a training grid out.
- Tick i (i = 0, 1, ..) first lands the PUSHED USER FRAME i on the user rows
  at undelayed col i + P: row n_q + 1 + k holds it at DELAYED positions
  i + P + dl[1 + k], the first of which (i + P + 1, the least-delayed row) is
  read by the model step of tick i + 1 (at p = i + 1 + P) — pushing at the
  top of the tick is always in time and never early.
- The pending logits entering tick i came from the step at delayed
  p = i + P - 1 (the prefill's last position for i = 0) and predict delayed
  position q = i + P:
    text sample            -> undelayed col q         (text of frame i)
    assistant audio head k -> undelayed col q - dl[1+k]
  Samples for cols < P fall on the <bos>+voice prefix and are discarded
  (the reference segment is never overwritten); sampled audio specials
  (AUDIO_PAD/BOS/EOS ids) are written back as AUDIO_PAD — the assistant
  legitimately "speaks" AUDIO_PAD while listening, but never feeds BOS/EOS
  back into the loop.
- Assistant frame f (undelayed col f + P) is COMPLETE once its most-delayed
  head has sampled it, i.e. at the tick where q - D = f + P -> f = i - D:
  the first D ticks return ``assistant_frame=None`` while the delay pipeline
  fills, then tick i returns frame i - D. Symmetrically, after the last user
  frame `run_file` ticks D extra times (user_frame=None) to drain the
  pipeline, so the assistant track length equals the user frame count —
  with or without a voice prefix (P cancels out of the pipeline depth).
- Tick i finally runs the model step at delayed p = i + P (input column from
  ``u`` as above; channel CHANNEL_ASSISTANT throughout, matching
  build_duplex) and stashes the logits for tick i + 1. Every read hits the
  prefix, an already-written sample, a pushed user frame, or a filler — the
  loop is causal by the same argument as `omni.infer.generate`. Delayed
  positions consumed by tick i: p = i + P, so `run_file` over T_user frames
  plus the D-tick flush needs T_user + D + P <= model.max_frames.

Without a voice (P = 1) every tensor fed to the model — and therefore every
sampled token — is bit-identical to the v2 loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from ..audio.codec import AudioCodec
from ..config import OmniConfig, SamplingConfig
from ..model.omni import OmniModel
from ..streams import (
    BOS,
    CHANNEL_ASSISTANT,
    TEXT_PAD,
    TEXT_STREAM,
    VOICE,
    VOICE_END,
    audio_pad_id,
    sanitize_codes,
    stream_delays,
)
from .generate import (
    OmniGenerator,
    _delayed_prompt,
    _input_column,
    _validate_voice_codes,
    sample_logits,
)

if TYPE_CHECKING:
    from ..text.tokenizer import ByteTokenizer, TextTokenizer

#: Sentinel for ``DuplexGenerator.reset``: "keep the stored voice" (distinct
#: from None, which clears it). Typed Any so it can default a Tensor|None arg.
_KEEP: Any = object()


@dataclass
class DuplexStep:
    """One 80 ms tick of the duplex loop.

    text_id: the sampled monologue token for this tick's frame (row 0;
        specials such as TEXT_PAD included — filter when decoding).
    assistant_frame: long [n_q] for the OLDEST newly completed frame —
        raw codec codes with sampled audio specials reduced to AUDIO_PAD
        (run `streams.sanitize_codes` before a codec decode). None while the
        delay pipeline fills (the first max_delay ticks) — and symmetrically
        the last max_delay pushed frames only complete during extra
        flush ticks.
    """

    text_id: int
    assistant_frame: torch.Tensor | None


class DuplexGenerator:
    """Streaming batch-1 duplex loop over a trained (or random-init) model.

    Requires ``cfg.model.duplex``. The model is moved to ``device`` and put in
    eval mode; ``reset()`` (called by ``__init__``) allocates a fresh decode cache,
    reseeds the sampling generator, and prefills the [<bos> + optional voice
    segment] prefix. Sampling runs on CPU with one shared ``torch.Generator``
    (seeded by ``seed``) so seeded runs reproduce across devices. Depth models
    (cfg.model.use_depth) route through ``prefill_hidden``/``step_hidden`` +
    ``model.depth.sample``; plain models use ``prefill``/``step`` parallel
    heads.

    ``voice_codes`` (DESIGN_V5): long ``[n_q, R]`` RAW codec codes of a
    reference speaker, pinned as a loss-shaped ``<voice>`` segment on the
    assistant rows directly after ``<bos>`` — one prefix serves the whole
    conversation. Stored across ``reset()`` calls; override or clear via
    ``reset(voice_codes=...)``. Without a voice the loop is bit-identical to
    the v2 behavior.
    """

    def __init__(
        self,
        model: OmniModel,
        cfg: OmniConfig,
        device: str | torch.device = "cpu",
        tokenizer: "TextTokenizer | ByteTokenizer | None" = None,
        sampling: SamplingConfig | None = None,
        seed: int | None = None,
        voice_codes: torch.Tensor | None = None,
    ):
        if not cfg.model.duplex:
            raise ValueError(
                "DuplexGenerator requires cfg.model.duplex=True; half-duplex "
                "tasks (tts/asr/s2s) run through omni.infer.OmniGenerator"
            )
        for f in OmniGenerator._STRUCTURAL_FIELDS:
            got, want = getattr(model.cfg, f), getattr(cfg.model, f)
            if got != want:
                raise ValueError(
                    f"cfg.model.{f}={want} does not match the model instance ({got}); "
                    "when loading a checkpoint, set cfg.model = model.cfg"
                )
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.sampling = sampling if sampling is not None else cfg.sampling
        self.seed = seed

        mc = cfg.model
        self.n_q = mc.n_codebooks
        self._cv = mc.audio_codec_vocab
        self._apad = audio_pad_id(self._cv)
        self._dl = stream_delays(self.n_q, mc.audio_delay_mode, duplex=True)
        self._D = max(self._dl)
        self._S = 1 + 2 * self.n_q
        self._use_depth = mc.use_depth
        if self._use_depth and getattr(self.model, "depth", None) is None:
            raise RuntimeError(
                "cfg.model.use_depth=True but this OmniModel has no depth "
                "transformer (model.depth); update omni.model to the v2 API"
            )
        self._dtype = next(self.model.parameters()).dtype
        self._voice_codes: torch.Tensor | None = (
            None
            if voice_codes is None
            else _validate_voice_codes(voice_codes, self.n_q, self._cv)
        )
        self.reset()

    # ----------------------------------------------------------------- state
    @torch.inference_mode()
    def reset(self, voice_codes: torch.Tensor | None = _KEEP) -> None:
        """Start a fresh conversation: new decode cache + RNG, prefill the prefix.

        The prefix is [<bos>] alone (P = 1) or [<bos> + <voice> + R reference
        frames + <voice_end>] (P = R + 2), prefilled with ONE model.prefill
        call over its delayed layout. After reset the pending logits predict
        frame 0 (delayed position P) and the next ``step`` call is tick 0.

        voice_codes: omitted -> keep the stored reference; a ``[n_q, R]`` raw
        code tensor -> replace it (persists for later resets); None -> clear.
        """
        if voice_codes is not _KEEP:
            self._voice_codes = (
                None
                if voice_codes is None
                else _validate_voice_codes(voice_codes, self.n_q, self._cv)
            )
        vc = self._voice_codes
        mc = self.cfg.model
        # Prefix length P: <bos> col + optional R+1 segment cols (R >= 1).
        self._p0 = 1 if vc is None else int(vc.shape[1]) + 2
        if self._p0 >= mc.max_frames:
            raise ValueError(
                f"voice reference of {self._p0 - 2} frames leaves no room to "
                f"tick within model.max_frames={mc.max_frames}"
            )
        # CUDA samples on-device (the 80 ms tick budget cannot afford a CPU
        # round-trip per codebook); CPU/MPS keep CPU sampling. Seeded runs
        # reproduce within a device type.
        self._on_dev = self.device.type == "cuda"
        self._gen: torch.Generator | None = None
        if self.seed is not None:
            self._gen = torch.Generator(device=self.device if self._on_dev else "cpu")
            self._gen.manual_seed(int(self.seed))
        self._cache = self.model.new_cache(1, self.device, self._dtype)
        # Undelayed buffer; delayed positions run 0..max_frames-1, so undelayed
        # cols reach at most max_frames (text col of the last tick).
        self._u = torch.full((self._S, mc.max_frames + 1), self._apad, dtype=torch.long)
        self._u[TEXT_STREAM] = TEXT_PAD  # never read before being written
        self._u[TEXT_STREAM, 0] = BOS
        if vc is not None:
            # grids.build_duplex voice segment at cols 1..R+1: <voice> over the
            # first reference frame, TEXT_PAD.., <voice_end> over audio APAD;
            # reference codes on ASSISTANT rows only, user rows stay APAD.
            self._u[TEXT_STREAM, 1] = VOICE
            self._u[TEXT_STREAM, self._p0 - 1] = VOICE_END
            self._u[1 : 1 + self.n_q, 1 : self._p0 - 1] = vc
        self._tick = 0
        self._hidden: torch.Tensor | None = None  # [1, d] (depth path)
        self._audio_logits: torch.Tensor | None = None  # [1, n_q, Va]
        # ONE prefill over delayed positions 0..P-1: row s at position p shows
        # undelayed col p - dl[s] (P = 1 degenerates to the bare <bos> column:
        # text BOS, audio rows all fillers since every audio delay >= 1).
        dgrid = _delayed_prompt(self._u, self._p0, self._dl, self._apad)
        grid = dgrid[None].to(self.device)  # [1, S, P]
        chan = torch.full(
            (1, self._p0), CHANNEL_ASSISTANT, dtype=torch.long, device=self.device
        )
        if self._use_depth:
            self._text_logits, self._hidden = self.model.prefill_hidden(grid, chan, self._cache)
        else:
            self._text_logits, self._audio_logits = self.model.prefill(grid, chan, self._cache)

    # ------------------------------------------------------------------ tick
    def _audio_sample_fn(self) -> Callable[[torch.Tensor, int], torch.Tensor]:
        sc = self.sampling

        def fn(logits: torch.Tensor, k: int) -> torch.Tensor:
            # [B, Va] -> long [B] on logits.device for the depth embedding
            # lookup; on-device sampling on CUDA (no per-codebook round-trip).
            del k
            if self._on_dev:
                return sample_logits(logits.float(), sc.audio_temperature, sc.audio_top_k, self._gen)
            out = sample_logits(logits.float().cpu(), sc.audio_temperature, sc.audio_top_k, self._gen)
            return out.to(logits.device)

        return fn

    @torch.inference_mode()
    def step(self, user_frame: torch.Tensor | None) -> DuplexStep:
        """One 80 ms tick: push user frame i, sample frame i, step the model.

        user_frame: long [n_q] RAW codec codes for user frame i (AUDIO_PAD
        entries allowed for non-speech; None means a full silence/AUDIO_PAD
        frame). Returns the sampled monologue token of frame i plus assistant
        frame i - max_delay once the delay pipeline has filled (None before).
        """
        sc = self.sampling
        i = self._tick
        P = self._p0  # undelayed prefix cols (<bos> + voice segment; 1 bare)
        p = i + P  # delayed position of this tick's model step
        if p >= self.cfg.model.max_frames:
            raise RuntimeError(
                f"duplex context exhausted: tick {i} needs delayed position {p} "
                f">= model.max_frames={self.cfg.model.max_frames} "
                f"(prefix {P} cols); call reset()"
            )

        # 1) User frame i -> user rows, undelayed col i + P. Its delayed
        #    positions are i + P + dl[1+k] per row; the earliest read is the
        #    step of tick i + 1 (at p = i + 1 + P), so landing it now is
        #    always in time.
        if user_frame is None:
            uf = torch.full((self.n_q,), self._apad, dtype=torch.long)
        else:
            uf = user_frame.to(device="cpu", dtype=torch.long).reshape(-1)
            if uf.shape != (self.n_q,):
                raise ValueError(f"user_frame must be [n_q={self.n_q}], got {tuple(user_frame.shape)}")
            ok = ((uf >= 0) & (uf < self._cv)) | (uf == self._apad)
            if not bool(ok.all()):
                raise ValueError("user_frame must hold raw codec codes or AUDIO_PAD")
        self._u[1 + self.n_q :, i + P] = uf

        # 2) Consume the pending logits (from the step at delayed p - 1 =
        #    i + P - 1; they predict delayed position q = i + P).
        q = i + P
        tl = self._text_logits[0] if self._on_dev else self._text_logits[0].cpu()
        tok = int(sample_logits(tl, sc.text_temperature, sc.text_top_k, self._gen))
        self._u[TEXT_STREAM, q] = tok  # text of frame i (col q = i + P)
        if self._use_depth:
            atoks = self.model.depth.sample(self._hidden, self._audio_sample_fn())[0].cpu()
        else:
            al = self._audio_logits[0] if self._on_dev else self._audio_logits[0].cpu()
            atoks = sample_logits(al, sc.audio_temperature, sc.audio_top_k, self._gen).cpu()
        for k in range(self.n_q):
            c = q - self._dl[1 + k]  # undelayed col of head k's sample
            if c >= P:  # cols < P are the <bos>+voice prefix: discard
                a = int(atoks[k])
                # sampled specials (incl. AUDIO_BOS/EOS) -> AUDIO_PAD
                self._u[1 + k, c] = a if a < self._cv else self._apad

        # 3) Frame f = i - D just completed: its most-delayed head wrote
        #    undelayed col q - D = f + P in (2); all other heads earlier.
        out_frame: torch.Tensor | None = None
        f = i - self._D
        if f >= 0:
            out_frame = self._u[1 : 1 + self.n_q, f + P].clone()

        # 4) Model step at delayed position p = i + P -> pending logits that
        #    tick i + 1 will consume.
        col = _input_column(self._u, p, self._dl, self._apad)
        grid1 = col[None].to(self.device)  # [1, S]
        chan1 = torch.full((1,), CHANNEL_ASSISTANT, dtype=torch.long, device=self.device)
        if self._use_depth:
            self._text_logits, self._hidden = self.model.step_hidden(grid1, chan1, self._cache)
        else:
            self._text_logits, self._audio_logits = self.model.step(grid1, chan1, self._cache)

        self._tick += 1
        return DuplexStep(text_id=tok, assistant_frame=out_frame)

    # ----------------------------------------------------------------- files
    @torch.inference_mode()
    def run_file(
        self, user_wav: torch.Tensor, codec: AudioCodec
    ) -> tuple[str, torch.Tensor]:
        """Offline duplex run over a whole user wav.

        user_wav: float32 mono [T_samples] at ``codec.sample_rate``. Encodes
        to T_user frames, resets (keeping any stored voice reference), ticks
        once per frame, then flushes max_delay extra ticks (user AUDIO_PAD) so
        the delay pipeline drains: exactly T_user assistant frames come out,
        with or without a voice prefix. Returns (monologue text with
        specials stripped — "" without a tokenizer, decoded assistant wav
        float32 [T_samples']); the assistant track is [n_q, T_user],
        sanitized before the codec decode.
        """
        mc = self.cfg.model
        if codec.n_codebooks != self.n_q:
            raise ValueError(f"codec has {codec.n_codebooks} codebooks, model wants {self.n_q}")
        if codec.codec_vocab != self._cv:
            raise ValueError(
                f"codec vocab {codec.codec_vocab} != model audio_codec_vocab {self._cv}"
            )
        if user_wav.ndim != 1:
            raise ValueError(f"wav must be mono [T_samples], got shape {tuple(user_wav.shape)}")
        codes = codec.encode(user_wav).to(device="cpu", dtype=torch.long)  # [n_q, T_user]
        t_user = int(codes.shape[1])
        # Delayed positions used: 0..P-1 (prefix prefill) + T_user + D ticks
        # (the last tick steps at p = T_user + D - 1 + P).
        if t_user + self._D + self._p0 > mc.max_frames:
            raise ValueError(
                f"user wav of {t_user} frames + {self._D} flush + {self._p0}-col "
                f"prefix (<bos> + voice segment) exceeds model.max_frames={mc.max_frames}"
            )

        self.reset()
        text_ids: list[int] = []
        frames: list[torch.Tensor] = []
        for t in range(t_user):
            st = self.step(codes[:, t])
            text_ids.append(st.text_id)
            if st.assistant_frame is not None:
                frames.append(st.assistant_frame)
        for _ in range(self._D):  # drain the delay pipeline
            st = self.step(None)
            text_ids.append(st.text_id)
            if st.assistant_frame is not None:
                frames.append(st.assistant_frame)

        track = (
            torch.stack(frames, dim=1)
            if frames
            else torch.zeros((self.n_q, 0), dtype=torch.long)
        )
        assert track.shape[1] == t_user, (
            f"assistant track {track.shape[1]} != user frames {t_user}"
        )
        if t_user:
            wav = codec.decode(sanitize_codes(track, self._cv))
        else:
            wav = torch.zeros(0, dtype=torch.float32)
        text = (
            self.tokenizer.decode(text_ids, skip_specials=True)
            if self.tokenizer is not None
            else ""
        )
        return text, wav
