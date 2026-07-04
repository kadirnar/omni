"""Frame-synchronous generation for the omni multistream decoder.

Implements the binding "Generation algorithm" of docs/INTERFACES.md as a
Moshi-style KV-cached frame loop (docs/research/inference-opt.md): one
`OmniModel.prefill` over the delayed prompt, then one `OmniModel.step` per
delayed position, writing samples back into a preallocated UNDELAYED buffer.
Depth models (cfg.model.use_depth, flat delays) route through
`prefill_hidden`/`step_hidden` and sample the codebooks of each frame
sequentially via `model.depth.sample`; the frame loop itself is identical.

Index math, re-derived from `streams.apply_delay` over
``dl = streams.stream_delays(n_q, mode)`` (stagger: [0, 1, .., n_q];
flat: [0, 1, 1, .., 1]) with flush horizon ``D = max(dl)`` (n_q stagger,
1 flat):

- Delayed position p of stream s shows undelayed column p - dl[s], so the
  input column fed at step p is ``undelayed[s, p - dl[s]]``, with filler
  PAD (text) / AUDIO_PAD (audio) where ``p - dl[s] < 0``.
- The logits returned by the step at position p predict position p + 1, hence
    text sample (delay 0)  -> undelayed text col p + 1
    audio head k           -> undelayed col (p + 1) - dl[1 + k],
  i.e. audio FRAME p + 1 - dl[1+k] (grid row 1 + k). Stagger: frame p - k;
  flat: every head lands on frame p (one whole frame per step).
- For a prompt of T0 undelayed columns (prefilled as delayed steps 0..T0-1),
  generated text cols and audio frames start at T0; head-k samples with
  p + 1 - dl[1+k] < T0 land in the prompt region and are discarded. Frame f
  is complete once the most-delayed head has produced it, i.e. after the
  step at p = f + D - 1.
- The delay pattern makes the loop causal: the input at step p only reads
  text col p (written after step p - 1) and audio cols p - dl[1+k] (head k's
  sample from step p - 1), so every read hits the prompt, an already-written
  sample, or a filler.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from ..audio.codec import AudioCodec
from ..config import OmniConfig, SamplingConfig
from ..grids import (
    build_asr_prompt,
    build_s2s_prompt,
    build_tts_prompt,
    prompt_forced_text,
)
from ..model.omni import OmniModel
from ..streams import (
    CHANNEL_ASSISTANT,
    END_OF_TURN,
    EOS,
    PAD,
    TEXT_PAD,
    TEXT_STREAM,
    Sample,
    audio_eos_id,
    audio_pad_id,
    sanitize_codes,
    stream_delays,
    trim_audio_at_eos,
)

if TYPE_CHECKING:
    from ..text.tokenizer import ByteTokenizer, TextTokenizer

#: Default reference-voice length in frames (10 s at 12.5 Hz). DESIGN_V5_VOICE
#: "Reference length": inference default R=125; training samples 3-20 s.
DEFAULT_VOICE_FRAMES = 125


def _validate_voice_codes(codes: torch.Tensor, n_q: int, codec_vocab: int) -> torch.Tensor:
    """Validate + copy reference-voice codes -> CPU long ``[n_q, R]``, R >= 1.

    The voice segment (grids.voice_segment) takes RAW codec codes only: no
    audio specials, no negatives. Raises ValueError on any violation.
    """
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.shape[0] != n_q:
        got = tuple(codes.shape) if isinstance(codes, torch.Tensor) else type(codes).__name__
        raise ValueError(f"voice codes must be a tensor [n_codebooks={n_q}, R], got {got}")
    codes = codes.to(device="cpu", dtype=torch.long).clone()
    if codes.shape[1] < 1:
        raise ValueError("voice reference must contain at least one frame")
    if not bool(((codes >= 0) & (codes < codec_vocab)).all()):
        raise ValueError(
            f"voice codes must be raw codec codes in [0, {codec_vocab}) "
            "(no audio specials; encode the reference wav, don't reuse grids)"
        )
    return codes


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample token ids from logits ``[..., V]`` -> long ``[...]``.

    ``temperature <= 0`` means argmax (greedy). Otherwise: divide by
    temperature, keep the ``top_k`` highest logits per row (no filtering when
    ``top_k <= 0`` or ``top_k >= V``), softmax, and draw with
    ``torch.multinomial`` using ``gen`` (``gen`` must live on the logits'
    device; None uses the global RNG).
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    x = logits.float() / temperature
    v = x.shape[-1]
    if 0 < top_k < v:
        kth = x.topk(top_k, dim=-1).values[..., -1:]  # [..., 1] k-th largest
        x = x.masked_fill(x < kth, float("-inf"))
    probs = torch.softmax(x, dim=-1)
    flat = probs.reshape(-1, v)  # multinomial wants [N, V]
    idx = torch.multinomial(flat, num_samples=1, generator=gen)
    return idx.reshape(logits.shape[:-1])


@dataclass
class GenResult:
    """Output of one generation.

    text_ids: every generated text-stream token in order (forced ids included,
        specials included; specials are filtered only for ``.text``).
    text: ``tokenizer.decode(text_ids, skip_specials=True)``; None when the
        generator has no tokenizer.
    audio_codes: long ``[n_q, frames]`` CPU tensor of RAW codec codes for the
        assistant segment (trimmed at AUDIO_EOS, specials replaced by code 0),
        directly decodable by an `AudioCodec`.
    frames: ``audio_codes.shape[1]``.
    """

    text_ids: list[int]
    text: str | None
    audio_codes: torch.Tensor
    frames: int


@dataclass
class StreamEvent:
    """One event from ``OmniGenerator.stream``.

    kind: "text" (token: generated text id), "frame" (frame: long [n_q] raw
    sanitized codes of one COMPLETE assistant audio frame, decodable
    incrementally), or "done" (result: the full GenResult, identical to a
    ``generate`` call with the same seed/arguments).
    """

    kind: str
    token: int | None = None
    frame: torch.Tensor | None = None
    result: "GenResult | None" = None


def _input_column(u: torch.Tensor, p: int, dl: list[int], apad: int) -> torch.Tensor:
    """Delayed input column at position p from the undelayed buffer.

    u long [S, cap], dl = per-row delays (``streams.stream_delays``, len S);
    returns long [S] with ``col[s] = u[s, p - dl[s]]`` and AUDIO_PAD where
    ``p - dl[s] < 0`` (text delay is 0, so row 0 always reads col p; the
    buffer's untouched tail already holds PAD/AUDIO_PAD).
    """
    col = torch.empty(len(dl), dtype=torch.long)
    col[TEXT_STREAM] = u[TEXT_STREAM, p]
    for s in range(1, len(dl)):
        j = p - dl[s]
        col[s] = u[s, j] if j >= 0 else apad
    return col


def _delayed_prompt(u: torch.Tensor, t0: int, dl: list[int], apad: int) -> torch.Tensor:
    """Delayed grid of the first t0 undelayed cols: ``d[s, p] = u[s, p - dl[s]]``.

    u long [S, cap] (prompt in cols 0..t0-1) -> long [S, t0]. Equals
    ``streams.apply_delay(prompt)[0][:, :t0]``: each stream shifted right by
    its delay with PAD/AUDIO_PAD fillers in the head.
    """
    dgrid = torch.full((len(dl), t0), apad, dtype=torch.long)
    dgrid[TEXT_STREAM] = PAD
    for s, d in enumerate(dl):
        if d < t0:
            dgrid[s, d:] = u[s, : t0 - d]
    return dgrid


class OmniGenerator:
    """Batch-1 frame-loop generator over a trained (or random-init) OmniModel.

    The model is moved to ``device`` and put in eval mode. Each ``generate``
    call allocates a fresh decode cache via ``model.new_cache`` (batch 1,
    model dtype); sampling runs
    on CPU with one shared ``torch.Generator`` so seeded runs reproduce across
    devices. ``tokenizer`` (TextTokenizer/ByteTokenizer duck type) is optional
    and only needed for ``tts`` and for decoding ``GenResult.text``.

    Reference-voice cloning (DESIGN_V5): ``set_voice`` pins a session voice;
    ``tts``/``s2s``/``asr`` also take one-shot ``voice_wav``/``voice_codes``
    overrides. The reference rides the prompt as a loss-shaped
    ``<voice>``-delimited segment (grids builders' ``voice_codes``), so the
    frame loop is untouched and ``generate`` needs no voice awareness.
    """

    # ModelConfig fields that must agree between cfg.model and model.cfg for
    # KV allocation / vocab checks / index math to be meaningful.
    _STRUCTURAL_FIELDS = (
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "d_ff",
        "max_frames",
        "text_vocab_size",
        "n_codebooks",
        "audio_codec_vocab",
        "audio_delay_mode",
        "use_depth",
        "duplex",
    )

    def __init__(
        self,
        model: OmniModel,
        cfg: OmniConfig,
        device: str | torch.device = "cpu",
        tokenizer: "TextTokenizer | ByteTokenizer | None" = None,
    ):
        for f in self._STRUCTURAL_FIELDS:
            got, want = getattr(model.cfg, f), getattr(cfg.model, f)
            if got != want:
                raise ValueError(
                    f"cfg.model.{f}={want} does not match the model instance ({got}); "
                    "when loading a checkpoint, set cfg.model = model.cfg"
                )
        if cfg.model.duplex:
            raise ValueError(
                "OmniGenerator handles half-duplex tasks only; duplex models "
                "stream through omni.infer.duplex.DuplexGenerator"
            )
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self._dtype = next(self.model.parameters()).dtype
        self._voice_codes: torch.Tensor | None = None  # session voice (set_voice)

    # ------------------------------------------------------------------ voice
    def set_voice(
        self,
        wav: torch.Tensor | None,
        codec: AudioCodec | None = None,
        *,
        voice_codes: torch.Tensor | None = None,
        max_frames: int = DEFAULT_VOICE_FRAMES,
    ) -> None:
        """Pin (or clear) the session reference voice (DESIGN_V5).

        wav: float32 mono ``[T_samples]`` at ``codec.sample_rate``, encoded via
        ``codec`` (required alongside wav); or pass pre-encoded ``voice_codes``
        long ``[n_q, R]`` raw codes instead (exclusive with wav). The stored
        reference is trimmed to its FIRST ``max_frames`` frames (default 125 =
        10 s, the DESIGN_V5 inference length). ``set_voice(None)`` clears.
        Subsequent ``tts``/``s2s``/``asr`` calls prepend the reference as a
        ``<voice>`` prompt segment unless overridden per call.
        """
        if wav is not None and voice_codes is not None:
            raise ValueError("set_voice takes wav OR voice_codes, not both")
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        if wav is not None:
            if codec is None:
                raise ValueError("set_voice(wav) needs codec= to encode the reference")
            self._check_codec(codec)
            voice_codes = self._encode_user(wav, codec)
        if voice_codes is None:
            self._voice_codes = None
            return
        mc = self.cfg.model
        codes = _validate_voice_codes(voice_codes, mc.n_codebooks, mc.audio_codec_vocab)
        self._voice_codes = codes[:, : int(max_frames)]

    def _resolve_voice(
        self,
        voice_wav: torch.Tensor | None,
        voice_codes: torch.Tensor | None,
        codec: AudioCodec,
    ) -> torch.Tensor | None:
        """Reference codes for one call: per-call override, else session state.

        ``voice_wav`` is encoded via the call's codec and trimmed to the
        default 125 frames; ``voice_codes`` are used as given (validated);
        both None falls back to the ``set_voice`` session reference.
        """
        if voice_wav is not None and voice_codes is not None:
            raise ValueError("pass voice_wav OR voice_codes, not both")
        mc = self.cfg.model
        if voice_wav is not None:
            codes = _validate_voice_codes(
                self._encode_user(voice_wav, codec), mc.n_codebooks, mc.audio_codec_vocab
            )
            return codes[:, :DEFAULT_VOICE_FRAMES]
        if voice_codes is not None:
            return _validate_voice_codes(voice_codes, mc.n_codebooks, mc.audio_codec_vocab)
        return self._voice_codes

    # ------------------------------------------------------------- core loop
    @torch.inference_mode()
    def generate(
        self,
        prompt: Sample,
        forced_text: list[int],
        *,
        sampling: SamplingConfig | None = None,
        max_frames: int | None = None,
        seed: int | None = None,
    ) -> GenResult:
        """Run the frame loop after an UNDELAYED prompt.

        prompt: Sample with grid [S, T0] (task read from ``prompt.task``).
        forced_text: ids fed as the text-stream input (and record) for the
            first ``len(forced_text)`` generated columns, overriding sampling;
            in tts mode the text is clamped to TEXT_PAD once the queue (which
            ends with END_OF_TURN) is exhausted.
        max_frames: cap on generated audio frames (default
            ``sampling.max_frames``), clamped so the delayed length
            T0 + frames + max_delay fits ``cfg.model.max_frames``.

        Stop: cb0 sampling AUDIO_EOS at frame f_eos ends generation after a
        trailing-codebook flush through step f_eos + max_delay - 1 (frames
        T0..f_eos-1 are returned; flat mode has max_delay 1, so no extra
        steps); in asr mode the audio stop is ignored and generation ends
        when the text head emits EOS or END_OF_TURN. Always capped at
        max_frames.

        Implemented as the sink of :meth:`stream` — ONE frame loop serves both
        entry points, so batch and streaming generation cannot drift apart.
        """
        for ev in self.stream(
            prompt, forced_text, sampling=sampling, max_frames=max_frames, seed=seed
        ):
            if ev.kind == "done":
                assert ev.result is not None
                return ev.result
        raise RuntimeError("stream ended without a 'done' event")  # unreachable

    # ------------------------------------------------------------ task sugar
    def _check_codec(self, codec: AudioCodec) -> None:
        mc = self.cfg.model
        if codec.n_codebooks != mc.n_codebooks:
            raise ValueError(
                f"codec has {codec.n_codebooks} codebooks, model wants {mc.n_codebooks}"
            )
        if codec.codec_vocab != mc.audio_codec_vocab:
            raise ValueError(
                f"codec vocab {codec.codec_vocab} != model audio_codec_vocab "
                f"{mc.audio_codec_vocab}"
            )

    def _encode_user(self, wav: torch.Tensor, codec: AudioCodec) -> torch.Tensor:
        if wav.ndim != 1:
            raise ValueError(f"wav must be mono [T_samples], got shape {tuple(wav.shape)}")
        return codec.encode(wav).to(device="cpu", dtype=torch.long)  # [n_q, T_fr]

    def s2s(
        self,
        wav: torch.Tensor,
        codec: AudioCodec,
        prefix_ids: list[int] | None = None,
        *,
        voice_wav: torch.Tensor | None = None,
        voice_codes: torch.Tensor | None = None,
        **kw,
    ) -> GenResult:
        """User speech wav float32 [T_samples] (at codec.sample_rate) -> spoken reply.

        ``prefix_ids`` (DESIGN_V4 §1): monologue control ids forced right after
        ``<assistant>`` — e.g. ``streams.turn_prefix(lang="en",
        response_style="calm")`` locks the reply language/emotion; None lets the
        model choose (it may emit its own perceived/response tags).
        ``voice_wav``/``voice_codes`` (DESIGN_V5): one-shot reference voice for
        this call, else the ``set_voice`` session reference; ONE ``<voice>``
        segment after ``<bos>`` pins the reply timbre (tags keep style
        authority). kw forwards ``sampling``/``max_frames``/``seed`` to
        ``generate``.
        """
        self._check_codec(codec)
        mc = self.cfg.model
        prompt = build_s2s_prompt(
            self._encode_user(wav, codec),
            mc.n_codebooks,
            mc.audio_codec_vocab,
            voice_codes=self._resolve_voice(voice_wav, voice_codes, codec),
        )
        forced = prompt_forced_text("s2s") + list(prefix_ids or [])
        return self.generate(prompt, forced, **kw)

    def tts(
        self,
        text: str,
        codec: AudioCodec,
        prefix_ids: list[int] | None = None,
        *,
        voice_wav: torch.Tensor | None = None,
        voice_codes: torch.Tensor | None = None,
        **kw,
    ) -> GenResult:
        """Speak ``text``: forced monologue + free audio; result decodable via codec.

        ``voice_wav``/``voice_codes``: one-shot reference voice (see ``s2s``).
        """
        self._check_codec(codec)
        if self.tokenizer is None:
            raise ValueError("tts needs a tokenizer (pass tokenizer= to OmniGenerator)")
        mc = self.cfg.model
        prompt = build_tts_prompt(
            mc.n_codebooks,
            mc.audio_codec_vocab,
            voice_codes=self._resolve_voice(voice_wav, voice_codes, codec),
        )
        text_ids = list(prefix_ids or []) + self.tokenizer.encode(text)
        forced = prompt_forced_text("tts", text_ids)
        return self.generate(prompt, forced, **kw)

    def asr(
        self,
        wav: torch.Tensor,
        codec: AudioCodec,
        prefix_ids: list[int] | None = None,
        *,
        voice_wav: torch.Tensor | None = None,
        voice_codes: torch.Tensor | None = None,
        **kw,
    ) -> GenResult:
        """Transcribe user speech; ``.text`` holds the transcript (audio is moot).

        A reference voice is accepted and threaded like ``s2s`` — the
        DESIGN_V5 invariance path (~10% of training asr grids carry an ignored
        segment so a cached session ``[<bos>+voice]`` prefix is never OOD).
        """
        self._check_codec(codec)
        mc = self.cfg.model
        prompt = build_asr_prompt(
            self._encode_user(wav, codec),
            mc.n_codebooks,
            mc.audio_codec_vocab,
            voice_codes=self._resolve_voice(voice_wav, voice_codes, codec),
        )
        forced = prompt_forced_text("asr") + list(prefix_ids or [])
        return self.generate(prompt, forced, **kw)

    # ------------------------------------------------------------- streaming
    def task_prompt(
        self,
        task: str,
        *,
        text: str | None = None,
        wav: torch.Tensor | None = None,
        codec: AudioCodec | None = None,
        prefix_ids: list[int] | None = None,
        voice_wav: torch.Tensor | None = None,
        voice_codes: torch.Tensor | None = None,
    ) -> tuple[Sample, list[int]]:
        """(prompt, forced_text) for one task — the exact inputs ``tts``/``s2s``/
        ``asr`` hand to ``generate``, exposed so streaming callers (the test
        console server) can drive ``stream`` with identical semantics.
        """
        mc = self.cfg.model
        if task == "tts":
            if text is None:
                raise ValueError("task 'tts' needs text=")
            if self.tokenizer is None:
                raise ValueError("tts needs a tokenizer (pass tokenizer= to OmniGenerator)")
            if codec is not None:
                self._check_codec(codec)
            vc = self._resolve_voice(voice_wav, voice_codes, codec) if codec else self._voice_codes
            prompt = build_tts_prompt(mc.n_codebooks, mc.audio_codec_vocab, voice_codes=vc)
            text_ids = list(prefix_ids or []) + self.tokenizer.encode(text)
            return prompt, prompt_forced_text("tts", text_ids)
        if task in ("s2s", "asr"):
            if wav is None or codec is None:
                raise ValueError(f"task {task!r} needs wav= and codec=")
            self._check_codec(codec)
            vc = self._resolve_voice(voice_wav, voice_codes, codec)
            build = build_s2s_prompt if task == "s2s" else build_asr_prompt
            prompt = build(
                self._encode_user(wav, codec), mc.n_codebooks, mc.audio_codec_vocab,
                voice_codes=vc,
            )
            return prompt, prompt_forced_text(task) + list(prefix_ids or [])
        raise ValueError(f"unknown task {task!r}: expected tts|s2s|asr")

    @torch.inference_mode()
    def stream(
        self,
        prompt: Sample,
        forced_text: list[int],
        *,
        sampling: SamplingConfig | None = None,
        max_frames: int | None = None,
        seed: int | None = None,
        stop: "object | None" = None,
    ) -> "Iterator[StreamEvent]":
        """``generate`` as an event stream, for live playback.

        Yields, in order: one ``StreamEvent("text", token=...)`` per generated
        text column (forced ids included, same order as ``GenResult.text_ids``);
        ``StreamEvent("frame", frame=[n_q] raw sanitized codes)`` for every
        COMPLETE assistant audio frame (a frame is emitted only once its
        most-delayed codebook has been produced — the same frames, in the same
        order, as the columns of the final ``GenResult.audio_codes``); finally
        one ``StreamEvent("done", result=GenResult)`` whose result matches a
        seeded ``generate`` call with identical arguments token-for-token
        (pinned by tests/test_serve.py parity tests).

        ``stop``: optional object with a truthy ``is_set()`` (e.g.
        ``threading.Event``) checked once per step for cooperative aborts;
        aborted streams still yield a valid ``done`` over what was produced.
        This is THE frame loop — ``generate`` consumes this stream, so the two
        entry points share one implementation by construction.
        """
        sc = sampling if sampling is not None else self.cfg.sampling
        mc = self.cfg.model
        n_q = mc.n_codebooks
        s_all = 1 + n_q
        cv = mc.audio_codec_vocab
        apad = audio_pad_id(cv)
        aeos = audio_eos_id(cv)
        dl = stream_delays(n_q, mc.audio_delay_mode)
        D = max(dl)
        use_depth = mc.use_depth
        if use_depth and getattr(self.model, "depth", None) is None:
            raise RuntimeError(
                "cfg.model.use_depth=True but this OmniModel has no depth "
                "transformer (model.depth); update omni.model to the v2 API"
            )
        if prompt.n_streams != s_all:
            raise ValueError(f"prompt has {prompt.n_streams} streams, model wants {s_all}")
        prompt.validate(cv, mc.text_vocab_size)
        forced = [int(i) for i in forced_text]
        for i in forced:
            if not 0 <= i < mc.text_vocab_size:
                raise ValueError(f"forced text id {i} outside text vocab {mc.text_vocab_size}")

        t0 = prompt.n_frames
        f_req = int(max_frames) if max_frames is not None else int(sc.max_frames)
        if f_req < 1:
            raise ValueError(f"max_frames must be >= 1, got {f_req}")
        f_max = min(f_req, mc.max_frames - t0 - D)
        if f_max < 1:
            raise ValueError(
                f"prompt of {t0} frames + {D} delay slack leaves no room to "
                f"generate within model.max_frames={mc.max_frames}"
            )
        # At most f_max + D text columns can ever be generated: a longer forced
        # queue would be silently cut mid-sentence (partial TTS with no error).
        if len(forced) > f_max + D:
            raise ValueError(
                f"forced_text has {len(forced)} ids but only {f_max + D} text "
                f"columns fit (max_frames={f_max}); raise max_frames (or "
                "model.max_frames) or shorten the text"
            )

        cap = t0 + f_max + D
        u = torch.full((s_all, cap), apad, dtype=torch.long)
        u[TEXT_STREAM] = PAD
        u[:, :t0] = prompt.grid.to(device="cpu", dtype=torch.long)

        cache = self.model.new_cache(1, self.device, self._dtype)
        dgrid = _delayed_prompt(u, t0, dl, apad)
        dchan = prompt.channel.to(device="cpu", dtype=torch.long)
        hidden: torch.Tensor | None = None
        audio_logits: torch.Tensor | None = None
        if use_depth:
            text_logits, hidden = self.model.prefill_hidden(
                dgrid[None].to(self.device), dchan[None].to(self.device), cache
            )
        else:
            text_logits, audio_logits = self.model.prefill(
                dgrid[None].to(self.device), dchan[None].to(self.device), cache
            )

        # CUDA samples on-device (one scalar/vector D2H per step instead of a
        # full-vocab logits copy, and zero copies inside the depth rollout —
        # the 80 ms budget cannot afford 32 round-trips per frame). CPU/MPS
        # keep CPU sampling; seeded runs reproduce within a device type.
        on_dev = self.device.type == "cuda"
        gen: torch.Generator | None = None
        if seed is not None:
            gen = torch.Generator(device=self.device if on_dev else "cpu")
            gen.manual_seed(int(seed))

        def depth_sample_fn(logits: torch.Tensor, k: int) -> torch.Tensor:
            # [B, Va] -> long [B] on logits.device (model.depth.sample embeds
            # the result for the next codebook).
            del k  # per-codebook settings are uniform in SamplingConfig
            if on_dev:
                return sample_logits(logits.float(), sc.audio_temperature, sc.audio_top_k, gen)
            out = sample_logits(logits.float().cpu(), sc.audio_temperature, sc.audio_top_k, gen)
            return out.to(logits.device)

        tts_mode = prompt.task == "tts"
        asr_mode = prompt.task == "asr"
        chan1 = torch.full((1,), CHANNEL_ASSISTANT, dtype=torch.long, device=self.device)

        text_record: list[int] = []
        fi = 0
        f_eos: int | None = None
        p_last = t0 + f_max + D - 2
        p = t0 - 1
        emitted = 0  # complete frames already yielded (asr never emits frames)

        def _complete(p_now: int) -> int:
            c = max(0, min(p_now - t0 - D + 2, f_max))
            if f_eos is not None:
                c = min(c, f_eos - t0)
            return c

        while True:
            if fi < len(forced):
                tok = forced[fi]
                fi += 1
            elif tts_mode:
                tok = TEXT_PAD
            else:
                tl = text_logits[0] if on_dev else text_logits[0].cpu()
                tok = int(sample_logits(tl, sc.text_temperature, sc.text_top_k, gen))
            u[TEXT_STREAM, p + 1] = tok
            text_record.append(tok)
            yield StreamEvent("text", token=tok)

            if use_depth:
                atoks = self.model.depth.sample(hidden, depth_sample_fn)[0].cpu()
            else:
                al = audio_logits[0] if on_dev else audio_logits[0].cpu()
                atoks = sample_logits(al, sc.audio_temperature, sc.audio_top_k, gen).cpu()
            for k in range(n_q):
                v = int(atoks[k])
                if v >= cv and not (k == 0 and v == aeos):
                    # Sampled audio specials never appear as inputs in training
                    # grids (AUDIO_BOS anywhere; PAD/EOS on codebooks > 0 —
                    # eos_frame puts AUDIO_EOS on cb0 only). Feed AUDIO_PAD
                    # back instead of an out-of-distribution token; the raw
                    # sample still drives the cb0 stop condition below.
                    v = apad
                f = p + 1 - dl[1 + k]
                if f >= t0:
                    u[1 + k, f] = v

            if not asr_mode and f_eos is None and t0 <= p < t0 + f_max and int(atoks[0]) == aeos:
                f_eos = p
                p_last = min(p_last, f_eos + D - 1)
            # A completed frame carrying AUDIO_EOS on cb0 ends the emitted
            # audio (mirrors trim_audio_at_eos in the final result).
            if not asr_mode:
                while emitted < _complete(p):
                    fr = u[1:, t0 + emitted]
                    if f_eos is None and int(fr[0]) == aeos:
                        break
                    yield StreamEvent("frame", frame=sanitize_codes(fr.clone(), cv))
                    emitted += 1
            if asr_mode and tok in (EOS, END_OF_TURN):
                break
            if p >= p_last:
                break
            if stop is not None and stop.is_set():
                break

            p += 1
            col = _input_column(u, p, dl, apad)
            if use_depth:
                text_logits, hidden = self.model.step_hidden(
                    col[None].to(self.device), chan1, cache
                )
            else:
                text_logits, audio_logits = self.model.step(
                    col[None].to(self.device), chan1, cache
                )

        if f_eos is not None:
            n_out = f_eos - t0
        else:
            n_out = max(0, min(p - t0 - D + 2, f_max))
        codes = u[1:, t0 : t0 + n_out]
        codes = trim_audio_at_eos(codes, cv)
        codes = sanitize_codes(codes, cv)
        # flush any frames completed on the final step (or cut short by EOS trim)
        if not asr_mode:
            while emitted < codes.shape[1]:
                yield StreamEvent("frame", frame=codes[:, emitted].clone())
                emitted += 1

        text = (
            self.tokenizer.decode(text_record, skip_specials=True)
            if self.tokenizer is not None
            else None
        )
        yield StreamEvent(
            "done",
            result=GenResult(
                text_ids=text_record,
                text=text,
                audio_codes=codes,
                frames=int(codes.shape[1]),
            ),
        )
