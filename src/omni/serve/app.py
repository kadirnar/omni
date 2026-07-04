"""Streaming test-console server: FastAPI + WebSockets over the omni model.

Protocol (all JSON messages are text frames; audio is binary frames):

``WS /ws/generate`` — half-duplex tasks (tts / s2s / asr)
  client -> {"task": "tts|s2s|asr", "text"?, "audio_wav_b64"?, "voice_wav_b64"?,
             "emotion"?, "lang"?, "seed"?, "max_frames"?}
  server -> {"type":"status", ...}                    once, after prefill
            {"type":"text","id":int,"piece":str,"special":str|null}   per token
            <binary>                                  int16 PCM chunks @ codec sr
            {"type":"metrics", ...}                   every few frames
            {"type":"done","text":...,"frames":...,"ttfa_ms":...,"rtf":...}
  client -> {"type":"stop"} at any point for a cooperative abort.

``WS /ws/duplex`` — full duplex (needs a duplex model)
  client -> {"seed"?, "voice_wav_b64"?} then binary frames of 1920 int16
            samples (80 ms @ 24 kHz); {"type":"stop"} ends the session.
  server -> {"type":"text",...} / <binary assistant PCM> / {"type":"metrics",...}

The server runs ONE generation at a time (the model is not thread-safe); a
second connection gets {"type":"error","message":"busy"}. This is a local
test tool: bind to localhost, no auth.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..audio.codec import AudioCodec, resample
from ..config import OmniConfig
from ..infer.generate import DEFAULT_VOICE_FRAMES, OmniGenerator
from ..model.omni import OmniModel
from ..streams import EMOTION_CLASSES, LANG_TAGS, N_RESERVED_SPECIALS, turn_prefix
from ..text.tokenizer import special_token_strings
from .streaming import ChunkDecoder, StreamingEncoder, pcm16_bytes

_STATIC = Path(__file__).parent / "static"


def _wav_from_b64(b64: str, target_sr: int) -> torch.Tensor:
    """base64 wav/flac bytes -> mono float32 [T] at target_sr (anti-aliased
    via the repo's shared resampler)."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    return resample(wav, int(sr), int(target_sr))


def _specials() -> list[str | None]:
    names = special_token_strings()
    return [n.strip("<>") for n in names]


def create_app(
    *,
    cfg: OmniConfig,
    model: OmniModel,
    codec: AudioCodec,
    tokenizer: Any,
    ckpt: str | None = None,
    device: str = "cpu",
) -> FastAPI:
    app = FastAPI(title="omni test console")
    gen = None if cfg.model.duplex else OmniGenerator(model, cfg, device=device, tokenizer=tokenizer)
    codec = codec.to(device)
    # Single-flight guard. A plain flag flipped with NO await between check
    # and set is atomic on the one event loop — the old `busy.locked()` +
    # `async with busy` pair let two simultaneous connections both pass the
    # check, silently queueing the second instead of rejecting it.
    busy = {"v": False}
    special_names = _specials()
    spf = int(round(codec.sample_rate / codec.frame_rate))
    frame_ms = 1000.0 / float(codec.frame_rate)

    def piece_of(tok: int) -> tuple[str, str | None]:
        if tok < N_RESERVED_SPECIALS:
            return "", special_names[tok]
        return tokenizer.decode([tok], skip_specials=True), None

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/info")
    async def info() -> dict:
        mc = cfg.model
        return {
            "preset": cfg.preset,
            "ckpt": ckpt,
            "params": model.param_counts()["total"],
            "duplex": mc.duplex,
            "tasks": ["duplex"] if mc.duplex else ["tts", "s2s", "asr"],
            "n_codebooks": mc.n_codebooks,
            "delay_mode": mc.audio_delay_mode,
            "use_depth": mc.use_depth,
            "codec": type(codec).__name__,
            "sample_rate": codec.sample_rate,
            "frame_ms": frame_ms,
            "samples_per_frame": spf,
            "voice_frames": DEFAULT_VOICE_FRAMES,
            "emotions": sorted(EMOTION_CLASSES),
            "langs": sorted(LANG_TAGS),
            "max_frames_default": cfg.sampling.max_frames,
        }

    async def _watch_stop(ws: WebSocket, flag: threading.Event) -> None:
        try:
            while not flag.is_set():
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    flag.set()
                    break
                text = msg.get("text")
                if text and json.loads(text).get("type") == "stop":
                    flag.set()
                    break
        except Exception:
            flag.set()

    @app.websocket("/ws/generate")
    async def ws_generate(ws: WebSocket) -> None:
        await ws.accept()
        if gen is None:
            await ws.send_json({"type": "error", "message": "duplex model: use /ws/duplex"})
            await ws.close()
            return
        if busy["v"]:
            await ws.send_json({"type": "error", "message": "busy: one generation at a time"})
            await ws.close()
            return
        busy["v"] = True
        try:
            try:
                req = json.loads(await ws.receive_text())
                task = req.get("task", "tts")
                sr = codec.sample_rate
                wav = _wav_from_b64(req["audio_wav_b64"], sr) if req.get("audio_wav_b64") else None
                voice = _wav_from_b64(req["voice_wav_b64"], sr) if req.get("voice_wav_b64") else None
                prefix = turn_prefix(lang=req.get("lang"), response_style=req.get("emotion"))
                prompt, forced = gen.task_prompt(
                    task,
                    text=req.get("text"),
                    wav=wav,
                    codec=codec,
                    prefix_ids=prefix or None,
                    voice_wav=voice,
                )
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                await ws.send_json({"type": "error", "message": str(e)})
                await ws.close()
                return

            stop = threading.Event()
            watcher = asyncio.create_task(_watch_stop(ws, stop))
            # unbounded: the producer thread must never block on put() after the
            # consumer bails early (error/disconnect), or shutdown would deadlock
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            t_start = time.perf_counter()

            def run() -> None:
                try:
                    last = time.perf_counter()
                    for ev in gen.stream(
                        prompt,
                        forced,
                        max_frames=req.get("max_frames"),
                        seed=req.get("seed"),
                        stop=stop,
                    ):
                        now = time.perf_counter()
                        dt_ms = (now - last) * 1000.0
                        last = now
                        asyncio.run_coroutine_threadsafe(queue.put((ev, dt_ms)), loop).result()
                except Exception as e:  # surfaced to the client, not swallowed
                    asyncio.run_coroutine_threadsafe(queue.put((e, 0.0)), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put((None, 0.0)), loop).result()

            worker = loop.run_in_executor(None, run)
            dec = ChunkDecoder(codec, crossfade_ms=0.0 if type(codec).__name__ == "FakeCodec" else 4.0)
            n_frames = 0
            ttfa_ms: float | None = None
            frame_ms_sum = 0.0
            prefill_ms: float | None = None
            try:
                while True:
                    ev, dt_ms = await queue.get()
                    if ev is None:
                        break
                    if isinstance(ev, Exception):
                        await ws.send_json({"type": "error", "message": str(ev)})
                        break
                    if prefill_ms is None:  # first event follows prompt prefill
                        prefill_ms = (time.perf_counter() - t_start) * 1000.0
                        await ws.send_json(
                            {"type": "status", "prefill_ms": round(prefill_ms, 2),
                             "prompt_frames": prompt.n_frames}
                        )
                    if ev.kind == "text":
                        piece, special = piece_of(ev.token)
                        await ws.send_json(
                            {"type": "text", "id": ev.token, "piece": piece, "special": special}
                        )
                    elif ev.kind == "frame":
                        n_frames += 1
                        frame_ms_sum += dt_ms
                        if ttfa_ms is None:
                            ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                        pcm = dec.feed(ev.frame)
                        if pcm.numel():
                            await ws.send_bytes(pcm16_bytes(pcm))
                        if n_frames % 5 == 0:
                            avg = frame_ms_sum / n_frames
                            await ws.send_json(
                                {"type": "metrics", "frames": n_frames,
                                 "ttfa_ms": round(ttfa_ms, 1),
                                 "ms_per_frame": round(avg, 2),
                                 "last_ms": round(dt_ms, 2),
                                 "rtf": round(frame_ms / max(avg, 1e-6), 2)}
                            )
                    elif ev.kind == "done":
                        tail = dec.flush()
                        if tail.numel():
                            await ws.send_bytes(pcm16_bytes(tail))
                        avg = frame_ms_sum / n_frames if n_frames else 0.0
                        await ws.send_json(
                            {"type": "done", "text": ev.result.text,
                             "text_ids": ev.result.text_ids,
                             "frames": ev.result.frames,
                             "stopped": stop.is_set(),
                             "prefill_ms": round(prefill_ms or 0.0, 2),
                             "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
                             "ms_per_frame": round(avg, 2),
                             "rtf": round(frame_ms / avg, 2) if avg > 0 else None,
                             "wall_s": round(time.perf_counter() - t_start, 3)}
                        )
            except WebSocketDisconnect:
                stop.set()
            finally:
                stop.set()
                watcher.cancel()
                await worker
                try:
                    await ws.close()
                except Exception:
                    pass
        finally:
            busy["v"] = False

    @app.websocket("/ws/duplex")
    async def ws_duplex(ws: WebSocket) -> None:
        await ws.accept()
        if not cfg.model.duplex:
            await ws.send_json({"type": "error", "message": "not a duplex model: use /ws/generate"})
            await ws.close()
            return
        if busy["v"]:
            await ws.send_json({"type": "error", "message": "busy: one generation at a time"})
            await ws.close()
            return
        busy["v"] = True
        try:
            from ..infer.duplex import DuplexGenerator

            try:
                req = json.loads(await ws.receive_text())
                voice_codes = None
                if req.get("voice_wav_b64"):
                    ref = _wav_from_b64(req["voice_wav_b64"], codec.sample_rate)
                    voice_codes = codec.encode(ref)[:, :DEFAULT_VOICE_FRAMES]
                dgen = DuplexGenerator(
                    model, cfg, device=device, tokenizer=tokenizer,
                    seed=req.get("seed"), voice_codes=voice_codes,
                )
                dgen.reset()
            except (ValueError, json.JSONDecodeError) as e:
                await ws.send_json({"type": "error", "message": str(e)})
                await ws.close()
                return
            await ws.send_json({"type": "status", "message": "duplex session live"})
            dec = ChunkDecoder(codec, crossfade_ms=0.0 if type(codec).__name__ == "FakeCodec" else 4.0)
            # Rolling-context mic encoder: per-chunk stateless encode diverges
            # from training-time whole-utterance codes (review finding).
            enc = StreamingEncoder(codec)
            ticks = 0
            tick_ms_sum = 0.0
            session_t0: float | None = None
            behind_warned = False
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("text"):
                        if json.loads(msg["text"]).get("type") == "stop":
                            break
                        continue
                    raw = msg.get("bytes") or b""
                    if not raw:
                        continue
                    if len(raw) % 2:
                        # torch.frombuffer would raise and kill the session
                        await ws.send_json(
                            {"type": "error",
                             "message": "binary frames must be int16 PCM (even byte count)"}
                        )
                        continue
                    pcm = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32767.0
                    if pcm.numel() < spf:
                        pcm = torch.nn.functional.pad(pcm, (0, spf - pcm.numel()))
                    if session_t0 is None:
                        session_t0 = time.perf_counter()
                    t0 = time.perf_counter()
                    # encode + model tick together off the event loop
                    chunk = pcm[:spf]
                    step = await asyncio.to_thread(lambda c=chunk: dgen.step(enc.feed(c)))
                    dt = (time.perf_counter() - t0) * 1000.0
                    ticks += 1
                    tick_ms_sum += dt
                    # same-clock health: how far compute lags the mic clock
                    behind_ms = max(
                        0.0,
                        (time.perf_counter() - session_t0) * 1000.0 - ticks * frame_ms,
                    )
                    if behind_ms > 2000.0 and not behind_warned:
                        behind_warned = True
                        await ws.send_json(
                            {"type": "error",
                             "message": f"falling behind real time by {behind_ms/1000:.1f}s "
                                        "(compute slower than the 80 ms frame clock)"}
                        )
                    piece, special = piece_of(step.text_id)
                    await ws.send_json(
                        {"type": "text", "id": step.text_id, "piece": piece, "special": special}
                    )
                    if step.assistant_frame is not None:
                        out = dec.feed(step.assistant_frame)
                        if out.numel():
                            await ws.send_bytes(pcm16_bytes(out))
                    if ticks % 12 == 0:
                        avg = tick_ms_sum / ticks
                        await ws.send_json(
                            {"type": "metrics", "frames": ticks,
                             "ms_per_frame": round(avg, 2), "last_ms": round(dt, 2),
                             "behind_ms": round(behind_ms, 1),
                             "rtf": round(frame_ms / max(avg, 1e-6), 2)}
                        )
            except WebSocketDisconnect:
                pass
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass
        finally:
            busy["v"] = False

    return app
