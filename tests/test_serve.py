"""Streaming test console: stream()/generate parity, ChunkDecoder, WS protocol."""

from __future__ import annotations

import base64
import copy
import io
import json

import pytest
import torch

from omni.audio.codec import FakeCodec
from omni.config import load_config
from omni.infer.generate import OmniGenerator, StreamEvent
from omni.model.omni import OmniModel
from omni.serve.streaming import ChunkDecoder, pcm16_bytes
from omni.text.tokenizer import ByteTokenizer


@pytest.fixture(scope="module")
def rig(test_cfg):
    torch.manual_seed(0)
    cfg = copy.deepcopy(test_cfg)
    model = OmniModel(cfg.model)
    model.init_weights()
    model.eval()
    gen = OmniGenerator(model, cfg, device="cpu", tokenizer=ByteTokenizer())
    codec = FakeCodec(
        n_codebooks=cfg.model.n_codebooks, codec_vocab=cfg.model.audio_codec_vocab
    )
    return cfg, model, gen, codec


def _wav(frames: int, codec) -> torch.Tensor:
    return torch.sin(torch.linspace(0, 70.0 * frames, codec.samples_per_frame * frames))


@pytest.mark.parametrize("task", ["tts", "s2s", "asr"])
def test_stream_generate_parity(rig, task) -> None:
    """stream() reassembled == generate() token-for-token (the binding contract)."""
    cfg, model, gen, codec = rig
    kw = dict(text="parity check") if task == "tts" else dict(wav=_wav(6, codec))
    prompt, forced = gen.task_prompt(task, codec=codec, voice_wav=_wav(10, codec), **kw)
    r_gen = gen.generate(prompt, forced, max_frames=16, seed=7)
    events = list(gen.stream(prompt, forced, max_frames=16, seed=7))
    assert events[-1].kind == "done"
    r_str = events[-1].result
    assert r_str.text_ids == r_gen.text_ids
    assert torch.equal(r_str.audio_codes, r_gen.audio_codes)
    assert [e.token for e in events if e.kind == "text"] == r_gen.text_ids
    frames = [e.frame for e in events if e.kind == "frame"]
    if task == "asr":
        assert frames == []
    else:
        assert len(frames) == r_gen.frames
        if frames:
            assert torch.equal(torch.stack(frames, dim=1), r_gen.audio_codes)


def test_stream_stop_aborts_early(rig) -> None:
    cfg, model, gen, codec = rig

    class AfterN:
        def __init__(self, n: int) -> None:
            self.n, self.seen = n, 0

        def is_set(self) -> bool:
            self.seen += 1
            return self.seen > self.n

    prompt, forced = gen.task_prompt("tts", text="stop me", codec=codec)
    events = list(gen.stream(prompt, forced, max_frames=40, seed=1, stop=AfterN(6)))
    done = events[-1]
    assert done.kind == "done"
    full = gen.generate(prompt, forced, max_frames=40, seed=1)
    assert done.result.frames < full.frames  # aborted early, still a valid result
    assert torch.equal(
        done.result.audio_codes, full.audio_codes[:, : done.result.frames]
    )  # prefix of the uninterrupted run (same seed, same sample stream)


def test_chunk_decoder_exact_on_fake(rig) -> None:
    """FakeCodec decodes frames independently: streamed == one-shot (xf=0)."""
    _, _, _, codec = rig
    codes = torch.randint(0, codec.codec_vocab, (codec.n_codebooks, 23))
    dec = ChunkDecoder(codec, crossfade_ms=0.0)
    parts = [dec.feed(codes[:, i]) for i in range(codes.shape[1])]
    parts.append(dec.flush())
    streamed = torch.cat([p for p in parts if p.numel()])
    full = codec.decode(codes).reshape(-1)
    assert streamed.shape == full.shape
    assert torch.allclose(streamed, full, atol=1e-6)


def test_chunk_decoder_crossfade_sample_count(rig) -> None:
    _, _, _, codec = rig
    codes = torch.randint(0, codec.codec_vocab, (codec.n_codebooks, 9))
    dec = ChunkDecoder(codec, crossfade_ms=4.0)
    total = sum(dec.feed(codes[:, i]).numel() for i in range(9)) + dec.flush().numel()
    assert total == 9 * codec.samples_per_frame  # holdback conserves samples


def test_pcm16_roundtrip() -> None:
    x = torch.linspace(-1, 1, 512)
    b = pcm16_bytes(x)
    assert len(b) == 1024
    back = torch.frombuffer(bytearray(b), dtype=torch.int16).float() / 32767.0
    assert torch.allclose(back, x, atol=1e-3)


# ---------------------------------------------------------------------------
# WebSocket protocol (in-process, no network)
# ---------------------------------------------------------------------------
def _wav_b64(wav: torch.Tensor, sr: int) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, wav.numpy(), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def client(rig):
    fastapi = pytest.importorskip("fastapi")  # serve extra
    from fastapi.testclient import TestClient

    from omni.serve.app import create_app

    cfg, model, gen, codec = rig
    app = create_app(cfg=cfg, model=model, codec=codec, tokenizer=ByteTokenizer())
    return TestClient(app)


def test_api_info(client, test_cfg) -> None:
    info = client.get("/api/info").json()
    assert info["tasks"] == ["tts", "s2s", "asr"]
    assert info["sample_rate"] == 24_000 and info["frame_ms"] == 80.0
    assert "angry" in info["emotions"] and "tr" in info["langs"]
    page = client.get("/")
    assert page.status_code == 200 and b"TEST CONSOLE" in page.content


def _drain(ws) -> tuple[list[dict], list[bytes]]:
    msgs, blobs = [], []
    while True:
        raw = ws.receive()
        if raw.get("type") == "websocket.close":
            break
        if raw.get("bytes") is not None:
            blobs.append(raw["bytes"])
            continue
        m = json.loads(raw["text"])
        msgs.append(m)
        if m["type"] in ("done", "error"):
            break
    return msgs, blobs


def test_ws_generate_tts_streams(client, rig) -> None:
    _, _, _, codec = rig
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({
            "task": "tts", "text": "stream me", "seed": 3, "max_frames": 12,
            "emotion": "calm", "lang": "en",
            "voice_wav_b64": _wav_b64(_wav(8, codec), codec.sample_rate),
        }))
        msgs, blobs = _drain(ws)
    kinds = [m["type"] for m in msgs]
    assert kinds[0] == "status" and "done" in kinds
    done = msgs[-1]
    assert done["frames"] == 12 and done["rtf"] is not None and done["ttfa_ms"] is not None
    texts = [m for m in msgs if m["type"] == "text"]
    assert [t["id"] for t in texts] == done["text_ids"]
    assert any(t["special"] == "calm" for t in texts)  # forced tag visible as a chip
    assert blobs, "no audio chunks streamed"
    n_samples = sum(len(b) for b in blobs) // 2
    assert n_samples == done["frames"] * codec.samples_per_frame


def test_ws_generate_stop(client) -> None:
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"task": "tts", "text": "long one", "seed": 0,
                                 "max_frames": 100}))
        ws.send_text(json.dumps({"type": "stop"}))
        msgs, _ = _drain(ws)
    done = msgs[-1]
    assert done["type"] == "done" and done["stopped"] is True
    assert done["frames"] < 100


def test_ws_generate_bad_request(client) -> None:
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"task": "s2s"}))  # missing audio
        msgs, _ = _drain(ws)
    assert msgs[-1]["type"] == "error"


def test_ws_duplex_session(test_cfg) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from omni.serve.app import create_app

    cfg = copy.deepcopy(test_cfg)
    cfg.model = __import__("dataclasses").replace(cfg.model, duplex=True)
    torch.manual_seed(1)
    model = OmniModel(cfg.model)
    model.init_weights()
    model.eval()
    codec = FakeCodec(n_codebooks=cfg.model.n_codebooks,
                      codec_vocab=cfg.model.audio_codec_vocab)
    client = TestClient(create_app(cfg=cfg, model=model, codec=codec,
                                   tokenizer=ByteTokenizer()))
    info = client.get("/api/info").json()
    assert info["tasks"] == ["duplex"]
    spf = codec.samples_per_frame
    with client.websocket_connect("/ws/duplex") as ws:
        ws.send_text(json.dumps({"seed": 5}))
        first = json.loads(ws.receive()["text"])
        assert first["type"] == "status"
        got_text = 0
        got_audio = 0
        for i in range(8):
            frame = (torch.sin(torch.linspace(0, 50 + i, spf)) * 20000).to(torch.int16)
            ws.send_bytes(frame.numpy().tobytes())
            # each tick answers with at least a text event (audio once pipeline fills)
            while True:
                raw = ws.receive()
                if raw.get("bytes") is not None:
                    got_audio += 1
                    continue
                m = json.loads(raw["text"])
                if m["type"] == "text":
                    got_text += 1
                    break
                if m["type"] == "error":
                    raise AssertionError(m["message"])
        ws.send_text(json.dumps({"type": "stop"}))
    assert got_text == 8
    assert got_audio > 0, "assistant audio never streamed back"


def test_streaming_encoder_matches_batch_encode() -> None:
    """Rolling-context mic encoding must agree with whole-utterance encoding
    (exactly true for FakeCodec's per-frame hashing; for Mimi the window is
    the approximation of its streaming encoder — see test_gpu_preflight)."""
    from omni.audio.codec import FakeCodec
    from omni.serve.streaming import StreamingEncoder

    codec = FakeCodec(n_codebooks=2)
    spf = codec.samples_per_frame
    wav = torch.sin(torch.linspace(0, 700, spf * 6))
    want = codec.encode(wav)
    enc = StreamingEncoder(codec)
    got = torch.stack(
        [enc.feed(wav[i * spf : (i + 1) * spf]) for i in range(6)], dim=1
    )
    assert torch.equal(got, want)
    enc.reset()
    assert torch.equal(enc.feed(wav[:spf]), want[:, 0])
