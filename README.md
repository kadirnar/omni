# omni

A speech-to-speech LLM trained from scratch. It listens (audio in), thinks (text
inner monologue), and speaks (audio out) — emotion-aware and multilingual.

- Reference-audio voice cloning: one speaker voice, held across every language
  (`--voice ref.wav`; one-time ~6 ms prefill, per-frame decode unchanged).
- One decoder-only transformer over parallel token streams at 12.5 Hz:
  **1 text stream + N audio streams** ([kyutai/mimi](https://huggingface.co/kyutai/mimi)
  codec tokens). The only pretrained part is the frozen codec.
- ASR, TTS, speech continuation, text LM, spoken dialogue (s2s), and full-duplex
  are all the **same grid format** with different columns masked.
- Runs fully offline on CPU with `FakeCodec` for development; trains with
  DDP/FSDP2 on a GPU node.

Design docs: [architecture](docs/DESIGN.md) ·
[audio quality (32-codebook + depth)](docs/DESIGN_V3_AUDIO.md) ·
[emotion + multilingual](docs/DESIGN_V4_EMOTION_I18N.md) ·
[voice cloning](docs/DESIGN_V5_VOICE.md) ·
[module APIs](docs/INTERFACES.md)

## Install

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/pytest          # 138 tests, CPU, no downloads
```

## Data format

**1. What you feed in** — dialogues as plain dicts (JSONL-friendly). Emotion and
language labels are optional per turn:

```json
{"turns": [
  {"user": "why is my order late again",
   "assistant": "I am sorry about that, let me check it right away",
   "user_emotion": "angry", "response_style": "calm", "lang": "en"}
]}
```

**2. What training reads** — binary shard dirs. Each sample is one token grid:

```
row 0        : text    <bos> <s2s> <user> ..user speech.. <end_of_turn>
                       <assistant> <lang_en> <emo_pcv> <angry> <emo_rsp> <calm> w1 w2 ...
rows 1..N    : audio   Mimi codebook tokens, one column = one 80 ms frame
loss_mask    : which positions train (user speech is input-only)
channel      : who is speaking (user / assistant)
```

On disk: `shard-00000.bin` (grid `uint16 [S,T]` + mask `uint8 [S,T]` + channel
`uint8 [T]` per sample), `shard-00000.idx.jsonl` (byte offsets), `meta.json`
(`n_codebooks`, `codec_vocab`, `text_vocab_size`, `duplex`). Grids are stored
undelayed; the codebook delay pattern is applied at batch time, so changing it
never invalidates data.

## Prepare data

```bash
# offline demo shards (no network, FakeCodec + SineTTS): all tasks incl. s2s
.venv/bin/python scripts/prepare_data.py fake --n 256 --out data/shards/fake \
    --preset tiny model.n_codebooks=2 model.text_vocab_size=320

# tokenizer (once, for real runs): 48k multilingual byte-BPE
.venv/bin/python scripts/train_tokenizer.py --dataset HuggingFaceFW/fineweb-edu:sample-10BT \
    --vocab-size 48000 --out data/tokenizer/omni_bpe.json

# text pretraining rows (Stage A)
.venv/bin/python scripts/prepare_data.py textlm --dataset HuggingFaceFW/fineweb-edu \
    --name sample-10BT --max-samples 100000 --lang en \
    --tokenizer data/tokenizer/omni_bpe.json --out data/shards/textlm --preset quality

# ASR/TTS/audio-LM from a speech corpus (Stage B); --emotion-column makes SER-tagged ASR
.venv/bin/python scripts/prepare_data.py asr --dataset openslr/librispeech_asr --name clean \
    --split train.100 --max-samples 20000 --codec mimi --lang en \
    --tokenizer data/tokenizer/omni_bpe.json --out data/shards/speech --preset quality

# spoken dialogues (Stage C): text dialogues -> TTS -> Mimi tokens
# emotion labels ride the dialogue dicts (--dialogues soda-emotional maps SODA's
# emotion field); assistant audio is voiced with the labeled style
.venv/bin/python scripts/prepare_data.py s2s --dialogues soda-emotional --tts vibevoice \
    --codec mimi --max-samples 50000 --tokenizer data/tokenizer/omni_bpe.json \
    --out data/shards/s2s --preset quality

# full-duplex conversations (optional)
.venv/bin/python scripts/prepare_data.py duplex --n 200 --out data/shards/duplex \
    --preset tiny model.duplex=true model.text_vocab_size=320
```

Or from Python:

```python
from omni.audio.codec import FakeCodec
from omni.config import load_config
from omni.data.prepare import prepare_s2s
from omni.data.synthesize import SineTTS, fake_dialogues
from omni.text.tokenizer import ByteTokenizer

cfg = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320",
                           "data.batch_size=2"])
dialogues = [{"turns": [{"user": "hello there", "assistant": "hi, how can I help",
                         "user_emotion": "happy", "response_style": "calm",
                         "lang": "en"}]}]
dialogues += list(fake_dialogues(15, seed=0))
prepare_s2s("data/shards/demo", dialogues=dialogues, tts=SineTTS(),
            codec=FakeCodec(n_codebooks=2), tokenizer=ByteTokenizer(),
            cfg=cfg, max_samples=16)
```

## Train

```bash
# single process (CPU or 1 GPU)
.venv/bin/python scripts/train.py --preset tiny --data data/shards/fake \
    model.n_codebooks=2 model.text_vocab_size=320 data.num_workers=0 \
    train.max_steps=200 --export checkpoints/tiny

# 8 GPUs: same script; <300M params uses DDP, larger uses FSDP2 automatically.
# --data DIR:WEIGHT mixes shard dirs; checkpoints resume automatically.
torchrun --standalone --nproc_per_node=8 scripts/train.py --preset quality \
    --data data/shards/s2s:0.6 --data data/shards/speech:0.25 --data data/shards/textlm:0.15 \
    --export checkpoints/quality
```

Or from Python:

```python
from omni.config import load_config
from omni.data.dataset import build_dataloader
from omni.model.omni import OmniModel
from omni.train.loop import Trainer

cfg = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320",
                           "data.batch_size=2", "data.num_workers=0",
                           "train.max_steps=20", "train.ckpt_dir=checkpoints/demo"])
model = OmniModel(cfg.model)
model.init_weights()
trainer = Trainer(cfg, model, build_dataloader(cfg, ["data/shards/demo"]))
metrics = trainer.fit()            # logs loss per head (text + each codebook)
trainer.export_model("checkpoints/demo-export")
```

W&B logging: add `train.wandb=true` (plus optional `train.wandb_project=...`
`train.wandb_run_name=...` `train.wandb_mode=offline`) to any train command —
per-head losses, lr, grad-norm, and throughput stream to the run, and resuming
a checkpoint resumes the *same* wandb run (`pip install 'omni[wandb]'`).

Presets (`--preset`): `tiny` ~22M CPU tests · `small` ~0.3B 1-GPU ·
`quality` ~1.0B 8-GPU from-scratch · `base` 0.76B legacy ·
**`qwen3-1.7b` / `qwen3-8b` / `llama32-3b` / `gemma3-4b`** — pretrained-LLM
backbones (the v6 production path). Stage recipe and mixtures:
[docs/DESIGN.md](docs/DESIGN.md) §4 (from scratch) /
[docs/DESIGN_V6](docs/DESIGN_V6_PRETRAINED_BACKBONE.md) §5 (backbone).

## Pretrained backbone (v6): skip text pretraining

Instead of pretraining a text LM, mount a pretrained decoder (Qwen3 / Llama 3 /
Gemma) as the temporal backbone: its tokenizer rides the text stream shifted by
+64 (`--tokenizer hf:<model_id>`, omni specials keep ids 0..63), new audio
embeddings + the depth transformer are grafted on, and training becomes two
cheap stages — Stage 1 aligns the new audio modules with the backbone FROZEN
(`model.freeze_backbone=true`, the default), Stage 2 unfreezes at a low LR
(`train.backbone_lr`) or uses LoRA (`model.lora_rank=32`, needs
`pip install 'omni[lora]'`). Exports write `adapters.safetensors` only — the
backbone is referenced by id, not copied.

```bash
# prepare with the backbone tokenizer (downloads tokenizer + model on first use)
.venv/bin/python scripts/prepare_data.py asr --dataset openslr/librispeech_asr \
    --name clean --split train.100 --codec mimi --lang en \
    --tokenizer hf:Qwen/Qwen3-1.7B-Base --out data/shards/speech --preset qwen3-1.7b

# Stage 1: frozen backbone, align audio modules (single GPU is fine at 1.7B)
.venv/bin/python scripts/train.py --preset qwen3-1.7b --data data/shards/speech \
    --export checkpoints/qwen3-s1

# Stage 2: unfreeze at low LR on the s2s + emotion mixture (8 GPUs)
torchrun --standalone --nproc_per_node=8 scripts/train.py --preset qwen3-8b \
    model.freeze_backbone=false --data data/shards/s2s:0.7 --data data/shards/speech:0.3 \
    --export checkpoints/qwen3-s2
```

Every other command (chat, serve, benchmark) takes the exported dir via
`--ckpt` unchanged. Design + rationale: [docs/DESIGN_V6](docs/DESIGN_V6_PRETRAINED_BACKBONE.md).

## Inference

```bash
# speak text / transcribe / spoken reply (use --codec mimi with real checkpoints)
.venv/bin/python scripts/chat.py --task tts --text "hello" --out hello.wav \
    --ckpt checkpoints/demo-export --codec fake --tokenizer byte
.venv/bin/python scripts/chat.py --task asr --in hello.wav \
    --ckpt checkpoints/demo-export --codec fake --tokenizer byte
.venv/bin/python scripts/chat.py --task s2s --in question.wav --out reply.wav \
    --ckpt checkpoints/demo-export --codec fake --tokenizer byte \
    --emotion calm --lang en \
    --voice me.wav                    # optional: 10s reference wav pins the speaker
                                      # voice — held across ALL languages
.venv/bin/python scripts/chat.py --task duplex --in user.wav --out assistant.wav \
    --ckpt checkpoints/duplex-export --codec fake --tokenizer byte
```

Or from Python:

```python
import torch
from omni.audio.codec import FakeCodec, save_wav   # build_codec("mimi") for real runs
from omni.config import load_config
from omni.infer.generate import OmniGenerator
from omni.model.omni import OmniModel
from omni.streams import turn_prefix
from omni.text.tokenizer import ByteTokenizer

model = OmniModel.from_pretrained("checkpoints/demo-export")
cfg = load_config("tiny")
cfg.model = model.cfg                              # keep checks coherent with weights
codec = FakeCodec(n_codebooks=model.cfg.n_codebooks)
gen = OmniGenerator(model, cfg, device="cpu", tokenizer=ByteTokenizer())

gen.set_voice(torch.randn(codec.sample_rate * 5).clamp(-1, 1), codec)  # reference wav
wav = torch.sin(torch.linspace(0, 500, codec.samples_per_frame * 10))  # your mic audio
reply = gen.s2s(wav, codec, seed=0,
                prefix_ids=turn_prefix(lang="en", response_style="calm"))
print(reply.text)                                  # inner monologue (specials stripped)
save_wav("reply.wav", codec.decode(reply.audio_codes), codec.sample_rate)
```

The model's monologue starts with its own read of your tone
(`<emo_pcv> <angry>`) and the register it chose (`<emo_rsp> <calm>`) before the
words — inspect `reply.text_ids`, or override the register with `prefix_ids`.

## Test console (streaming UI)

```bash
uv pip install -p .venv/bin/python -e ".[serve]"
.venv/bin/python scripts/serve.py --ckpt checkpoints/quality-export --codec mimi \
    --tokenizer data/tokenizer/omni_bpe.json          # or: --preset tiny (wiring demo)
# -> open http://127.0.0.1:7860
```

Pick a task, talk or type, and the reply **streams**: audio plays as frames are
generated, the inner monologue appears live (control tags render as chips — you
watch the model pick a language and emotion before it speaks), and a per-frame
budget meter shows every frame's latency against the 80 ms real-time line,
with TTFA / ms-per-frame / realtime-× readouts. Reference-voice and duplex
(mic-to-model, same clock) modes included. Local tool — binds to 127.0.0.1,
no auth.

## Status

Code-complete and contract-tested on CPU (100 tests, offline). Needs a GPU box
for: the 48k tokenizer run, per-language codec Gate-0, and the Stage A→D
training campaign. Remaining engineering queue (depth compute amortization, CFG
decode, eval battery, synthesis QC, DPO): see the implementation queues in
[DESIGN_V3](docs/DESIGN_V3_AUDIO.md) / [DESIGN_V4](docs/DESIGN_V4_EMOTION_I18N.md).
