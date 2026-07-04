# Module Interfaces (binding contract for implementation)

Every module below is owned by exactly one implementer. Import *only* what other
modules pin here. `src/omni/config.py`, `src/omni/streams.py`, `src/omni/grids.py`,
and `pyproject.toml` are already implemented and FROZEN — read them first; do not
edit them (report concerns instead).

Global rules
- Python ≥3.11, torch ≥2.6 APIs (dev venv has torch 2.12.1, CPU/MPS only — no CUDA here).
- Type hints everywhere; docstrings state tensor shapes like `[B, S, T]`.
- `S = 1 + n_codebooks` streams; row 0 is text (`streams.TEXT_STREAM`). Frames = 12.5 Hz steps.
- Grids on disk and in `Sample` are UNDELAYED; the model/generator consume DELAYED grids
  (`streams.apply_delay`). Loss/targets: logits at position `p` predict grid position `p+1`;
  `loss_mask[s, p+1]` gates that target.
- Optional deps (`kokoro`, `torchao`, `wandb`) must be imported lazily inside the function
  that needs them, with a clear `ImportError` message. No new dependencies.
- No network at import time, ever. Nothing may download models/datasets unless the user
  invoked that path explicitly (`--codec mimi`, HF dataset prep, `--tts kokoro`).
- Determinism: accept explicit `seed` args; use local `torch.Generator` for sampling,
  `random.Random(seed)` for python-level shuffles.
- No einops; plain torch. Keep MPS/CPU-safe (guard bf16/fused paths behind device checks).

---

## src/omni/audio/codec.py  (agent A)

```python
class AudioCodec(abc.ABC):
    sample_rate: int          # Hz of encode input / decode output
    frame_rate: float         # frames per second (12.5 for Mimi)
    n_codebooks: int
    codec_vocab: int          # real codes per codebook (2048 for Mimi); specials live above

    @abc.abstractmethod
    def encode(self, wav: torch.Tensor) -> torch.Tensor: ...
        # wav float32 [T_samples] or [B, T_samples], values ~[-1, 1], at self.sample_rate
        # -> long codes [n_q, T_frames] (unbatched in) or [B, n_q, T_frames]
    @abc.abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...
        # long [n_q, T] or [B, n_q, T], RAW codes only (caller sanitizes specials)
        # -> float32 wav [T_samples] or [B, T_samples]
    def to(self, device: str | torch.device) -> "AudioCodec": ...

class MimiCodec(AudioCodec):
    def __init__(self, model_id: str = "kyutai/mimi", n_codebooks: int = 8,
                 device: str | torch.device = "cpu"): ...
    # transformers.MimiModel.from_pretrained(model_id); .eval(); no_grad; encode(...,
    # num_quantizers/num codebooks limited to n_codebooks — check the transformers API);
    # sample_rate 24000, frame_rate 12.5, codec_vocab 2048.

class FakeCodec(AudioCodec):
    def __init__(self, n_codebooks: int = 8, codec_vocab: int = 2048): ...
    # Offline test stand-in. sample_rate 24000, frame_rate 12.5 (1920 samples/frame).
    # encode: deterministic per-80ms-frame hash -> codes (same wav -> same codes).
    # decode: (codebook, code) -> fixed small sinusoid; sum, clamp to [-1, 1]; nonzero.
    # encode(decode(x)) need NOT roundtrip.

def build_codec(name: str, *, model_id: str = "kyutai/mimi", n_codebooks: int = 8,
                device: str | torch.device = "cpu") -> AudioCodec  # "mimi" | "fake"

def resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor
    # THE one resampler (2026-07 amendment): anti-aliased Kaiser-windowed sinc
    # (torchaudio "kaiser_best" formulation), plain torch, mono [T] -> [ceil].
    # Data prep, voice references, and serve all route through it — the two
    # divergent linear interpolators it replaces aliased >Nyquist content.
def load_wav(path: str | Path, target_sr: int) -> torch.Tensor   # mono float32 [T], soundfile + resample()
def save_wav(path: str | Path, wav: torch.Tensor, sr: int) -> None
```

## src/omni/text/tokenizer.py  (agent A)

```python
class TextTokenizer:
    # wraps tokenizers.Tokenizer (byte-level BPE). Ids 0..63 are EXACTLY
    # streams.SPECIAL_TOKENS plus streams.RESERVED_SPECIAL_FORMAT fillers; BPE ids start at 64.
    @property
    def vocab_size(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...        # never auto-adds specials
    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "TextTokenizer": ...

class ByteTokenizer:
    # Same duck-typed API, zero artifacts: byte b -> id 64 + b; vocab_size = 320.
    # decode: bytes(ids-64), errors="replace", specials skipped.

def train_bpe(texts: Iterable[str], vocab_size: int = 32_768,
              out_path: str | Path | None = None) -> TextTokenizer
def build_tokenizer(path: str | None) -> TextTokenizer | ByteTokenizer
    # None or "byte" -> ByteTokenizer(); else TextTokenizer.load(path)
```

`scripts/train_tokenizer.py`: `--dataset HF_ID[:config[:split]] | --text-file PATH...`,
`--field text`, `--max-docs N`, `--vocab-size 32768`, `--out data/tokenizer/omni_bpe.json`.

NOTE: `model.text_vocab_size` must equal the tokenizer's `vocab_size` when preparing data
(ByteTokenizer runs use override `model.text_vocab_size=320`).

---

## src/omni/data/synthesize.py  (agent B)

```python
class TTSBackend(abc.ABC):
    voices: list[str]
    @abc.abstractmethod
    def synth(self, text: str, voice: str) -> tuple[torch.Tensor, int]  # (wav float32 [T], sr)

class SineTTS(TTSBackend):   # no-dep deterministic pseudo-speech from text hash (~0.08 s/word)
class KokoroTTS(TTSBackend): # lazy `import kokoro`; helpful ImportError otherwise
def build_tts(name: str) -> TTSBackend  # "sine" | "kokoro"

def fake_dialogues(n: int, seed: int = 0) -> Iterator[dict]
    # deterministic toy dialogues: {"turns": [{"user": str, "assistant": str}, ...]} (1-3 turns)
def load_text_dialogues(dataset_id: str, split: str, max_dialogues: int) -> Iterator[dict]
    # supports at least allenai/soda and HuggingFaceH4/ultrachat_200k -> same dict shape;
    # filter unspeakable turns (>60 words, code/URLs/markdown)
```

## src/omni/data/prepare.py  (agent B)

Shard format (binding, versions 1 and 2 — v2 added 2026-07 for backbone
tokenizers whose vocab exceeds uint16): a shard dir holds `meta.json` +
`shard-%05d.bin` + `shard-%05d.idx.jsonl`.
- `meta.json`: `{"version": 1|2, "n_codebooks": int, "codec_vocab": int,
  "text_vocab_size": int, "n_samples": int, "n_shards": int}` (+ optional
  `"duplex": bool`, `"tokenizer_id": str` provenance)
- version 1 (text_vocab_size <= 65536), per sample, contiguous bytes in the
  `.bin`: grid `uint16 [S, T]` C-order, then loss_mask `uint8 [S, T]`, then
  channel `uint8 [T]` (little-endian).
- version 2 (any text vocab; writers pick it automatically when needed):
  text row `uint32 [T]`, audio rows `uint16 [S-1, T]`, then loss_mask
  `uint8 [S, T]`, then channel `uint8 [T]`. Readers support both.
- idx line: `{"offset": int, "frames": int, "task": str}` (offset in bytes into the .bin).

```python
class ShardWriter:
    def __init__(self, out_dir: str | Path, *, n_codebooks: int, codec_vocab: int,
                 text_vocab_size: int, shard_mb: int = 512): ...
    def add(self, sample: Sample) -> None    # validates, converts, rolls shards
    def close(self) -> None                  # writes meta.json

def prepare_fake(out_dir, *, n_samples: int, cfg: OmniConfig, seed: int = 0) -> None
    # SineTTS + FakeCodec + ByteTokenizer; mixture over all tasks
    # (textlm/audiolm/asr/tts/s2s); respects cfg.data.max_sample_frames (truncate).
def prepare_textlm(out_dir, *, dataset_id: str, name: str | None, split: str,
                   tokenizer, cfg, max_samples: int, streaming: bool = True) -> None
    # packs token runs into full-length rows (concat docs, chunk to max_sample_frames)
def prepare_asr_tts(out_dir, *, dataset_id: str, name: str | None, split: str,
                    codec: AudioCodec, tokenizer, cfg, max_samples: int,
                    tasks: tuple[str, ...] = ("asr", "tts", "alm")) -> None
    # librispeech-style rows (audio + transcript) -> build_asr/build_tts/build_audiolm,
    # cycling tasks; resample to codec.sample_rate; skip > max frames
def prepare_s2s(out_dir, *, dialogues: Iterable[dict], tts: TTSBackend,
                codec: AudioCodec, tokenizer, cfg, max_samples: int, seed: int = 0) -> None
    # random user voice per dialogue, fixed small assistant-voice pool, build_s2s
```

## src/omni/data/dataset.py  (agent B)

```python
class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, shard_dir: str | Path): ...   # reads meta + all idx lines eagerly
    frames: list[int]                                # undelayed T per sample (bucketing)
    meta: dict
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Sample: ...     # np.memmap opened lazily PER WORKER

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[ShardDataset], weights: list[float] | None = None,
                 seed: int = 0): ...
    # 2026-07 amendment: weights are SAMPLING PROPORTIONS honored by
    # RESAMPLING (repeat/subsample counts — survives any downstream shuffle);
    # weights=None = natural mixing (every sample once). len = sum of lens in
    # both forms; exposes .frames and .counts. The old order-only interleave
    # was a silent no-op under BucketBatchSampler's shuffle.

class BucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], batch_size: int, *, shuffle: bool = True,
                 seed: int = 0, rank: int = 0, world_size: int = 1, drop_last: bool = True): ...
    def set_epoch(self, epoch: int) -> None: ...
    # sort into length buckets -> batches of similar length -> shuffle batches;
    # rank r takes batches r, r+W, r+2W... (equal count per rank, drop remainder)

def collate(samples: list[Sample], codec_vocab: int) -> dict[str, torch.Tensor]
    # apply_delay each -> right-pad to batch max T' (text PAD / audio AUDIO_PAD,
    # mask False, channel repeats last) -> {"grid": long [B,S,T'],
    # "loss_mask": bool [B,S,T'], "channel": long [B,T']}

def build_dataloader(cfg: OmniConfig, shard_dirs: dict[str, float] | list[str], *,
                     rank: int = 0, world_size: int = 1, epoch: int = 0,
                     shuffle: bool = True) -> torch.utils.data.DataLoader
    # asserts every shard meta matches cfg.model (n_codebooks, codec_vocab, text_vocab_size);
    # infinite iteration is the Trainer's job (it may cycle epochs and call set_epoch)
```

`scripts/prepare_data.py` subcommands (all take `--preset/--config`, `--out`, overrides):
- `fake --n 200`
- `textlm --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --split train --tokenizer PATH --max-samples N`
- `asr --dataset openslr/librispeech_asr --name clean --split train.100 --codec mimi|fake --tokenizer ... --max-samples N`
- `s2s --dialogues soda|ultrachat|fake --tts sine|kokoro --codec mimi|fake --max-samples N`
`scripts/synthesize_data.py`: standalone dialogue→wav→shards pipeline (thin wrapper over
`prepare_s2s`, plus `--dump-wav DIR` to inspect audio).

---

## src/omni/model/layers.py  (agent C)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5): ...   # fp32 internal, cast back

def precompute_rope(head_dim: int, max_pos: int, theta: float,
                    device=None) -> torch.Tensor    # float32 [max_pos, head_dim//2, 2] cos/sin
def apply_rope(x: torch.Tensor, rope: torch.Tensor, pos: int = 0) -> torch.Tensor
    # x [B, H, T, head_dim]; uses rope[pos : pos+T]; fp32 math, cast back to x.dtype

class LayerKVCache:
    k: torch.Tensor; v: torch.Tensor   # [B, n_kv_heads, max_frames, head_dim]
    pos: int                           # filled length
    def append(self, k_new, v_new) -> tuple[torch.Tensor, torch.Tensor]  # views [:, :, :pos+T]

class KVCache:
    layers: list[LayerKVCache]; pos: int
    @classmethod
    def allocate(cls, cfg: ModelConfig, batch: int, device, dtype) -> "KVCache"

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig): ...   # GQA; no biases
    def forward(self, x: torch.Tensor, rope: torch.Tensor,
                cache: LayerKVCache | None = None, pos: int = 0) -> torch.Tensor
    # training (cache None): F.scaled_dot_product_attention(is_causal=True)
    # decode (cache set): append then attend over cache[:pos+T]; for T==1 no mask needed;
    # for prefill T>1 with empty cache use is_causal=True

class MLP(nn.Module):     # SwiGLU: w1,w3: d->d_ff, w2: d_ff->d, no biases
class Block(nn.Module):
    def forward(self, x, rope, cache=None, pos: int = 0) -> torch.Tensor  # pre-norm
```

## src/omni/model/omni.py  (agent C)

```python
@dataclass
class ModelOutput:
    text_logits: torch.Tensor    # float32 [B, T, text_vocab_size]
    audio_logits: torch.Tensor   # float32 [B, n_q, T|P, audio_vocab_size]
    audio_positions: torch.Tensor | None = None
    # 2026-07 amendment (CSM depth-loss amortization): training-mode depth
    # forwards with cfg.depth_loss_ratio < 1 compute audio logits on a random
    # position subset [P] (sorted; position p scores grid column p+1);
    # audio_positions carries it and loss() gathers accordingly. None (eval,
    # ratio 1.0, non-depth models) keeps the full-T v1 shape.

class OmniModel(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    # text_emb: Embedding(text_vocab_size, d); audio_embs: ModuleList of n_q
    # Embedding(audio_vocab_size, d); channel_emb: Embedding(2, d); blocks; norm;
    # text_head Linear(d, text_vocab, bias=False) tied to text_emb if cfg.tie_text_head;
    # audio_heads: ModuleList of n_q Linear(d, audio_vocab_size, bias=False).
    def embed(self, grid: torch.Tensor, channel: torch.Tensor) -> torch.Tensor
        # [B, S, T] + [B, T] -> [B, T, d]: text_emb + sum audio_embs + channel_emb
    def forward(self, grid: torch.Tensor, channel: torch.Tensor) -> ModelOutput
        # full-sequence training path; grad checkpoint per block if cfg.grad_checkpoint
        # and self.training (torch.utils.checkpoint, use_reentrant=False)
    def loss(self, out: ModelOutput, grid, loss_mask,
             weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
             ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
        # weights = (text_w, audio_w, semantic_extra_w_on_cb0).
        # next-token: logits[:, :p-1] vs grid[:, :, 1:]; target mask loss_mask[:, :, 1:].
        # per-head mean CE over that head's masked targets; 2026-07 amendment:
        # total divides by the CONSTANT configured weight sum
        # text_w + audio_w*(semantic_w + n_q - 1) — heads without targets
        # contribute exactly 0, so per-head gradient scale no longer swings
        # with batch task composition. Implemented via ignore_index CE (no
        # per-head host syncs); loss carries its own zero anchor so every
        # head always receives a grad (DDP requirement).
        # metrics: {"loss": total, "loss/text": .., "loss/audio_0": .. } (detached;
        # per-head keys still appear only when that head had targets)
    def prefill(self, grid, channel, cache: KVCache) -> tuple[torch.Tensor, torch.Tensor]
        # [B, S, T] delayed prompt -> logits for LAST position: (text [B, Vt], audio [B, n_q, Va])
    def step(self, tokens: torch.Tensor, channel: torch.Tensor, cache: KVCache
             ) -> tuple[torch.Tensor, torch.Tensor]
        # one decode step: tokens [B, S] (input column at position cache.pos), channel [B]
    def init_weights(self) -> None    # normal(0, init_std); out-projections (attn out, mlp w2)
                                      # scaled by 1/sqrt(2*n_layers); embeddings std 0.02
    def param_counts(self) -> dict[str, int]   # {"total": .., "non_embedding": ..}
    def save_pretrained(self, save_dir: str | Path) -> None    # model.safetensors + config.yaml (ModelConfig fields)
    @classmethod
    def from_pretrained(cls, save_dir: str | Path, map_location="cpu") -> "OmniModel"
```

Numerics: heads computed in model dtype then cast `.float()` for logits. RoPE positions are
DELAYED-grid absolute positions (prefill offset 0; step at pos=cache.pos).

---

## src/omni/train/distributed.py  (agent D)

```python
@dataclass
class DistContext:
    rank: int; local_rank: int; world_size: int
    device: torch.device; is_main: bool; initialized: bool

def setup_distributed(device_preference: str = "auto") -> DistContext
    # if RANK in env: init_process_group(nccl if cuda else gloo), cuda.set_device(local_rank)
    # else single process; device auto: cuda > mps > cpu (mps only when explicitly asked)
def cleanup(ctx) -> None
def pick_strategy(cfg: OmniConfig, ctx, trainable_params: int) -> str  # "none"|"ddp"|"fsdp2"
    # world_size==1 -> none; strategy fixed if not auto; auto: fsdp2 iff TRAINABLE
    # params >= cfg.train.fsdp_threshold_params and cuda, else ddp (2026-07: frozen
    # v6 backbones with small adapters stay on ddp; full finetunes shard)
def wrap_model(model: torch.nn.Module, cfg: OmniConfig, ctx) -> torch.nn.Module
    # fsdp2 (cuda only): per-layer fully_shard(mp_policy=MixedPrecisionPolicy(bf16, fp32))
    # over model.blocks OR the HF backbone's decoder .layers (2026-07: HFOmniModel
    # support), then root fully_shard; ddp: DDP(model) after .to(device); none: .to(device).
    # Apply BEFORE optimizer creation. Order with grad-checkpoint: AC is inside the model.
def grad_sync_ctx(wrapped, strategy: str, is_last_microbatch: bool)  # contextmanager
def autocast_ctx(cfg, ctx, strategy: str)
    # cuda + (ddp|none) + precision bf16 -> torch.autocast("cuda", bfloat16); else nullcontext
    # (fsdp2 handles dtype via MixedPrecisionPolicy; cpu/mps stay fp32)
def is_distributed_env() -> bool
```

## src/omni/train/loop.py  (agent D)

```python
def build_lr_lambda(tc: TrainConfig) -> Callable[[int], float]
    # warmup 0->1 over warmup_steps; cosine: decay to min_lr_ratio at max_steps;
    # wsd: constant 1.0 until 0.8*max_steps then linear to min_lr_ratio; constant: 1.0

class Trainer:
    def __init__(self, cfg: OmniConfig, model: OmniModel,
                 train_loader: DataLoader, val_loader: DataLoader | None = None,
                 ctx: DistContext | None = None): ...
    def fit(self) -> dict[str, float]
        # cycles the loader (set_epoch on its batch_sampler when present); accum_steps
        # microbatches per optimizer step; loss/accum_steps backward under grad_sync_ctx;
        # clip_grad_norm_(grad_clip); AdamW(fused=cuda); LambdaLR(build_lr_lambda);
        # logs every log_every (loss, per-head, lr, grad_norm, steps/s, frames/s) on rank0
        # (wandb lazy iff cfg.train.wandb); eval every eval_every (mean val loss over
        # eval_steps batches, model.eval + no_grad); save_checkpoint every save_every and
        # at the end; returns final metrics dict.
    def save_checkpoint(self, step: int) -> None
        # <ckpt_dir>/step_%08d/ ; world_size>1: DCP dcp.save with
        # get_state_dict(model, opt) (+ StateDictOptions defaults) storing
        # {"model", "optim"} plus a rank0 extra.pt {step, sched, rng(all ranks? rank0 ok), cfg};
        # world_size==1: torch.save single file trainer_state.pt.
        # Always: rank0 "latest" pointer file <ckpt_dir>/latest.txt with the dir name.
        # 2026-07 amendments: FROZEN parameters (v6 backbone) are excluded from the
        # saved model state (lean checkpoints; resume tolerates their absence), and
        # rank0 prunes to the newest cfg.train.save_keep step dirs (0 = keep all).
    def maybe_resume(self) -> int   # if cfg.train.resume and latest.txt exists: restore
                                    # model/opt/sched/step (DCP load when distributed).
                                    # 2026-07: refuses structurally-incompatible
                                    # checkpoints (model shape / freeze / lora config
                                    # diff) with a clear error instead of a raw
                                    # state_dict crash; stage transitions go through
                                    # scripts/train.py --init-from (omni.model.load_weights:
                                    # exported weights into a freshly built model,
                                    # fresh optimizer/schedule).
    def export_model(self, out_dir: str | Path) -> None
        # consolidated weights: get_model_state_dict(..., StateDictOptions(
        # full_state_dict=True, cpu_offload=True)) when sharded; rank0
        # model.save_pretrained-equivalent (safetensors + config.yaml)
```

`scripts/train.py`:
```
python scripts/train.py --preset tiny --data data/shards/fake --max-steps? (via overrides)
torchrun --standalone --nproc_per_node=8 scripts/train.py --preset base \
    --data data/shards/stageB:0.7 --data data/shards/textlm:0.3 [k=v ...]
```
`--data DIR[:WEIGHT]` repeatable; `--val-data DIR` optional; positional `key=value`
overrides go to `load_config`; `--export DIR` exports consolidated weights at the end.
Prints param counts and effective config on rank0. Exits nonzero on failure.

---

## src/omni/infer/generate.py  (agent E)

```python
def sample_logits(logits: torch.Tensor, temperature: float, top_k: int,
                  gen: torch.Generator | None = None) -> torch.Tensor
    # [.., V] -> long [..]; temperature <= 0 means argmax

@dataclass
class GenResult:
    text_ids: list[int]           # sampled text stream (specials filtered for .text)
    text: str | None
    audio_codes: torch.Tensor     # long [n_q, T_frames] raw codes (specials stripped/trimmed)
    frames: int

class OmniGenerator:
    def __init__(self, model: OmniModel, cfg: OmniConfig,
                 device: str | torch.device = "cpu",
                 tokenizer=None): ...
    @torch.inference_mode()
    def generate(self, prompt: Sample, forced_text: list[int], *,
                 sampling: SamplingConfig | None = None,
                 max_frames: int | None = None, seed: int | None = None) -> GenResult
    def s2s(self, wav: torch.Tensor, codec: AudioCodec, **kw) -> GenResult
    def tts(self, text: str, codec: AudioCodec, **kw) -> GenResult   # decode-able via codec
    def asr(self, wav: torch.Tensor, codec: AudioCodec, **kw) -> GenResult  # .text transcript
```

Generation algorithm (binding):
- Maintain a growing UNDELAYED grid `[S, T_total]`; prompt occupies cols `0..T0-1`.
- Input column at delayed step `p`: stream s reads `undelayed[s, p - delays(n_q)[s]]`
  when in range, else `PAD` (text) / `AUDIO_PAD` (audio). Channel: prompt channel for
  prompt cols, `CHANNEL_ASSISTANT` afterwards.
- Prefill the model with delayed steps `0..T0-1` (`model.prefill` over the derived
  delayed grid), then loop `model.step`.
- After the step at position `p` (predicting position `p+1`): text sample belongs to
  undelayed text col `p+1`; audio head k's sample belongs to audio FRAME `p - k`
  (row k+1, col p-k). Discard samples that fall inside the prompt region or before
  the assistant segment start; forced text (from `forced_text`, then `TEXT_PAD` after
  `END_OF_TURN` was consumed/emitted in tts mode) overrides sampled text as INPUT and
  in the record.
- Stop: when cb0 (row 1) samples `AUDIO_EOS` at frame f_eos, keep stepping until all
  codebooks of frame `f_eos - 1` have been produced (i.e. through step `f_eos + n_q - 1`),
  then return frames `assistant_start..f_eos-1`. For asr: stop when the text head emits
  `EOS` or `END_OF_TURN`. Always cap at `max_frames` (default `sampling.max_frames`,
  clamped so the delayed length fits `cfg.model.max_frames`).
- Sampling per SamplingConfig: text (text_temperature, text_top_k); audio per codebook
  (audio_temperature, audio_top_k); one shared torch.Generator seeded by `seed`.
  2026-07: CUDA samples ON-DEVICE (device generator; no full-vocab copies or
  per-codebook round-trips inside the depth rollout); CPU/MPS keep CPU
  sampling. Seeded runs reproduce within a device type. Sampled audio
  specials that never appear as training inputs (AUDIO_BOS anywhere, PAD/EOS
  on codebooks > 0) are written back as AUDIO_PAD; forced_text longer than
  the frame budget raises instead of truncating. `generate` is the sink of
  `stream` — one frame loop serves both.
- KV cache: `KVCache.allocate(cfg.model, batch=1, device, model dtype)`.

A teacher-forcing parity path must exist for tests: `generate` with
`forced_text` covering every step and forced audio via `prompt` extension is not needed —
instead tests compare `model.forward` logits with sequential `model.prefill/step` calls.

## src/omni/infer/chat.py  (agent E)

```python
def main(argv=None) -> int
# --ckpt DIR (from Trainer.export_model / save_pretrained)  --preset/--config fallback for
#   random-init demo runs
# --task s2s|tts|asr  --in in.wav  --text "..."  --out out.wav  --codec fake|mimi
# --tokenizer byte|PATH  --device auto|cpu|cuda|mps  --seed N  --max-frames N
# prints transcript/text to stdout; writes wav via audio.codec.save_wav for s2s/tts;
# exits nonzero with a clear message on bad args.
```
`scripts/chat.py`: thin wrapper calling `omni.infer.chat.main()`.

## src/omni/optim/perf.py  (agent E)

```python
def apply_compile(model: OmniModel, mode: str = "default") -> OmniModel
    # per-Block regional torch.compile; no-op with a warning on mps; never at import
def quantize_int8(model: OmniModel) -> OmniModel   # torchao lazy import
def benchmark_decode(model, cfg, device, *, n_frames: int = 100, batch: int = 1) -> dict
    # random tiny prompt; returns {"steps_per_s": .., "rtf": steps_per_s / 12.5, ...}
def benchmark_forward(model, cfg, device, *, batch: int, frames: int, steps: int = 10) -> dict
    # fwd+bwd tokens/sec + peak memory (cuda only for memory)
```
`scripts/benchmark.py`: `--preset/--config --ckpt? --device --compile --int8`, prints a table.

---

## tests/  (agent F)

`tests/conftest.py` fixtures (session-scoped where cheap):
- `test_cfg`: `load_config("tiny", [...])` shrunk for speed: n_codebooks=2, d_model=64,
  n_layers=2, n_heads=2, n_kv_heads=1, d_ff=128, max_frames=128,
  data.max_sample_frames=120, text_vocab_size=320 (ByteTokenizer), codec="fake",
  batch_size=2, num_workers=0, log_every=1.
- `byte_tok`, `fake_codec` (matching n_codebooks/codec_vocab), `fake_shards` (tmp_path
  factory calling prepare_fake with ~24 samples).

Required files/coverage (each test < ~30 s CPU; whole suite < ~4 min):
- `test_streams.py`: delay/undelay roundtrip on random grids (n_q in {2, 8}); filler and
  mask-shift invariants; sanitize/trim helpers.
- `test_grids.py`: every builder's layout/mask against the docstring spec (bos/task cols,
  user seg fully unmasked, `<assistant>` unmasked, eos-frame on cb0, text-longer-than-audio
  case in assistant segment).
- `test_tokenizer.py`: ByteTokenizer roundtrip; train_bpe on a toy corpus -> specials at
  pinned ids, save/load roundtrip, vocab_size honored.
- `test_codec.py`: FakeCodec determinism, shapes both batched/unbatched, decode length =
  frames * 1920; save/load wav roundtrip via tmp file.
- `test_model.py`: forward shapes; loss finite + per-head metrics keys; overfit: 80 steps
  Adam on one fixed batch drops loss > 60%; **KV parity**: forward logits vs
  prefill+step over the same delayed grid, allclose (atol 1e-4 fp32); grad_checkpoint
  forward matches non-checkpointed.
- `test_data.py`: ShardWriter/ShardDataset roundtrip equality (grid/mask/channel/task);
  collate shapes/dtypes incl. delay slack; BucketBatchSampler: deterministic for fixed
  seed+epoch, disjoint+equal-count across ranks, batches length-homogeneous;
  MixDataset determinism; build_dataloader meta-mismatch raises.
- `test_train.py`: Trainer 6 steps on fake shards (strategy none, cpu) -> finite metrics;
  save_checkpoint + fresh Trainer maybe_resume resumes step and matches optimizer state;
  export_model -> OmniModel.from_pretrained loads and forward-matches (same input, eval).
- `test_generate.py`: tiny random-init model: tts/asr/s2s smoke via FakeCodec (stops by
  max_frames, shapes valid, wav finite); sample_logits temperature-0 argmax; seeded
  reproducibility of generate.
- `test_e2e.py`: tmp dir: prepare_fake -> train 30 steps -> loss drops vs step-0 -> export
  -> OmniGenerator.tts -> decode via FakeCodec -> non-silent wav; runs on CPU < 2 min.

Tests must not download anything or require CUDA. Use `pytest.mark.slow` +
`RUN_SLOW=1` env gate for anything heavier (e.g. real Mimi/kokoro) — none required.

---

# Extensions v2 (depth transformer, word-aligned monologue, full duplex)

Contract for the roadmap stretch features. The v1 sections above stay binding;
everything here is additive and OFF by default (`audio_delay_mode="stagger"`,
`use_depth=False`, `duplex=False` reproduce v1 bit-for-bit — the 62 existing
tests must pass unmodified). Frozen-core support already landed and is tested:
`ModelConfig` gained `audio_delay_mode/use_depth/depth_*/duplex` (+ mode-aware
`max_delay`, duplex-aware `n_streams`; flat⇔use_depth locked by assertion);
`streams` gained `delays(n_q, mode)`, `max_delay(n_q, mode)`,
`stream_delays(n_q, mode, duplex)`, `n_streams`, `infer_n_codebooks`, and
mode/duplex kwargs on `apply_delay`/`undelay`; `grids` gained `word_frames` on
`assistant_speech_segment`/`build_tts`/`build_s2s` plus `build_duplex` (see
docstrings — they are the layout spec).

## src/omni/model/  (agent M) — depth transformer + duplex embeddings

```python
class DepthTransformer(nn.Module):                     # layers.py
    def __init__(self, cfg: ModelConfig, audio_embs: nn.ModuleList): ...
    # Shares the backbone's audio embedding tables (no copies). in_proj:
    # Linear(d_model, depth_d_model, bias=False); pos_emb: Embedding(n_codebooks,
    # depth_d_model); depth_n_layers pre-norm blocks (RMSNorm, causal SDPA,
    # SwiGLU d_ff = 4 * depth_d_model, no biases, NO RoPE); final RMSNorm;
    # heads: ModuleList of n_codebooks Linear(depth_d_model, audio_vocab_size, bias=False).
    def forward(self, h: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor
        # h [N, d_model] backbone hidden (post-final-norm); teacher [N, n_q] codebook ids
        # of the frame being predicted. Position-k input: in_proj(h) for k==0 else
        # in_proj(audio_embs[k-1](teacher[:, k-1])); +pos_emb(k). Head k reads position k.
        # -> float32 logits [N, n_q, audio_vocab_size].
    def sample(self, h: torch.Tensor, sample_fn) -> torch.Tensor
        # h [B, d_model]; sample_fn(logits [B, Va], codebook_index: int) -> long [B];
        # sequential decode over codebooks -> long [B, n_q].

OmniModel additions (omni.py):
    self.depth: DepthTransformer | None        # built iff cfg.use_depth (then NO audio_heads)
    self.user_audio_embs: nn.ModuleList | None # built iff cfg.duplex (n_q tables, audio_vocab_size)
    def step_hidden(self, tokens: Long[B, S], channel: Long[B], cache: KVCache
                    ) -> tuple[Tensor, Tensor]   # (text_logits [B,Vt] fp32, hidden [B,d] post-norm)
    def prefill_hidden(self, grid: Long[B, S, T], channel: Long[B, T], cache: KVCache
                       ) -> tuple[Tensor, Tensor]  # same, for the LAST position
    # step_hidden/prefill_hidden exist for ALL models; step()/prefill() raise RuntimeError
    # with a pointed message when cfg.use_depth (audio logits are sequential there).
```
Semantics:
- `embed`: duplex grids sum rows `1..n_q` via `audio_embs` and rows `n_q+1..2n_q`
  via `user_audio_embs`; non-duplex unchanged.
- `forward` with use_depth: backbone -> h [B,T,d]; audio_logits at position p
  (predicting delayed col p+1) = depth.forward teacher-forced with
  `teacher = grid[:, 1:1+n_q, p+1]` (flat mode puts a frame's codebooks on one
  col). Vectorize as one depth call over N = B*T shifted positions; the last
  position (no p+1) gets AUDIO_PAD teacher and is loss-masked anyway.
  `ModelOutput` shapes are UNCHANGED ([B, n_q, T, Va]).
- `loss`: audio targets/mask are ALWAYS sliced to the assistant group —
  `grid[:, 1:1+n_q]`, `loss_mask[:, 1:1+n_q]` (identical to v1 when not duplex).
- init_weights covers depth (out-projections scaled 1/sqrt(2*depth_n_layers));
  save_pretrained/from_pretrained work unchanged (new config fields serialize;
  old checkpoints load with defaults).
- KV parity gate extends to depth models: forward audio logits == per-step
  step_hidden + depth.forward(teacher) logits (atol 1e-4).

## src/omni/infer/  + optim (agent I) — mode-aware generation + duplex loop

- `generate.py`: derive all index math from
  `streams.stream_delays(n_q, cfg.model.audio_delay_mode)` (flush horizon =
  max delay: n_q stagger, 1 flat). Non-depth models keep the existing
  prefill/step path; depth models use prefill_hidden/step_hidden +
  `model.depth.sample` with a sample_fn built from SamplingConfig (audio
  temperature/top_k per codebook, one shared seeded Generator).
  `OmniGenerator.__init__` raises ValueError for duplex models.
- `src/omni/infer/duplex.py` (new):
```python
@dataclass
class DuplexStep:
    text_id: int
    assistant_frame: torch.Tensor | None  # long [n_q] raw codes; None while the
                                          # delay pipeline fills / after flush drains
class DuplexGenerator:
    def __init__(self, model: OmniModel, cfg: OmniConfig, device="cpu",
                 tokenizer=None, sampling: SamplingConfig | None = None,
                 seed: int | None = None): ...   # requires cfg.model.duplex
    def reset(self) -> None            # fresh KVCache, feeds the <bos> column
    def step(self, user_frame: torch.Tensor | None) -> DuplexStep
        # one 80 ms tick; user_frame = long [n_q] raw codes (None -> silence /
        # AUDIO_PAD). Undelayed-buffer scheme over stream_delays(n_q, mode, True):
        # pushed user frames land on user rows, sampled tokens on text/assistant
        # rows; AUDIO_PAD substituted where sampled ids are specials.
    def run_file(self, user_wav: torch.Tensor, codec: AudioCodec
                 ) -> tuple[str, torch.Tensor]
        # encode wav -> tick every frame -> flush max_delay extra ticks ->
        # assistant track [n_q, T_user] -> sanitize -> decode; returns
        # (monologue text with specials stripped, wav). Assistant track length
        # must equal the user frame count.
```
- `chat.py`: add `--task duplex --in user.wav --out assistant.wav` (works with
  `--codec fake`; prints the monologue text).
- `perf.py benchmark_decode`: route depth models through step_hidden +
  depth.sample so it works for all configs; same result keys.

## src/omni/data/  (agent D) — aligners + duplex data

```python
# synthesize.py
class WordAligner(abc.ABC):
    @abc.abstractmethod
    def align(self, wav: torch.Tensor, sr: int, text: str
              ) -> list[tuple[float, float, str]]  # (start_s, end_s, word), monotonic
class UniformAligner(WordAligner)   # words spread uniformly over the wav duration; no deps
class WhisperAligner(WordAligner)   # lazy transformers whisper (return_timestamps="word");
                                    # helpful ImportError; NOT exercised offline
def build_aligner(name: str | None) -> WordAligner | None   # "uniform"|"whisper"|"none"/None
def word_frames_from_alignment(alignment, tokenizer, frame_rate: float = 12.5
                               ) -> list[tuple[int, list[int]]]
    # (start_s, _, word) -> (int(start_s * frame_rate), encode(" " + word) except first word)
def fake_duplex_dialogues(n: int, seed: int = 0) -> Iterator[dict]
    # deterministic: {"events": [{"speaker": "user"|"assistant", "start_s": float,
    # "text": str}, ...]} — 1-3 exchanges, alternating with occasional small overlap

# prepare.py
# prepare_asr_tts / prepare_s2s gain `aligner: WordAligner | None = None` -> pass
# word_frames(_per_turn) to the grid builders (packed layout when None).
def prepare_duplex(out_dir, *, n_conversations: int, cfg: OmniConfig, tts: TTSBackend,
                   codec: AudioCodec, tokenizer, seed: int = 0) -> None
    # per dialogue: place each event's synthesized+encoded codes onto a shared
    # AUDIO_PAD timeline (two [n_q, T] tracks; non-speech stays AUDIO_PAD),
    # assistant monologue word-aligned via UniformAligner on absolute frames,
    # grids.build_duplex, truncate to cfg.data.max_sample_frames.
# ShardWriter meta.json gains "duplex": bool (absent == false; version stays 1).

# dataset.py
# ShardDataset derives S duplex-aware (streams.n_streams) when parsing records.
def collate(samples, codec_vocab, *, mode: str = "stagger", duplex: bool = False) -> dict
# build_dataloader passes cfg.model.audio_delay_mode / cfg.model.duplex and raises
# on meta["duplex"] != cfg.model.duplex.
```
- `scripts/prepare_data.py`: new `duplex --n N` subcommand (SineTTS + FakeCodec,
  offline); `asr`/`s2s` gain `--align uniform|whisper|none` (default none).

## tests/  (agent T) — new files only; existing tests untouched

- `tests/test_depth.py`: flat-mode delay roundtrips; config gate (flat⇔depth);
  forward shapes + finite loss with use_depth; **depth KV parity** (forward vs
  prefill_hidden/step_hidden + depth.forward teacher, atol 1e-4); step()/prefill()
  raise on depth models; 80-step overfit drops loss >50%; save/from_pretrained
  roundtrip incl. depth weights; OmniGenerator tts smoke on a depth model with
  seeded reproducibility; 5-step Trainer run on flat-collated fake shards.
- `tests/test_duplex.py`: build_duplex layout/mask properties beyond the basics;
  duplex shard write/read + collate (S = 1+2n_q, mode/duplex kwargs); duplex
  model forward/loss (user rows never contribute); DuplexGenerator: pipeline-fill
  Nones then steady frames, seeded determinism, run_file returns [n_q, T_user]
  track + finite wav; e2e prepare_duplex -> 20-step train -> loss drops.
- `tests/test_aligned.py`: word_frames placement (gaps -> TEXT_PAD, collisions
  shift right, text-beyond-audio extends segment); UniformAligner determinism +
  monotonicity; word_frames_from_alignment tokenization; prepare tts with
  uniform aligner -> loadable shards whose text rows differ from packed layout.
- conftest.py: may ADD fixtures (e.g. depth_cfg, duplex_cfg derived from
  test_cfg overrides); nothing existing may change. Suite stays offline, CPU,
  < ~6 min total.

---

# Extensions v5 (reference-voice cloning — docs/DESIGN_V5_VOICE.md is the binding spec)

Landed in the frozen core and contract-tested: `streams` ids `VOICE=49`,
`VOICE_END=50`, `ACCENT_KEEP=51`; `grids._GridBuilder.voice_segment(ref_codes)`
plus an optional `voice_codes: torch.Tensor | None = None` kwarg on
`build_asr/build_tts/build_s2s/build_duplex` and the three prompt builders
(segment sits between `<bos>` and the task tag; duplex shifts frame f to col
f+R+2; loss masked on every row over the segment; defaults reproduce v4
bit-identically). The "Grid layout (binding)" section of DESIGN_V5_VOICE.md
governs every detail below.

## src/omni/data/  (agent D)

```python
# prepare.py
def sample_voice_ref(codes: torch.Tensor, *, min_frames: int = 37, max_frames: int = 250,
                     rng: random.Random) -> torch.Tensor
    # random contiguous chunk [n_q, R] of a reference utterance's codes (3-20 s @12.5 Hz);
    # clamps to the utterance length; never returns 0 frames.
# prepare_s2s / prepare_asr_tts gain: voice_p: float = 0.0 (probability a sample carries a
# voice segment). Reference rule (binding): same synthetic voice, DIFFERENT utterance than
# any target in the sample — for s2s synthesize one extra short utterance (a few words from
# the dialogue-independent _fake-sentence pool or the next dialogue's text) with the SAME
# asst_voice; for asr/tts use the previous same-speaker row's codes (speaker_column:
# str | None = None names the dataset speaker id column; no column -> voice_p must be 0,
# raise ValueError). Budget: when a segment of R frames is present, reduce the
# _fit_s2s_turns / max_sample_frames budget by R + 2. prepare_duplex gains voice_p
# likewise (reference = extra same-voice utterance; passes voice_codes to build_duplex).
```
`scripts/prepare_data.py`: `--voice-p FLOAT` on `s2s`/`duplex` (and `asr` with
`--speaker-column COL`). The full cross-lingual VoicePairBank/manifest pipeline
(DESIGN_V5 §3) is GPU-box scope — implement only the hooks above now, keep
signatures forward-compatible.

## src/omni/infer/ + optim  (agent I)

```python
# generate.py — OmniGenerator
def set_voice(self, wav: torch.Tensor | None, codec: AudioCodec | None = None,
              *, voice_codes: torch.Tensor | None = None, max_frames: int = 125) -> None
    # encodes + stores reference codes (None clears); trims to max_frames.
# tts/s2s/asr gain voice_wav: torch.Tensor | None = None and voice_codes: ... = None
# kwargs (one-shot override; falls back to the set_voice session state; asr accepts and
# threads it — invariance path). Thread into the prompt builders' voice_codes.
# generate()'s frame loop is UNCHANGED (segment rides the prompt prefill); the
# max_frames clamp must account for the longer prompt (existing guard already does
# if it derives from prompt.n_frames — verify).

# duplex.py — DuplexGenerator gains voice_codes: torch.Tensor | None = None on
# __init__ (stored) and reset(voice_codes=...) override. reset() prefills the delayed
# [<bos> + voice segment] block (R+2 undelayed cols) via ONE prefill call instead of
# the single <bos> column; all tick index math offsets by R+2 (conversation frame f at
# undelayed col f+R+2); run_file capacity check adds R+2. Without a voice this must be
# BIT-IDENTICAL to today (seeded tests exist).

# chat.py: --voice REF.wav on tts/s2s/duplex (load_wav -> codec.encode -> trimmed codes);
# warn-and-continue (stderr) when given with asr. Composes with --emotion/--lang.

# perf.py: benchmark_decode(..., voice_frames: int = 0) — random voice codes of that many
# frames in the prompt; report "voice_prefill_ms" (prefill wall time) separately and the
# usual decode keys so the with/without step-time delta is measurable.
# scripts/benchmark.py: --voice-frames N (default 0).
```

## tests/test_voice.py  (agent T; conftest may ADD a fixture, change nothing)

Required coverage (CPU, offline, tiny dims): voice_segment invariants beyond the
landed sanity checks (apply_delay roundtrip stagger+flat, budget edge R=1, reject
R=0 and specials in refs); prepare_s2s voice_p=1.0 with SineTTS -> every sample
carries `<voice>`, reference codes differ from every assistant segment in the
sample, budget respected (n_frames <= max_sample_frames), voice_p=0.0 bit-stable
vs v4; prepare_asr_tts speaker_column guard (ValueError when voice_p>0 without
column); OmniGenerator: set_voice + per-call voice_wav smoke on FakeCodec (seeded
determinism; voice vs no-voice prompts differ; generated frames cap respected);
DuplexGenerator: voice reset -> run_file output length invariant unchanged, and
no-voice path seeded-identical to pre-v5; benchmark_decode voice_frames=32 returns
voice_prefill_ms > 0 and the standard keys; chat --voice CLI smoke (tts, fake
codec, tmp wav as reference).

---

## Return format for every implementation agent

Final structured output: `{"files_written": [...], "self_checked": bool,
"notes": "...", "interface_concerns": "..."}` — `self_checked` true only if you ran
`.venv/bin/python -c "import omni.<your.module>"` (and ideally a tiny functional check)
successfully from the project root.

---

## 2026-07-03 review-fix amendments (binding; full report in docs/research/ARCHITECTURE_REVIEW_2026-07-03.md)

Contract deltas from the post-review fix pass, beyond the in-place edits above:

- `streams.delays` gained mode `"lead"` (text 0, semantic cb0 1, acoustics 2 —
  Moshi's ablated semantic-lead pattern); `max_delay = max(delays)`;
  `ModelConfig.audio_delay_mode` accepts it and `use_depth` locks to
  `mode in ("flat", "lead")`. `turn_prefix` raises on intensity without a
  response style; `grids` reject empty assistant speech segments.
- `ModelConfig.depth_loss_ratio` (default 1.0; depth presets 0.0625): fraction
  of frame positions the depth-transformer loss trains on per step (CSM
  amortization). Training-mode-only; eval/decode always full.
- `TrainConfig.save_keep` (default 3): keep-last-K checkpoint retention.
- Config overrides are strict: unknown boolean spellings and wrong-length
  tuples raise instead of silently coercing.
- Both model classes expose `optim_param_groups(train_cfg)`; weight decay
  skips 1-D params and every embedding table; HF models add the backbone_lr
  tier for trainable backbone/LoRA params.
- `omni.model.load_weights(model, dir)` + `scripts/train.py --init-from DIR`:
  stage-transition warm start (exported weights into a freshly built model,
  fresh optimizer); `omni.model.structural_diff` backs both this and resume's
  compatibility check.
- Transcript tokenization in every prep path (textlm/asr/tts/s2s and
  word-aligned monologues) drops reserved ids < 64 — free text can never
  inject control tokens. `_check_meta` accepts shard text_vocab_size smaller
  than the model's when a backbone is set (padded-embedding vocab derivation);
  mixed shard dirs must agree on text_vocab_size and tokenizer_id.
- `serve.streaming.StreamingEncoder`: rolling-context mic encoding for the
  duplex console (approximates streaming Mimi; stateless per-chunk encode
  diverged from training codes). `create_app(..., device=)` and
  `scripts/serve.py --device`; duplex sessions report `behind_ms` and reject
  odd-length PCM frames instead of dying.
- Codec `decode` validates RAW codes (specials must be stripped with
  `streams.sanitize_codes` first) and codebook counts, for Mimi and FakeCodec
  alike. `tests/test_gpu_preflight.py` (RUN_SLOW=1) pins the Mimi/tokenizer
  contracts on the GPU node; see docs/GPU_DAY_CHECKLIST.md.

## 2026-07-03 multilingual amendments (binding; runbook in docs/MULTILINGUAL.md)

- `TTSBackend.synth(text, voice, style=None, lang=None)`: all backends accept
  a `streams.LANG_TAGS` key (route/hint or safely ignore). `build_tts(name,
  lang=None)` seeds Chatterbox's default language.
- `prepare_asr_tts(..., lang_column=)`: per-row language from a dataset
  column, normalized by `prepare._lang_label` (locale spellings -> LANG_TAGS
  keys; unknown rows skipped + one summary warning); exclusive with `lang`.
  Dialogue turns / duplex events accept `"lang"` keys (any spelling,
  normalized) which reach both the `<lang_XX>` monologue prefix and
  `synth(lang=)`.
- `WhisperAligner`: default model is the MULTILINGUAL `openai/whisper-tiny`
  (was the English-only `.en`); `language=` pins decode language.
  `build_aligner(name, lang=None)`.
- `scripts/train_tokenizer.py --dataset` is repeatable with `@WEIGHT`
  (deterministic weighted interleave) and combinable with `--text-file` —
  the DESIGN_V4 §3 multilingual 48k-BPE mix.
- `prepare_fake` / `fake_dialogues` rotate languages (None/en/tr/zh resp.
  en/tr/zh/fr) with per-language word pools (non-ASCII, UTF-8 through
  ByteTokenizer) — offline smokes exercise multilingual grids by default.
  `synthesize.FAKE_LANGS` names the rotation.
