# Omni-S2S Design

> Produced from a 5-agent web-research pass (2026-07-02); raw reports in docs/research/. The data-recipe report failed mid-run, so dataset ids below derive from the architecture report + general knowledge — verify each downloads ungated in week 1 (the synthetic Kokoro path is the fallback).

# Omni-S2S Design Brief (v1)

From-scratch, half-duplex speech-to-speech LLM: one decoder-only transformer, fully discrete audio I/O over a frozen codec, PyTorch-native. Dev on Mac (CPU/MPS, tiny config), train on one node with 8 CUDA GPUs.

## 1. Architecture template & token layout

**Template: Moshi's multistream frame-grid LM (Kyutai), simplified Mini-Omni/MusicGen style** — a single temporal transformer (no depth transformer), per-stream parallel output heads, MusicGen per-codebook delay pattern, Mini-Omni packed text-ahead inner monologue. Upgrade path when needed: Sesame CSM-1B depth decoder; Moshi dual-stream full duplex.

**Streams: 9 parallel** — 1 text stream + 8 Mimi codebook streams on a 12.5 Hz frame grid (1 step = 80 ms).

**Input = sum of embeddings.** Nine embedding tables (text 32,768+64 specials; each audio codebook 2,051) map to d_model and are summed per step, plus a 2-entry channel embedding (USER vs ASSISTANT audio). RoPE only; no absolute positions.

**Output heads: 9 linear heads** — text head weight-tied to the text embedding; 8 untied audio heads (one per codebook). One forward per frame predicts all 9 next-step tokens.

**Acoustic delay pattern (exact, in frames):** text = 0; audio cb0 (semantic) = 1; cb1 = 2; cb2 = 3; cb3 = 4; cb4 = 5; cb5 = 6; cb6 = 7; cb7 = 8. Codebook k of audio frame t sits at grid step t+k+1, so each codebook is predicted with all lower codebooks of that frame already in context (no depth model needed) and all audio conditions on the current text token. Cost: frame t fully decodable at step t+8 (+640 ms). Latency escape hatch: Moshi delays [0,1,1,1,1,1,1,1] plus a small depth transformer (CSM style), or 4 codebooks.

**Special tokens.** Text stream: `<pad>` (batch pad, loss-ignored), `<bos>`, `<eos>`, `<text_pad>` (monologue filler), `<user>`, `<assistant>`, `<end_of_turn>`, task tags `<asr>`, `<tts>`, `<s2s>`, `<alm>`; 64 ids reserved. Audio streams (per codebook, vocab 2051): 0–2047 Mimi codes, 2048 = `<audio_pad>` (delay filler, non-speech frames, "ungenerated" placeholder at inference), 2049 = `<audio_bos>`, 2050 = `<audio_eos>` (emitted on cb0 to end speech).

**Inner-monologue alignment: packed text-ahead (Mini-Omni), not word-aligned.** Assistant text tokens are laid out one per frame from turn start, ending with `<end_of_turn>` then `<text_pad>` until audio finishes. BPE runs ~4–6 tokens per second of speech vs the grid's 12.5, so text leads audio by a growing margin; audio heads always see current and past text. Zero word-timestamp preprocessing. Upgrade if pacing/stop-timing misbehaves: Moshi word-aligned monologue (whisperX timestamps, pad/EPAD placement).

**Half-duplex turn format (user speech → assistant text+speech):**

```
step:     0     1 … Tu                          Tu+1           Tu+2 …
text:     <bos> <user> <text_pad> …             <end_of_turn>  <assistant> w1 w2 … <end_of_turn> <text_pad> …
audio_k:  <audio_pad>  user Mimi codes          <audio_pad>    assistant Mimi codes (delay k+1,
                       (delay k+1, ch=USER)                    ch=ASSISTANT) … <audio_eos>
```

Loss: cross-entropy on text + all 8 audio heads over assistant-turn steps only (user steps masked); equal head weights initially (cb0 up-weighting is a tuning knob). The same grid encodes every task: ASR = user audio → text-only reply; TTS = assistant text+audio turn; audio-LM = audio-only rows; text-LM = text-only rows; multi-turn = concatenated turns.

## 2. Codec

**Primary: Mimi — HF `kyutai/mimi`** (CC-BY-4.0; `transformers.MimiModel`, runs CUDA/CPU/MPS). 24 kHz audio, **12.5 Hz frame rate**, RVQ, **codebook size 2048**, **8 codebooks used** (`encode(..., num_quantizers=8)`, 1.1 kbps). Codebook 0 is WavLM-distilled (semantic): content on cb0, acoustics on cb1–7. Causal streaming encode/decode (80 ms) keeps the full-duplex door open. Tiny/smoke runs may drop to 4 codebooks (~0.55 kbps).

**Runner-up: SNAC** `hubertsiuzdak/snac_24khz` (MIT, pip `snac`): 3 hierarchical codebooks (~12/23/47 Hz), vocab 4096, 0.98 kbps, 19.8M-param codec, flattened 84 tok/s single stream (Orpheus-style). Adopt only if the multistream machinery must be abandoned. Rejected: XCodec2 (CC-BY-NC), EnCodec/DAC (75–86 Hz, too many steps), CosyVoice S3 tokens (no waveform decoder — drags in a flow-matching vocoder).

## 3. Backbone configs

Llama-style decoder: pre-RMSNorm, SwiGLU, GQA, RoPE, no biases, tied text embedding/head, fp32 logits, `F.scaled_dot_product_attention(is_causal=True)`. Param counts include all embeddings/heads.

| config | total params | d_model | n_layers | n_heads | n_kv_heads | d_ff (SwiGLU) | context (frames @12.5 Hz) | rope_theta |
|---|---|---|---|---|---|---|---|---|
| tiny (CPU smoke) | ~22M (~6M non-emb) | 256 | 6 | 4 | 2 | 768 | 512 (41 s) | 500,000 |
| small (1 GPU) | ~250M | 1024 | 16 | 16 | 4 | 2816 | 1024 (82 s) | 500,000 |
| base (8 GPU) | ~0.76B | 1536 | 26 | 12 | 4 | 4096 | 2048 (164 s) | 500,000 |

Head_dim 64/64/128. One model class, three config presets; tiny must run data→train→generate→WAV end-to-end on Mac CPU.

## 4. Stage-wise training recipe

Optimizer everywhere: fused AdamW (betas 0.9/0.95, wd 0.1), grad-clip 1.0, WSD schedule (1–2% linear warmup, stable, cosine/sqrt decay), bf16 compute with fp32 masters.

**Stage A — text-LM pretrain** (text stream only; audio rows `<audio_pad>`). Data: `HuggingFaceFW/fineweb-edu` (sample-10BT; base pulls from sample-100BT), ~5% dialogue flavor from `allenai/soda` and `HuggingFaceH4/ultrachat_200k`. Budget: tiny 0.2B, small 5B, base 15B tokens.

**Stage B — speech multitask pretrain** (full grid, all heads). Mixture by positions: **TTS 30% / ASR 20% / audio-LM 20% / text-LM replay 30%**. Ungated permissive data: `openslr/librispeech_asr` (960 h, CC-BY-4.0), `parler-tts/mls_eng` (44.5k h CC-BY-4.0 — take 8–12k h), `MLCommons/peoples_speech` (clean CC-BY subset, 3–5k h), `mythicinfinity/libritts_r` (585 h, CC-BY-4.0, TTS-grade). Optional with NC caveat: `amphion/Emilia-Dataset`. Target ~12–15k h ≈ 550–680M frames, 1–2 epochs (small: 3–5k h suffices).

**Stage C — S2S instruct.** Mixture: **S2S 60% / TTS 15% / ASR 10% / text-SFT 15%**. Data: `gpt-omni/VoiceAssistant-400K` (~470k exchanges; alt `worstchan/VoiceAssistant-400K-SLAM-Omni`), `ICTNLP/InstructS2S-200K` pairs with assistant audio re-synthesized locally, plus 300–500k synthetic dialogues (below). ~2–4k h.

**Stage D (optional polish).** 200–500 h with one fixed assistant voice: re-synthesize Stage-C assistant turns with a single Kokoro voice (Moshi's final-finetune trick) for a consistent persona.

**Fully-synthetic fallback (also the top-up path).** Local TTS: **`hexgrad/Kokoro-82M`** (Apache-2.0, ~50 voices, ONNX faster-than-real-time on CPU, ~100x RT on one GPU → 5k h ≈ 50 GPU-h). Runner-up for speaker diversity: `FunAudioLLM/CosyVoice2-0.5B` (Apache-2.0, zero-shot voices); Piper for bulk-cheap. Text sources: `allenai/soda` (1.5M dialogues, CC-BY-4.0), `HuggingFaceH4/ultrachat_200k` (MIT), `OpenAssistant/oasst2` (Apache-2.0). Filter to speakable turns (<60 words; no code/URLs/markdown), random user voice + 0.9–1.1x speed perturb, fixed assistant-voice pool. This pipeline alone can regenerate Stage B (TTS/ASR from fineweb sentences) and Stage C end-to-end if any HF audio set turns out gated or broken.

## 5. Multi-GPU & systems

**Decision rule: DDP below ~300M (tiny, small); FSDP2 at/above (base).** A 1B model with fp32 masters + Adam (~16 GB/GPU) technically fits DDP, but FSDP2 buys activation headroom and native bf16-param/fp32-reduce. Pin torch ≥2.8 (current stable), torchtitan pattern: per-transformer-block `fully_shard` + root, `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, optimizer constructed after sharding. Order per block: activation checkpointing (`use_reentrant=False`) → optional regional `torch.compile` → `fully_shard` (torch 2.9+ cannot trace through FSDP2 hooks; compile ships off by default). DDP path: autocast-bf16 + `no_sync()`; FSDP2 grad accumulation via `set_requires_gradient_sync(last_microbatch)`. The same script runs single-process CPU/MPS (sharding skipped) for Mac smoke tests. Launch: `torchrun --standalone --nproc_per_node=8`. No DeepSpeed, no Accelerate.

**Checkpointing.** DCP (`dcp.save`/`async_save` + `get_state_dict`) sharded every N steps; rank0 consolidated safetensors export (`StateDictOptions(full_state_dict=True, cpu_offload=True)`) at stage boundaries; RNG + dataloader cursor stored in the checkpoint.

**Dataloader for pre-tokenized shards.** Offline GPU job: resample to 24 kHz mono → batched `MimiModel.encode` (8 cbs) + BPE text → **undelayed** frame grids `[T, 9]` uint16 with turn/loss metadata → ~512 MB memmap `.bin` shards + JSONL index. Delay shifting, packing to context length (document boundaries kept; no cross-doc attention masking initially), and loss masks are applied in collate — changing the delay pattern never forces re-tokenization. DataLoader: deterministic (epoch, rank, worker) shard assignment, `num_workers=4–8`, `pin_memory`, `persistent_workers`, `drop_last`. Graduate to mosaicml-streaming only if resumption/scale demands it.

## 6. Inference design

**Decode loop (Moshi LMGen, scaled down).** One KV-cached backbone step per 80 ms frame emits 9 tokens from parallel heads. Preallocated static/ring KV cache (fixed batch, context = config frames). Delay handling via a circular token cache `(B, 9, max_delay+2)`: write sampled tokens at `(offset+delay) % CT`, read inputs at `offset % CT`, substitute `<audio_pad>` for not-yet-started streams, emit waveform only once `offset > 8`. Cache-indexing, not the transformers `build_delay_pattern_mask` approach — only the former streams.

**Per-stream sampling defaults.** Audio heads: **temperature 0.8, top_k 250** (independently per codebook). Text head: **temperature 0.7, top_k 25**. No top-p, **no repetition penalty on audio** (silence codes legitimately repeat). Debug fallback at tiny/small scale: near-greedy top_k 1 (Mini-Omni ships this at 0.5B).

**Streaming plan.** Silero VAD gates the half-duplex loop: while the user speaks, streaming-Mimi-encode each 80 ms frame and prefill the backbone incrementally; on end-of-speech append `<end_of_turn><assistant>` and decode frame-synchronously; first audio ≈ 9 frames + compute ≈ 0.8–1.0 s. Use kyutai's `moshi` package (`mimi.streaming()`) or `rustymimi` for frame-wise decode — HF transformers Mimi streams encode cleanly but its decoder transposed-convs are stateless (chunk-boundary artifacts). Real-time budget is 12.5 steps/s: 0.25–0.76B is far under 80 ms/step on GPU and comfortably real-time on M-series (Moshi-7B-on-M3 precedent).

**Speed/quantization.** CUDA: `torch.compile(step_fn, mode="reduce-overhead", fullgraph=True)` over the fixed-shape step (gpt-fast recipe), then torchao `Int8WeightOnlyConfig` (Int4 later for serving). Mac demo: MLX q4/q8 export or fp16 MPS eager; keep an eager fallback everywhere.

## 7. Text tokenizer

**Train our own byte-level BPE: vocab 32,768 + 64 reserved specials** (HF `tokenizers`, GPT-2-style byte fallback, digit splitting, whitespace pretokenization), trained on the Stage-A mix. Rationale: 128k-class open vocabs (Llama-3, Qwen2.5) would put 33–50M embedding params on a 22M tiny model; 32k matches Moshi's own choice, keeps heads cheap at 12.5 Hz, and is license-clean (our artifact). Runner-up: GPT-2's 50,257 byte-level BPE (MIT) if tokenizer training must be skipped — but custom 32k is a half-day task.

## Build order

1. Codec round-trip + tokenizer + shard writer (Mac). 2. Tiny model overfits 100 TTS utterances (Mac CPU). 3. Small Stage A→B on 1 GPU; gate on ASR-WER and Whisper-judged TTS intelligibility. 4. Stage C on small → first S2S demo. 5. Base on 8 GPUs (FSDP2). 6. Latency + quantization pass. 7. Stretch: depth transformer, word-aligned monologue, dual-stream full duplex.
# Decisions

## Architecture template
CHOICE: Moshi-style multistream frame-grid LM simplified to Mini-Omni/MusicGen form: one temporal transformer, 9 parallel streams (1 text + 8 Mimi codebooks), sum-of-embeddings input, 9 parallel output heads, no depth transformer
RATIONALE: Fully discrete audio I/O is the only proven from-scratch recipe (Moshi, GLM-4-Voice, MiMo-Audio) and the project allows only a pretrained frozen codec — a Whisper encoder would be a second pretrained component. Single transformer + parallel heads is the simplest thing proven at 0.5B (Mini-Omni) and keeps tiny-config CPU smoke tests trivial.
ALT: Architectures report's primary pick (Whisper continuous encoder + CosyVoice semantic tokens + SLAM-Omni grouping) — rejected as it assumes a pretrained text LLM init and needs an external flow-matching vocoder; Sesame CSM depth-decoder kept as upgrade path.

## Duplex / turn format
CHOICE: Half-duplex, single shared audio stream group (9 streams total) with USER/ASSISTANT channel embedding and turn-marker tokens; user frames input-only (loss-masked)
RATIONALE: Halves stream count vs Moshi's 17-stream dual group; one frame-grid format covers ASR/TTS/audio-LM/text-LM/S2S with task tags; matches the half-duplex baseline requirement.
ALT: Moshi dual audio-stream group (full-duplex-native) — deferred to stretch goal; GLM-4-Voice 13:26 interleaving — rejected (worse latency structure, fixed ratio brittleness).

## Codec
CHOICE: kyutai/mimi, 12.5 Hz, 8 codebooks of 2048 (1.1 kbps), frozen; encode/decode via transformers.MimiModel; kyutai moshi package for streaming decode
RATIONALE: Lowest frame rate available, semantic codebook 0 (WavLM-distilled) aids understanding without a Whisper encoder, causal streaming, CC-BY-4.0, de-risked by Moshi/CSM/LFM2-Audio; both codec and inference reports converge on it.
ALT: Runner-up SNAC snac_24khz (MIT, 3 codebooks, 84 tok/s flattened, Orpheus-proven); rejected XCodec2 (NC license), EnCodec/DAC (75–86 Hz), CosyVoice S3 tokens (no waveform decoder).

## Delay pattern
CHOICE: Text delay 0; audio codebook k delay = k+1, i.e. [1,2,3,4,5,6,7,8] frames (MusicGen-style stagger, text-ahead by 1)
RATIONALE: With parallel heads (no depth transformer), a strict stagger guarantees each codebook is predicted with all lower codebooks of the same frame in context, avoiding intra-frame conditional independence; MusicGen showed delay ≈ flatten quality at 1/K compute.
ALT: Moshi delays [0,1,1,...,1] + small depth transformer (CSM style) — the documented escape hatch if the 640 ms audio lag hurts; 4-codebook variant for lower latency/quality.

## Inner-monologue alignment
CHOICE: Packed text-ahead (Mini-Omni): assistant text one token per frame from turn start, then <text_pad>; no word-level timestamps anywhere in data prep
RATIONALE: BPE at ~4–6 tok/s vs 12.5 Hz grid means text naturally leads audio; eliminates the whisperX/forced-alignment preprocessing pass over all training audio; explicitly recommended by the architectures report.
ALT: Moshi word-aligned monologue with PAD/EPAD placement via whisper-timestamped — upgrade path if pacing or stop-timing misbehaves.

## Backbone configs
CHOICE: Llama-style (RMSNorm, SwiGLU, GQA, RoPE theta 500k, tied text emb/head): tiny ~22M (d256/L6/H4/KV2/ff768/ctx512), small ~250M (d1024/L16/H16/KV4/ff2816/ctx1024), base ~0.76B (d1536/L26/H12/KV4/ff4096/ctx2048 frames)
RATIONALE: Standard, torchtitan-compatible shapes; head_dim 64/64/128 SDPA-friendly; context sized in 12.5 Hz frames (41 s / 82 s / 164 s) to cover multi-turn dialogues; param counts include the 9 embedding tables/heads.
ALT: Wider-shallower base (d2048/L20 ≈ 1.0B) rejected to stay safely under 1B with embeddings included.

## Text tokenizer
CHOICE: From-scratch byte-level BPE, vocab 32,768 + 64 reserved specials, trained with HF tokenizers on the Stage-A text mix
RATIONALE: Right-sized for a 22M tiny model (128k vocabs would be 33–50M embedding params), matches Moshi's 32k choice, no external license obligations.
ALT: GPT-2 50,257 BPE (MIT) as zero-effort runner-up; Llama-3/Qwen tokenizers rejected (vocab too large, license attribution).

## Training stages and mixtures
CHOICE: Stage A text-LM pretrain (0.2B/5B/15B tokens by size) → Stage B speech multitask (TTS 30/ASR 20/audio-LM 20/text replay 30, ~12–15k h) → Stage C S2S instruct (S2S 60/TTS 15/ASR 10/text-SFT 15, ~2–4k h) → optional Stage D single-voice polish (200–500 h)
RATIONALE: Random-init backbone needs language first (Moshi's ordering); multitask Stage B teaches listening (ASR) and speaking (TTS) modes in the exact grid format Stage C composes into S2S; Stage D is Moshi's consistent-persona trick.
ALT: SLAM-Omni single-stage training — rejected because it relies on a pretrained text LLM; pure audio-LM pretrain (Moshi's 7M h) — out of compute scope.

## Datasets
CHOICE: Stage A: HuggingFaceFW/fineweb-edu + allenai/soda + HuggingFaceH4/ultrachat_200k. Stage B: openslr/librispeech_asr, parler-tts/mls_eng (8–12k h), MLCommons/peoples_speech (clean subset), mythicinfinity/libritts_r. Stage C: gpt-omni/VoiceAssistant-400K, ICTNLP/InstructS2S-200K (assistant audio re-synthesized), synthetic SODA dialogues. Fallback TTS: hexgrad/Kokoro-82M (runner-up FunAudioLLM/CosyVoice2-0.5B)
RATIONALE: All primary sets are ungated with CC-BY/MIT/Apache-class licenses and HF-downloadable; Kokoro is Apache-2.0 and ~100x real-time on one GPU, making a fully synthetic Stage B+C regeneration path cheap (~50 GPU-h per 5k h).
ALT: GigaSpeech (gated) and Emilia (CC-BY-NC) listed only as optional; the data research report was null, so these picks come from the architecture report plus general knowledge and need early verification.

## Multi-GPU strategy
CHOICE: DDP (+autocast bf16) below ~300M; FSDP2 per-block fully_shard with MixedPrecisionPolicy(bf16 param, fp32 reduce) and fp32 masters for base; AC (use_reentrant=False) → optional regional compile → fully_shard; DCP sharded checkpoints + rank0 consolidated safetensors; uint16 memmap shards with delays applied at collate
RATIONALE: Torchtitan-verified pattern on torch ≥2.8; 250M fits DDP trivially and keeps the Mac single-process path identical; FSDP2 at 0.76B buys activation headroom; storing undelayed grids means delay-pattern changes never force re-tokenization.
ALT: DeepSpeed/Accelerate rejected (unnecessary indirection single-node ≤1B); mosaicml-streaming deferred until memmap shards become limiting.

## Inference design
CHOICE: Moshi LMGen-style loop: one KV-cached step per 80 ms frame, static/ring KV cache, circular delay token cache written at (offset+delay)%CT; sampling audio temp 0.8 / top_k 250 per codebook, text temp 0.7 / top_k 25, no top-p, no audio repetition penalty; Silero-VAD half-duplex driver; torch.compile reduce-overhead on CUDA; torchao Int8WeightOnly for serving, MLX q4/q8 for Mac
RATIONALE: These are Moshi's shipped defaults and the inference report's explicit recommendation; cache-indexed delays (not transformers' delay-pattern mask) are the only approach that streams; 12.5 steps/s real-time budget gives huge headroom at ≤0.76B.
ALT: Mini-Omni near-greedy (top_k 1) kept as small-scale debug fallback; transformers build_delay_pattern_mask rejected (batch, non-streaming).

# Risks

- Reports disagree on input modality: the architectures report recommends a continuous Whisper encoder for speech understanding at <1B scale (its primary SLAM-Omni/Mini-Omni pick), while the codec and inference reports recommend fully-discrete Mimi in/out. The brief chooses discrete Mimi because the backbone is random-init and only a frozen codec is sanctioned; accept the known risk of weaker ASR/understanding at small scale, mitigated by the semantic codebook 0, an ASR-heavy Stage B, and the option to bolt on a Whisper adapter later without changing the output side.
- Reports disagree on output tokens: architectures report prefers single-codebook CosyVoice/GLM-4-Voice semantic tokens (+ external flow-matching vocoder), codecs report explicitly rejects that as 'not a codec'. Choosing Mimi RVQ costs 8 streams + delay machinery; if multistream complexity stalls the project, the SNAC single-flattened-stream fallback (84 tok/s) is the documented retreat.
- The data research report was null: all dataset ids, hours, gating and license claims (mls_eng, peoples_speech, VoiceAssistant-400K contents/license, InstructS2S-200K audio availability) come from the architecture report and general knowledge — verify each dataset downloads ungated and contains actual waveforms in week 1; the fully-synthetic Kokoro path is the insurance policy.
- The full [1..8] delay stagger adds 640 ms between text start and first decodable audio frame (~0.8-1.0 s to first sound in half-duplex). If unacceptable, switching to Moshi delays [0,1,1,...] requires adding a depth transformer (CSM-style) — a planned but non-trivial architecture change requiring retraining.
- Parallel per-codebook heads with a MusicGen delay are proven for music (EnCodec) and SNAC speech (Mini-Omni) but less validated for 8-codebook Mimi speech without a depth transformer; monitor acoustic quality early (codebook-wise teacher-forced NLL and resynthesis MOS-proxy) at the small scale before committing base-scale compute.
- Packed (unaligned) text monologue may produce pacing drift, premature audio EOS, or run-on speech versus Moshi's word-aligned monologue; the fallback (whisperX word timestamps + aligned <text_pad> placement) changes data prep and requires re-sharding, so keep the shard writer's text-placement logic pluggable.
- Sampling defaults (audio 0.8/250, text 0.7/25) come from a 7B Moshi; at 150-250M they may be too hot (Mini-Omni ships effectively greedy at 0.5B) — treat as a per-size tuning sweep, not a constant.
- Stage A from-scratch text pretraining (15B tokens for base) dominates the schedule and is the largest compute line-item; WSD scheduling with early-decay checkpoints lets you bank a usable model before the full budget lands.
- HF transformers' Mimi decode is not fully streaming (stateless transposed convs cause chunk-boundary artifacts); the live demo depends on kyutai's moshi package or rustymimi — pin versions and smoke-test streaming decode on both CUDA and MPS early.
- torch.compile interacts badly with FSDP2 hooks on torch >=2.9 (must compile per-block before sharding, or fullgraph=False) and inductor-Metal on MPS is immature — ship compile off by default and treat it purely as an optimization flag.
- Synthetic Kokoro-generated audio has limited speaker/prosody/noise diversity; a model trained mostly on it may fail on real microphone speech — always mix real ASR corpora into the user-side audio and apply speed/voice perturbation.
- Loss weighting across 1 text + 8 audio heads is unvalidated at this scale (reports are silent); starting equal risks either text degradation or muffled acoustics — add per-head loss logging from day one and treat cb0/text up-weighting as the first tuning knob.