# Multilingual Training — End-to-End Runbook

How to train omni multilingually at every stage, and why the architecture
supports it. Extends DESIGN_V4 §3 (which chose the token design and language
tiers); this documents the now-implemented plumbing (2026-07-03 multilingual
pass; contract deltas in INTERFACES.md).

## Why the architecture is multilingual-suitable

- **Language is a text-stream concern only.** `<lang_XX>` reserved ids 37..48
  ride row 0 exactly like Whisper's language tokens; the audio path (Mimi
  codes, delay pattern, depth transformer) is language-agnostic — no
  per-language heads, embeddings, or branches anywhere in the model. Adding a
  language is a data + tag change, never an architecture change.
- **The 12-language inventory** (en zh fr de es ja ko tr ru it pt nl) is
  pinned in `streams.LANG_TAGS`. Expansion path: reserved ids 52..63 are
  still free → up to 12 more languages as an additive frozen-contract
  amendment; beyond 24 needs a layout revision (surface it then).
- **Tokenizers:** the v6 backbone path inherits a natively multilingual
  tokenizer/LM (Qwen3 covers all 12 tags incl. Turkish — the reason it is the
  recommended backbone). The from-scratch path uses the 48k byte-BPE
  (DESIGN_V4 §3) — byte fallback makes every script *representable*; the 48k
  merge budget with ≥~20% Chinese in the tokenizer corpus keeps CJK at
  ~1 token/char so the 12.5 tokens/s monologue budget holds. ByteTokenizer
  (tests) handles all scripts by construction (UTF-8 bytes).
- **The model learns language identification for free:** `<lang_XX>` sits in
  the assistant monologue prefix (`turn_prefix`), so in s2s the model
  *predicts* the tag from user audio (language ID) unless the caller forces
  it (`chat --lang`, serve dropdown, `prefix_ids=turn_prefix(lang=...)`) —
  same auto-predict/forceable mechanics as the emotion tags.
- **Cross-lingual voice cloning** is a data recipe, not a code path:
  DESIGN_V5 pairs reference/target across languages (50% ref_lang != tgt_lang)
  over the same `<voice>` segment.

## Data preparation per source type

- **Text LM** (per-language corpora): one dir per language,
  `prepare_data.py textlm --lang XX` tags every row.
- **ASR/TTS single-language corpora** (LibriSpeech, MLS configs):
  `--lang XX`.
- **ASR/TTS multilingual corpora** (Common Voice, FLEURS, VoxPopuli):
  `--lang-column COL` reads the language per row (normalized by
  `prepare._lang_label`: "en-US"/"cmn"/"pt_BR"/"Turkish" all resolve; rows
  outside the 12-tag inventory are skipped with a summary warning).
- **Dialogue s2s**: per-turn `"lang"` keys on the dialogue dicts (any locale
  spelling; normalized). The turn's language reaches BOTH the `<lang_XX>`
  monologue tag and the TTS backend (`synth(lang=)`).
- **Duplex**: per-event or per-dialogue `"lang"` keys, threaded to synthesis.
- **Response-language=mirror** (DESIGN_V4 stage C): give each dialogue turn
  the user's language as its `lang` — the tag teaches reply-in-kind.

## TTS backends (synthetic data)

`TTSBackend.synth(text, voice, style=, lang=)` — all backends accept `lang`:

| backend | language behavior |
|---|---|
| sine (CI) | deterministic per-language shift (a fake "accent"); `lang=None` bit-identical to before |
| vibevoice | En/Zh from the text itself (no control input) |
| cosyvoice3 | `lang` joins the instruct prompt (natively multilingual) |
| soulx | Zh/En from text |
| chatterbox | `language_id` per call — the Tier-2 engine (23 langs incl. Turkish) |

Alignment: `--align whisper` now defaults to the MULTILINGUAL
`openai/whisper-tiny` (the old `.en` default silently broke 11 languages);
`build_aligner("whisper", lang=XX)` pins the decode language for
single-language corpora, None auto-detects per utterance. `--align uniform`
is language-agnostic.

## Tokenizer training (from-scratch path only)

```bash
python scripts/train_tokenizer.py \
    --dataset HuggingFaceFW/fineweb-edu:sample-10BT@35 \
    --dataset uonlp/CulturaX:zh@20 --dataset uonlp/CulturaX:fr@10 \
    --dataset uonlp/CulturaX:de@10 --dataset uonlp/CulturaX:tr@10 \
    --dataset uonlp/CulturaX:es@5  --dataset uonlp/CulturaX:it@10 \
    --field text --max-docs 2000000 --vocab-size 48000 \
    --out data/tokenizer/omni_bpe_48k.json
```

Repeatable `--dataset ...@WEIGHT` interleaves deterministically by weight
(the DESIGN_V4 En35/Zh20/... mix). After training, verify
`tok.encode("<lang_zh>")[0] == 38` and a CJK round-trip (pinned offline by
tests/test_multilingual.py on a toy corpus). The v6 backbone path skips all
of this — `--tokenizer hf:Qwen/Qwen3-1.7B-Base` (shard format v2 handles the
150k+ vocab).

## Training mixture

One shard dir per (language, task) slice; weights are sampling PROPORTIONS
(post-fix MixDataset — shuffle-proof):

```bash
torchrun ... scripts/train.py --preset qwen3-1.7b \
    --data shards/asr-en:0.20 --data shards/asr-zh:0.10 --data shards/asr-tr:0.05 \
    --data shards/tts-en:0.15 --data shards/tts-zh:0.10 --data shards/tts-tr:0.05 \
    --data shards/s2s-mixed:0.20 --data shards/textlm-multi:0.15 [...]
```

Language proportions are therefore exact and independent of corpus sizes —
small languages (tr) oversample to their share instead of drowning. Mixed
dirs must share one tokenizer (`tokenizer_id` is cross-checked).

## Verification

- Offline: tests/test_multilingual.py (normalization, non-ASCII round-trips
  through every tokenizer/grid/model layer, tag rotation in fake data,
  language-tagged batches reaching a real forward/loss, mixture proportions,
  forced-tag generation). Every `prepare_data.py fake` smoke now rotates
  None/en/tr/zh with real non-ASCII text — multilingual coverage is the
  default, not an option.
- GPU day: Gate-0 (Mimi resynthesis ceiling) PER LANGUAGE before committing
  to a language tier (DESIGN_V4 §3 — Mimi's training data skews English;
  measure zh/ja/tr explicitly). Then per-language ASR/TTS evals from the
  DESIGN_V3 eval-battery queue item.

## Inference

`chat.py --lang XX` / serve's lang dropdown / `turn_prefix(lang=...)` force
the reply language; omitting them lets the model predict the tag (reply in
the user's language, if the mirror recipe trained it). ASR of any of the 12
languages needs no flag — the tag is part of the transcript prefix the model
emits.
