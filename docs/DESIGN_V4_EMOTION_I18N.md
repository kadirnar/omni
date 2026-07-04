# Omni-S2S v4: Emotion + Multilingual Amendment

> Produced 2026-07-02 from a 3-agent research pass (raw reports: docs/research/emotion-*.md, multilingual.md). Amends DESIGN.md §1/§4/§7 and DESIGN_V3_AUDIO.md §3/§4. Frozen-core touch: additive streams.SPECIAL_TOKENS names at ids 11..48 + text_vocab_size=48000 on quality/small presets; grids.py untouched (all new tokens ride text_ids).

# Omni-S2S v4 amendment: emotion perception→response + multilingual

> Amends DESIGN.md (§1 special tokens, §4 stages, §7 tokenizer) and DESIGN_V3_AUDIO.md (§3 data, §4 tags). Frozen-core touch: `streams.SPECIAL_TOKENS` (additive names at ids 11..48) + one preset field in `config.py`; `grids.py` stays untouched by threading all new tokens through existing `text_ids`.

## 1. Emotion mechanism — discrete tags in the monologue + auxiliary SER (mechanism a+d)

Implicit-only systems are demonstrably tone-deaf (ParaS2SBench ~3/5 incl. GPT-4o); explicit perceive-then-respond gives the only measured gains (OSUM-EChat ablations; EACL-2026 multi-task RL 65→77%; ParaS2S SFT 2.89→4.08). We use **language-neutral special tokens, not free-text captions**: captions burn our 4–6 tok/s text-ahead budget and are English-biased, whereas tags transfer across languages and cost ~5 frames one-time.

**Grid placement (s2s assistant turn), all threaded through the assistant `text_ids` — no grid-builder change:**
```
[user audio, ch=USER] <end_of_turn> <assistant> <lang_tr> <emo_pcv> <angry> <emo_rsp> <calm> w1 w2 … <end_of_turn> <text_pad>…
```
`<assistant>` is injected (not a loss target); `<lang_XX> <emo_pcv> PCV <emo_rsp> RSP w1…` are all text-stream loss targets. The model has consumed all user audio before emitting `<emo_pcv> PCV`, so this is ParalinGPT ordering (current-sentiment → response-sentiment → text). Because text leads audio, the assistant audio (depth transformer) sees `<emo_rsp> RSP` in-context and realizes it (EMOVA-style, in-band). Optional `<intensity_*>` after `<emo_rsp> RSP` only in the expressive-SFT subset.

**Taxonomy — one shared 12-class inventory** (both fields draw from it; generation-side self-consistency lets a shared set serve both). SER-trained/eval on the coarse 8: neutral, happy, sad, angry, surprised, fearful, disgusted, sarcastic (7-class MELD SOTA is only ~59%; EACL collapsed to 3-class for cross-corpus robustness — we cap perception at 8). Response adds calm, empathetic, excited, serious (12 total). 16+ perceived classes: no system learns them reliably; 40-class (EmoNet) rejected. Plus 3 intensity tokens. Response styles stay self-consistent because we synthesize them with known labels.

**Auxiliary SER task:** tag `<ser>` (id 11), realized in v1 as **SER-tagged ASR** — prepend `<emo_pcv> PCV` to the ASR transcript `text_ids` and reuse `build_asr` (keeps grids.py frozen). Mixture share **~4% of Stage-B understanding tokens** (OSUM: SER rides the big understanding stage; only ~200 h empathetic dialogue needed afterward). A dedicated emotion-only `build_ser` is an optional additive follow-up.

**Inference behavior:** perceived emotion = **AUTO** (sampled from the text head; the model's SER read of user audio; logged/evaluated). Response emotion = **AUTO by default, user-forceable** (pass `<emo_rsp> RSP` as forced_text, like TTS text forcing) — this satisfies "if the input voice is angry, respond accordingly" out of the box while allowing override. Optional constrained decode: after a marker, mask text logits to the valid class block.

## 2. Emotion data pipeline

**Text scripting:** seed from `allenai/soda` (CC-BY-4.0; use its `xReact` emotional-reaction field) and regenerate into (user-emotion, assistant-strategy) dialogues — angry/frustrated/anxious/sad/excited user + empathetic/calm/warm assistant — emitting per-turn `{user_emotion, response_style, lang}` labels that map to our token ids. NC sets (DailyDialog/EmpatheticDialogues/ESConv) are eval-only.

**Audio synthesis (all Apache/MIT):**
- *User side:* `stepfun-ai/Step-Audio-EditX` bracket-tag editing (`[Angry]`, `[Whispering]`) of neutral VibeVoice/CosyVoice3 output → **minimal pairs** (same text/voice, ±emotion); plus `Fun-CosyVoice3-0.5B-2512` instruct — natural-language prefix ending `<|endofprompt|>` (`"You are a helpful assistant. Please speak very angrily.<|endofprompt|>"`) over thousands of Emilia-cloned voices.
- *Multilingual:* `ResembleAI/chatterbox` (MIT, 23 langs incl. **Turkish**, emotion-exaggeration param); `maya-research/maya1` (English inline `<laugh>`/`angry` tags).
- *Assistant side:* CosyVoice3/Step-Audio-EditX Comforting/Warm/Empathetic on a small consistent assistant-voice pool.
- Explicitly **exclude IndexTTS-2** (bilibili license bans training other models), F5/Voxtral (CC-BY-NC).

**Scale:** committed minimum **~300–500 h emotion-labeled S2S** (OSUM-EChat reached its gains at ~200 h) as ~10–15% of Stage-C S2S; scale target ~1,500 h (~500k exchanges) if budget allows.

**QC:** ensemble SER judge (`emotion2vec+` + Empathic-Insight-Voice) keep-top-confidence; whisper-large-v3 WER round-trip (reject >5%); speaker-sim gate. **Calibrate judges on CREMA-D** because SER is unreliable on synthetic audio.

**Real corpora for perception (critical — synthetic SER does NOT transfer to real voices, arXiv 2603.16483):** CREMA-D (ODbL, commercial-OK, EN, 6 emotions), JVNV (CC-BY-SA, JA), MSP-Podcast (sign license; 400 h naturalistic EN — largest, not HF-hosted → schedule the sign-off early), plus SER-pseudo-labeled Emilia-YODAS. Feed these as SER-tagged ASR grids so perception is learned from **real** voices. Avoid MELD/IEMOCAP/ESD/RAVDESS for training (copyright/NC).

## 3. Multilingual plan

**v1 set:** Tier-1 **En, Fr, De**; Tier-1-gated **Zh** (codec risk, §below); Tier-2 **Tr** (understand+respond, accept lower voice naturalness). **Ja/Ko/Es deferred** but token-reserved.

**Turkish viability (user is likely Turkish):** viable but thin. Text is ample — fineweb-2 tr (41.9B words, ODC-By), aya (Apache, has Turkish), oasst (small). Audio: CommonVoice tr ~130 h (CC0) + YODAS2-tr (low hundreds h, noisy). **CosyVoice3/VibeVoice cannot synthesize Turkish** → the assistant-voice path is **Chatterbox Multilingual** (MIT, Turkish + emotion). Plan ~500–1,000 h total; accept Tier-2 MOS.

**Per-language hours (commercially clean):** En ~137k (Emilia-YODAS 92k + MLS 44.5k), Fr ~8.5k, De ~7.6k, Ko 7.3k (if wanted). Zh is license-constrained (~0.3k CC-BY + CommonVoice CC0) → cover with VibeVoice/CosyVoice3 synthesis; tolerate CC-BY-NC Emilia-Zh only if NC acceptable.

**Tokenizer revision:** train a **48k byte-BPE** (up from 32k), specials still pinned at 0..63. Mix ≈ En 35 / Zh 20 / Fr,De,Tr 10 each / Es 5 / code+misc 10 (fineweb-2 + ASR transcripts). **CJK must be in-vocab** (~6–8k chars): with UTF-8 byte fallback Mandarin ≈ 13.5 tok/s > the 12.5 Hz budget; in-vocab ≈ 3.5–4.5. Per-language monologue rates all clear 12.5: En 3.5–4.4, Fr/De 4–5.5, **Tr 4–5** (Latin, byte-fallback harmless), Zh 4–4.5. 64k only if Ja/Ko enter scope (byte-fallback kana/hangul also blows the budget → deferred with those languages). Embedding cost at d1536: 48k → ~74M (acceptable); shard uint16 limit (≤65536) safe.

**Language tags + policy:** Whisper convention — `<lang_XX>` at assistant turn start. Train both **forced (locked)** and **model-predicted (auto/LID)**. Default response-language policy = **mirror the detected user language**; forceable to pin output.

**Mimi per-language risk + Gate-0:** Mandarin is the expected failure — DualCodec shows Mimi's semantic cb0 (WavLM/English) lacks tone. Gate-0 (1 day, CPU-OK): Mimi encode→decode FLEURS + CommonVoice per language at 8/16/32 codebooks; measure ΔWER (whisper-large-v3), DistillMOS, and a Mandarin tone minimal-pair probe. **Pass: ΔWER <5 pts abs at 32 cb.** If Zh fails, drop it to Tier-2 — **never swap codecs** (every alternative breaks the 12.5 Hz grid).

**Stage mixture changes:** Stage A → multilingual text mix (drives tokenizer + text pretrain). Stage B → make ASR/TTS multilingual + carve **~4% SER-tagged ASR** on real SER corpora; keep TTS 30/ASR 20/audio-LM 20/text-replay 30 shape. Stage C → within S2S 60%, ~10–15% emotion-labeled empathetic dialogues + multilingual S2S with response-language=mirror; TTS 15/ASR 10/text-SFT 15 unchanged.

## 4. Evaluation additions
- **SER accuracy:** IEMOCAP 4-class + MELD 7-class (research eval; expect ~55–60% ceiling) + CREMA-D, scored on the `<ser>`/SER-tagged-ASR head.
- **Emotion-appropriateness:** ParaS2SBench (emotion 6-way + sarcasm, judge content+style 1–5), SD-Eval, StepEval-Audio-Paralinguistic, EChat-eval; judge = GPT-4o-Audio or an open LLM judge scoring perceived-emotion correctness + response-tone appropriateness (EACL recipe).
- **Per-language gates:** extend v3 Gate B/C/D with per-language whisper-large-v3 WER + DistillMOS breakdown; Turkish accepts a lower MOS floor; Gate-0 per-language resynthesis (ΔWER<5) precedes training.

## 5. Reserved-token fit
All new tokens are named specials at ids **11..48** (38 ids), leaving **49..63 reserved (15)**. They ride through existing `text_ids` so `grids.py`/`apply_delay` are untouched; only `streams.SPECIAL_TOKENS` gains names (numeric ids 0..10 unchanged → all 62 existing tests still pass). Everything fits; nothing must move to plain BPE text.

# Reserved special-token allocation (binding)

Free ids = 11..63 (53). Allocation uses 11..48 (38); 49..63 (15) stay reserved. All added to streams.SPECIAL_TOKENS as named specials; numeric ids 0..10 unchanged.

TASK TAG (1):
  11 <ser>            (speech-emotion / paralinguistic understanding; v1 realized via SER-tagged ASR)

EMOTION MARKERS (2):
  12 <emo_pcv>        (perceived-emotion field marker; auto-predicted from user audio)
  13 <emo_rsp>        (response-style field marker; auto, user-forceable)

INTENSITY (3, used only in the expressive-SFT subset):
  14 <intensity_lo>   15 <intensity_md>   16 <intensity_hi>

EMOTION/STYLE CLASSES (12, shared by both fields; SER trained/eval on the first 8):
  17 <neutral>  18 <happy>  19 <sad>  20 <angry>  21 <surprised>
  22 <fearful>  23 <disgusted>  24 <sarcastic>
  25 <calm>  26 <empathetic>  27 <excited>  28 <serious>

PARALINGUISTICS (8, DESIGN_V3 §4; SoulX/Orpheus/Maya1 tag-rich synthesis; loss-active on text, realized as nonverbal audio):
  29 <laugh>  30 <sigh>  31 <chuckle>  32 <gasp>  33 <cough>  34 <breath>  35 <sniffle>  36 <yawn>

LANGUAGE TAGS (12, Whisper-style, turn-start; v1 uses en/fr/de/zh/tr):
  37 <lang_en>  38 <lang_zh>  39 <lang_fr>  40 <lang_de>  41 <lang_es>  42 <lang_ja>
  43 <lang_ko>  44 <lang_tr>  45 <lang_ru>  46 <lang_it>  47 <lang_pt>  48 <lang_nl>

STILL RESERVED (15): 49..63  -> future languages, extra emotion classes, or new markers.

FITS: 38 used + 15 reserved = 53. Nothing moves to plain BPE text. (Paralinguistic tags must be registered specials so tokenizer.encode('<laugh>') maps to id 29 rather than byte-encoding; verify HF-tokenizers matches AddedTokens under add_special_tokens=False.)
# Decisions

## Emotion mechanism
CHOICE: Discrete perceived+response emotion special tokens in the inner monologue plus an auxiliary SER task (mechanism a+d); placement '<assistant> <lang_XX> <emo_pcv> PCV <emo_rsp> RSP w1 w2...' threaded through assistant text_ids
RATIONALE: Implicit-only S2S is tone-deaf (~3/5 on ParaS2SBench incl. GPT-4o); explicit perceive-then-respond gives the only measured gains (OSUM-EChat ablations, EACL-2026 RL 65->77%, ParaS2S 2.89->4.08). Language-neutral tags transfer across languages and cost ~5 frames one-time vs a full CoT paragraph.
ALT: Free-text emotional CoT/captions (EmoOmni): works but burns the 4-6 tok/s text-ahead budget and is English-biased. Implicit-only (Moshi/Qwen-Omni): demonstrably fails.

## Emotion taxonomy
CHOICE: One shared 12-class inventory (neutral/happy/sad/angry/surprised/fearful/disgusted/sarcastic + calm/empathetic/excited/serious) used by both fields; SER trained/evaluated on the coarse 8; 3 intensity tokens; markers <emo_pcv>/<emo_rsp>
RATIONALE: 7-class MELD SOTA is only ~59% and fine-grained taxonomies fail cross-corpus (EACL collapsed to 3-class); response side is self-consistent because we synthesize with known labels; a shared inventory + markers costs 17 ids vs 20+ for namespaced tokens.
ALT: EMOVA 24 style-combos or EmoNet 40 fine-grained classes: rejected as non-generalizable at perception; 16+ perceived classes: no system learns them.

## Auxiliary SER task
CHOICE: Task tag <ser> (id 11), realized in v1 as SER-tagged ASR (prepend <emo_pcv> PCV to the ASR transcript, reuse build_asr); ~4% of Stage-B understanding tokens; grounded on REAL SER audio (CREMA-D/JVNV/MSP-Podcast + pseudo-labeled Emilia)
RATIONALE: OSUM-EChat: SER rides the big understanding stage then only ~200 h empathetic dialogue is needed; reusing build_asr keeps grids.py frozen; real audio is mandatory because SER does not transfer from synthetic to real voices (arXiv 2603.16483).
ALT: Synthetic-only SER (fails to transfer); implicit understanding (no forced supervision to read the acoustic evidence in cb1-31); dedicated build_ser (deferred additive follow-up).

## Inference emotion + language behavior
CHOICE: Perceived emotion AUTO (sampled, logged); response emotion AUTO by default but user-forceable via forced_text; language AUTO=mirror user language (Whisper-style LID) but lockable
RATIONALE: Auto response matches the user requirement ('angry input -> respond accordingly') with zero user action, while forcing reuses the existing TTS forced_text path; mirror-language is the natural conversational default and both forced/predicted train from one Whisper convention.
ALT: Always user-forced emotion (defeats the perception goal); fixed output language (breaks multilingual dialogue).

## v1 language set
CHOICE: Tier-1 En/Fr/De, Tier-1-gated Zh, Tier-2 Tr (understand+respond, lower voice naturalness); Ja/Ko/Es deferred but token-reserved
RATIONALE: Commercially-clean hours exist for En/Fr/De/Ko; Turkish text is ample (fineweb-2 41.9B words) and Chatterbox (MIT) can voice Turkish with emotion; Zh is both license-constrained and the Mimi tone-risk case.
ALT: Ja/Ko in v1 (would force a 64k tokenizer and byte-fallback rate risk); English-only (ignores the likely-Turkish user).

## Tokenizer revision
CHOICE: 48k byte-BPE with CJK (~6-8k chars) forced in-vocab; multilingual training mix (En35/Zh20/Fr,De,Tr10/Es5/misc10)
RATIONALE: Mandarin with byte fallback is ~13.5 tok/s > the 12.5 Hz budget; in-vocab CJK gives ~3.5-4.5; all target languages incl. Turkish stay under 12.5; +~24M embedding params at d1536 is acceptable and fits uint16 shards.
ALT: Keep 32k (English-biased, breaks Mandarin rate); 64k (only justified if Ja/Ko enter scope).

## Reserved-token layout
CHOICE: 38 named specials at ids 11..48 threaded through text_ids; 49..63 stay reserved; only streams.SPECIAL_TOKENS gains names
RATIONALE: Grid builders place any text ids, so emotion/lang/paralinguistic tokens need no grids.py change; numeric ids 0..10 are unchanged so the 62 existing tests pass; everything fits with 15 ids of headroom.
ALT: Fused namespaced tokens like <user_emo:angry>/<style:calm> (2x id cost, ~20 ids just for emotion); moving tags to plain BPE text (loses language-neutrality and pinned ids).

## Mimi Mandarin risk handling
CHOICE: Keep Mimi-32; Gate-0 per-language resynthesis (ΔWER<5 at 32 cb); if Zh fails, demote to Tier-2, never swap codec
RATIONALE: DualCodec shows Mimi's semantic cb0 lacks tone; but 32 acoustic codebooks can carry tone and every alternative codec breaks the 12.5 Hz frame grid that the whole stack depends on.
ALT: Swap to XCodec2/NanoCodec (25/50 Hz, destroys the frame grid and all trained weights).

## Emotion data scale
CHOICE: Committed minimum ~300-500 h emotion-labeled S2S (~10-15% of Stage C); scale target ~1,500 h
RATIONALE: OSUM-EChat reached its empathy gains at ~200 h and ParaS2S matched SFT with 1/5 the annotations via RL; over-committing hours is wasteful before the mechanism is validated.
ALT: Report-2's flat 1,500-2,500 h (deferred as a scale target, not the entry bar).

# Implementation queue (ordered)

1. 1. src/omni/streams.py (FROZEN — additive, needs INTERFACES.md version bump): extend SPECIAL_TOKENS with names at ids 11..48 (see reserved_token_plan); add TASK_SER + 'ser' to TASK_TAGS; add lookup dicts EMOTION_CLASSES, LANG_TAGS, PARALING_TAGS, INTENSITY and a helper turn_prefix(lang,pcv,rsp,intensity)->list[int]. Numeric ids 0..10 unchanged so the 62 existing tests pass; RESERVED_SPECIAL_FORMAT fillers now only cover 49..63.
2. 2. src/omni/text/tokenizer.py: no code change (special_token_strings() auto-reads SPECIAL_TOKENS). Retrain the BPE with the multilingual mix at vocab 48k; verify literal '<laugh>' etc. encode to their special ids.
3. 3. scripts/train_tokenizer.py: accept a multi-source / pre-mixed corpus and --vocab-size 48000; document CJK-coverage requirement (mix >=~20% Chinese so common characters land in-vocab).
4. 4. src/omni/config.py (FROZEN — additive): set text_vocab_size=48000 on the quality (and small) presets used for multilingual runs; keep the 32768 default and the ByteTokenizer=320 CI override untouched. Optional: a comment noting emotion/SER mixture is set via --data weights, not a new field.
5. 5. src/omni/data/synthesize.py: add StepAudioEditXTTS (bracket-tag [Angry] editing), ChatterboxTTS (MIT, 23 langs incl. Turkish, exaggeration), Maya1TTS (EN inline tags) behind TTSBackend; add emotion/instruct kwargs to CosyVoice3TTS.synth ('...<|endofprompt|>'); extend the dialogue schema to carry per-turn {user_emotion, response_style, lang}; add load_soda_emotional() reading xReact; register backends in build_tts.
6. 6. src/omni/data/prepare.py: thread turn_prefix ids into the text_ids passed to build_asr/build_tts/build_s2s (prepend <lang_XX>, <emo_pcv> PCV, <emo_rsp> RSP) — NO grids.py change since builders place any text ids; add SER-tagged ASR path (prepend <emo_pcv> PCV to transcript, ~4% Stage-B); add language-tag threading to prepare_textlm/prepare_asr_tts. Shard format unchanged (ids<64, uint16).
7. 7. grids.py (FROZEN): intentionally UNTOUCHED — all new tokens ride in text_ids. Flag: a dedicated emotion-only build_ser would be additive but is deferred to keep the freeze.
8. 8. src/omni/infer/generate.py + chat.py: allow forced_text to carry locked <lang_XX>/<emo_rsp> RSP (auto-sampled otherwise); optional constrained decode masking text logits to the class/lang block after a marker; parse the emitted <emo_pcv> PCV for logging; add chat.py --emotion/--lang flags.
9. 9. scripts/synthesize_data.py: emotion QC pipeline — ensemble SER judge (emotion2vec+ + Empathic-Insight-Voice) top-confidence keep, whisper WER round-trip, speaker-sim gate, judges calibrated on CREMA-D; pass per-turn labels through.
10. 10. NEW src/omni/eval/ additions (extends v3 eval): SER accuracy scorer (IEMOCAP/MELD/CREMA-D), emotion-appropriateness harness (ParaS2SBench/SD-Eval/StepEval LLM-judge), per-language WER/MOS breakdown, Gate-0 per-language Mimi-32 resynthesis (ΔWER<5).
11. 11. (Later, behind Gate C) src/omni/train/dpo.py: add SER-correctness + LLM-judged tone-appropriateness terms to the reward (EACL recipe, ParaS2S GRPO).
# Risks

- Mandarin Mimi cb0 lacks tone (DualCodec: excessive Chinese WER); mitigation is Gate-0 per-language (ΔWER<5 at 32 cb) and demoting Zh to Tier-2 — NOT a codec swap, which would break the 12.5 Hz grid and invalidate the whole stack.
- SER does not transfer from synthetic to real voices (arXiv 2603.16483): perception training MUST include real SER audio. CREMA-D (ODbL) is clean, but MSP-Podcast (the largest naturalistic set) is not HF-hosted and needs an institutional sign-off — a scheduling blocker; JVNV covers only Japanese.
- HF tokenizers must match registered special strings ('<laugh>' etc.) as AddedTokens under add_special_tokens=False; if it byte-encodes them instead, paralinguistic tags leak into BPE text — verify on the retrained 48k tokenizer before mass synthesis.
- Turkish data is thin (~130 h CommonVoice + noisy YODAS2) and its only emotion-capable voice is Chatterbox (MIT, but watermarked); expect lower Turkish MOS — accept Tier-2 and gate it separately.
- Text-lead budget: up to 7 special tokens at turn start (~560 ms) consume the audio-ahead margin; monolingual/neutral turns should omit intensity and can omit <lang_en>; monitor first-audio timing.
- Fine-grained emotion is unreliable (MELD 59%): keep SER at 8 coarse classes; if even 8-way perception underperforms cross-corpus, fall back to a 3-class sentiment head (EACL precedent) while keeping richer response-side styles.
- streams.py is a frozen contract file; adding names requires an INTERFACES.md version bump and a full test pass. Additive (ids 0..10 unchanged) so existing tests should pass, but this is the one coordinated frozen-core edit.
- Emotion-appropriateness eval leans on a GPT-4o-Audio judge (external cost/availability); an open LLM judge is weaker and needs its own calibration.
- config.py text_vocab_size bump to 48k must stay a preset-level change: any run whose shards were prepared at 32k will trip prepare.py's tokenizer/config mismatch check — re-prepare or pin consistently.
- Report disagreement on emotion-data scale (300-500 h vs 1,500-2,500 h): committed to the OSUM-backed minimum with a scale target; if empathy quality stalls at 500 h, the cure is more data or a judge-reward GRPO pass, both already staged.