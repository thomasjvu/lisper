---
library_name: peft
base_model: google/gemma-4-E2B-it
license: gemma
tags:
- gemma-4
- audio
- lora
- unsloth
- speech
- lisper
---

# Lisper Gemma 4 E2B Audio LoRA

This is the Lisper Gemma 4 E2B LoRA adapter for raw-audio lisp coaching.

Lisper is a hackathon prototype for low-pressure /s/ practice. It classifies a short speech clip as `clear`, `frontal`, `lateral`, `dental`, or `palatal`, then returns one concise reason, one corrective cue, and one encouragement line.

## Model Lineage

- Base model: `google/gemma-4-E2B-it`
- Fine-tuning: Unsloth supervised fine-tuning with QLoRA / LoRA
- Trainable parameters: about `29.86M` of `5.15B`
- Training rows: `16,000`
- Validation rows: `2,000`
- Held-out test rows: `2,000`
- Training steps: `4,000`
- Selected checkpoint: `checkpoint-2500`

This is not a dense full-parameter fine-tune. The base model was frozen and the learned update is stored as LoRA adapter weights.

## Evaluation

The release-quality evaluation is the v18 hybrid acoustic+Gemma path:

- Held-out rows: `2,000`
- Hard errors: `0`
- Verdict: `pass`
- Class match: `0.976`
- Clear/non-clear match: `0.989`
- Exact four-line format: `1.0`
- Reason/cue/encouragement present: `1.0`

The evaluated pipeline uses acoustic features for the lisp-class hint and Gemma for structured coaching text and tone. Do not interpret these metrics as a pure direct-Gemma raw-audio classification result.

See `eval_summary.json` and `publish_verdict.json` for the public summary.

## Companion Artifacts

- Merged full checkpoint: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Browser q4f16 ONNX/WebGPU package: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Server-side demo Space: `thomasjvu/lisper-zerogpu`

## Limitations

- The lisp dataset is synthetically generated from speaker-disjoint source speech.
- This is a practice assistant, not a medical diagnosis tool or a replacement for a speech-language pathologist.
- The browser q4f16 package is large for consumer devices, so a ZeroGPU fallback is provided.
