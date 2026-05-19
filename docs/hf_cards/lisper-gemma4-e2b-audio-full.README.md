---
library_name: transformers
base_model: google/gemma-4-E2B-it
license: gemma
tags:
- gemma-4
- audio
- merged
- unsloth
- speech
- lisper
---

# Lisper Gemma 4 E2B Audio Full Checkpoint

This is the merged Lisper checkpoint: `google/gemma-4-E2B-it` with the trained Lisper LoRA adapter folded into a standalone `safetensors` model.

## Model Lineage

- Base model: `google/gemma-4-E2B-it`
- Training: Unsloth supervised fine-tuning with QLoRA / LoRA
- Merge method: base + LoRA adapter merged into a 16-bit checkpoint
- Weight file: `model.safetensors`
- Training rows: `16,000`
- Validation rows: `2,000`
- Held-out test rows: `2,000`

This is not a dense full-parameter fine-tune. It is a merged base+LoRA checkpoint for easier deployment.

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

## Deployment

- Browser/WebGPU demo: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Server-side ZeroGPU fallback: `thomasjvu/lisper-zerogpu`
- Adapter-only package: `thomasjvu/lisper-gemma4-e2b-audio-lora`

Use the q4f16 ONNX/WebGPU package for browser demos. Use this merged checkpoint as the server-side correctness reference.

## Limitations

- The lisp dataset is synthetically generated from speaker-disjoint source speech.
- This is a practice assistant, not a medical diagnosis tool or a replacement for a speech-language pathologist.
