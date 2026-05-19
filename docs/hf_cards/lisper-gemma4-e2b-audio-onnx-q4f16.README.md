---
library_name: transformers.js
base_model: google/gemma-4-E2B-it
license: gemma
tags:
- gemma-4
- audio
- onnx
- webgpu
- q4f16
- unsloth
- speech
- lisper
---

# Lisper Gemma 4 E2B Audio ONNX q4f16

This is the browser/WebGPU q4f16 package for the Lisper Gemma 4 E2B audio fine-tune.

## Component Layout

- `onnx/embed_tokens_q4f16.onnx`: trained Lisper embedding component.
- `onnx/decoder_model_merged_q4f16.onnx`: trained Lisper decoder component.
- `onnx/audio_encoder_q4f16.onnx`: Gemma 4 E2B q4f16 audio encoder component.
- `onnx/vision_encoder_q4f16.onnx`: Gemma 4 E2B q4f16 vision encoder component.

The LoRA training targeted language/text modules; audio modules were not targeted, and vision fine-tuning was disabled. Reusing the official audio/vision components keeps the browser package compatible with the public Gemma 4 E2B ONNX runtime contract while using the trained Lisper text stack.

## Evaluation

The release-quality evaluation result belongs to the full v18 hybrid acoustic+Gemma pipeline, not to a separate browser-only q4f16 eval:

- Held-out rows: `2,000`
- Hard errors: `0`
- Verdict: `pass`
- Class match: `0.976`
- Clear/non-clear match: `0.989`
- Exact four-line format: `1.0`

This q4f16 package is the browser demo artifact for consumer-device testing.

## App Use

Configure the app with:

```bash
VITE_LISPER_BROWSER_MODEL_ID=thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16
VITE_LISPER_BROWSER_DTYPE=q4f16
```

Expected required browser payload is about `3.15 GB`. Keep `q4f16` as the primary browser dtype for the hackathon package.

## Companion Artifacts

- LoRA adapter: `thomasjvu/lisper-gemma4-e2b-audio-lora`
- Merged full checkpoint: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Server-side ZeroGPU fallback: `thomasjvu/lisper-zerogpu`
