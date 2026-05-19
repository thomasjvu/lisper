---
title: Lisper ZeroGPU
emoji: 🐍
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
hardware: zerogpu
short_description: Raw-audio lisp coaching on ZeroGPU.
---

# Lisper ZeroGPU

This Space is the server-side companion to the browser WebGPU demo.

- Browser demo target: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- ZeroGPU default target: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Purpose: provide a reliable server-side fallback for users whose browser or laptop cannot comfortably run the q4f16 ONNX model. The live path uses the v18 acoustic gate by default; server-side Gemma generation is optional and disabled until endpoint stability is verified.

## Important Runtime Notes

ZeroGPU is for inference demos, not training. For hackathon submission, this Space should stay on the validated Gemma 4 E2B story. Larger Gemma variants are post-submission experiments and should not be linked as the submitted app.

Hardware note: this Space still requires Hugging Face ZeroGPU access on the owning account. If hardware remains `cpu-basic`, enable PRO/ZeroGPU access or select another GPU runtime before using it for a demo.

Status: this Space has been smoke-tested on `zero-a10g` with the fine-tuned E2B full model. Live analysis defaults to the v18 acoustic gate plus template coaching because the displayed class is anchored to that sidecar; optional Gemma generation can be re-enabled with `LISPER_ZERO_GPU_USE_GEMMA_GENERATION=1` after endpoint stability is verified.

Input handling:

- The app rejects silent, empty, too-short, or very low-energy recordings before calling Gemma. This prevents confident but falsified coaching on empty microphone captures.
- The primary recorder bypasses Gradio's microphone component and captures raw microphone PCM through the browser Web Audio API, then encodes a small WAV payload client-side. This avoids the blank/silent recordings seen from Gradio's built-in microphone recorder on some browser/device combinations.
- The Gradio `Audio` component is kept as upload-only fallback. If the browser recorder reports a near-zero peak/RMS, the issue is browser permission/input capture before the backend sees the file.
- Live classifications are gated before Gemma generation. The Space can return `rejected_audio` or `inconclusive` instead of forcing `clear`, `palatal`, or another class when the clip is silent, noisy, missing usable /s/ evidence, or classifier confidence is weak.
- Live analysis prefers the v18 ExtraTrees acoustic hint artifact in `acoustic_extratrees_v18.joblib`. In `auto` mode, a narrow KNN fallback can override only when the clip is extremely close to a known synthetic non-clear exemplar. If the acoustic artifact is missing, the app reports analysis unavailable instead of letting Gemma freely guess the class.

Set these Space variables/secrets:

- `LISPER_ZERO_GPU_MODEL_ID`: model repo to load. Defaults to `thomasjvu/lisper-gemma4-e2b-audio-full`.
- `LISPER_ZERO_GPU_ADAPTER_ID`: optional PEFT/LoRA adapter repo to load on top of `LISPER_ZERO_GPU_MODEL_ID`. Leave empty for merged full-model repos.
- `LISPER_ZERO_GPU_DTYPE`: `float16`, `bfloat16`, or `float32`. Defaults to `float16`.
- `LISPER_ZERO_GPU_AUDIO_DTYPE`: optional override for Gemma audio features. Adapter deployments default to `bfloat16`.
- `LISPER_ZERO_GPU_LOAD_IN_4BIT`: defaults to `1` when `LISPER_ZERO_GPU_ADAPTER_ID` is set, otherwise `0`.
- `LISPER_ZERO_GPU_ACOUSTIC_HINT`: defaults to `1`. Set to `0` only when intentionally testing direct Gemma audio classification.
- `LISPER_ZERO_GPU_ACOUSTIC_MODEL`: `auto`, `extratrees`, or `knn`. Defaults to `auto`, which uses v18 ExtraTrees when `acoustic_extratrees_v18.joblib` is present and allows only distance-gated KNN synthetic-exemplar overrides.
- `LISPER_ZERO_GPU_KNN_OVERRIDE_MAX_DISTANCE`: defaults to `0.25`. Lower values are safer; higher values make the KNN synthetic-exemplar override more aggressive.
- `LISPER_ZERO_GPU_KNN_OVERRIDE_MIN_CONFIDENCE`: defaults to `0.90`.
- `LISPER_ZERO_GPU_LIVE_CLEAR_MIN_CONFIDENCE`: defaults to `0.85`.
- `LISPER_ZERO_GPU_LIVE_NONCLEAR_MIN_CONFIDENCE`: defaults to `0.55`.
- `LISPER_ZERO_GPU_MIN_SIBILANT_FRAME_RATIO`: defaults to `0.015`; increase this to reject more clips without enough /s/ or /z/ evidence.
- `LISPER_ZERO_GPU_ALIGN_AUDIO_TOKENS`: defaults to `0` for adapter deployments and `1` for merged-model deployments.
- `LISPER_ZERO_GPU_MAX_SEQ_LENGTH`: defaults to `2048`.
- `LISPER_ZERO_GPU_SIZE`: `large` or `xlarge`. Defaults to `large`.
- `LISPER_ZERO_GPU_MAX_NEW_TOKENS`: defaults to `96`.
- `LISPER_ZERO_GPU_EAGER_LOAD`: defaults to `0`. Keep model loading inside the Analyze GPU call so the public page and recorder stay responsive.
- `LISPER_ZERO_GPU_USE_GEMMA_GENERATION`: defaults to `0`. Keep disabled for the reliable live demo path; enable only when intentionally testing server-side Gemma generation.
- `HF_TOKEN`: required if the selected model repo is private or gated.

## Model Lineage

The currently validated Lisper fine-tune is Gemma 4 E2B. The E4B LoRA run completed training, but the Kaggle full-model merge failed due local disk pressure after training and it has not passed the same held-out eval/publish-verdict gate. Keep E4B out of the hackathon submission path.

For hackathon claims, keep this distinction precise:

- The browser artifact is the E2B q4f16 ONNX/WebGPU package.
- The quality gate was the v18 hybrid acoustic+Gemma evaluation path.
- This ZeroGPU Space is the server fallback for the same E2B submission story, not a separate submitted model.
