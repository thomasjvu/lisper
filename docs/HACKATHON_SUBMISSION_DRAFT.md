# Lisper Hackathon Submission Draft

## One-Line Summary

Lisper is a lisp-focused speech practice coach that uses a fine-tuned Gemma 4 E2B model to turn short speech clips into structured, low-pressure pronunciation feedback.

## Short Description

Lisper helps users practice /s/ and related sounds without needing to start from a clinical workflow. A user records or uploads a short target sentence, and Lisper returns a compact coaching response:

```text
Detected class: clear | frontal | lateral | dental | palatal
Reason: one short explanation tied to airflow or tongue placement
Corrective cue: one practical next step
Encouragement: one supportive sentence
```

The app rejects silent, empty, too-short, and very low-energy clips before analysis so it does not invent confident feedback from unusable audio.

## Problem

Most speech tools are broad articulation apps, clinician-facing systems, or generic speech recognition products. Lisp-specific practice is underserved, especially for users who want private, immediate feedback before working with a speech-language pathologist, caregiver, or teacher.

Lisper is not a diagnosis tool. It is a practice assistant that helps users notice patterns and try one correction at a time.

## Technical Approach

### Model Stack

- Base model: `google/gemma-4-E2B-it`
- Fine-tuning: Unsloth QLoRA / LoRA supervised fine-tuning
- Public LoRA adapter: `thomasjvu/lisper-gemma4-e2b-audio-lora`
- Public merged model: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Browser package: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Hosted fallback: `thomasjvu/lisper-zerogpu`

The model was not dense full-parameter fine-tuned. We trained LoRA adapter weights with Unsloth, then merged those adapter deltas into Gemma 4 E2B for deployment.

### Dataset

The training/evaluation dataset was built from LibriSpeech-style source speech with deterministic synthetic lisp transformations.

- Train rows: `16,000`
- Validation rows: `2,000`
- Held-out test rows: `2,000`
- Splits: speaker-disjoint train/validation/test
- Labels: `clear`, `frontal`, `lateral`, `dental`, `palatal`

Each row contains raw audio, prompt text, a lisp label, and a schema-locked coaching target.

### Evaluation

The release-quality evaluation uses the v18 hybrid acoustic+Gemma path.

- Held-out rows: `2,000`
- Hard errors: `0`
- Verdict: `pass`
- Class match: `0.976`
- Clear/non-clear match: `0.989`
- Exact four-line schema format: `1.0`
- Coaching fields present: `1.0`

This is intentionally described as hybrid evaluation: an ExtraTrees acoustic sidecar supplies the lisp-class hint, while Gemma produces the structured coaching explanation, corrective cue, and encouragement. This avoids overstating pure generative audio classification quality while still using Gemma as the user-facing coaching model.

### Deployment

Lisper has two demo paths:

- Browser/WebGPU: q4f16 ONNX package for local inference on compatible browsers.
- Hugging Face ZeroGPU: Gradio fallback using the merged Gemma 4 E2B model plus the v18 acoustic sidecar.

The browser model is the lightweight local demo artifact. The ZeroGPU Space is the public fallback for users whose browser or laptop cannot comfortably run the WebGPU model.

## Why Gemma 4

Gemma 4 was chosen because it supports multimodal raw-audio workflows while still offering an E2B variant small enough for hackathon iteration and browser deployment. We use Gemma as a coaching engine, not only as a labeler: the goal is actionable speech-practice feedback.

## Why Unsloth

Unsloth made fine-tuning feasible on free Kaggle GPUs. QLoRA reduced memory use enough to iterate under hackathon constraints while producing a real adapter that could be merged and packaged for deployment.

## What Worked

- Completed Gemma 4 E2B LoRA training with Unsloth.
- Built a speaker-disjoint synthetic lisp dataset.
- Passed the v18 held-out hybrid evaluation gate on 2,000 rows.
- Published LoRA, merged, and q4f16 ONNX artifacts.
- Built a browser/WebGPU demo path and a ZeroGPU fallback.
- Added audio validation so empty recordings are rejected instead of classified.

## Limitations

- The dataset uses synthetic lisp transformations, not clinically collected lisp recordings.
- The quality claim is for the v18 hybrid acoustic+Gemma path, not pure direct-Gemma audio classification.
- The browser q4f16 package is still large for consumer devices, around 3 GB.
- E4B and 31B are future targets; they are not part of this submitted release.
- Lisper is a practice assistant, not medical diagnosis or a replacement for a speech-language pathologist.

## Demo Script

1. Open the ZeroGPU Space or browser app.
2. Record or upload a short target sentence such as “Sally sells seashells.”
3. Show the recording-finalization state after pressing stop.
4. Submit the clip and show the four-line feedback response.
5. Submit an empty/silent recording and show that Lisper rejects it.
6. Explain that the release-quality path combines acoustic evidence with Gemma-generated coaching.

## Screenshots

- Desktop ZeroGPU Space: `docs/assets/screenshots/lisper-zerogpu-app.png`
- Mobile-size ZeroGPU Space: `docs/assets/screenshots/lisper-zerogpu-app-mobile.png`

## Submission Links

Primary links to share:

- GitHub: https://github.com/thomasjvu/lisper
- Live demo Space: https://huggingface.co/spaces/thomasjvu/lisper-zerogpu
- LoRA adapter: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-lora
- Merged model: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-full
- Browser q4f16 ONNX package: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16

Kaggle links:

- Training dataset: https://www.kaggle.com/datasets/thomasjvu/lisper-gemma4-audio
- Adapter artifact dataset: https://www.kaggle.com/datasets/thomasjvu/lisper-gemma4-audio-lora
- Unsloth training notebook: https://www.kaggle.com/code/thomasjvu/lisper-gemma-4-audio-unsloth-training
- Evaluation notebook: https://www.kaggle.com/code/thomasjvu/lisper-gemma-4-audio-adapter-eval

Optional supporting links:

- q4f16 eval summary: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16/blob/main/eval_summary.json
- q4f16 publish verdict: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16/blob/main/publish_verdict.json

Do not submit E4B or 31B links for this release. Those are future experiments and have not passed the same E2B gate.
