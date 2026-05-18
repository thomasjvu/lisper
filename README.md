# Lisper

Lisper is a Gemma 4 speech-practice coach for lisp correction. It gives users a short, low-pressure response for /s/ practice:

```text
Detected class: clear|frontal|lateral|dental|palatal
Reason: one brief reason tied to tongue placement or airflow
Corrective cue: one concrete next-step cue
Encouragement: one brief supportive line
```

This is a practice assistant, not a medical diagnosis tool or a replacement for a speech-language pathologist.

## Hackathon Model Story

The submitted model family is Gemma 4 E2B fine-tuned with Unsloth LoRA.

- Base model: `google/gemma-4-E2B-it`
- LoRA adapter: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-lora
- Merged full checkpoint: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-full
- Browser q4f16 package: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16
- ZeroGPU fallback Space: https://huggingface.co/spaces/thomasjvu/lisper-zerogpu

E4B and q2f16 are experimental follow-up paths and are not part of the submitted model gate.

## Evaluation

The release-quality evaluation is the v18 hybrid acoustic+Gemma held-out path:

- Held-out rows: `2,000`
- Hard errors: `0`
- Class match: `0.976`
- Clear/non-clear match: `0.989`
- Exact four-line format: `1.0`

The lisp-class evidence comes from acoustic features; Gemma provides the structured coaching response and tone. These metrics should not be described as a pure direct-Gemma raw-audio classification result.

## App Runtime

The browser app uses the q4f16 ONNX/WebGPU package:

```bash
VITE_LISPER_BROWSER_MODEL_ID=thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16
VITE_LISPER_BROWSER_DTYPE=q4f16
```

The ZeroGPU Space uses the same E2B model lineage server-side plus the packaged v18 ExtraTrees acoustic sidecar.

## Run Locally

```bash
npm install
npm run start
```

Gemma Lab:

```bash
npm run start:gemma-lab
```

Production checks:

```bash
npm run build:gemma-lab
npm run build:web
```

## Kaggle Artifacts

- Dataset: https://www.kaggle.com/datasets/thomasjvu/lisper-gemma4-audio
- Adapter artifact dataset: https://www.kaggle.com/datasets/thomasjvu/lisper-gemma4-audio-lora
- Training notebook: https://www.kaggle.com/code/thomasjvu/lisper-gemma-4-audio-unsloth-training
- Eval notebook: https://www.kaggle.com/code/thomasjvu/lisper-gemma-4-audio-adapter-eval
