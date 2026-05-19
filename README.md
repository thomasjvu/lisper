# Lisper

Lisper is a Gemma 4 speech-practice coach for lisp correction. It combines a fine-tuned Gemma 4 E2B coaching model, a browser WebGPU app, and a ZeroGPU fallback Space for users whose devices cannot comfortably run the browser model.

This is a practice assistant, not a medical diagnosis tool or a replacement for a speech-language pathologist.

## Project Layout

```text
lisper-app/                  Vite + React Native Web app and Gemma Lab
spaces/lisper-zerogpu/       Hugging Face ZeroGPU fallback Space
src/model/                   Dataset, training, eval, export, and publish utilities
notebooks/                   Kaggle training/eval notebooks and upload metadata
docs/                        Hackathon writeups, model cards, screenshots, and notes
scripts/                     Deployment helpers
videos/lisper-hackathon/     Demo video composition source
```

Large local datasets, Kaggle output bundles, Python virtualenvs, build outputs, and rendered videos are intentionally ignored.

## Submitted Model Story

- Base model: `google/gemma-4-E2B-it`
- Fine-tuning: Unsloth LoRA
- LoRA adapter: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-lora
- Merged full checkpoint: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-full
- Browser q4f16 package: https://huggingface.co/thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16
- ZeroGPU fallback Space: https://huggingface.co/spaces/thomasjvu/lisper-zerogpu

The release-quality gate is the v18 hybrid acoustic+Gemma held-out path: `2,000` rows, `0` hard errors, `0.976` class match, and `1.0` exact output-format rate.

E4B and q2f16 are experimental follow-up paths and are not part of the submitted model gate.

## Run The App

```bash
cd lisper-app
npm install
npm run start
```

Gemma Lab:

```bash
cd lisper-app
npm run start:gemma-lab
```

Production checks:

```bash
cd lisper-app
npm run build:gemma-lab
npm run build:web
```

## ZeroGPU Space

The Space is deployed from `spaces/lisper-zerogpu/`:

```bash
scripts/deploy_lisper_zerogpu_space.sh
```

The live path uses the v18 acoustic gate by default and returns `rejected_audio` or `inconclusive` instead of forcing a lisp label when the recording is silent, noisy, missing usable /s/ evidence, or out of domain.

## Kaggle Artifacts

- Dataset: https://www.kaggle.com/datasets/thomasjvu/lisper-gemma4-audio
- Adapter artifact dataset: https://www.kaggle.com/datasets/thomasjvu/lisper-gemma4-audio-lora
- Training notebook: https://www.kaggle.com/code/thomasjvu/lisper-gemma-4-audio-unsloth-training
- Eval notebook: https://www.kaggle.com/code/thomasjvu/lisper-gemma-4-audio-adapter-eval
