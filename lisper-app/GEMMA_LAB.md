# Gemma Lab

Standalone browser test harness for Gemma 4 WebGPU packages. It is separate from the main Lisper app route and is intended for fast manual testing of model loading, text chat, raw audio, image input, and video-frame vision prompts.

## Run

```bash
cd /Users/area/repos/lisper/lisper-app
npm run dev:gemma-lab
```

Open `http://127.0.0.1:5174/gemma-lab.html`. The dev server also redirects `/` to the lab page so it does not launch the normal app by accident.

## Build

```bash
cd /Users/area/repos/lisper/lisper-app
npm run build:gemma-lab
```

The production output is written to `dist-gemma-lab/`.

## Model Presets

- `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`: trained q4f16 package, expected browser payload around 3.15 GB.
- `thomasjvu/lisper-gemma4-e2b-audio-onnx-q2f16-experimental`: decoder-only q2f16 experiment with q4f16 embed/audio/vision components, expected browser payload around 2.61 GB.
- `onnx-community/gemma-4-E2B-it-ONNX`: public base q4f16 reference package.

## Test Modes

- `Chat`: text-only generation.
- `Image`: one uploaded image through the vision encoder.
- `Audio`: uploaded or recorded audio through the audio encoder.
- `Video Frames`: sampled video frames sent as images. Gemma 4 does not expose a separate temporal video encoder here.
- `Combined`: sends available audio plus image/video frames together.

First model load downloads multi-GB ONNX assets into the browser cache. Use a Chromium browser with WebGPU enabled for realistic testing.

If you tested before the JSEP/external-data config fix, use `Clear Browser Model Cache` in the lab before retrying. The previous broken `config.json` can persist inside the browser Cache API even after the Hugging Face repo is fixed.
