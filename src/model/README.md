# Lisper Model Training

This directory contains the training-side helpers for Lisper's Gemma 4 work.

The current hackathon web demo does **not** bundle local fine-tuned weights. As of April 21, 2026, the web build uses the public browser-ready ONNX model `onnx-community/gemma-4-E2B-it-ONNX` through `@huggingface/transformers`, while this folder prepares data and notes for future fine-tuning/export work.

## Quick Start

### 1. Prepare Environment

```bash
# Create conda environment
conda create -n lisper python=3.11
conda activate lisper

# Install dependencies
pip install -r requirements.txt

# Or install Unsloth directly
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

### 2. Prepare Dataset

```bash
# Add more raw speech before the real run
curl -L -o data/raw/LibriSpeech/train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf data/raw/LibriSpeech/train-clean-100.tar.gz -C data/raw/LibriSpeech

# Build the hackathon-sized raw-audio dataset
python dataset.py --build-multimodal --profile hackathon

# Audit split counts, label counts, and speaker leakage
python dataset.py --audit

# Export split-wise multimodal JSONL bundles for Kaggle / Unsloth
python finetune.py --prepare
```

This produces:
- `data/processed/gemma4_audio/manifest.jsonl`
- `data/processed/gemma4_audio/messages/train.jsonl`
- `data/processed/gemma4_audio/messages/val.jsonl`
- `data/processed/gemma4_audio/messages/test.jsonl`
- compatibility exports at `data/processed/lisper_{train,val,test}.jsonl`
- refreshed preview audio under `data/synthetic/`

The `hackathon` profile targets:
- `4,000` unique source utterances total
- `20,000` expanded training examples after label expansion
- `150+` speakers before the build will start
- all available LibriSpeech subsets under `data/raw/LibriSpeech/`

### 3. Fine-tune Model

**Option A: Kaggle + Unsloth (Recommended)**
1. Generate the audio dataset and bundle with the commands above.
2. Upload only `data/processed/gemma4_audio/` to Kaggle as a Dataset. The current processed bundle is about `11 GB`.
3. Open [notebooks/kaggle_gemma4_audio_unsloth.ipynb](/Users/area/repos/lisper/notebooks/kaggle_gemma4_audio_unsloth.ipynb) in Kaggle.
4. Leave `RUN_FULL_TRAIN = False` for the first pass, run the smoke test, and only enable full training after the smoke pass completes without OOM or NaNs.
5. If you want Hub pushes, set the `HF_TOKEN` secret and update `HF_ADAPTER_REPO` and `HF_FULL_MODEL_REPO`.
6. Leave `EXPORT_MERGED_MODEL = True` so Kaggle writes both the LoRA adapter and a standalone merged Gemma E2B checkpoint.
7. Follow the full runbook in `docs/KAGGLE_TRAINING.md`.

**Option B: Local GPU**
Use the same generated bundle, but the checked-in notebook assumes Kaggle paths. Copy its cells into a local notebook or adapt the `DATASET_ROOT` path.

### 4. Export Model

```bash
python export.py --format quantization-plan
python export.py --format web-notes
```

For the hackathon run, the preferred standalone full-model export path is the Kaggle notebook itself:
- save LoRA adapter under `adapter/`
- save merged base+LoRA model under `merged_model/`
- publish the adapter and merged model to separate private Hugging Face repos

For consumer-device deployment, quantize after the merged checkpoint is valid. The current app target is ONNX/WebGPU `q4f16` only, matching the public Gemma 4 WebGPU layout used by Transformers.js. GGUF is optional native-local packaging, not the browser runtime target. See [QUANTIZATION.md](/Users/area/repos/lisper/docs/QUANTIZATION.md).

Current v16 size reality:
- merged E2B checkpoint on Hub: about `10.25 GB`
- local merged directory: about `9.6 GiB`
- expected q4 ONNX/WebGPU download: about `4.0 GB`
- expected q4f16 ONNX/WebGPU download: about `3.4 GB`
- q4 and q4f16 together: about `7.3 GB`

These estimates come from the public `onnx-community/gemma-4-E2B-it-ONNX` repo, which uses the same `google/gemma-4-E2B-it` architecture. The trained Lisper export should be in the same range because LoRA merging changes weight values, not tensor shapes.

Local-only ONNX/WebGPU helper:

```bash
python src/model/export_onnx_webgpu.py --mode probe
python src/model/export_onnx_webgpu.py --mode size-estimate
python src/model/export_onnx_webgpu.py --mode copy-metadata
```

Full export is deliberately not launched as a paid Hugging Face Job. If you build a local export env, note the current dependency edge: Gemma 4 E2B needs Transformers `5.x`, while released `optimum-onnx` still pins `transformers<4.58`. Do not upload a partial ONNX export; only promote a browser repo after the full audio/vision/embed/decoder ONNX layout exists and held-out eval passes.

The local dependency probe that works is:

```bash
tmpdir=/tmp/lisper-onnx-probe
rm -rf "$tmpdir"
mkdir -p "$tmpdir"
cd "$tmpdir"
uv venv
. .venv/bin/activate
uv pip install \
  "transformers==5.5.0" \
  "torch==2.6.0" \
  "optimum@git+https://github.com/huggingface/optimum.git" \
  "onnxruntime==1.20.1" \
  "onnx==1.17.0" \
  "onnxslim==0.1.48" \
  requests \
  huggingface-hub
uv pip install --no-deps "optimum-onnx@git+https://github.com/huggingface/optimum-onnx.git"
cd /Users/area/repos/lisper
python src/model/export_onnx_webgpu.py --mode probe
```

Current local export result:

- `optimum-cli export onnx` successfully loaded the merged v16 weights.
- Export then failed because Optimum does not yet provide a native ONNX config for `model_type="gemma4"`.
- The browser q4f16 artifact has since been promoted through the custom component-export path at `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`.
- The current quality gate is satisfied by the v18 hybrid eval path, not by a pure re-export of a v18 merged ONNX checkpoint.

The remaining work is not more training. The model is trained and merged. The blocker is implementing or obtaining the Gemma4-specific ONNX export config that emits the browser layout used by Transformers.js:
- `onnx/audio_encoder_q4f16.onnx`
- `onnx/vision_encoder_q4f16.onnx`
- `onnx/embed_tokens_q4f16.onnx`
- `onnx/decoder_model_merged_q4f16.onnx`

After those fp16 ONNX components exist, run q4f16 quantization only. Do not generate q4 unless needed. For the hackathon demo, the app uses the uploaded q4f16 package while the evaluated release verdict is tied to the v18 hybrid acoustic+Gemma path.

Custom component-export scaffold:

```bash
python src/model/inspect_onnx_contract.py \
  data/processed/gemma4_audio/artifacts/exports/onnx-webgpu/reference-e2b-q4f16-wrappers \
  --output data/processed/gemma4_audio/artifacts/exports/onnx-webgpu/reference-e2b-q4f16-contract.json

python src/model/export_gemma4_onnx_components.py --component embed_tokens --dry-run
python src/model/export_gemma4_onnx_components.py --component audio_encoder --dry-run
python src/model/export_gemma4_onnx_components.py --component vision_encoder --dry-run
```

`decoder_model_merged` is intentionally guarded in the custom scaffold. It must match the official KV-cache input/output contract before any trained ONNX repo is uploaded.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-4-E2B-it |
| Max Seq Length | 2048 |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Target Modules | all-linear |

## Unsloth Benefits

- **2x faster** training
- **70% less VRAM**
- **No approximation** - exact same results

## Training Data Format

```json
{
  "id": "hackathon_train_61-70968-0000_lateral",
  "audio_path": "data/processed/gemma4_audio/audio/hackathon/train/lateral/61-70968-0000_lateral.wav",
  "lisp_type": "lateral",
  "instruction": "Analyze this pronunciation attempt for lisp type and give concise corrective coaching...",
  "expected_feedback": "Detected class: lateral\nReason: Air is leaking over the sides of the tongue...\nCorrective cue: Lift the sides of the tongue...\nEncouragement: Good attempt..."
}
```

The exported Unsloth / Kaggle bundle uses `messages` chat-format records with audio-first user content:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "audio", "audio": "data/processed/gemma4_audio/audio/hackathon/train/lateral/example.wav" },
        { "type": "text", "text": "You are Lisper, a speech therapy assistant for lisp practice...\n\nAnalyze this pronunciation attempt..." }
      ]
    },
    { "role": "assistant", "content": [{ "type": "text", "text": "Detected class: lateral..." }] }
  ]
}
```

## Browser Demo Note

The working web demo path is intentionally separate from this folder:

- Web runtime: `@huggingface/transformers`
- Browser model: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Browser dtype: `q4f16` by default
- Bundler: Vite fallback for web, because Expo Metro could not bundle `onnxruntime-web` cleanly in this repo

Do not point the browser directly at the Kaggle adapter artifacts. The browser path uses the uploaded trained q4f16 ONNX/WebGPU package. The backend path remains useful for debugging the raw merged 16-bit checkpoint:

- browser app records/uploads audio
- local or hosted backend loads the merged Gemma 4 + Lisper checkpoint
- backend returns parsed assessment/coaching JSON
- browser ONNX path is the default demo/runtime for consumer-device testing

Local handoff command:

```bash
pip install -r src/model/requirements-serve.txt
python3 -m uvicorn src.model.inference_server:app --host 127.0.0.1 --port 8000
```

If you want to force the standalone merged checkpoint explicitly:

```bash
LISPER_MODEL_ARTIFACT_KIND=merged python3 -m uvicorn src.model.inference_server:app --host 127.0.0.1 --port 8000
```

Then start the web app with backend inference:

```bash
cd lisper-app
VITE_LISPER_INFERENCE_URL=http://127.0.0.1:8000 npm run web
```

The app defaults to the uploaded trained browser artifact:

```bash
cd lisper-app
VITE_LISPER_BROWSER_MODEL_ID=thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16 \
VITE_LISPER_BROWSER_DTYPE=q4f16 \
npm run web
```

For public-base A/B testing only:

```bash
cd lisper-app
VITE_LISPER_BROWSER_MODEL_ID=onnx-community/gemma-4-E2B-it-ONNX \
VITE_LISPER_BROWSER_DTYPE=q4f16 \
npm run web
```

Apple Silicon note:

- The LoRA adapter runtime in this repo is the CUDA/Unsloth path.
- On Apple Silicon or CPU, the supported path is the repaired `merged_model/` checkpoint plus standard Hugging Face `transformers`.
- `src/model/inference_server.py` now auto-selects the merged runtime on non-CUDA machines when the merged checkpoint is valid.
- Whether an M2 can hold the merged model depends on unified memory. The model family is "edge-capable" in the sense that smaller or quantized deployments are possible, but the raw merged `16-bit` checkpoint used here is still much heavier than a mobile/web export.
- The preferred M2/browser deployment artifact is the trained ONNX/WebGPU 4-bit export produced after v16, not the raw 16-bit merged checkpoint.
- Use `requirements.txt` for Kaggle/CUDA training. Use `requirements-serve.txt` for the local merged-model backend.

## Fine-Tuning Terms

- `SFT`: supervised fine-tuning on labeled examples
- `LoRA`: train lightweight low-rank adapter weights instead of updating all base weights
- `QLoRA`: LoRA with a quantized base model so training fits on smaller GPUs
- `Merged model`: base model plus trained LoRA folded into one standalone checkpoint
- `Full-parameter fine-tune`: updates all model weights directly

Lisper currently uses:
- `Unsloth QLoRA / LoRA SFT`

Lisper does not currently use:
- dense full-parameter fine-tuning

That means:
- the main trained artifact is the adapter directory
- the standalone "full model" deliverable is a merged base+LoRA export, not a dense full-weight retrain

## Files

- `dataset.py` - Raw-audio dataset generation, split assignment, and manifest audit
- `finetune.py` - Manifest-to-chat export for Kaggle / Unsloth
- `export.py` - Model export
- `test_base.py` - Base-model smoke test
- `requirements.txt` - CUDA/Unsloth training dependencies
- `requirements-serve.txt` - local/backend inference dependencies
