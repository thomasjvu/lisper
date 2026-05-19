# Quantization Plan

This project should keep two separate deliverables:

- Canonical full model: merged base Gemma 4 E2B plus Lisper LoRA, saved as normal `safetensors`.
- Consumer artifact: a quantized ONNX/WebGPU build derived from the canonical merged checkpoint for the web app.

## Recommendation

Use ONNX/WebGPU `q4f16` as the first browser and consumer-device target. It matches the public Gemma 4 E2B WebGPU artifact layout used by the app today and remains the safest browser path after the v18 hybrid eval passed.

Use `q4` only if we later need the smaller variant and it passes held-out eval. Use `q8` as a conservative local baseline. Treat `q3` and `q2` as experiments only; they may fit smaller devices, but they are likely to damage exact class and coaching behavior.

## Practical Limits

- 16-bit merged checkpoint: canonical artifact, too heavy for many consumer machines.
- 8-bit: safer quality, less useful for phones or small-memory laptops.
- q4f16: primary ONNX/WebGPU target for the browser app, Apple Silicon, and consumer GPUs.
- 4-bit q4: smaller fallback only after q4f16 works and eval passes.
- 3-bit: only if 4-bit cannot fit.
- 2-bit: smallest experiment, not a release target unless held-out eval passes.

Unsloth's Gemma 4 local guidance reports Gemma 4 E2B around `5 GB` RAM in 4-bit and around `15 GB` in full 16-bit precision. The app still needs extra memory for audio processing, KV cache, Python/runtime overhead, and the web/native shell, so those numbers are not total app memory.

## Runtime Reality

The raw-audio path matters. A quantized text-only runtime is not enough for Lisper unless it can also handle Gemma 4 E2B audio inputs. For the first deployment pass:

- Keep the backend `transformers` merged-model path and the v18 hybrid eval artifacts as the correctness reference.
- Use the existing trained q4f16 ONNX/WebGPU candidate for browser smoke testing.
- Treat the current browser package as the demo runtime; do not claim it is a pure v18 ONNX re-export unless a v18 merged checkpoint is exported and validated.
- Promote quantized variants only if class detection and cue formatting stay materially close to the passing release path.

## ONNX vs GGUF

The app target is ONNX/WebGPU because the current browser path already uses `@huggingface/transformers` and ONNX Runtime Web. GGUF is useful for native local/server inference experiments, but it is not the primary browser deployment format.

That means the current closeout order is:

- validate the merged safetensors checkpoint
- upload the full merged checkpoint to Hugging Face
- export an ONNX/WebGPU-compatible model variant
- quantize the ONNX variant to q4f16 first
- run browser smoke tests against q4f16
- run eval and require a passing `publish_verdict.json` before submission promotion

## Repo Command

Generate a local quantization handoff after the merged checkpoint exists:

```bash
python3 src/model/export.py \
  --format quantization-plan \
  --model data/processed/gemma4_audio/artifacts/YOUR_RUN/lisper-gemma4-audio/merged_model \
  --output data/processed/gemma4_audio/artifacts/exports/quantization
```

Validate a merged checkpoint before trying to quantize it:

```bash
python3 src/model/export.py \
  --format validate-merged \
  --model data/processed/gemma4_audio/artifacts/YOUR_RUN/lisper-gemma4-audio/merged_model
```

## Promotion Gate

A quantized model is submission-ready only when:

- the source merged checkpoint validates as complete
- the quantized runtime accepts audio inputs
- the ONNX/WebGPU bundle loads in the app runtime
- held-out eval finishes without hard crashes
- class-match and clear-vs-lisp separation do not materially regress
- generated coaching still contains reason, corrective cue, and encouragement
