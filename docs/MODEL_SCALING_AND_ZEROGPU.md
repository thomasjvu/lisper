# Model Scaling And ZeroGPU Plan

## Current Model State

The only fully trained and evaluated Lisper Gemma checkpoint is Gemma 4 E2B.

- Browser demo: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Server-side full model target: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Training method: Unsloth LoRA SFT, then merged full checkpoint
- Quality gate: v18 hybrid acoustic+Gemma held-out eval, `2000` rows, `0` hard errors, `publish_verdict.json` pass

Do not claim E4B or 31B are trained until they have gone through the same train, merge, eval, and publish-verdict path.

## Why E2B Can Still Lag On A Mac M2

“Edge capable” does not mean every browser WebGPU path is smooth on every edge device.

The current browser artifact is an ONNX q4f16 multimodal package. The browser has to download/cache multiple external-data ONNX components, initialize ONNX Runtime Web, run q4f16 dequantization and fp16 compute through WebGPU, and handle audio/vision preprocessing in the same tab. On an M2 with other work running, unified-memory pressure and browser runtime overhead can dominate even though the base parameter count is small enough for edge deployment.

ZeroGPU gives us a practical fallback: keep the static browser q4f16 demo for install-free local inference, and offer a remote Gradio path for users whose laptop browser is too slow.

## E4B Path, Post-Submission Only

E4B is feasible as the next scaling experiment, but it is a new run, not a config flag on the existing model. It is not part of the hackathon submission path.

Current launch state:

- Base checkpoint: `google/gemma-4-E4B-it`
- Private Kaggle notebook: `thomasjvu/lisper-gemma-4-e4b-audio-unsloth-training`
- Private LoRA repo target: `thomasjvu/lisper-gemma4-e4b-audio-lora`
- Private full model repo target: `thomasjvu/lisper-gemma4-e4b-audio-full`
- Preferred merge-only Kaggle notebook: `thomasjvu/lisper-gemma-4-e4b-audio-merge`
- Recipe: E2B v18-derived audio recipe ported to E4B as `v19-e4b-*`, LoRA rank reduced to `8` for memory headroom
- Status: Kaggle version 1 completed the 25-step smoke pass and full 4,000-step LoRA training, then failed during local merged-model export because Kaggle disk was insufficient for saving an 8B merged checkpoint
- Recovered artifact: the E4B LoRA adapter was downloaded from the failed Kaggle output and uploaded to `thomasjvu/lisper-gemma4-e4b-audio-lora`
- Current deploy path: load adapter `thomasjvu/lisper-gemma4-e4b-audio-lora` through Unsloth `FastVisionModel` on ZeroGPU until a push-to-Hub merge job creates `thomasjvu/lisper-gemma4-e4b-audio-full`
- Runtime smoke: the private E4B adapter Space completed an authenticated audio request on `zero-a10g`; this proves serving works, but it is not a quality gate and should not be linked in submission materials
- Disk mitigation: the regenerated E4B Kaggle notebook now keeps only one training checkpoint, deletes older `checkpoint-*` directories before merge/export, and uses Unsloth `push_to_hub_merged` when `HF_TOKEN` is available instead of writing the full merged model to `/kaggle/working` first

Required steps:

1. Smoke and full LoRA training are complete.
2. Deploy and smoke-test the recovered adapter on ZeroGPU. Complete for basic runtime only.
3. Add `HF_TOKEN` as a Kaggle notebook secret before any E4B merge rerun; without it the notebook must fall back to local merged save and can hit the same ephemeral disk limit.
4. Push and run the merge-only Kaggle notebook from `notebooks/kaggle_upload/lisper-gemma-4-e4b-audio-merge/` so we do not spend another full training run.
5. Run the same v18-style held-out eval and require a local `publish_verdict.json`.
6. Browser q4f16 export is optional and should be attempted only after the model gate passes.

Expected tradeoff: better reasoning capacity, higher VRAM and latency. E4B may be reasonable on ZeroGPU later; it is not the browser target, server target, or submitted model for the hackathon.

## 31B Path

31B is not a free-Kaggle/browser target.

It should be treated as a cloud inference or cloud training experiment:

- Training likely needs paid A100/H100-class infrastructure or another sponsored GPU source.
- Full fp16 inference may exceed the default ZeroGPU slice after model, KV cache, processor, and framework overhead.
- Quantized inference may be possible, but it needs a separate packaging and latency validation pass.
- It should not block the hackathon submission because the current E2B gate already passed.

## ZeroGPU Deployment Role

The ZeroGPU Space in `spaces/lisper-zerogpu/` is for server-side inference demos.

Current remote status:

- Private Space package uploaded: `thomasjvu/lisper-zerogpu`
- Space variables configured for `thomasjvu/lisper-gemma4-e2b-audio-full`
- `HF_TOKEN` secret configured
- Hardware active: `zero-a10g`
- Smoke test passed through the authenticated Gradio API with a synthetic lisp clip

Runtime note:

- The Space uses the official Gemma 4 `processor.apply_chat_template(... tokenize=True, return_dict=True, return_tensors="pt")` path for audio.
- The hosted Transformers/Gemma 4 audio stack can still produce an audio placeholder/features mismatch. The Space includes a narrow alignment shim that adjusts the audio placeholder run to match the valid soft-token count produced by the audio encoder before calling `generate`.
- Treat this as a remote demo fallback. The model-quality claim remains the v18 hybrid held-out eval gate, not this ad hoc Space smoke test.

Default target:

```text
LISPER_ZERO_GPU_MODEL_ID=thomasjvu/lisper-gemma4-e2b-audio-full
LISPER_ZERO_GPU_DTYPE=float16
LISPER_ZERO_GPU_SIZE=large
LISPER_ZERO_GPU_MAX_NEW_TOKENS=96
```

Use `xlarge` only for larger checkpoints or if the E2B full checkpoint shows memory pressure. Keep model repos private until submission packaging is finalized.

## Deploy

From the repo root:

```bash
scripts/deploy_lisper_zerogpu_space.sh
```

Defaults:

```text
SPACE_ID=thomasjvu/lisper-zerogpu
SPACE_PRIVATE=1
```

For the separate private E4B experiment, use the wrapper only after configuring the recovered E4B adapter path and only when intentionally working on post-submission scaling:

```bash
scripts/deploy_lisper_zerogpu_e4b_space.sh
```

The wrapper uploads the same ZeroGPU source to `thomasjvu/lisper-zerogpu-e4b`. Configure it with `LISPER_ZERO_GPU_MODEL_ID=google/gemma-4-E4B-it`, `LISPER_ZERO_GPU_ADAPTER_ID=thomasjvu/lisper-gemma4-e4b-audio-lora`, `LISPER_ZERO_GPU_LOAD_IN_4BIT=1`, and start with `LISPER_ZERO_GPU_SIZE=large`. Keep `LISPER_ZERO_GPU_EAGER_LOAD=0` until the Space variables and gated/private model access are configured.

Current E4B adapter Space smoke command:

```bash
uvx --with gradio_client --with huggingface_hub python -c "from huggingface_hub import get_token; from gradio_client import Client, handle_file; client=Client('thomasjvu/lisper-zerogpu-e4b', token=get_token()); print(client.predict(handle_file('/Users/area/repos/lisper/data/synthetic/frontal/1081-125237-0034_frontal.wav'), 'Sally sells seashells.', api_name='/analyze'))"
```

## E4B Kaggle Disk Cleanup

The failed E4B training run exhausted the notebook's ephemeral `/kaggle/working` disk while trying to save a full merged checkpoint locally. There is nothing useful to delete inside that completed failed run; Kaggle tears down the working directory after the version finishes. The fix is to avoid recreating the same pressure in the next run.

Use the merge-only notebook after adding `HF_TOKEN` as a Kaggle secret:

```bash
/Users/area/Library/Python/3.14/bin/kaggle kernels push -p notebooks/kaggle_upload/lisper-gemma-4-e4b-audio-merge
```

The upload generators default to the currently used `thomasjvu` owner. To generate metadata for the temporary `alkahestai` Kaggle account instead, run the generator with `KAGGLE_OWNER=alkahestai` after logging the CLI into that account.

If a full training rerun is ever needed, the regenerated E4B training notebook now keeps only one checkpoint and deletes older checkpoints before merge/export:

```bash
/Users/area/Library/Python/3.14/bin/kaggle kernels push -p notebooks/kaggle_upload/lisper-gemma-4-e4b-audio-unsloth-training
```

After Space upload, set variables/secrets in the Hugging Face UI:

- `LISPER_ZERO_GPU_MODEL_ID`
- `LISPER_ZERO_GPU_ADAPTER_ID`, only when serving a base model plus LoRA adapter
- `LISPER_ZERO_GPU_DTYPE`
- `LISPER_ZERO_GPU_LOAD_IN_4BIT`, defaults to `1` for adapter deployments
- `LISPER_ZERO_GPU_MAX_SEQ_LENGTH`
- `LISPER_ZERO_GPU_SIZE`
- `LISPER_ZERO_GPU_MAX_NEW_TOKENS`
- `HF_TOKEN`, if the selected model is private or gated

Then request ZeroGPU hardware. If the Hub API returns `Subscribe to PRO ... to use ZeroGPU`, the Space will remain on CPU and should not be used for the demo until PRO access is enabled or another GPU runtime is selected.

Current verification command used:

```bash
uvx --with gradio_client --with huggingface_hub python -c "from huggingface_hub import get_token; from gradio_client import Client, handle_file; client=Client('thomasjvu/lisper-zerogpu', token=get_token()); print(client.predict(handle_file('/Users/area/repos/lisper/data/synthetic/frontal/1081-125237-0034_frontal.wav'), 'Sally sells seashells.', api_name='/analyze'))"
```

## Submission Language

Use this wording:

```text
Lisper uses a fine-tuned Gemma 4 E2B model. The browser demo runs a q4f16 ONNX/WebGPU package for local inference, while the optional Hugging Face ZeroGPU Space provides a server-side fallback for devices that cannot comfortably run the browser model.
```

Avoid saying:

```text
We trained E4B/31B.
```

until those checkpoints pass the same evaluation gate.
