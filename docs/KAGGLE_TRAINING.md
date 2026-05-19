# Kaggle Training Runbook

This is the exact handoff for the current Lisper Gemma 4 audio bundle.

## Ready State

- Upload target: `data/processed/gemma4_audio/`
- Current size: about `11 GB`
- Manifest: `data/processed/gemma4_audio/manifest.jsonl`
- Split JSONL files:
  - `data/processed/gemma4_audio/messages/train.jsonl`
  - `data/processed/gemma4_audio/messages/val.jsonl`
  - `data/processed/gemma4_audio/messages/test.jsonl`
- Notebook: `notebooks/kaggle_gemma4_audio_unsloth.ipynb`

Current verified counts:
- `16000` train rows
- `2000` val rows
- `2000` test rows
- `204` train speakers, `44` val speakers, `43` test speakers
- balanced labels across `clear`, `frontal`, `lateral`, `dental`, `palatal`
- `0` speaker leakage across splits

## What To Upload

Upload only `data/processed/gemma4_audio/` to Kaggle as a Dataset. Do not upload:
- `data/raw/`
- the full repo
- local caches or checkpoints

The notebook already knows how to resolve either of these Kaggle layouts:
- dataset root contains `bundle.json`
- dataset root contains `gemma4_audio/bundle.json`
- dataset root contains `data/processed/gemma4_audio/bundle.json`
- dataset root contains `messages.zip` and `audio.zip` from `kaggle datasets create --dir-mode zip`

It also scans all attached Dataset mount names under `/kaggle/input/`, so the Kaggle Dataset slug does not need to be `lisper-gemma4-audio`.

## Kaggle Steps

1. Create a private Kaggle Dataset from `data/processed/gemma4_audio/`.
2. Create a new Kaggle Notebook with a GPU accelerator.
3. Attach the Dataset you just uploaded.
4. Upload `notebooks/kaggle_gemma4_audio_unsloth.ipynb` into Kaggle, or paste its cells into the new Notebook.
5. If you plan to push adapters to Hugging Face, add a Kaggle secret named `HF_TOKEN`.
6. Open the config cell near the top of the notebook and set:
   - `RUN_BASELINE_EVAL = False` unless you are explicitly running base-vs-tuned generation eval
   - `RUN_SMOKE_TRAIN = True` for the corrected `50-100` step smoke pass on a new recipe
   - `RUN_FULL_TRAIN = False` until the smoke pass succeeds
   - `RUN_TUNED_EVAL = False`; use the separate held-out eval notebook for release gating
   - `PUSH_TO_HUB = False` unless your `HF_TOKEN` secret is set
   - `PUSH_MERGED_TO_HUB = False` unless your `HF_TOKEN` secret is set
   - `HF_ADAPTER_REPO = "your-hf-username/lisper-gemma4-e2b-audio-lora"` when pushing
   - `HF_FULL_MODEL_REPO = "your-hf-username/lisper-gemma4-e2b-audio-full"` when pushing the merged checkpoint
   - `EXPORT_MERGED_MODEL = True` so Kaggle saves a standalone merged Gemma E2B checkpoint
   - `RESUME_FROM_CHECKPOINT = ""` unless you are resuming the exact same recipe
   - `SELECT_BEST_CHECKPOINT = True` to save the best validation checkpoint into `adapter/`

If you want repo-generated Kaggle metadata instead of the UI flow, run:

```bash
python src/model/kaggle_prepare_upload.py \
  --username YOUR_KAGGLE_USERNAME \
  --dataset-license YOUR_KAGGLE_LICENSE_NAME
```

That writes:
- `data/processed/gemma4_audio/dataset-metadata.json`
- `notebooks/kaggle_upload/lisper-gemma-4-audio-unsloth-training/kernel-metadata.json`
- `notebooks/kaggle_upload/lisper-gemma-4-audio-unsloth-training/push_commands.txt`

The helper does not upload anything by itself. It only prepares metadata and push commands.

Current Kaggle resources:
- Dataset: `thomasjvu/lisper-gemma4-audio`
- Training notebook: `thomasjvu/lisper-gemma-4-audio-unsloth-training`
- Adapter dataset: `thomasjvu/lisper-gemma4-audio-lora`
- Eval notebook: `thomasjvu/lisper-gemma-4-audio-adapter-eval`

## Run Order

Run the notebook in this order:

1. Install cell
2. Dataset load + path resolution cell
3. Model load cell
4. LoRA setup and optional smoke-training cell

Stop there on the first pass. Promotion gate for continuing:
- model loads successfully
- a batch runs
- loss is finite
- no OOM
- no NaNs

Only after that, go back to the config cell and set:
- `RUN_SMOKE_TRAIN = False`
- `RUN_FULL_TRAIN = True`
- `RUN_BASELINE_EVAL = True` if you want the base-vs-tuned comparison file
- `PUSH_TO_HUB = True` only if `HF_TOKEN` is present and `HF_ADAPTER_REPO` is set
- `PUSH_MERGED_TO_HUB = True` only if `HF_TOKEN` is present and `HF_FULL_MODEL_REPO` is set

Then run:

5. Full-train cell
6. Tuned-eval / comparison cell

## Expected Outputs

Kaggle writes artifacts to `/kaggle/working/lisper-gemma4-audio/`.

Important files:
- `baseline_eval.json` if baseline eval is enabled
- `artifacts.json`
- `adapter/training_metadata.json`
- `comparison.json` if baseline eval is enabled
- `adapter/` for the saved LoRA adapter
- `merged_model/` for the standalone merged Gemma E2B checkpoint

After the run, download those outputs into:
- `data/processed/gemma4_audio/artifacts/`

Run the held-out adapter eval in the separate eval notebook after training. The eval output directory should contain:
- `tuned_eval.json`
- `tuned_eval_rows.jsonl`
- `publish_verdict.json`

Latest passing held-out eval:
- Kaggle eval notebook: `alkahestai/lisper-gemma-4-audio-adapter-eval`
- Eval version: `16`
- Model source: `alkahestai/lisper-gemma4-audio-lora-v18`
- Eval limit: `2000`
- Verdict: `pass`
- Hard errors: `0`
- Class match: `0.976`
- Clear/non-clear match: `0.989`
- Exact four-line format: `1.0`
- Local artifacts: `data/processed/gemma4_audio/artifacts/eval_v18_hybrid_full_v16/lisper-gemma4-audio-eval/`

## Corrected Training Recipe

- Base model: `google/gemma-4-E2B-it`
- Method: Unsloth QLoRA
- Current recipe name: `v17-schema-lock` when the bundle is regenerated from this repo
- Max sequence length: `2048`
- LoRA rank/alpha: `16 / 16`
- Target modules: `all-linear`
- Finetune MLP modules: `True`
- Per-device batch size: `1`
- Gradient accumulation: `4`
- Learning rate: `2e-4`
- Smoke steps: `100`
- Full-train steps: `4000`
- Eval/save cadence: `500`

Why the retrain is needed:
- notebook version `11` finished successfully, but it silently narrowed the recipe down to `1024`, `r=8`, q/k/v/o only, and `1500` steps
- eval v4 shows the resulting adapter is not submission-ready on exact class output or format compliance
- the current notebook now honors the stronger bundle config and writes `training_metadata.json` plus best-checkpoint metadata for publishing
- the next bundle prompt explicitly locks `Detected class` to exactly one of `clear`, `frontal`, `lateral`, `dental`, or `palatal`

## Completed Kaggle Runs

### Version 10 Smoke Run

- Kaggle notebook: `thomasjvu/lisper-gemma-4-audio-unsloth-training`
- Accelerator: Kaggle GPU, observed `Tesla P100`
- Result: completed 10-step Unsloth smoke train with saved adapter
- Local adapter: `data/processed/gemma4_audio/artifacts/lisper-gemma4-audio/smoke/checkpoint-10/adapter_model.safetensors`
- Train loss: `15.87` at step 5, `15.58` at step 10
- Eval loss on validation: `5.30`

### Version 11 Full Run

- Kaggle notebook: `thomasjvu/lisper-gemma-4-audio-unsloth-training`
- CLI accelerator request: `--accelerator NvidiaTeslaT4`
- Observed GPUs: `2 x Tesla T4`
- Result: completed 1500-step full train with saved final adapter
- Unsloth confirmation: notebook log shows the Unsloth banner, Gemma 4 patching, and `Num GPUs = 2`
- Local adapter: `data/processed/gemma4_audio/artifacts/full_run_v11/lisper-gemma4-audio/adapter/adapter_model.safetensors`
- Local final checkpoint: `data/processed/gemma4_audio/artifacts/full_run_v11/lisper-gemma4-audio/full_train/checkpoint-1500/`
- Trainable parameters: `4,644,864 / 5,127,822,880`
- Runtime: `4801` seconds, about `80` minutes
- Effective examples seen: `6000`, or `0.375` epoch with batch size `1` and gradient accumulation `4`
- Final train loss: `1.075`
- Validation loss checkpoints: `4.92` at step 250, `4.98` at 500, `4.85` at 750, `4.84` at 1000, `4.84` at 1250, `4.84` at 1500
- Log check: no traceback, OOM, RuntimeError, or NaN matches

### Version 13 Adapter + Merged Export Run

- Kaggle notebook: `thomasjvu/lisper-gemma-4-audio-unsloth-training`
- Remote status: `COMPLETE`
- Observed GPU: `Tesla P100-PCIE-16GB`
- Unsloth confirmation: log shows `Unsloth 2026.4.8` and Gemma 4 patching
- Result: completed `1500` steps, selected checkpoint `checkpoint-400`, wrote both `adapter/` and `merged_model/`
- Local adapter root: `data/processed/gemma4_audio/artifacts/full_run_v13/lisper-gemma4-audio/adapter/`
- Local merged root: `data/processed/gemma4_audio/artifacts/full_run_v13/lisper-gemma4-audio/merged_model/`
- Final train loss: `0.9895`
- Best eval loss: `4.7103`
- Effective progress: `0.375` epoch

What v13 proved:
- the adapter export path is healthy
- the notebook can save a merged checkpoint directory
- the run still did not execute the intended stronger full recipe

What v13 did not do:
- it did not use the intended `4000`-step recipe
- it did not push artifacts to Hugging Face during the run because both Hub push flags were `False`
- it did not produce a locally valid merged standalone weight file from the downloaded Kaggle output

Important v13 mismatch:
- `training_metadata.json` still reports `full_train_max_steps = 1500`
- `save_steps = 200`
- `eval_steps = 200`
- this means Kaggle ran with a stale bundle/config despite the local repo having been updated

Important merged-model blocker:
- the downloaded local file `merged_model/model.safetensors` is currently incomplete from an interrupted download
- treat the merged standalone model as invalid until the export is revalidated or re-uploaded
- do not promote or publish the merged repo from this artifact as-is

Repo-side fix after v13:
- the training notebook source now embeds the expected local `bundle.json` SHA-256 and expected `full_train_max_steps`, `save_steps`, and `eval_steps`
- future Kaggle runs now report `stale_kaggle_bundle` when the uploaded dataset bundle is old, but train from the notebook-embedded corrected config so a stale dataset upload cannot silently force the old recipe
- `kaggle_prepare_upload.py` now re-syncs the Kaggle notebook copies before generating upload metadata so the notebook and bundle guard stay aligned

## Corrected Retrain Status

Version `16` of `thomasjvu/lisper-gemma-4-audio-unsloth-training` completed on April 25, 2026 and is the current valid training/export artifact.

Why the docs jump from local `v13` to Kaggle `v16`:
- `v13` is the latest completed local artifact download, but it used the stale `1500`-step recipe and the merged `model.safetensors` download is incomplete.
- `v14` added stale-bundle detection and corrected notebook-embedded config.
- `v15` used the corrected recipe but failed early when Kaggle did not expose an `HF_TOKEN` secret for Hub pushes.
- `v16` kept the corrected recipe, trained to completion, exported the adapter and merged checkpoint locally in Kaggle, and skipped Hub push because the Kaggle `HF_TOKEN` secret was missing.

What changed in v16:
- uses notebook-embedded corrected config: `2048` max sequence length, `r=16`, `all-linear`, MLP LoRA enabled, `4000` max steps
- keeps the stale uploaded Kaggle dataset usable for audio/message files while ignoring its older training hyperparameters
- completed `4000` training steps over `1.0` epoch
- selected `checkpoint-2500` as the best checkpoint with metric `4.851842880249023`
- wrote `adapter/` and `merged_model/` under `/kaggle/working/lisper-gemma4-audio/`
- skipped Hub push with `hf_token_missing_skip_hub_push`, so local download plus local Hugging Face upload is required

v16 metrics:
- observed GPU: `Tesla P100-PCIE-16GB`
- trainable parameters: `29,859,840 / 5,153,037,856`
- train runtime: `14599.68` seconds
- final training loss: `0.7217483117580413`
- merged save method: `merged_16bit`
- local merged validation: `model.safetensors` is complete, `10,246,621,886` bytes, `2011` tensors

Current Hub closeout:
- v16 adapter uploaded to `thomasjvu/lisper-gemma4-e2b-audio-lora`
- adapter repo commit: `f441f337c290a244f28470fb33380f357278f4f3`
- v16 merged checkpoint uploaded to `thomasjvu/lisper-gemma4-e2b-audio-full`
- merged repo commit: `73d84f6f314e95376e3769b99cfdb9c537729bee`
- merged Hub storage: about `10.28 GB`, including `model.safetensors`, processor, tokenizer, config, chat template, and model card
- do not promote v13; the local v13 merged `model.safetensors` was incomplete and is superseded by v16

Consumer-device quantization target:
- keep the merged `16-bit` checkpoint as the canonical full-model artifact
- use the uploaded ONNX/WebGPU `q4f16` package first for the browser app, Apple Silicon browser testing, and small GPU deployment
- treat ONNX/WebGPU `q4` as a smaller fallback only after q4f16 works and passes eval
- test `q3` and `q2` only as experiments, not default hackathon artifacts
- treat GGUF as optional native-local packaging, not the default app runtime
- the trained q4f16 browser repo is `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- q4f16 artifact commit: `e77aa9b08b5a0f364567d955508961b37fa33b30`
- latest model-card commit: `e4e0bfcc300e8c70214cb7187662f497683f502a`
- required q4f16 payload: `3,147,142,026` bytes across the ONNX wrappers, external data files, README, and validation summary
- the app defaults to the trained q4f16 repo; set `VITE_LISPER_BROWSER_MODEL_ID=onnx-community/gemma-4-E2B-it-ONNX` only when explicitly A/B testing against the public base model

## Held-Out Eval Status

- Eval path: [kaggle_gemma4_audio_eval.py](/Users/area/repos/lisper/notebooks/kaggle_gemma4_audio_eval.py)
- Kaggle eval notebook version: `4`
- Remote status: `COMPLETE`
- Observed GPU: `Tesla P100-PCIE-16GB`
- Runtime: about `1987` seconds, roughly `33` minutes
- Current issue: some held-out utterances trigger `ValueError('Audio features and audio tokens do not match ...')` during Gemma 4 generation
- Confirmed clip groups:
  - `1069-133699-0039_*`, duration `11.77s`
  - `121-121726-0010_*`, duration `9.81s`
- Root cause inference: this is a clip-level Gemma audio preprocessing mismatch during generation, not a label-specific synthesis issue

Current eval retry policy:
- keep the full clip on the first attempt
- if the exact Gemma audio mismatch occurs, retry once with the first `8.0` seconds only
- write the retry clip under `/kaggle/working/lisper-gemma4-audio-eval/tmp_audio/`
- record `used_truncation`, `original_duration_seconds`, `eval_duration_seconds`, and `retry_reason` in the output rows
- continue past hard failures instead of aborting the notebook

Eval v4 outputs:
- local copy: `data/processed/gemma4_audio/artifacts/eval_v4/tuned_eval.json`
- local rows: `data/processed/gemma4_audio/artifacts/eval_v4/tuned_eval_rows.jsonl`
- summary:
  - `120` rows evaluated
  - `110` successful rows
  - `10` hard-error rows
  - `20` truncated retries
  - `0.0` class match
  - `0.7333` clear-match over all rows
  - `0.0833` reason presence
  - `0.0833` corrective-cue presence
  - `0.0667` encouragement presence

What the retry fix accomplished:
- all `1069-133699-0039_*` rows succeeded after truncation from `11.77s` to `8.0s`
- all `121-121726-0010_*` rows succeeded after truncation from `9.81s` to `8.0s`

Remaining hard-error rows:
- `121-127105-0032_*`, original duration `3.17s`
- `121-127105-0034_*`, original duration `7.41s`

Historical eval v4 conclusion: the eval notebook was robust enough to finish and audit failures, but that adapter was not submission-ready on generative held-out quality. This was superseded by the v18 hybrid full eval, which passes the release gate.

Repo-side eval fixes after eval v4:
- eval now loads the adapter with `max_seq_length` from `bundle["model_config"]`, defaulting to `2048`, instead of hard-coding `1024`
- eval uses `EVAL_MAX_NEW_TOKENS=96` by default to reduce rambling past the four-line schema
- eval embeds the publish verdict in `tuned_eval.json` and writes `publish_verdict.json`
- local downloaded eval artifacts can regenerate `publish_verdict.json` with `python3 src/model/eval_verdict.py`

## Hugging Face Publish Flow

Private-first publish path:
1. Finish the corrected Kaggle retrain.
2. Run the held-out eval notebook and confirm `publish_verdict.json` passes the release gate.
3. Stage and publish the adapter with:

```bash
python3 src/model/publish_hf_adapter.py \
  --repo-id YOUR_HF_USERNAME/lisper-gemma4-e2b-audio-lora \
  --artifact-kind adapter \
  --artifact-dir data/processed/gemma4_audio/artifacts/YOUR_RUN/adapter \
  --eval-json data/processed/gemma4_audio/artifacts/YOUR_EVAL/tuned_eval.json \
  --verdict-json data/processed/gemma4_audio/artifacts/YOUR_EVAL/publish_verdict.json
```

4. Stage and publish the standalone merged model with:

```bash
python3 src/model/publish_hf_adapter.py \
  --repo-id YOUR_HF_USERNAME/lisper-gemma4-e2b-audio-full \
  --artifact-kind merged \
  --artifact-dir data/processed/gemma4_audio/artifacts/YOUR_RUN/merged_model \
  --eval-json data/processed/gemma4_audio/artifacts/YOUR_EVAL/tuned_eval.json \
  --verdict-json data/processed/gemma4_audio/artifacts/YOUR_EVAL/publish_verdict.json
```

Use `--dry-run` first if you want to inspect the staged repo contents locally. The helper writes a full model card, includes eval artifacts, and refuses to publish a failing artifact unless `--allow-failed-verdict` is set.

The merged artifact is a standalone Gemma E2B checkpoint created by merging the trained LoRA into the base weights. It is not a dense full-parameter fine-tune.

Current publication status as of May 4, 2026:
- private adapter repo exists and contains the v16 adapter: `thomasjvu/lisper-gemma4-e2b-audio-lora`
- private merged/full repo exists and contains the v16 merged checkpoint: `thomasjvu/lisper-gemma4-e2b-audio-full`
- trained q4f16 ONNX/WebGPU repo exists: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- current local `publish_verdict.json` passes the release gate from the v18 hybrid full held-out eval, with `2000` successful rows and `0` hard errors
- the app demo path should use the q4f16 browser repo, while the model-quality claim should cite the v18 hybrid eval artifact
- do not publish or promote any v13 artifact; v13 is superseded by v16

## Fine-Tuning Terms

- `SFT` means supervised fine-tuning on example input/output pairs.
- `LoRA` means we do not update all Gemma weights directly. We freeze the base model and train small low-rank update matrices on top of selected layers.
- `QLoRA` means LoRA with a quantized base model, usually `4-bit`, so training fits on smaller GPUs.
- `Merged model` means the trained LoRA updates are folded back into the base model weights and saved as one standalone checkpoint.
- `Full-parameter fine-tune` means training all model weights directly. That is not what this project is doing on Kaggle.

For Lisper:
- the training method is `Unsloth QLoRA / LoRA SFT`
- the adapter is the primary trained artifact
- the standalone full-model deliverable is a merged base+LoRA checkpoint, not a dense full-parameter fine-tune

## Local Sanity Command

Run this before you switch to Kaggle if you want the repo to print the handoff summary again:

```bash
python src/model/finetune.py --train
```

## MCP And Skill Setup

This machine now has:
- a Codex MCP entry named `kaggle` pointing at `https://www.kaggle.com/mcp`
- the third-party Kaggle skill installed under `~/.codex/skills/kaggle`
- local Python packages `kaggle` and `kagglehub`

Current auth state:
- Kaggle CLI is installed at `/Users/area/Library/Python/3.14/bin/kaggle`.
- Current local Kaggle CLI auth can be switched by replacing `~/.kaggle/access_token`.
- Use `python3 src/model/kaggle_prepare_account_handoff.py --username alkahestai` to prepare private notebook copies owned by `alkahestai` while reading the shared `thomasjvu` datasets.
- The Kaggle MCP entry needs `KAGGLE_API_TOKEN` in the environment that launches Codex, then a full Codex restart.

To make the Kaggle CLI / skill usable, store the token for local tools too:

```bash
mkdir -p ~/.kaggle
printf '%s' 'YOUR_KAGGLE_API_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```
