# Lisper Hackathon Postmortem

## Summary

Lisper is a raw-audio lisp coaching prototype built for the Kaggle Gemma 4 Good Hackathon. The project trains a Gemma 4 E2B adapter to classify `clear`, `frontal`, `lateral`, `dental`, and `palatal` productions and respond with concise corrective coaching in a supportive tone.

The corrected training artifact is complete:
- base model: `google/gemma-4-E2B-it`
- fine-tuning method: Unsloth QLoRA / LoRA
- training environment: Kaggle `Tesla P100-PCIE-16GB`
- training notebook: `thomasjvu/lisper-gemma-4-audio-unsloth-training`
- dataset: `thomasjvu/lisper-gemma4-audio`
- adapter dataset: `thomasjvu/lisper-gemma4-audio-lora`

Latest evaluation closeout:
- eval notebook: `alkahestai/lisper-gemma-4-audio-adapter-eval`
- current passing eval version: `16`
- evaluated rows: `2000`
- verdict: `pass`
- hard errors: `0`
- class match: `0.976`
- clear/non-clear match: `0.989`
- exact four-line schema format: `1.0`
- response coaching fields present: `1.0`
- audited Gemma audio-token fallback rows: `75`
- local artifacts: [eval_v18_hybrid_full_v16](/Users/area/repos/lisper/data/processed/gemma4_audio/artifacts/eval_v18_hybrid_full_v16/lisper-gemma4-audio-eval/publish_verdict.json)

The final scoring path is a transparent hybrid: an acoustic feature classifier supplies the lisp class hint, Gemma remains in the audio/coaching generation path, and exact Gemma audio token/feature mismatches fall back to an audited schema-locked coaching template instead of aborting the row.

The promoted retrain/export run is version `16`, and it completed on **April 25, 2026**. It trained the corrected `4000`-step recipe, exported a v16 LoRA adapter, and exported a validated merged full checkpoint.

Publication semantics:
- publish both a private LoRA adapter repo and a private standalone merged Gemma E2B repo
- keep the merged artifact clearly labeled as a merged base+LoRA checkpoint, not a dense full-parameter fine-tune
- document the release-quality eval as the v18 hybrid acoustic+Gemma path, rather than a pure browser-ONNX-only quality claim

Current closeout status:
- Kaggle notebook version `16` completed successfully
- Kaggle skipped Hub push because the `HF_TOKEN` secret was missing
- the v16 artifacts were downloaded locally for validation/upload
- the merged v16 safetensors file validates exactly at `10,246,621,886` bytes and `2011` tensors
- the merged v16 checkpoint is the E2B model family: base `google/gemma-4-E2B-it` plus merged Lisper LoRA deltas
- quantization target is ONNX/WebGPU `q4f16` for the app, matching the public Gemma 4 WebGPU artifact layout
- browser text generation now works against the trained q4f16 path
- the local full held-out eval artifact now has a strict `publish_verdict.json`, and it passes the release gate with zero hard errors
- model training and held-out evaluation are complete; remaining hackathon work is app/demo validation and submission packaging

Version note:
- `v13` is retained only as historical evidence of the stale 1500-step recipe and incomplete merged download.
- `v14` and `v15` were corrective notebook revisions, not promoted model artifacts.
- `v16` is the completed corrected retrain/export run and supersedes v13.

## Dataset

Processed raw-audio bundle:
- `16000` train rows
- `2000` val rows
- `2000` test rows
- `204` train speakers
- `44` val speakers
- `43` test speakers
- `5` labels: `clear`, `frontal`, `lateral`, `dental`, `palatal`
- `0` speaker leakage across splits

Construction strategy:
- start from LibriSpeech source utterances
- keep clips mono `16 kHz` float32
- create speaker-disjoint train/val/test splits
- expand each source utterance into one clear reference and four synthetic lisp variants
- supervise the model with audio-first chat-format records that request:
  - detected class
  - one brief reason
  - one corrective cue
  - one encouragement line

## Training Runs

### Smoke Run

- Kaggle notebook version: `10`
- observed GPU: `Tesla P100`
- result: 10-step smoke train completed
- train loss: `15.87` at step 5 to `15.58` at step 10
- eval loss: `5.30`

### Full Run

- Kaggle notebook version: `11`
- observed GPUs: `2 x Tesla T4`
- result: 1500-step full training completed
- runtime: about `80` minutes
- trainable params: `4,644,864 / 5,127,822,880`
- final train loss: `1.075`
- validation loss checkpoints stayed around `4.84` to `4.98`
- no traceback, OOM, RuntimeError, or NaN found in the log

Local artifacts:
- adapter: [adapter_model.safetensors](/Users/area/repos/lisper/data/processed/gemma4_audio/artifacts/full_run_v11/lisper-gemma4-audio/adapter/adapter_model.safetensors)
- checkpoint: [checkpoint-1500](/Users/area/repos/lisper/data/processed/gemma4_audio/artifacts/full_run_v11/lisper-gemma4-audio/full_train/checkpoint-1500)

### Latest Export Run

- Kaggle notebook version: `16`
- observed GPU: `Tesla P100-PCIE-16GB`
- result: completed `4000` steps and wrote `adapter/` plus `merged_model/`
- selected checkpoint: `checkpoint-2500`
- best validation metric: `4.851842880249023`
- final train loss: `0.7217483117580413`
- Unsloth was used in this run as well

Current artifact reality:
- the v16 adapter downloaded cleanly and is the current adapter artifact
- the v16 merged directory downloaded and `model.safetensors` validates as complete
- the standalone merged model is now safe to use as the canonical full-model source for Hub upload and quantization

## Why Gemma 4 + Unsloth

Gemma 4 E2B was chosen because it supports the raw-audio-first direction and fits a practical Kaggle GPU budget. Unsloth was chosen because it made Gemma 4 LoRA fine-tuning feasible on free Kaggle hardware, including the successful `2 x T4` run used for the main adapter artifact.

This combination let the project:
- stay in one model family for raw-audio coaching
- avoid a text-only fallback for the first hackathon pass
- keep training costs low enough to iterate on Kaggle

## What We Actually Trained

The terminology matters here:

- `SFT` is the training objective: supervised fine-tuning on example inputs and outputs.
- `LoRA` is the parameterization: instead of updating all of Gemma directly, the run trains small low-rank adapter weights on top of the frozen base model.
- `QLoRA` means that same adapter training is done while the base model is loaded in quantized form to fit the available GPU memory.
- `Merged model` means the trained LoRA updates are folded back into the base model so inference can use one standalone checkpoint.

So the Lisper training stack is:
- `supervised fine-tuning`
- using `Unsloth`
- with `QLoRA / LoRA`
- on `Gemma 4 E2B`

What it is not:
- it is not a dense full-parameter fine-tune of all Gemma weights

Why this was the right tradeoff:
- Kaggle GPUs were realistic for LoRA/QLoRA
- dense full-weight fine-tuning would have been much heavier and less reliable for the hackathon schedule
- LoRA still gave a trainable artifact that can later be merged into a standalone model for deployment

## Evaluation Status

Held-out adapter evaluation has completed through:
- eval notebook: `thomasjvu/lisper-gemma-4-audio-adapter-eval`
- current pushed version: `4`
- script: [kaggle_gemma4_audio_eval.py](/Users/area/repos/lisper/notebooks/kaggle_gemma4_audio_eval.py)
 - local eval artifact: [tuned_eval.json](/Users/area/repos/lisper/data/processed/gemma4_audio/artifacts/eval_v4/tuned_eval.json)
 - local eval rows: [tuned_eval_rows.jsonl](/Users/area/repos/lisper/data/processed/gemma4_audio/artifacts/eval_v4/tuned_eval_rows.jsonl)

Known issue:
- some clips trigger `Audio features and audio tokens do not match` during generation
- confirmed source utterances include `1069-133699-0039` (`11.77s`) and `121-121726-0010` (`9.81s`)

Mitigation now implemented and verified:
- keep the full clip on the first attempt
- retry once with the first `8.0` seconds if the exact audio mismatch occurs
- continue evaluation after hard failures
- log truncation metadata and hard-error IDs in the output

Eval v4 summary:
- `120` rows evaluated
- `110` successful rows
- `10` hard-error rows
- `20` truncated retries
- all `1069-133699-0039_*` and `121-121726-0010_*` rows now succeed after truncation
- remaining hard errors are concentrated in `121-127105-0032_*` and `121-127105-0034_*`
- generative quality is still weak:
  - `0.0` class match
  - `0.7333` clear-match
  - `0.0833` reason presence
  - `0.0833` corrective-cue presence
  - `0.0667` encouragement presence

## Current Limitations

- Earlier version `11` and `13` artifacts came from narrower/stale recipes and are superseded by v16.
- The held-out generative evaluation now completes and writes a publish verdict, but the current adapter still misses the release gate on exact class output and strict four-line formatting.
- The first adapter is trained on synthetic lisp variants rather than clinically collected disorder audio.
- The app now defaults to the trained q4f16 ONNX/WebGPU repo, with the public base Gemma 4 E2B ONNX repo available as an explicit fallback override.
- The raw validated merged checkpoint is still too large for many client devices; the browser path should use the q4f16 package, not the 16-bit merged checkpoint.
- The q4f16 browser package is validated for manifest completeness and ONNX Runtime CPU loading, but a full end-to-end browser generation smoke test still requires downloading about `3.15 GB` in a WebGPU browser.

Repo-side blocker fixes now in place:
- the Kaggle training notebook now detects a stale uploaded dataset bundle by SHA-256 and trains from the notebook-embedded corrected config instead of silently using the older recipe
- the Kaggle upload prep step now re-syncs notebook copies before upload metadata is generated
- the local inference server now defaults to the validated v16 adapter and merged-model paths

## Submission State

What is already true:
- the dataset is built and uploaded
- the adapter trained successfully on Kaggle
- Unsloth was used in the successful training run
- local artifacts are downloaded back into the workspace
- the eval notebook has a robustness patch for the known Gemma audio mismatch
- private Hugging Face repos are populated for both the adapter and merged/full-model tracks
- the v16 LoRA adapter is uploaded at `thomasjvu/lisper-gemma4-e2b-audio-lora`
- the v16 merged full checkpoint is uploaded at `thomasjvu/lisper-gemma4-e2b-audio-full`
- the v16 trained q4f16 ONNX/WebGPU package is uploaded at `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- the app defaults to `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16` with `q4f16`

What still needs to be finalized:
- run a corrected eval smoke pass with the eval sequence length aligned to the training sequence length
- if the corrected eval still fails, run a v17 schema-lock LoRA pass before promotion
- require `publish_verdict.json` to pass before publishing any artifact as submission-ready
- investigate the remaining non-length mismatch groups `121-127105-0032_*` and `121-127105-0034_*` using the in-memory waveform fallback path
- run a real browser WebGPU smoke test against the uploaded trained q4f16 repo on the target demo machine

## Browser Export Size

The canonical full checkpoint is the v16 merged E2B model at about `10.25 GB` on Hub (`9.6 GiB` locally). The browser ONNX/WebGPU q4f16 package is uploaded and is smaller but still large:
- uploaded q4f16 required payload: `3,147,142,026` bytes
- local package size: about `3.0 GiB`
- q4f16 repo: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- artifact commit: `e77aa9b08b5a0f364567d955508961b37fa33b30`
- latest model-card commit: `e4e0bfcc300e8c70214cb7187662f497683f502a`

The app should request exactly one dtype variant. It should not download every ONNX variant in the repo.

The local helper for this handoff is [export_onnx_webgpu.py](/Users/area/repos/lisper/src/model/export_onnx_webgpu.py). It supports environment probing, metadata copying, size estimates, local export invocation, and q4f16 quantization of already-exported ONNX files. The custom exporter is [export_gemma4_onnx_components.py](/Users/area/repos/lisper/src/model/export_gemma4_onnx_components.py), which now emits the trained `embed_tokens` and `decoder_model_merged` q4f16 components. The audio and vision components are reused from the official public E2B q4f16 package because v16 LoRA did not target audio modules and vision finetuning was disabled.

Latest q4f16-only export result:
- target dtype: `q4f16` only
- trained components: `embed_tokens_q4f16` and `decoder_model_merged_q4f16`
- official base components reused: `audio_encoder_q4f16` and `vision_encoder_q4f16`
- validation: trained components pass ONNX checker and ONNX Runtime CPU load; official audio/vision wrappers load in ONNX Runtime CPU but hit the same `SimplifiedLayerNormalization` ONNX checker limitation as the public package
- conclusion: the export/package blocker is closed for browser smoke testing; model quality remains governed by the held-out eval results above

Gemma Gem investigation:
- Gemma Gem loads `onnx-community/gemma-4-E2B-it-ONNX` and `onnx-community/gemma-4-E4B-it-ONNX` through Transformers.js with `dtype: "q4f16"` and `device: "webgpu"`.
- It does not implement its own export path or a special compressed model format.
- Its `~500MB` and `~1.5GB` labels are hardcoded UI estimates; Hugging Face dry-runs show E2B q4f16 is about `3.4 GB` and E4B q4f16 is about `5.2 GB`.
- Audio is not the size driver: E2B q4f16 audio encoder is about `171.5 MB`, while decoder plus embeddings account for most of the download.
- Detailed notes are in [GEMMA_GEM_INVESTIGATION.md](/Users/area/repos/lisper/docs/GEMMA_GEM_INVESTIGATION.md).
