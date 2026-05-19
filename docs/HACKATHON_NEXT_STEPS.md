# Hackathon Next Steps

This file is the operational closeout checklist. The model-training/eval gate is complete; the remaining work is app validation, demo assets, and public submission packaging.

## Current Decision

There is one canonical submission path:

- App/browser demo: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Server fallback demo: `thomasjvu/lisper-zerogpu`
- Submitted trained model family: Gemma 4 E2B, Unsloth LoRA fine-tune
- Submitted evaluation result: v18 hybrid acoustic+Gemma held-out eval

Do not include the E4B Space in the hackathon submission. It is a private post-submission experiment, not a passed model gate.

Use the current q4f16 ONNX/WebGPU package for the browser demo:

- `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`

Use the v18 hybrid held-out eval as the release-quality evaluation result:

- eval notebook: `alkahestai/lisper-gemma-4-audio-adapter-eval`, version `16`
- eval artifact: `data/processed/gemma4_audio/artifacts/eval_v18_hybrid_full_v16/lisper-gemma4-audio-eval/`
- verdict: `pass`
- rows: `2000`
- hard errors: `0`
- class match: `0.976`
- clear/non-clear match: `0.989`
- exact four-line format: `1.0`

Do not describe the browser q4f16 artifact as a pure v18 ONNX export unless we actually export and validate a v18 merged checkpoint into q4f16.

## Model Inventory

### Base Model

- ID: `google/gemma-4-E2B-it`
- Role: foundation model used for LoRA training and fallback A/B tests
- Submission wording: "We fine-tuned Gemma 4 E2B with Unsloth LoRA."
- Do not present this as the trained Lisper model by itself.

### LoRA Adapter

- Current evaluated source: `alkahestai/lisper-gemma4-audio-lora-v18`
- Public/owner target: `thomasjvu/lisper-gemma4-e2b-audio-lora`
- Role: compact trainable delta on top of Gemma 4 E2B
- Submission wording: "The adapter contains the Lisper fine-tuning deltas."

### Merged Full Model

- Target repo: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Role: standalone Gemma 4 E2B checkpoint with the LoRA merged into base weights
- Submission wording: "This is a merged base+LoRA checkpoint, not a dense full-parameter fine-tune."

### Browser q4f16 Model

- ID: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- Role: browser/WebGPU demo package used by the app
- Expected payload: about `3.15 GB`
- Submission wording: "The browser demo runs a q4f16 ONNX/WebGPU package for consumer-device testing."

### ZeroGPU Space

- ID: `thomasjvu/lisper-zerogpu`
- Role: server-side fallback for users whose browser cannot comfortably run the q4f16 package
- Model source: `thomasjvu/lisper-gemma4-e2b-audio-full`
- Acoustic sidecar: packaged v18 ExtraTrees hint model, `acoustic_extratrees_v18.joblib`
- Submission wording: "The Hugging Face ZeroGPU Space runs the same E2B model lineage server-side, with the v18 acoustic sidecar used for the held-out class hint."
- Do not link `thomasjvu/lisper-zerogpu-e4b` in submission materials.

### q2f16 Experiment

- Role: size experiment only
- Submission wording: do not mention unless asked; keep it out of the main submission path.

### E4B Experiment

- Role: private follow-up experiment only
- Current state: LoRA adapter exists and has a runtime smoke, but no merged full E4B checkpoint and no passing held-out eval/publish verdict
- Submission wording: do not include it in the main model list; if asked, say E4B is future work

## What The Evaluation Means

"Quality gate" is internal shorthand. In public materials, call it a held-out evaluation.

Use this wording:

> We evaluated the final hackathon pipeline on 2,000 held-out synthetic lisp/clear audio rows. The evaluated pipeline combines Gemma 4 audio/coaching generation with an acoustic feature classifier for the lisp-class hint and an audited fallback for exact Gemma audio token/feature mismatch cases. It completed with 0 hard errors, 97.6% class match, 98.9% clear/non-clear match, and 100% exact four-line response format.

Avoid this wording:

> The q4f16 browser model alone gets 97.6% class accuracy.

That would be too strong unless we separately run the same held-out eval directly against the q4f16 browser package.

## Step-By-Step Remaining Work

1. Freeze the model story.
   - Use q4f16 ONNX/WebGPU for the demo runtime.
   - Use v18 hybrid eval as the model-quality result.
   - Keep q2f16 experimental.

2. Validate Gemma Lab manually in a WebGPU browser.
   - Load `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`.
   - Run text-only generation.
   - Run one short audio prompt.
   - Run one image prompt.
   - Run one video-frame prompt.
   - Run one combined audio+image prompt if time allows.
   - Save screenshots of the model-loaded state and at least one useful response.

3. Validate the main app demo path.
   - Confirm the app reports the q4f16 model ID, not the public base model.
   - Complete the assessment flow with microphone or uploaded audio.
   - Complete one practice attempt.
   - Confirm fallback errors are readable if WebGPU/model loading fails.

4. Do one final app build pass.
   - `npm run build:gemma-lab`
   - `npm run build:web`
   - Do not run extra heavy checks in parallel.

5. Finalize public artifacts.
   - Make the GitHub app repo public or submission-accessible.
   - Make the Hugging Face LoRA, merged, and q4f16 repos public only when ready to submit.
   - Ensure model cards include the held-out eval wording above.
   - Keep q2f16 private or clearly marked experimental.

6. Finalize Kaggle artifacts.
   - Link the training notebook.
   - Link the eval notebook and `publish_verdict.json`.
   - Link the dataset bundle.
   - Make sure ownership is under `thomasjvu` for final submission, even if `alkahestai` was used for extra GPU quota.

7. Record the demo.
   - Show the problem: lisp-specific coaching gap.
   - Show the app loading/running q4f16 in browser.
   - Show audio practice and corrective cue.
   - Show the evaluation result table.
   - Keep the video to 2-3 minutes.

8. Submit.
   - Include GitHub, Hugging Face, Kaggle notebook/dataset/model links.
   - Include the evaluation metrics table.
   - Include limitations: synthetic lisp data, not clinical diagnosis, hybrid eval path.

## Completion Criteria

We can call the project submission-ready when:

- Gemma Lab browser smoke passes on the target machine.
- Main app demo flow passes on the target machine.
- Public model cards match the model inventory above.
- Demo video and screenshots are done.
- Submission links are public and consistent.
