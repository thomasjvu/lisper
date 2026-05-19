# Hackathon Completion Checklist

This is the current completion gate after the browser text-generation fix and v18 eval remediation. For the operational submission checklist, see [HACKATHON_NEXT_STEPS.md](/Users/area/repos/lisper/docs/HACKATHON_NEXT_STEPS.md).

## Current State

- GitHub app repo is pushed and browser text generation works.
- Current eval model source is v18 LoRA: `alkahestai/lisper-gemma4-audio-lora-v18`.
- Current browser target remains q4f16: `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`.
- Training and held-out model evaluation are complete for the hackathon closeout gate.
- App/submission work remains: browser demo validation, final UI/demo polish, and public submission packaging.
- The 20-row hybrid eval smoke gate passed with `hard_error_count=0` and all tracked metrics at `1.0`.
- The first 120-row gate failed only because two source utterances produced 10 Gemma audio-token hard errors; successful rows were perfect.
- Eval v14 passed the 120-row gate with `hard_error_count=0`; 10 rows used the audited acoustic-template fallback for exact Gemma audio-token mismatches.
- Eval v15 passed the 500-row intermediate gate with `hard_error_count=0`.
- Eval v16 passed the full 2,000-row held-out gate with `hard_error_count=0`.
- Full eval artifacts are stored under `data/processed/gemma4_audio/artifacts/eval_v18_hybrid_full_v16/`.
- App build checks pass: `npm run build:gemma-lab` and `npm run build:web`.

## Submission Readiness Gate

The model/eval gate is complete. Do not call the full hackathon submission complete until all are true:

- `publish_verdict.json` status is `pass`.
- `hard_error_count` is `0`.
- `class_match_successful_only >= 0.60`.
- `clear_match_successful_only >= 0.90`.
- `format_exact_successful_only >= 0.95`.
- `has_encouragement_successful_only >= 0.90`.
- Gemma Lab can generate in Chromium WebGPU with the trained q4f16 repo.
- The main app can use the same trained q4f16 browser model config.

## Next Execution Order

1. Verify the app demo path against the final q4f16 browser target in a WebGPU browser.
2. Use the current q4f16 artifact as the browser demo target for hackathon submission, with the v18 hybrid eval path documented as the held-out evaluation result.
3. Treat a pure v18 q4f16 ONNX re-export as post-gate hardening unless a downloaded v18 merged checkpoint is available and can be exported quickly.
4. Keep q2f16 experimental and out of the submission path.

Plain-English version:
- The trained/evaluated model work is complete enough for the hackathon.
- The browser app uses the q4f16 ONNX/WebGPU package because that is what can run locally in Chrome/WebGPU.
- The public writeup should cite the v18 held-out eval as the evaluation result, not claim that the q4f16 browser package alone was separately evaluated at those metrics.

Current local state: the Kaggle CLI is authenticated as `alkahestai`, and the private `thomasjvu` datasets grant `alkahestai` `WRITER` access through Kaggle dataset collaborator metadata. Kaggle notebook collaborator access is not exposed by this CLI, so spending `alkahestai` GPU quota uses the prepared private notebook copies under the `alkahestai` account.

If `thomasjvu` is out of GPU quota, use the `alkahestai` handoff. Push the eval notebook with `EVAL_LIMIT=20` first, then repush with `EVAL_LIMIT=120` only after the smoke run completes:

```bash
python3 src/model/kaggle_prepare_account_handoff.py --username alkahestai
/Users/area/Library/Python/3.14/bin/kaggle kernels push -p notebooks/kaggle_upload/alkahestai/lisper-gemma-4-audio-unsloth-training
/Users/area/Library/Python/3.14/bin/kaggle kernels push -p notebooks/kaggle_upload/alkahestai/lisper-gemma-4-audio-adapter-eval
```

The target account still needs Kaggle auth active locally before pushing. Do not run the GPU notebooks while authenticated as `thomasjvu` if that account is out of quota.

## Submission Assets

- Update the model cards with the final passing eval summary.
- Make HF repos public only at submission time.
- Keep v13 references marked historical only.
- Record demo footage from the q4f16 browser path, not the raw 16-bit checkpoint.
