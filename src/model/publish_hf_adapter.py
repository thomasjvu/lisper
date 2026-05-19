#!/usr/bin/env python3
"""Stage and publish a Lisper Gemma 4 audio artifact to Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
DEFAULT_BUNDLE_PATH = REPO_ROOT / "data" / "processed" / "gemma4_audio" / "bundle.json"
DEFAULT_EVAL_PATH = REPO_ROOT / "data" / "processed" / "gemma4_audio" / "artifacts" / "eval_v4" / "tuned_eval.json"
DEFAULT_VERDICT_PATH = (
    REPO_ROOT / "data" / "processed" / "gemma4_audio" / "artifacts" / "eval_v4" / "publish_verdict.json"
)
DEFAULT_ADAPTER_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "full_run_v16"
    / "lisper-gemma4-audio"
    / "adapter"
)
DEFAULT_MERGED_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "full_run_v16"
    / "lisper-gemma4-audio"
    / "merged_model"
)
DEFAULT_PUBLISH_ROOT = REPO_ROOT / "data" / "processed" / "gemma4_audio" / "artifacts" / "hf_publish"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def build_model_card(
    repo_id: str,
    artifact_kind: str,
    bundle: dict,
    training_metadata: dict | None,
    eval_payload: dict | None,
    verdict_payload: dict | None,
) -> str:
    summary = bundle["summary"]
    model_config = bundle["model_config"]
    build_config = bundle.get("build_config") or {}
    eval_summary = (eval_payload or {}).get("summary") or {}
    verdict_reasons = (verdict_payload or {}).get("reasons") or []

    training_lines = [
        f"- Base model: `{model_config['model_name']}`",
        "- Fine-tuning method: Unsloth QLoRA / LoRA",
        f"- Max sequence length: `{model_config['max_seq_length']}`",
        f"- LoRA rank / alpha: `{model_config['lora_rank']} / {model_config['lora_alpha']}`",
        f"- Target modules: `{model_config['target_modules']}`",
        f"- Gradient accumulation: `{model_config['gradient_accumulation_steps']}`",
        f"- Learning rate: `{model_config['learning_rate']}`",
        f"- Planned max steps: `{model_config['full_train_max_steps']}`",
    ]
    if training_metadata:
        training_lines.extend(
            [
                f"- Selected checkpoint: `{training_metadata.get('selected_checkpoint_path')}`",
                f"- Best eval loss: `{training_metadata.get('best_metric')}`",
                f"- Completed global step: `{training_metadata.get('global_step')}`",
                f"- GPUs: `{training_metadata.get('gpu_info')}`",
            ]
        )
        if artifact_kind == "merged":
            training_lines.append(f"- Merged model dir: `{training_metadata.get('merged_model_dir')}`")

    eval_lines = [
        f"- Successful rows: `{eval_summary.get('success_count', 0)} / {eval_summary.get('count', 0)}`",
        f"- Hard errors: `{eval_summary.get('hard_error_count', 0)}`",
        f"- Class match (successful only): `{eval_summary.get('class_match_successful_only', 0.0):.4f}`",
        f"- Clear-match (successful only): `{eval_summary.get('clear_match_successful_only', 0.0):.4f}`",
        f"- Exact format (successful only): `{eval_summary.get('format_exact_successful_only', 0.0):.4f}`",
        f"- Encouragement present (successful only): `{eval_summary.get('has_encouragement_successful_only', 0.0):.4f}`",
    ]
    if verdict_payload:
        eval_lines.append(f"- Publish verdict: `{verdict_payload.get('status', 'unknown')}`")
    if verdict_reasons:
        eval_lines.append("- Release gate reasons:")
        eval_lines.extend([f"  - {reason}" for reason in verdict_reasons])

    speaker_count = build_config.get("available_summary", {}).get("available_speakers")
    source_utterances = build_config.get("available_summary", {}).get("available_source_utterances")
    library_name = "peft" if artifact_kind == "adapter" else "transformers"
    artifact_title = "LoRA adapter" if artifact_kind == "adapter" else "merged full model"
    artifact_lines = [
        "- Adapter weights and processor files" if artifact_kind == "adapter" else "- Full merged model weights and processor files",
        "- `training_metadata.json`",
        "- `eval_summary.json`",
        "- `publish_verdict.json` when available",
    ]
    tag_lines = [
        "- unsloth",
        "- gemma4",
        "- audio",
        "- speech-therapy",
        "- pronunciation",
        "- kaggle",
        "- hackathon",
        "- peft" if artifact_kind == "adapter" else "- merged-model",
        "- lora" if artifact_kind == "adapter" else "- full-model",
    ]

    return f"""---
base_model: google/gemma-4-E2B-it
library_name: {library_name}
pipeline_tag: text-generation
license: gemma
tags:
{chr(10).join(tag_lines)}
---

# {repo_id}

Lisper is a raw-audio lisp coaching prototype for the Kaggle Gemma 4 Good Hackathon. This repository contains a {artifact_title} for `google/gemma-4-E2B-it` trained to classify `clear`, `frontal`, `lateral`, `dental`, and `palatal` productions and answer with brief, supportive corrective coaching.

## Intended Use

- Analyze a short `16 kHz` mono speech clip
- Return exactly four labeled lines:
  - `Detected class`
  - `Reason`
  - `Corrective cue`
  - `Encouragement`

This artifact is intended for hackathon experimentation, not clinical deployment.

## Training Data

- Train rows: `{summary['record_counts']['train']}`
- Validation rows: `{summary['record_counts']['val']}`
- Test rows: `{summary['record_counts']['test']}`
- Train speakers: `{summary['speaker_counts']['train']}`
- Validation speakers: `{summary['speaker_counts']['val']}`
- Test speakers: `{summary['speaker_counts']['test']}`
- Labels: `{summary['lisp_type_counts']['train']}`
- Available source speakers before expansion: `{speaker_count}`
- Available source utterances before expansion: `{source_utterances}`

The dataset is speaker-disjoint and built from LibriSpeech source clips expanded into one clear reference plus four synthetic lisp variants per source utterance.

## Training Procedure

{chr(10).join(training_lines)}

## Evaluation Snapshot

{chr(10).join(eval_lines)}

## Limitations

- The training data uses synthetic lisp variants rather than clinically collected disorder audio.
- Held-out evaluation currently shows the adapter still needs improvement on exact label behavior and schema adherence.
- This model should be treated as a private experimental artifact until the release gate passes cleanly.

## Repository Contents

{chr(10).join(artifact_lines)}

## Source Project

- Local project repo: `{REPO_ROOT}`
- Bundle metadata: `{repo_relative(DEFAULT_BUNDLE_PATH)}`
"""


def stage_publish_dir(
    artifact_dir: Path,
    staging_dir: Path,
    model_card: str,
    training_metadata: dict | None,
    eval_payload: dict | None,
    verdict_payload: dict | None,
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    shutil.copytree(artifact_dir, staging_dir)
    (staging_dir / "README.md").write_text(model_card, encoding="utf-8")
    if training_metadata is not None:
        (staging_dir / "training_metadata.json").write_text(
            json.dumps(training_metadata, indent=2),
            encoding="utf-8",
        )
    if eval_payload is not None:
        (staging_dir / "eval_summary.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    if verdict_payload is not None:
        (staging_dir / "publish_verdict.json").write_text(
            json.dumps(verdict_payload, indent=2), encoding="utf-8"
        )


def resolve_hf_cli() -> str:
    """Return the available Hugging Face CLI command."""

    hf = shutil.which("hf")
    if hf:
        return hf
    legacy = shutil.which("huggingface-cli")
    if legacy:
        return legacy
    raise RuntimeError(
        "No Hugging Face CLI found. Install the current CLI with "
        "`curl -LsSf https://hf.co/cli/install.sh | bash -s`, or make `huggingface-cli` available on PATH."
    )


def publish_to_hub(repo_id: str, staging_dir: Path, private: bool) -> None:
    cli = resolve_hf_cli()
    if Path(cli).name == "hf":
        create_cmd = [cli, "repos", "create", repo_id, "--type", "model", "--exist-ok"]
        if private:
            create_cmd.append("--private")
        subprocess.run(create_cmd, check=True)
        upload_cmd = [cli, "upload-large-folder", repo_id, str(staging_dir), "--type", "model"]
        if private:
            upload_cmd.append("--private")
        subprocess.run(upload_cmd, check=True)
        return

    upload_cmd = [cli, "upload-large-folder", repo_id, str(staging_dir), "--repo-type", "model"]
    if private:
        upload_cmd.append("--private")
    subprocess.run(upload_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a Lisper Gemma 4 audio artifact to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo id")
    parser.add_argument(
        "--artifact-kind",
        choices=("adapter", "merged"),
        default="adapter",
        help="Whether the published directory is a LoRA adapter or a merged full model",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory to publish. Defaults to the standard adapter or merged path based on --artifact-kind.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH, help="Bundle metadata path")
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_PATH, help="Held-out evaluation JSON")
    parser.add_argument(
        "--verdict-json",
        type=Path,
        default=DEFAULT_VERDICT_PATH,
        help="Optional publish verdict JSON. Required for strict publication unless --allow-failed-verdict is used.",
    )
    parser.add_argument("--publish-root", type=Path, default=DEFAULT_PUBLISH_ROOT, help="Local staging root")
    parser.add_argument("--allow-failed-verdict", action="store_true", help="Stage/publish even if verdict fails")
    parser.add_argument("--dry-run", action="store_true", help="Only stage files locally")
    parser.add_argument("--public", action="store_true", help="Create a public Hugging Face repo instead of private")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or args.adapter_dir or (
        DEFAULT_ADAPTER_DIR if args.artifact_kind == "adapter" else DEFAULT_MERGED_DIR
    )

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    if args.artifact_kind == "merged":
        merged_weights = artifact_dir / "model.safetensors"
        if not merged_weights.exists() or merged_weights.stat().st_size <= 0:
            raise RuntimeError(
                f"Merged artifact is not publishable because {merged_weights} is missing or empty."
            )
    if not args.bundle_path.exists():
        raise FileNotFoundError(f"Bundle metadata not found: {args.bundle_path}")

    bundle = load_json(args.bundle_path)
    training_metadata_path = artifact_dir / "training_metadata.json"
    if not training_metadata_path.exists() and args.artifact_kind == "merged":
        candidate = artifact_dir.parent / "adapter" / "training_metadata.json"
        if candidate.exists():
            training_metadata_path = candidate
    training_metadata = load_json(training_metadata_path) if training_metadata_path.exists() else None
    eval_payload = load_json(args.eval_json) if args.eval_json and args.eval_json.exists() else None
    verdict_payload = load_json(args.verdict_json) if args.verdict_json and args.verdict_json.exists() else None

    if verdict_payload and verdict_payload.get("status") != "pass" and not args.allow_failed_verdict:
        raise RuntimeError(
            {
                "repo_id": args.repo_id,
                "publish_blocked": True,
                "verdict_status": verdict_payload.get("status"),
                "reasons": verdict_payload.get("reasons", []),
            }
        )

    if args.verdict_json and not verdict_payload and not args.allow_failed_verdict:
        raise RuntimeError("Verdict file was requested but could not be loaded.")

    staging_dir = args.publish_root / args.repo_id.replace("/", "__")
    model_card = build_model_card(
        args.repo_id,
        args.artifact_kind,
        bundle,
        training_metadata,
        eval_payload,
        verdict_payload,
    )
    stage_publish_dir(
        artifact_dir,
        staging_dir,
        model_card,
        training_metadata,
        eval_payload,
        verdict_payload,
    )

    summary = {
        "repo_id": args.repo_id,
        "artifact_kind": args.artifact_kind,
        "artifact_dir": str(artifact_dir),
        "staging_dir": str(staging_dir),
        "private_repo": not args.public,
        "dry_run": args.dry_run,
        "verdict_status": verdict_payload.get("status") if verdict_payload else None,
    }
    print(json.dumps(summary, indent=2))

    if not args.dry_run:
        publish_to_hub(args.repo_id, staging_dir, private=not args.public)


if __name__ == "__main__":
    main()
