#!/usr/bin/env python3
"""Prepare Gemma 4 E2B audio fine-tuning bundles for Kaggle + Unsloth."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
MODEL_DIR = REPO_ROOT / "src" / "model"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MULTIMODAL_DIR = PROCESSED_DIR / "gemma4_audio"
MANIFEST_FILE = MULTIMODAL_DIR / "manifest.jsonl"
BUILD_CONFIG_FILE = MULTIMODAL_DIR / "build_config.json"
MESSAGE_DIR = MULTIMODAL_DIR / "messages"
TRAINING_JSONL = PROCESSED_DIR / "lisper_train.jsonl"
VAL_JSONL = PROCESSED_DIR / "lisper_val.jsonl"
TEST_JSONL = PROCESSED_DIR / "lisper_test.jsonl"
BUNDLE_FILE = MULTIMODAL_DIR / "bundle.json"

SYSTEM_PROMPT = (
    "You are Lisper, a speech therapy assistant for lisp practice. "
    "Use the audio as the primary evidence. "
    "The audio clip is already attached in this message; do not ask the user to provide audio. "
    "Return exactly four labeled lines in this order: "
    "Detected class, Reason, Corrective cue, Encouragement. "
    "The Detected class value must be exactly one of: clear, frontal, lateral, dental, palatal. "
    "Do not use any other diagnostic category."
)
SCHEMA_LOCK_INSTRUCTION = (
    "The Detected class value must be exactly one of: clear, frontal, lateral, dental, palatal. "
    "Do not use any other diagnostic category."
)


def repo_relative(path: Path) -> str:
    """Return a repo-relative path string."""

    return str(path.relative_to(REPO_ROOT))


def load_manifest(manifest_path: Path = MANIFEST_FILE) -> list[dict]:
    """Load manifest rows from JSONL."""

    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No manifest rows found in {manifest_path}")
    return rows


def load_build_config(build_config_path: Path = BUILD_CONFIG_FILE) -> dict | None:
    """Load build metadata if present."""

    if not build_config_path.exists():
        return None
    return json.loads(build_config_path.read_text(encoding="utf-8"))


def load_bundle_metadata(bundle_path: Path = BUNDLE_FILE) -> dict | None:
    """Load the generated Kaggle handoff bundle if present."""

    if not bundle_path.exists():
        return None
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def directory_size_bytes(path: Path) -> int:
    """Return the total byte size of all files under a directory."""

    if not path.exists():
        return 0
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def format_bytes(size_bytes: int) -> str:
    """Format a byte count for human-readable output."""

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def build_messages(row: dict) -> list[dict]:
    """Convert one manifest row into Gemma 4 multimodal chat format."""

    instruction = str(row["instruction"])
    if SCHEMA_LOCK_INSTRUCTION not in instruction:
        instruction = f"{instruction} {SCHEMA_LOCK_INSTRUCTION}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio_path"]},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{instruction}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": row["expected_feedback"]}],
        },
    ]


def prepare_for_training(manifest_path: Path = MANIFEST_FILE) -> dict[str, list[dict]]:
    """Prepare split-wise training records from the manifest."""

    rows = load_manifest(manifest_path)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for row in rows:
        record = {
            "id": row["id"],
            "profile": row.get("profile"),
            "split": row["split"],
            "subset_name": row["subset_name"],
            "speaker_id": row["speaker_id"],
            "chapter_id": row["chapter_id"],
            "source_utterance_id": row["source_utterance_id"],
            "audio_path": row["audio_path"],
            "source_audio_path": row["source_audio_path"],
            "target_text": row["target_text"],
            "practice_level": row["practice_level"],
            "focus_score": row.get("focus_score"),
            "lisp_type": row["lisp_type"],
            "severity": row["severity"],
            "duration_seconds": row["duration_seconds"],
            "augmentation_recipe": row.get("augmentation_recipe"),
            "augmentation_parameters": row.get("augmentation_parameters"),
            "instruction": row["instruction"],
            "expected_feedback": row["expected_feedback"],
            "messages": build_messages(row),
        }
        splits[row["split"]].append(record)

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: row["id"])

    return splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def export_split_jsonls(splits: dict[str, list[dict]]) -> dict[str, str]:
    """Export split JSONL files for Kaggle and local reuse."""

    output_paths = {
        "train": MESSAGE_DIR / "train.jsonl",
        "val": MESSAGE_DIR / "val.jsonl",
        "test": MESSAGE_DIR / "test.jsonl",
    }
    compatibility_paths = {
        "train": TRAINING_JSONL,
        "val": VAL_JSONL,
        "test": TEST_JSONL,
    }

    for split_name, rows in splits.items():
        write_jsonl(output_paths[split_name], rows)
        write_jsonl(compatibility_paths[split_name], rows)

    return {split_name: repo_relative(path) for split_name, path in output_paths.items()}


def summarize_splits(splits: dict[str, list[dict]]) -> dict:
    """Summarize the exported bundle."""

    lisp_counts = {
        split_name: dict(Counter(row["lisp_type"] for row in rows))
        for split_name, rows in splits.items()
    }
    speaker_counts = {
        split_name: len({row["speaker_id"] for row in rows})
        for split_name, rows in splits.items()
    }
    return {
        "record_counts": {split_name: len(rows) for split_name, rows in splits.items()},
        "lisp_type_counts": lisp_counts,
        "speaker_counts": speaker_counts,
        "subset_counts": {
            split_name: dict(Counter(row["subset_name"] for row in rows))
            for split_name, rows in splits.items()
        },
    }


def get_model_config(train_row_count: int = 0) -> dict:
    """Model configuration for Gemma 4 E2B multimodal LoRA."""

    smoke_test_steps = 100 if train_row_count >= 4000 else 75
    full_train_max_steps = 4000 if train_row_count >= 16000 else 1000
    save_steps = 500 if train_row_count >= 16000 else 200

    audio_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "dense",
        "linear",
        "input_proj",
        "output_proj",
        "embedding_projection",
        "relative_k_proj",
        "per_layer_input_gate",
        "per_layer_projection",
    ]

    return {
        "model_name": "google/gemma-4-E2B-it",
        "recipe_name": "v17b-audio-targeted",
        "schema_lock": True,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "full_finetuning": False,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "target_modules": audio_target_modules,
        "use_gradient_checkpointing": "unsloth",
        "finetune_vision_layers": True,
        "finetune_audio_layers": True,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "smoke_test_steps": smoke_test_steps,
        "full_train_max_steps": full_train_max_steps,
        "save_steps": save_steps,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "eval_steps": save_steps,
        "quality_gate": {
            "min_class_match_successful_only": 0.60,
            "min_clear_match_successful_only": 0.90,
            "min_format_exact_successful_only": 0.95,
            "min_has_encouragement_successful_only": 0.90,
            "require_zero_hard_errors": True,
        },
    }


def write_bundle_metadata(
    manifest_path: Path,
    split_paths: dict[str, str],
    summary: dict,
    build_config: dict | None,
) -> dict:
    """Write bundle metadata for notebook consumption."""

    bundle = {
        "manifest_path": repo_relative(manifest_path),
        "split_files": split_paths,
        "model_config": get_model_config(summary["record_counts"]["train"]),
        "summary": summary,
        "build_config": build_config,
        "notebook_path": "notebooks/kaggle_gemma4_audio_unsloth.ipynb",
        "artifact_targets": {
            "kaggle_output_dir": "/kaggle/working/lisper-gemma4-audio",
            "hub_adapter_repo_placeholder": "your-hf-username/lisper-gemma4-e2b-audio-lora",
            "hub_full_model_repo_placeholder": "your-hf-username/lisper-gemma4-e2b-audio-full",
            "local_download_dir": repo_relative(MULTIMODAL_DIR / "artifacts"),
            "local_publish_dir": repo_relative(MULTIMODAL_DIR / "artifacts" / "hf_publish"),
            "merged_model_dirname": "merged_model",
            "verdict_filename": "publish_verdict.json",
            "training_metadata_filename": "training_metadata.json",
            "private_hub_repo": True,
        },
    }
    BUNDLE_FILE.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def print_kaggle_handoff(summary: dict, model_config: dict, build_config: dict | None) -> None:
    """Print the concrete Kaggle training handoff for the current bundle."""

    record_counts = summary["record_counts"]
    lisp_counts = summary["lisp_type_counts"]
    speaker_counts = summary["speaker_counts"]
    dataset_size = format_bytes(directory_size_bytes(MULTIMODAL_DIR))

    print("\nProcessed bundle:")
    print(f"  {MULTIMODAL_DIR} ({dataset_size})")
    print(f"  manifest: {MANIFEST_FILE}")
    print(f"  train jsonl: {MESSAGE_DIR / 'train.jsonl'}")
    print(f"  val jsonl:   {MESSAGE_DIR / 'val.jsonl'}")
    print(f"  test jsonl:  {MESSAGE_DIR / 'test.jsonl'}")

    print("\nCurrent counts:")
    print(
        "  rows: "
        f"train={record_counts['train']}, val={record_counts['val']}, test={record_counts['test']}"
    )
    print(
        "  speakers: "
        f"train={speaker_counts['train']}, val={speaker_counts['val']}, test={speaker_counts['test']}"
    )
    print(
        "  labels per split: "
        f"train={lisp_counts['train']}, val={lisp_counts['val']}, test={lisp_counts['test']}"
    )
    if build_config:
        print(
            "  build profile: "
            f"{build_config['profile']} "
            f"from {build_config['available_summary']['available_speakers']} speakers / "
            f"{build_config['available_summary']['available_source_utterances']} source utterances"
        )

    print("\nNotebook:")
    print(f"  {REPO_ROOT / 'notebooks' / 'kaggle_gemma4_audio_unsloth.ipynb'}")

    print("\nKaggle run order:")
    print("  1. Upload only data/processed/gemma4_audio as a Kaggle Dataset.")
    print("  2. Create a Kaggle Notebook with a GPU accelerator and attach that Dataset.")
    print("  3. Upload or paste notebooks/kaggle_gemma4_audio_unsloth.ipynb into the Kaggle Notebook.")
    print("  4. Run the install cell, then the data-loading cell, then the smoke-training cell.")
    print("  5. Leave RUN_FULL_TRAIN = False until the smoke pass completes without OOM or NaNs.")
    print("  6. Keep SELECT_BEST_CHECKPOINT = True so the notebook saves the best validation checkpoint.")
    print("  7. If you want Hub pushes, add the HF_TOKEN secret and set HF_ADAPTER_REPO / HF_FULL_MODEL_REPO.")
    print("  8. Leave EXPORT_MERGED_MODEL = True so the notebook also writes a standalone merged Gemma E2B checkpoint.")
    print("  9. Set RUN_FULL_TRAIN = True and run the full-train cell.")
    print(" 10. Run notebooks/kaggle_gemma4_audio_eval.py against the saved adapter dataset for held-out eval.")
    print(" 11. Download /kaggle/working/lisper-gemma4-audio into data/processed/gemma4_audio/artifacts.")
    print(" 12. Publish only after publish_verdict.json passes the release gate.")

    print("\nTraining config:")
    print(f"  model: {model_config['model_name']}")
    print(
        "  smoke/full steps: "
        f"{model_config['smoke_test_steps']} / {model_config['full_train_max_steps']}"
    )
    print(
        "  batching: "
        f"per_device={model_config['per_device_train_batch_size']} "
        f"x grad_accum={model_config['gradient_accumulation_steps']}"
    )
    print(f"  learning_rate: {model_config['learning_rate']}")


def main() -> None:
    """Command-line entrypoint."""

    parser = argparse.ArgumentParser(description="Prepare Lisper Gemma 4 training bundles")
    parser.add_argument("--prepare", action="store_true", help="Export split JSONL training bundles")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE, help="Source manifest path")
    parser.add_argument("--train", action="store_true", help="Show Kaggle training handoff details")
    parser.add_argument("--test", action="store_true", help="Print bundle and model configuration")
    args = parser.parse_args()

    print("=" * 60)
    print("Lisper - Gemma 4 E2B Audio Fine-tuning")
    print("=" * 60)

    if args.prepare:
        splits = prepare_for_training(args.manifest)
        split_paths = export_split_jsonls(splits)
        summary = summarize_splits(splits)
        bundle = write_bundle_metadata(args.manifest, split_paths, summary, load_build_config())
        print(json.dumps(bundle, indent=2))
        print("\nNext step:")
        print(f"  Open {REPO_ROOT / bundle['notebook_path']} in Kaggle and point it at the exported dataset bundle.")
    elif args.train:
        bundle = load_bundle_metadata()
        if bundle:
            print_kaggle_handoff(
                bundle["summary"],
                bundle["model_config"],
                bundle.get("build_config"),
            )
        else:
            splits = prepare_for_training(args.manifest)
            summary = summarize_splits(splits)
            print_kaggle_handoff(
                summary,
                get_model_config(summary["record_counts"]["train"]),
                load_build_config(),
            )
    elif args.test:
        splits = prepare_for_training(args.manifest)
        print(
            json.dumps(
                {
                    "summary": summarize_splits(splits),
                    "model_config": get_model_config(len(splits["train"])),
                    "build_config": load_build_config(),
                },
                indent=2,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
