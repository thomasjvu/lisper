#!/usr/bin/env python3
"""Prepare Kaggle notebook upload folders for a different runner account."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
UPLOAD_ROOT = NOTEBOOK_DIR / "kaggle_upload"
TRAINING_NOTEBOOK = NOTEBOOK_DIR / "kaggle_gemma4_audio_unsloth.ipynb"
EVAL_SCRIPT = NOTEBOOK_DIR / "kaggle_gemma4_audio_eval.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_push_commands(path: Path, kaggle_cli: str, folders: list[Path]) -> None:
    lines = ["# Switch Kaggle auth to the target account before running these."]
    lines.extend(f'{kaggle_cli} kernels push -p "{folder}"' for folder in folders)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_training_kernel(username: str, dataset_source: str, output_root: Path) -> Path:
    kernel_slug = "lisper-gemma-4-audio-unsloth-training"
    folder = output_root / kernel_slug
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINING_NOTEBOOK, folder / TRAINING_NOTEBOOK.name)
    write_json(
        folder / "kernel-metadata.json",
        {
            "id": f"{username}/{kernel_slug}",
            "title": "Lisper Gemma 4 Audio Unsloth Training",
            "code_file": TRAINING_NOTEBOOK.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [dataset_source],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        },
    )
    return folder


def prepare_eval_kernel(username: str, dataset_source: str, adapter_source: str, output_root: Path) -> Path:
    kernel_slug = "lisper-gemma-4-audio-adapter-eval"
    folder = output_root / kernel_slug
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EVAL_SCRIPT, folder / EVAL_SCRIPT.name)
    write_json(
        folder / "kernel-metadata.json",
        {
            "id": f"{username}/{kernel_slug}",
            "title": "Lisper Gemma 4 Audio Adapter Eval",
            "code_file": EVAL_SCRIPT.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [dataset_source, adapter_source],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        },
    )
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle upload folders for another account")
    parser.add_argument("--username", default="alkahestai", help="Kaggle account that will own the pushed kernels")
    parser.add_argument("--dataset-source", default="thomasjvu/lisper-gemma4-audio")
    parser.add_argument("--adapter-source", default="thomasjvu/lisper-gemma4-audio-lora")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=UPLOAD_ROOT / "alkahestai",
        help="Output folder for account-specific Kaggle kernel folders",
    )
    parser.add_argument("--kaggle-cli", default="/Users/area/Library/Python/3.14/bin/kaggle")
    args = parser.parse_args()

    if not TRAINING_NOTEBOOK.exists():
        raise FileNotFoundError(TRAINING_NOTEBOOK)
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(EVAL_SCRIPT)

    training_folder = prepare_training_kernel(args.username, args.dataset_source, args.output_root)
    eval_folder = prepare_eval_kernel(args.username, args.dataset_source, args.adapter_source, args.output_root)
    push_commands = args.output_root / "push_commands.txt"
    write_push_commands(push_commands, args.kaggle_cli, [training_folder, eval_folder])
    print(
        json.dumps(
            {
                "username": args.username,
                "dataset_source": args.dataset_source,
                "adapter_source": args.adapter_source,
                "training_folder": str(training_folder),
                "eval_folder": str(eval_folder),
                "push_commands": str(push_commands),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
