#!/usr/bin/env python3
"""Prepare Kaggle dataset and notebook metadata for the Lisper training run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
MULTIMODAL_DIR = REPO_ROOT / "data" / "processed" / "gemma4_audio"
NOTEBOOK_SOURCE = REPO_ROOT / "notebooks" / "kaggle_gemma4_audio_unsloth.ipynb"
NOTEBOOK_UPLOAD_ROOT = REPO_ROOT / "notebooks" / "kaggle_upload"
DATASET_METADATA_FILE = MULTIMODAL_DIR / "dataset-metadata.json"
SYNC_NOTEBOOK_SCRIPT = REPO_ROOT / "src" / "model" / "sync_kaggle_training_notebooks.py"


def write_json(path: Path, payload: dict) -> None:
    """Write formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_kaggle_cli() -> str:
    """Return the preferred Kaggle CLI command string."""

    user_bin = Path.home() / "Library" / "Python" / "3.14" / "bin" / "kaggle"
    if user_bin.exists():
        return str(user_bin)
    discovered = shutil.which("kaggle")
    if discovered:
        return discovered
    return "python3 -m kaggle"


def build_dataset_metadata(
    username: str,
    dataset_slug: str,
    dataset_title: str,
    dataset_license: str,
) -> dict:
    """Build Kaggle dataset metadata for the processed audio bundle."""

    return {
        "title": dataset_title,
        "id": f"{username}/{dataset_slug}",
        "subtitle": "Processed Gemma 4 audio fine-tuning bundle for Lisper",
        "description": (
            "Speaker-disjoint raw-audio lisp-practice dataset for Gemma 4 / Unsloth fine-tuning. "
            "Includes processed audio clips, manifest metadata, split-wise chat JSONL exports, "
            "and the bundle metadata consumed by the Kaggle training notebook."
        ),
        "licenses": [{"name": dataset_license}],
        "keywords": [],
    }


def build_kernel_metadata(
    username: str,
    kernel_slug: str,
    kernel_title: str,
    dataset_slug: str,
    notebook_filename: str,
) -> dict:
    """Build Kaggle kernel metadata for the training notebook."""

    return {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_title,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [f"{username}/{dataset_slug}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def write_push_instructions(
    instructions_path: Path,
    kaggle_cli: str,
    dataset_dir: Path,
    kernel_dir: Path,
) -> None:
    """Write a short command file for Kaggle upload/push."""

    instructions = [
        "# Run these after KAGGLE_API_TOKEN is configured.",
        f'{kaggle_cli} datasets create -p "{dataset_dir}" --dir-mode zip',
        f'{kaggle_cli} datasets version -p "{dataset_dir}" -m "Update Lisper Gemma 4 audio bundle" --dir-mode zip',
        f'{kaggle_cli} kernels push -p "{kernel_dir}"',
    ]
    instructions_path.write_text("\n".join(instructions) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Prepare Kaggle upload metadata for Lisper")
    parser.add_argument("--username", required=True, help="Your Kaggle username")
    parser.add_argument(
        "--dataset-slug",
        default="lisper-gemma4-audio",
        help="Kaggle dataset slug for the processed bundle",
    )
    parser.add_argument(
        "--kernel-slug",
        default="lisper-gemma-4-audio-unsloth-training",
        help="Kaggle notebook slug",
    )
    parser.add_argument(
        "--dataset-title",
        default="Lisper Gemma 4 Audio Bundle",
        help="Kaggle dataset title",
    )
    parser.add_argument(
        "--kernel-title",
        default="Lisper Gemma 4 Audio Unsloth Training",
        help="Kaggle notebook title",
    )
    parser.add_argument(
        "--dataset-license",
        required=True,
        help="Kaggle license name to place in dataset-metadata.json",
    )
    args = parser.parse_args()

    if not MULTIMODAL_DIR.exists():
        raise RuntimeError(f"Processed dataset bundle not found: {MULTIMODAL_DIR}")
    if not NOTEBOOK_SOURCE.exists():
        raise RuntimeError(f"Training notebook not found: {NOTEBOOK_SOURCE}")
    if not SYNC_NOTEBOOK_SCRIPT.exists():
        raise RuntimeError(f"Notebook sync script not found: {SYNC_NOTEBOOK_SCRIPT}")

    subprocess.run(["python3", str(SYNC_NOTEBOOK_SCRIPT)], check=True)

    dataset_metadata = build_dataset_metadata(
        args.username,
        args.dataset_slug,
        args.dataset_title,
        args.dataset_license,
    )
    write_json(DATASET_METADATA_FILE, dataset_metadata)

    kernel_dir = NOTEBOOK_UPLOAD_ROOT / args.kernel_slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook_target = kernel_dir / NOTEBOOK_SOURCE.name
    shutil.copy2(NOTEBOOK_SOURCE, notebook_target)

    kernel_metadata = build_kernel_metadata(
        args.username,
        args.kernel_slug,
        args.kernel_title,
        args.dataset_slug,
        notebook_target.name,
    )
    kernel_metadata_file = kernel_dir / "kernel-metadata.json"
    write_json(kernel_metadata_file, kernel_metadata)

    instructions_file = kernel_dir / "push_commands.txt"
    kaggle_cli = resolve_kaggle_cli()
    write_push_instructions(instructions_file, kaggle_cli, MULTIMODAL_DIR, kernel_dir)

    output = {
        "dataset_metadata": str(DATASET_METADATA_FILE),
        "kernel_metadata": str(kernel_metadata_file),
        "notebook_bundle_dir": str(kernel_dir),
        "kaggle_cli": kaggle_cli,
        "next_steps": [
            f'{kaggle_cli} datasets create -p "{MULTIMODAL_DIR}" --dir-mode zip',
            f'{kaggle_cli} kernels push -p "{kernel_dir}"',
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
