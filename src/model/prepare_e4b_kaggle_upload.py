#!/usr/bin/env python3
"""Prepare the Gemma 4 E4B Kaggle training notebook from the validated E2B recipe."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
SOURCE_NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_gemma4_audio_unsloth.ipynb"
LOCAL_NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_gemma4_e4b_audio_unsloth.ipynb"
UPLOAD_DIR = REPO_ROOT / "notebooks" / "kaggle_upload" / "lisper-gemma-4-e4b-audio-unsloth-training"
UPLOAD_NOTEBOOK = UPLOAD_DIR / SOURCE_NOTEBOOK.name
KERNEL_METADATA = UPLOAD_DIR / "kernel-metadata.json"
PUSH_COMMANDS = UPLOAD_DIR / "push_commands.txt"

KAGGLE_CLI = str(Path.home() / "Library" / "Python" / "3.14" / "bin" / "kaggle")
KAGGLE_OWNER = os.environ.get("KAGGLE_OWNER", "thomasjvu").strip() or "thomasjvu"
KERNEL_SLUG = "lisper-gemma-4-e4b-audio-unsloth-training"
DATASET_SOURCE = "thomasjvu/lisper-gemma4-audio"
ADAPTER_REPO = "thomasjvu/lisper-gemma4-e4b-audio-lora"
FULL_REPO = "thomasjvu/lisper-gemma4-e4b-audio-full"


def patch_source(source: str) -> str:
    """Patch notebook source text for the E4B experiment."""

    replacements = {
        "Gemma 4 E2B": "Gemma 4 E4B",
        "Gemma E2B": "Gemma E4B",
        "google/gemma-4-E2B-it": "google/gemma-4-E4B-it",
        "thomasjvu/lisper-gemma4-e2b-audio-lora": ADAPTER_REPO,
        "thomasjvu/lisper-gemma4-e2b-audio-full": FULL_REPO,
        "/kaggle/working/lisper-gemma4-audio": "/kaggle/working/lisper-gemma4-e4b-audio",
        '"recipe_name": "v17b-audio-targeted"': '"recipe_name": "v19-e4b-audio-targeted"',
        '"lora_rank": 16': '"lora_rank": 8',
        "RUN_SMOKE_TRAIN = False": "RUN_SMOKE_TRAIN = True",
        '"smoke_test_steps": 100': '"smoke_test_steps": 25',
        'config["recipe_name"] = "v18-strong-audio-dynamic"': 'config["recipe_name"] = "v19-e4b-strong-audio-dynamic"',
        "save_total_limit=3,": "save_total_limit=1,",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    source = source.replace(
        "import shutil\n\nfrom huggingface_hub import HfApi, login\n",
        "import gc\nimport shutil\n\nfrom huggingface_hub import HfApi, login\n",
    )
    source = source.replace(
        """def export_merged_model(model, processor, merged_dir: Path) -> dict:
    save_pretrained_merged = getattr(model, "save_pretrained_merged", None)
    if save_pretrained_merged is None:
        raise AttributeError("Unsloth model is missing save_pretrained_merged; cannot export merged model.")
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    save_pretrained_merged(str(merged_dir), processor, save_method=MERGED_SAVE_METHOD)
    return {
        "merged_dir": str(merged_dir),
        "save_method": MERGED_SAVE_METHOD,
    }
""",
        """def cleanup_training_checkpoints(output_dir: Path, keep_last: int = 1) -> dict:
    checkpoint_dirs = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    keep = set(checkpoint_dirs[-keep_last:]) if keep_last > 0 else set()
    removed = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir in keep:
            continue
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        removed.append(str(checkpoint_dir))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "removed_checkpoints": removed,
        "kept_checkpoints": [str(path) for path in checkpoint_dirs if path in keep],
    }


def export_merged_model(model, processor, merged_dir: Path) -> dict:
    save_pretrained_merged = getattr(model, "save_pretrained_merged", None)
    if save_pretrained_merged is None:
        raise AttributeError("Unsloth model is missing save_pretrained_merged; cannot export merged model.")
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    save_pretrained_merged(str(merged_dir), processor, save_method=MERGED_SAVE_METHOD)
    return {
        "merged_dir": str(merged_dir),
        "save_method": MERGED_SAVE_METHOD,
    }


def push_merged_model_to_hub(model, processor, repo_id: str, token: str) -> dict:
    push_to_hub_merged = getattr(model, "push_to_hub_merged", None)
    if push_to_hub_merged is None:
        raise AttributeError("Unsloth model is missing push_to_hub_merged; cannot direct-push merged model.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    push_to_hub_merged(repo_id, processor, save_method=MERGED_SAVE_METHOD, token=token)
    return {
        "merged_repo": repo_id,
        "save_method": MERGED_SAVE_METHOD,
        "pushed_to_hub_directly": True,
    }
""",
    )
    source = source.replace(
        """    adapter_dir = OUTPUT_DIR / "adapter"
    full_trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    merged_dir = None
    if EXPORT_MERGED_MODEL:
        merged_dir = OUTPUT_DIR / "merged_model"
        merged_summary = export_merged_model(model, processor, merged_dir)
        print(merged_summary)
""",
        """    adapter_dir = OUTPUT_DIR / "adapter"
    full_trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    cleanup_summary = cleanup_training_checkpoints(Path(full_trainer.args.output_dir), keep_last=1)
    print(cleanup_summary)
    merged_dir = None
    merged_summary = None
    if EXPORT_MERGED_MODEL:
        if PUSH_MERGED_TO_HUB and hf_token:
            merged_summary = push_merged_model_to_hub(model, processor, HF_FULL_MODEL_REPO, hf_token)
        else:
            merged_dir = OUTPUT_DIR / "merged_model"
            merged_summary = export_merged_model(model, processor, merged_dir)
        print(merged_summary)
""",
    )
    source = source.replace(
        """    (adapter_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "artifacts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
""",
        """    metadata["checkpoint_cleanup"] = cleanup_summary
    metadata["merged_export"] = merged_summary
    if merged_summary and merged_summary.get("pushed_to_hub_directly"):
        metadata["merged_model_repo"] = HF_FULL_MODEL_REPO
    (adapter_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "artifacts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
""",
    )
    source = source.replace(
        """        if PUSH_MERGED_TO_HUB:
            if merged_dir is None:
                raise RuntimeError("PUSH_MERGED_TO_HUB=True requires EXPORT_MERGED_MODEL=True")
            upload_folder_to_hub(
                api,
                HF_FULL_MODEL_REPO,
                merged_dir,
                commit_message="Upload merged Lisper Gemma 4 audio model",
            )
            print({"pushed_merged_to_hub": HF_FULL_MODEL_REPO, "private_repo": True})
""",
        """        if PUSH_MERGED_TO_HUB:
            if merged_summary and merged_summary.get("pushed_to_hub_directly"):
                print({"pushed_merged_to_hub": HF_FULL_MODEL_REPO, "private_repo": True, "direct_unsloth_push": True})
            else:
                if merged_dir is None:
                    raise RuntimeError("PUSH_MERGED_TO_HUB=True requires EXPORT_MERGED_MODEL=True")
                upload_folder_to_hub(
                    api,
                    HF_FULL_MODEL_REPO,
                    merged_dir,
                    commit_message="Upload merged Lisper Gemma 4 audio model",
                )
                print({"pushed_merged_to_hub": HF_FULL_MODEL_REPO, "private_repo": True})
""",
    )
    source = re.sub(
        r"# Default to the real run recipe\. Run a short smoke test first when you change the recipe\.",
        "# E4B run: execute a short smoke phase before full training to catch OOM early.",
        source,
    )
    return source


def transform_notebook(notebook: dict) -> dict:
    """Return an E4B-patched notebook payload."""

    transformed = dict(notebook)
    cells = []
    for cell in notebook["cells"]:
        updated = dict(cell)
        source = "".join(cell.get("source", []))
        updated["source"] = patch_source(source).splitlines(keepends=True)
        cells.append(updated)
    transformed["cells"] = cells
    return transformed


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_NOTEBOOK.exists():
        raise RuntimeError(f"Missing source notebook: {SOURCE_NOTEBOOK}")

    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    transformed = transform_notebook(notebook)

    write_json(LOCAL_NOTEBOOK, transformed)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCAL_NOTEBOOK, UPLOAD_NOTEBOOK)

    write_json(
        KERNEL_METADATA,
        {
            "id": f"{KAGGLE_OWNER}/{KERNEL_SLUG}",
            "title": "Lisper Gemma 4 E4B Audio Unsloth Training",
            "code_file": UPLOAD_NOTEBOOK.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [DATASET_SOURCE],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        },
    )

    PUSH_COMMANDS.write_text(
        "\n".join(
            [
                "# E4B training notebook push command.",
                f'{KAGGLE_CLI} kernels push -p "{UPLOAD_DIR}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "local_notebook": str(LOCAL_NOTEBOOK),
                "upload_notebook": str(UPLOAD_NOTEBOOK),
                "kernel_metadata": str(KERNEL_METADATA),
                "kernel": f"{KAGGLE_OWNER}/{KERNEL_SLUG}",
                "kaggle_owner": KAGGLE_OWNER,
                "adapter_repo": ADAPTER_REPO,
                "full_repo": FULL_REPO,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
