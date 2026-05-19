#!/usr/bin/env python3
"""Prepare a Kaggle merge-only notebook for the trained Gemma 4 E4B LoRA."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
UPLOAD_DIR = REPO_ROOT / "notebooks" / "kaggle_upload" / "lisper-gemma-4-e4b-audio-merge"
NOTEBOOK_PATH = UPLOAD_DIR / "kaggle_gemma4_e4b_audio_merge.ipynb"
KERNEL_METADATA = UPLOAD_DIR / "kernel-metadata.json"
PUSH_COMMANDS = UPLOAD_DIR / "push_commands.txt"

KAGGLE_CLI = str(Path.home() / "Library" / "Python" / "3.14" / "bin" / "kaggle")
KAGGLE_OWNER = os.environ.get("KAGGLE_OWNER", "thomasjvu").strip() or "thomasjvu"
KERNEL_SLUG = "lisper-gemma-4-e4b-audio-merge"
ADAPTER_REPO = "thomasjvu/lisper-gemma4-e4b-audio-lora"
FULL_REPO = "thomasjvu/lisper-gemma4-e4b-audio-full"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook_payload() -> dict:
    return {
        "cells": [
            markdown_cell(
                """# Lisper Gemma 4 E4B Merge Only

This notebook does not retrain. It loads the recovered E4B LoRA adapter and uses Unsloth's direct Hub merge path to create the private full-model repo.

Required Kaggle secret: `HF_TOKEN` with read access to the private adapter repo and write access to the full model repo.
"""
            ),
            code_cell(
                """!pip install --quiet --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 numpy==2.0.2 pillow==11.3.0
!pip install --quiet --no-cache-dir unsloth unsloth_zoo
!pip install --quiet --no-cache-dir "transformers!=5.0.0,!=5.1.0,<=5.5.0,>=4.51.3" "peft" "accelerate" "bitsandbytes" "huggingface_hub" "hf_transfer"
"""
            ),
            code_cell(
                f"""import gc
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, login

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing Kaggle HF_TOKEN secret.")
    return token


HF_TOKEN = get_hf_token()
ADAPTER_REPO = "{ADAPTER_REPO}"
FULL_REPO = "{FULL_REPO}"
MAX_SEQ_LENGTH = 2048
MERGED_SAVE_METHOD = "merged_16bit"

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=FULL_REPO, repo_type="model", private=True, exist_ok=True)

print(
    json.dumps(
        {{
            "adapter_repo": ADAPTER_REPO,
            "full_repo": FULL_REPO,
            "cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }},
        indent=2,
    )
)
"""
            ),
            code_cell(
                """import unsloth  # noqa: F401
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name=ADAPTER_REPO,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
    token=HF_TOKEN,
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

push_to_hub_merged = getattr(model, "push_to_hub_merged", None)
if push_to_hub_merged is None:
    raise AttributeError("Loaded Unsloth model does not expose push_to_hub_merged.")

push_to_hub_merged(
    FULL_REPO,
    processor,
    save_method=MERGED_SAVE_METHOD,
    token=HF_TOKEN,
)

print(
    json.dumps(
        {
            "status": "pushed_merged_model",
            "adapter_repo": ADAPTER_REPO,
            "full_repo": FULL_REPO,
            "save_method": MERGED_SAVE_METHOD,
        },
        indent=2,
    )
)
"""
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    write_json(NOTEBOOK_PATH, notebook_payload())
    write_json(
        KERNEL_METADATA,
        {
            "id": f"{KAGGLE_OWNER}/{KERNEL_SLUG}",
            "title": "Lisper Gemma 4 E4B Audio Merge",
            "code_file": NOTEBOOK_PATH.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        },
    )
    PUSH_COMMANDS.write_text(
        "\n".join(
            [
                "# E4B merge-only notebook push command. Confirm Kaggle HF_TOKEN secret first.",
                f'{KAGGLE_CLI} kernels push -p "{UPLOAD_DIR}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK_PATH),
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
