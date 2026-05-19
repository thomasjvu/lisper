#!/usr/bin/env python3
"""Sync the Kaggle training notebook cells across local and upload copies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
LOCAL_BUNDLE_PATH = REPO_ROOT / "data" / "processed" / "gemma4_audio" / "bundle.json"
NOTEBOOK_PATHS = [
    REPO_ROOT / "notebooks" / "kaggle_gemma4_audio_unsloth.ipynb",
    REPO_ROOT
    / "notebooks"
    / "kaggle_upload"
    / "lisper-gemma-4-audio-unsloth-training"
    / "kaggle_gemma4_audio_unsloth.ipynb",
    REPO_ROOT
    / "notebooks"
    / "kaggle_upload"
    / "lisper-gemma4-audio-unsloth"
    / "kaggle_gemma4_audio_unsloth.ipynb",
    REPO_ROOT
    / "notebooks"
    / "kaggle_upload"
    / "alkahestai"
    / "lisper-gemma-4-audio-unsloth-training"
    / "kaggle_gemma4_audio_unsloth.ipynb",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


LOCAL_BUNDLE = json.loads(LOCAL_BUNDLE_PATH.read_text(encoding="utf-8"))
EXPECTED_BUNDLE_SHA256 = file_sha256(LOCAL_BUNDLE_PATH)
EXPECTED_MODEL_CONFIG_JSON = json.dumps(LOCAL_BUNDLE["model_config"], sort_keys=True)
EXPECTED_FULL_TRAIN_MAX_STEPS = int(LOCAL_BUNDLE["model_config"]["full_train_max_steps"])
EXPECTED_SAVE_STEPS = int(LOCAL_BUNDLE["model_config"]["save_steps"])
EXPECTED_EVAL_STEPS = int(LOCAL_BUNDLE["model_config"]["eval_steps"])
EXPECTED_MODEL_CONFIG_JSON_ESCAPED = EXPECTED_MODEL_CONFIG_JSON.replace("\\", "\\\\").replace("'", "\\'")

CELL_2 = """from __future__ import annotations

import hashlib
import json
import os
import zlib
import zipfile
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import soundfile as sf
import torch
from scipy import signal

try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


def _disable_torch_compile(model=None, *args, **kwargs):
    if model is None:
        return lambda fn: fn
    return model


torch.compile = _disable_torch_compile
from datasets import Dataset

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
DATASET_ROOTS = sorted(path for path in KAGGLE_INPUT_ROOT.iterdir() if path.is_dir()) if KAGGLE_INPUT_ROOT.exists() else []
PREFERRED_BUNDLE_SLUGS = [
    slug.strip()
    for slug in os.environ.get(
        "PREFERRED_BUNDLE_SLUGS",
        "lisper-gemma4-audio-v17b-control,lisper-gemma4-audio-v17-control,lisper-gemma4-audio",
    ).split(",")
    if slug.strip()
]
PREFERRED_BUNDLE_CANDIDATES = []
for slug in PREFERRED_BUNDLE_SLUGS:
    preferred_root = KAGGLE_INPUT_ROOT / slug
    PREFERRED_BUNDLE_CANDIDATES.extend(
        [
            preferred_root / "bundle.json",
            preferred_root / "gemma4_audio/bundle.json",
            preferred_root / "data/processed/gemma4_audio/bundle.json",
        ]
    )
BUNDLE_CANDIDATES = []
for dataset_root in DATASET_ROOTS:
    BUNDLE_CANDIDATES.extend(
        [
            dataset_root / "data/processed/gemma4_audio/bundle.json",
            dataset_root / "gemma4_audio/bundle.json",
            dataset_root / "bundle.json",
        ]
    )
if KAGGLE_INPUT_ROOT.exists():
    BUNDLE_CANDIDATES.extend(sorted(KAGGLE_INPUT_ROOT.rglob("bundle.json")))
if not BUNDLE_CANDIDATES:
    BUNDLE_CANDIDATES = [
        Path("/kaggle/input/lisper-gemma4-audio/data/processed/gemma4_audio/bundle.json"),
        Path("/kaggle/input/lisper-gemma4-audio/gemma4_audio/bundle.json"),
        Path("/kaggle/input/lisper-gemma4-audio/bundle.json"),
    ]
ALL_BUNDLE_CANDIDATES = PREFERRED_BUNDLE_CANDIDATES + BUNDLE_CANDIDATES
BUNDLE_PATH = next((path for path in ALL_BUNDLE_CANDIDATES if path.exists()), ALL_BUNDLE_CANDIDATES[0])

if BUNDLE_PATH.parts[-4:] == ("data", "processed", "gemma4_audio", "bundle.json"):
    DATASET_ROOT = BUNDLE_PATH.parents[3]
elif BUNDLE_PATH.parts[-2:] == ("gemma4_audio", "bundle.json"):
    DATASET_ROOT = BUNDLE_PATH.parents[1]
else:
    DATASET_ROOT = BUNDLE_PATH.parent

OUTPUT_DIR = Path("/kaggle/working/lisper-gemma4-audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_INPUT_DIR = OUTPUT_DIR / "input_extracted"
EXTRACTED_DATASET_ROOT = EXTRACTED_INPUT_DIR / "data/processed/gemma4_audio"


def existing_dataset_search_roots() -> list[Path]:
    roots = [DATASET_ROOT, BUNDLE_PATH.parent, EXTRACTED_INPUT_DIR, EXTRACTED_DATASET_ROOT]
    for root in DATASET_ROOTS:
        roots.extend(
            [
                root,
                root / "data",
                root / "data/processed",
                root / "data/processed/gemma4_audio",
                root / "gemma4_audio",
            ]
        )
        if root.name == "datasets":
            for owner_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                for dataset_dir in sorted(path for path in owner_dir.iterdir() if path.is_dir()):
                    roots.extend(
                        [
                            dataset_dir,
                            dataset_dir / "data",
                            dataset_dir / "data/processed",
                            dataset_dir / "data/processed/gemma4_audio",
                            dataset_dir / "gemma4_audio",
                        ]
                    )
    seen = set()
    out = []
    for root in roots:
        key = str(root)
        if root.exists() and root.is_dir() and key not in seen:
            out.append(root)
            seen.add(key)
    return out


DATASET_SEARCH_ROOTS = existing_dataset_search_roots()
RESOLVE_CACHE = {}


def unpack_dataset_archive(archive_name: str) -> None:
    for root in (BUNDLE_PATH.parent, DATASET_ROOT):
        archive_path = root / archive_name
        if not archive_path.exists():
            continue
        marker_path = EXTRACTED_DATASET_ROOT / f".{archive_name}.extracted"
        if marker_path.exists():
            return
        EXTRACTED_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive_path} to {EXTRACTED_DATASET_ROOT}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(EXTRACTED_DATASET_ROOT)
        marker_path.touch()
        return


for archive_name in ("messages.zip", "audio.zip"):
    unpack_dataset_archive(archive_name)

# Default to the real run recipe. Run a short smoke test first when you change the recipe.
RUN_BASELINE_EVAL = False
RUN_SMOKE_TRAIN = False
RUN_FULL_TRAIN = True
RUN_TUNED_EVAL = False
PUSH_TO_HUB = True
PUSH_MERGED_TO_HUB = True
DERIVE_V18_AUDIO = True
HF_ADAPTER_REPO = "thomasjvu/lisper-gemma4-e2b-audio-lora"
HF_FULL_MODEL_REPO = "thomasjvu/lisper-gemma4-e2b-audio-full"
EXPORT_MERGED_MODEL = True
MERGED_SAVE_METHOD = "merged_16bit"
RESUME_FROM_CHECKPOINT = ""
SELECT_BEST_CHECKPOINT = True
EVAL_LIMIT = 120
FULL_EVAL_LIMIT = 500
TARGET_SAMPLE_RATE = 16000
V18_DERIVED_AUDIO_DIR = OUTPUT_DIR / "derived_v18_audio"
EXPECTED_BUNDLE_SHA256 = "__EXPECTED_BUNDLE_SHA256__"
EXPECTED_MODEL_CONFIG = json.loads('__EXPECTED_MODEL_CONFIG_JSON__')
EXPECTED_FULL_TRAIN_MAX_STEPS = __EXPECTED_FULL_TRAIN_MAX_STEPS__
EXPECTED_SAVE_STEPS = __EXPECTED_SAVE_STEPS__
EXPECTED_EVAL_STEPS = __EXPECTED_EVAL_STEPS__


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

bundle = json.loads(BUNDLE_PATH.read_text())
bundle_sha256 = compute_file_sha256(BUNDLE_PATH)
bundle_hash_matches = bundle_sha256 == EXPECTED_BUNDLE_SHA256
if not bundle_hash_matches:
    print(
        {
            "warning": "stale_kaggle_bundle",
            "bundle_path": str(BUNDLE_PATH),
            "expected_bundle_sha256": EXPECTED_BUNDLE_SHA256,
            "actual_bundle_sha256": bundle_sha256,
        }
    )
config = dict(EXPECTED_MODEL_CONFIG)
if isinstance(config.get("target_modules"), str) and config["target_modules"] != "all-linear":
    config["target_modules"] = [module.strip() for module in config["target_modules"].split(",") if module.strip()]
config["smoke_test_steps"] = int(config.get("smoke_test_steps", 100))
config["full_train_max_steps"] = int(config.get("full_train_max_steps", 4000))
config["save_steps"] = int(config.get("save_steps", 500))
config["eval_steps"] = int(config.get("eval_steps", config["save_steps"]))


def resolve_dataset_path(relative_path: str) -> str:
    cached = RESOLVE_CACHE.get(relative_path)
    if cached is not None:
        return cached
    relative = Path(relative_path)
    candidates = [
        DATASET_ROOT / relative,
        BUNDLE_PATH.parent / relative,
        EXTRACTED_INPUT_DIR / relative,
        EXTRACTED_DATASET_ROOT / relative,
    ]
    if relative.parts[:2] == ("data", "processed"):
        candidates.extend(
            [
                DATASET_ROOT / Path(*relative.parts[2:]),
                BUNDLE_PATH.parent / Path(*relative.parts[2:]),
                EXTRACTED_INPUT_DIR / Path(*relative.parts[2:]),
            ]
        )
    if relative.parts[:3] == ("data", "processed", "gemma4_audio"):
        candidates.extend(
            [
                DATASET_ROOT / Path(*relative.parts[3:]),
                BUNDLE_PATH.parent / Path(*relative.parts[3:]),
                EXTRACTED_DATASET_ROOT / Path(*relative.parts[3:]),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            resolved = str(candidate)
            RESOLVE_CACHE[relative_path] = resolved
            return resolved

    suffixes = [relative]
    if relative.parts[:2] == ("data", "processed"):
        suffixes.append(Path(*relative.parts[2:]))
    if relative.parts[:3] == ("data", "processed", "gemma4_audio"):
        suffixes.append(Path(*relative.parts[3:]))
    for root in DATASET_SEARCH_ROOTS:
        for suffix in suffixes:
            candidate = root / suffix
            if candidate.exists():
                resolved = str(candidate)
                RESOLVE_CACHE[relative_path] = resolved
                return resolved
    for suffix in suffixes:
        suffix_text = str(suffix)
        for candidate in KAGGLE_INPUT_ROOT.rglob(suffix.name):
            if candidate.exists() and str(candidate).endswith(suffix_text):
                resolved = str(candidate)
                RESOLVE_CACHE[relative_path] = resolved
                return resolved

    raise FileNotFoundError(relative_path)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def safe_filter(audio: np.ndarray, sample_rate: int, cutoff, btype: str, order: int = 3) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if isinstance(cutoff, tuple):
        low = max(40.0, min(float(cutoff[0]), nyquist * 0.75))
        high = min(float(cutoff[1]), nyquist * 0.95)
        if low >= high:
            return audio.astype(np.float32)
        cutoff_value = (low, high)
    else:
        cutoff_value = min(float(cutoff), nyquist * 0.95)
    sos = signal.butter(order, cutoff_value, btype=btype, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return normalize_audio(audio), int(sample_rate)


def write_audio_file(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), normalize_audio(audio), TARGET_SAMPLE_RATE, subtype="FLOAT")


def synthesize_v18_audio(clear_audio: np.ndarray, sample_rate: int, lisp_type: str, row_id: str) -> np.ndarray:
    audio = normalize_audio(clear_audio)
    if lisp_type == "clear":
        return audio

    seed = zlib.crc32(row_id.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, audio.size).astype(np.float32)

    if lisp_type == "frontal":
        high_sibilants = safe_filter(audio, sample_rate, (4200, min(7600, sample_rate / 2 - 200)), "bandpass", order=3)
        low_sibilants = safe_filter(audio, sample_rate, (1400, 3600), "bandpass", order=3)
        low_noise = safe_filter(noise, sample_rate, (1800, 3600), "bandpass", order=2)
        result = audio * 0.60 - high_sibilants * 0.26 + low_sibilants * 0.42 + low_noise * 0.095
    elif lisp_type == "lateral":
        mid_sibilants = safe_filter(audio, sample_rate, (2400, 5200), "bandpass", order=3)
        high_noise = safe_filter(noise, sample_rate, (4800, min(7600, sample_rate / 2 - 200)), "bandpass", order=2)
        result = audio * 0.58 + mid_sibilants * 0.28 + high_noise * 0.16
    elif lisp_type == "dental":
        high_sibilants = safe_filter(audio, sample_rate, (4200, min(7600, sample_rate / 2 - 200)), "bandpass", order=3)
        low_sibilants = safe_filter(audio, sample_rate, (1400, 3600), "bandpass", order=3)
        low_noise = safe_filter(noise, sample_rate, (1800, 3600), "bandpass", order=2)
        result = audio * 0.62 - high_sibilants * 0.34 + low_sibilants * 0.55 + low_noise * 0.055
    elif lisp_type == "palatal":
        high_sibilants = safe_filter(audio, sample_rate, (4200, min(7600, sample_rate / 2 - 200)), "bandpass", order=3)
        low_passed = safe_filter(audio, sample_rate, 3200, "lowpass", order=4)
        nasal_band = safe_filter(audio, sample_rate, (850, 2300), "bandpass", order=3)
        result = low_passed * 0.82 + nasal_band * 0.36 - high_sibilants * 0.28
    else:
        raise ValueError(f"Unsupported lisp type: {lisp_type}")
    return normalize_audio(result)


def absolutize_audio_paths(example):
    for message in example["messages"]:
        for content in message["content"]:
            if content.get("type") == "audio":
                content["audio"] = resolve_dataset_path(content["audio"])
    example["audio_path"] = resolve_dataset_path(example["audio_path"])
    try:
        example["source_audio_path"] = resolve_dataset_path(example["source_audio_path"])
    except FileNotFoundError:
        # The small control datasets intentionally omit raw LibriSpeech files.
        # v18 derivation replaces this with the processed clear WAV path.
        pass
    return example


def load_jsonl_rows(relative_path: str) -> list[dict]:
    dataset_path = Path(resolve_dataset_path(relative_path))
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]
    return [absolutize_audio_paths(row) for row in rows]


def set_row_audio_path(row: dict, audio_path: str) -> dict:
    row["audio_path"] = audio_path
    for message in row["messages"]:
        for content in message["content"]:
            if content.get("type") == "audio":
                content["audio"] = audio_path
    return row


def derive_v18_rows(rows: list[dict], split_name: str) -> list[dict]:
    clear_by_source = {
        row["source_utterance_id"]: row["audio_path"]
        for row in rows
        if row["lisp_type"] == "clear"
    }
    derived_rows = []
    generated = 0
    for index, row in enumerate(rows, start=1):
        updated = json.loads(json.dumps(row))
        clear_path = clear_by_source[updated["source_utterance_id"]]
        updated["source_audio_path"] = clear_path
        if updated["lisp_type"] == "clear":
            derived_path = clear_path
        else:
            derived_path = V18_DERIVED_AUDIO_DIR / split_name / updated["lisp_type"] / f"{updated['id']}.wav"
            if not derived_path.exists():
                clear_audio, sample_rate = load_audio_file(clear_path)
                derived_audio = synthesize_v18_audio(clear_audio, sample_rate, updated["lisp_type"], updated["id"])
                write_audio_file(derived_path, derived_audio)
                generated += 1
            derived_path = str(derived_path)
        derived_rows.append(set_row_audio_path(updated, derived_path))
        if index % 500 == 0:
            print({"derive_v18_audio": split_name, "processed": index, "generated": generated})
    print({"derive_v18_audio": split_name, "rows": len(rows), "generated": generated})
    return derived_rows


train_rows = load_jsonl_rows(bundle["split_files"]["train"])
val_rows = load_jsonl_rows(bundle["split_files"]["val"])
test_rows = load_jsonl_rows(bundle["split_files"]["test"])

if DERIVE_V18_AUDIO:
    config["recipe_name"] = "v18-strong-audio-dynamic"
    train_rows = derive_v18_rows(train_rows, "train")
    val_rows = derive_v18_rows(val_rows, "val")
    test_rows = derive_v18_rows(test_rows, "test")

train_dataset = Dataset.from_list(train_rows)
val_dataset = Dataset.from_list(val_rows)
test_dataset = Dataset.from_list(test_rows)

USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
GPU_INFO = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
print(json.dumps(bundle["summary"], indent=2))
print(
    {
        "bundle_sha256": bundle_sha256,
        "bundle_hash_matches": bundle_hash_matches,
        "use_bf16": USE_BF16,
        "gpu_count": len(GPU_INFO),
        "gpus": GPU_INFO,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "test_rows": len(test_dataset),
        "config": config,
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT or None,
        "export_merged_model": EXPORT_MERGED_MODEL,
        "push_adapter_to_hub": PUSH_TO_HUB,
        "push_merged_to_hub": PUSH_MERGED_TO_HUB,
    }
)
assert config["full_train_max_steps"] == EXPECTED_FULL_TRAIN_MAX_STEPS, (
    f"Expected full_train_max_steps={EXPECTED_FULL_TRAIN_MAX_STEPS}, got {config['full_train_max_steps']}"
)
assert config["save_steps"] == EXPECTED_SAVE_STEPS, (
    f"Expected save_steps={EXPECTED_SAVE_STEPS}, got {config['save_steps']}"
)
assert config["eval_steps"] == EXPECTED_EVAL_STEPS, (
    f"Expected eval_steps={EXPECTED_EVAL_STEPS}, got {config['eval_steps']}"
)


def extract_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def score_response(example: dict, response: str) -> dict[str, bool]:
    fields = extract_fields(response)
    detected = fields.get("detected class", "").lower()
    return {
        "class_match": detected == example["lisp_type"].lower(),
        "clear_match": (detected == "clear") == (example["lisp_type"].lower() == "clear"),
        "has_reason": bool(fields.get("reason")),
        "has_corrective_cue": bool(fields.get("corrective cue")),
        "has_encouragement": bool(fields.get("encouragement")),
    }


def aggregate_scores(rows: list[dict]) -> dict[str, float]:
    total = len(rows) or 1
    keys = rows[0].keys() if rows else []
    return {key: sum(1 for row in rows if row[key]) / total for key in keys}


def generate_single_response(model, processor, example: dict, max_new_tokens: int = 128) -> str:
    messages = [message for message in example["messages"] if message["role"] != "assistant"]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def evaluate_model(model, processor, dataset, limit: int, output_path: Path) -> dict:
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    rows = []
    for index, example in enumerate(dataset.select(range(min(limit, len(dataset))))):
        response = generate_single_response(model, processor, example)
        scores = score_response(example, response)
        rows.append(
            {
                "id": example["id"],
                "ground_truth": example["lisp_type"],
                "response": response,
                **scores,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"evaluated {index + 1} examples")
    summary = {"count": len(rows), **aggregate_scores(rows)}
    output_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return summary
""".replace("__EXPECTED_BUNDLE_SHA256__", EXPECTED_BUNDLE_SHA256).replace(
    "__EXPECTED_FULL_TRAIN_MAX_STEPS__",
    str(EXPECTED_FULL_TRAIN_MAX_STEPS),
).replace(
    "__EXPECTED_SAVE_STEPS__",
    str(EXPECTED_SAVE_STEPS),
).replace(
    "__EXPECTED_EVAL_STEPS__",
    str(EXPECTED_EVAL_STEPS),
).replace(
    "__EXPECTED_MODEL_CONFIG_JSON__",
    EXPECTED_MODEL_CONFIG_JSON_ESCAPED,
)

CELL_4 = """from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=config["finetune_vision_layers"],
    finetune_language_layers=config["finetune_language_layers"],
    finetune_attention_modules=config["finetune_attention_modules"],
    finetune_mlp_modules=config["finetune_mlp_modules"],
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    random_state=3407,
    use_gradient_checkpointing=config["use_gradient_checkpointing"],
    target_modules=config["target_modules"],
    modules_to_save=None,
)

data_collator = UnslothVisionDataCollator(model, processor)
smoke_eval_dataset = val_dataset.select(range(min(FULL_EVAL_LIMIT, len(val_dataset))))

if RUN_SMOKE_TRAIN:
    smoke_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "smoke"),
        max_seq_length=config["max_seq_length"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=5,
        max_steps=config["smoke_test_steps"],
        logging_steps=5,
        eval_steps=min(config["smoke_test_steps"], 25),
        save_steps=config["smoke_test_steps"],
        save_strategy="steps",
        eval_strategy="steps",
        fp16=not USE_BF16,
        bf16=USE_BF16,
        optim="adamw_8bit",
        remove_unused_columns=False,
        dataset_num_proc=1,
        report_to="none",
        seed=3407,
    )

    smoke_trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=smoke_eval_dataset,
        args=smoke_args,
    )
    smoke_result = smoke_trainer.train()
    smoke_metadata = {
        "smoke_steps": config["smoke_test_steps"],
        "metrics": smoke_result.metrics,
        "gpu_info": GPU_INFO,
        "config": config,
    }
    (OUTPUT_DIR / "smoke_metadata.json").write_text(json.dumps(smoke_metadata, indent=2), encoding="utf-8")
    print(smoke_result)
else:
    print("Smoke training skipped. RUN_FULL_TRAIN=", RUN_FULL_TRAIN)
"""

CELL_5 = """import shutil

from huggingface_hub import HfApi, login


def get_hf_token() -> str | None:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        return hf_token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def resolve_resume_checkpoint(raw_path: str) -> str | None:
    if not raw_path:
        return None
    checkpoint_path = Path(raw_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in checkpoint: {checkpoint_path}")
    adapter_config = json.loads(adapter_config_path.read_text())
    expected_r = int(config["lora_rank"])
    actual_r = int(adapter_config.get("r", -1))
    if actual_r != expected_r:
        raise RuntimeError({"resume_checkpoint": str(checkpoint_path), "expected_r": expected_r, "actual_r": actual_r})
    expected_targets = config["target_modules"]
    actual_targets = adapter_config.get("target_modules")
    if expected_targets != "all-linear":
        expected_targets = sorted(expected_targets)
        actual_targets = sorted(actual_targets or [])
        if expected_targets != actual_targets:
            raise RuntimeError(
                {
                    "resume_checkpoint": str(checkpoint_path),
                    "expected_target_modules": expected_targets,
                    "actual_target_modules": actual_targets,
                }
            )
    return str(checkpoint_path)


def build_training_metadata(
    full_result,
    trainer,
    full_eval_dataset,
    adapter_dir: Path,
    merged_dir: Path | None,
    resume_checkpoint: str | None,
) -> dict:
    selected_checkpoint = trainer.state.best_model_checkpoint or str(Path(trainer.args.output_dir))
    return {
        "bundle_path": str(BUNDLE_PATH),
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_dir) if merged_dir else None,
        "model_name": config["model_name"],
        "gpu_info": GPU_INFO,
        "config": config,
        "resume_from_checkpoint": resume_checkpoint,
        "selected_best_checkpoint": bool(trainer.state.best_model_checkpoint),
        "selected_checkpoint_path": selected_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "full_eval_rows": len(full_eval_dataset),
        "train_metrics": full_result.metrics,
        "export_merged_model": EXPORT_MERGED_MODEL,
        "merged_save_method": MERGED_SAVE_METHOD if merged_dir else None,
    }


def upload_folder_to_hub(api: HfApi, repo_id: str, folder_path: Path, commit_message: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        commit_message=commit_message,
    )


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


if RUN_FULL_TRAIN:
    hf_token = get_hf_token() if (PUSH_TO_HUB or PUSH_MERGED_TO_HUB) else None
    if (PUSH_TO_HUB or PUSH_MERGED_TO_HUB) and not hf_token:
        print(
            {
                "warning": "hf_token_missing_skip_hub_push",
                "adapter_repo": HF_ADAPTER_REPO if PUSH_TO_HUB else None,
                "full_model_repo": HF_FULL_MODEL_REPO if PUSH_MERGED_TO_HUB else None,
            }
        )

    full_eval_dataset = val_dataset.select(range(min(FULL_EVAL_LIMIT, len(val_dataset))))
    full_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "full_train"),
        max_seq_length=config["max_seq_length"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=max(25, min(100, config["full_train_max_steps"] // 20)),
        max_steps=config["full_train_max_steps"],
        logging_steps=10,
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_total_limit=3,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=SELECT_BEST_CHECKPOINT,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not USE_BF16,
        bf16=USE_BF16,
        optim="adamw_8bit",
        remove_unused_columns=False,
        dataset_num_proc=1,
        report_to="none",
        seed=3407,
    )

    full_trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=full_eval_dataset,
        args=full_args,
    )
    resume_checkpoint = resolve_resume_checkpoint(RESUME_FROM_CHECKPOINT)
    full_result = full_trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(full_result)

    adapter_dir = OUTPUT_DIR / "adapter"
    full_trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    merged_dir = None
    if EXPORT_MERGED_MODEL:
        merged_dir = OUTPUT_DIR / "merged_model"
        merged_summary = export_merged_model(model, processor, merged_dir)
        print(merged_summary)

    metadata = build_training_metadata(
        full_result,
        full_trainer,
        full_eval_dataset,
        adapter_dir,
        merged_dir,
        resume_checkpoint,
    )
    (adapter_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "artifacts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        {
            "selected_checkpoint_path": metadata["selected_checkpoint_path"],
            "best_metric": metadata["best_metric"],
            "merged_model_dir": metadata["merged_model_dir"],
        }
    )

    if (PUSH_TO_HUB or PUSH_MERGED_TO_HUB) and hf_token:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        if PUSH_TO_HUB:
            upload_folder_to_hub(
                api,
                HF_ADAPTER_REPO,
                adapter_dir,
                commit_message="Upload Lisper Gemma 4 audio adapter",
            )
            print({"pushed_adapter_to_hub": HF_ADAPTER_REPO, "private_repo": True})
        if PUSH_MERGED_TO_HUB:
            if merged_dir is None:
                raise RuntimeError("PUSH_MERGED_TO_HUB=True requires EXPORT_MERGED_MODEL=True")
            upload_folder_to_hub(
                api,
                HF_FULL_MODEL_REPO,
                merged_dir,
                commit_message="Upload merged Lisper Gemma 4 audio model",
            )
            print({"pushed_merged_to_hub": HF_FULL_MODEL_REPO, "private_repo": True})
"""


def main() -> None:
    for notebook_path in NOTEBOOK_PATHS:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        notebook["cells"][2]["source"] = CELL_2.splitlines(keepends=True)
        notebook["cells"][4]["source"] = CELL_4.splitlines(keepends=True)
        notebook["cells"][5]["source"] = CELL_5.splitlines(keepends=True)
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated {notebook_path}")


if __name__ == "__main__":
    main()
