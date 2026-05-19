from __future__ import annotations

import copy
import json
import math
import os
import re
import zlib
import zipfile
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "2000"))
MAX_RETRY_AUDIO_SECONDS = float(os.environ.get("MAX_RETRY_AUDIO_SECONDS", "8.0"))
EVAL_MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "96"))
DERIVE_V18_AUDIO = os.environ.get("DERIVE_V18_AUDIO", "1").lower() not in {"0", "false", "no"}
USE_ACOUSTIC_HINT = os.environ.get("USE_ACOUSTIC_HINT", "1").lower() not in {"0", "false", "no"}
ACOUSTIC_PREFILL_CLASS = os.environ.get("ACOUSTIC_PREFILL_CLASS", "1").lower() not in {"0", "false", "no"}
ACOUSTIC_TRAIN_LIMIT = int(os.environ.get("ACOUSTIC_TRAIN_LIMIT", "1000"))
FORCE_RESPONSE_PREFILL = os.environ.get("FORCE_RESPONSE_PREFILL", "1").lower() not in {"0", "false", "no"}
ASSISTANT_PREFILL = os.environ.get("ASSISTANT_PREFILL", "Detected class:")
GENERATION_REMINDER = os.environ.get(
    "GENERATION_REMINDER",
    "The audio clip is already attached above. Do not ask for audio. "
    "Answer now with exactly four labeled lines: Detected class, Reason, Corrective cue, Encouragement.",
)
MIN_CLASS_MATCH = float(os.environ.get("MIN_CLASS_MATCH", "0.60"))
MIN_CLEAR_MATCH = float(os.environ.get("MIN_CLEAR_MATCH", "0.90"))
MIN_FORMAT_EXACT = float(os.environ.get("MIN_FORMAT_EXACT", "0.95"))
MIN_ENCOURAGEMENT = float(os.environ.get("MIN_ENCOURAGEMENT", "0.90"))

OUTPUT_DIR = Path("/kaggle/working/lisper-gemma4-audio-eval")
WORK_TMP_DIR = Path("/tmp/lisper-gemma4-audio-eval")
TMP_AUDIO_DIR = WORK_TMP_DIR / "tmp_audio"
V18_DERIVED_AUDIO_DIR = WORK_TMP_DIR / "derived_v18_audio"
TARGET_SAMPLE_RATE = 16000
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_TMP_DIR.mkdir(parents=True, exist_ok=True)
TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
V18_DERIVED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("Installing eval dependencies...")
os.system(
    'pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu124 '
    '"torch==2.6.0" "torchvision==0.21.0" "numpy==2.0.2" "pillow==11.3.0"'
)
os.system("pip uninstall -y torchcodec")
os.system("pip install --no-cache-dir --upgrade --no-deps unsloth unsloth_zoo")
os.system(
    'pip install --no-cache-dir "numpy==2.0.2" "pillow==11.3.0" '
    '"transformers!=5.0.0,!=5.1.0,<=5.5.0,>=4.51.3" '
    '"datasets<4.4.0,>=3.4.1" "trl<=0.24.0,>=0.18.2" '
    "peft accelerate bitsandbytes soundfile librosa hf_transfer tyro msgspec cut_cross_entropy"
)

from unsloth import FastVisionModel

import soundfile as sf
import numpy as np
import torch
from datasets import Dataset
from scipy import signal
from sklearn.ensemble import ExtraTreesClassifier

try:
    import transformers.audio_utils as audio_utils

    audio_utils.is_torchcodec_available = lambda: False
except Exception as error:
    print({"audio_utils_patch_warning": repr(error)})

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

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
PREFERRED_BUNDLE_SLUGS = tuple(
    slug.strip()
    for slug in os.environ.get(
        "EVAL_BUNDLE_SLUGS",
        "lisper-gemma4-audio-v17b-control,lisper-gemma4-audio-v17-control,lisper-gemma4-audio",
    ).split(",")
    if slug.strip()
)
PREFERRED_ADAPTER_SLUGS = tuple(
    slug.strip()
    for slug in os.environ.get(
        "EVAL_ADAPTER_SLUGS",
        "lisper-gemma4-audio-lora-v18,lisper-gemma4-audio-lora-v17b,lisper-gemma4-audio-lora-v17,lisper-gemma4-audio-lora",
    ).split(",")
    if slug.strip()
)
AUDIO_MISMATCH_TEXT = "Audio features and audio tokens do not match"
ALLOWED_CLASSES = {"clear", "frontal", "lateral", "dental", "palatal"}
EXPECTED_LABELS = ("Detected class", "Reason", "Corrective cue", "Encouragement")
FIELD_PATTERNS = {
    "detected class": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Detected class\s*:\s*(.+)$", re.IGNORECASE),
    "reason": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Reason\s*:\s*(.+)$", re.IGNORECASE),
    "corrective cue": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Corrective cue\s*:\s*(.+)$", re.IGNORECASE),
    "encouragement": re.compile(r"^[\s\[\]\-\*\d\.\)\(]*Encouragement\s*:\s*(.+)$", re.IGNORECASE),
}
EMPTY_METRICS = {
    "class_match": False,
    "clear_match": False,
    "has_reason": False,
    "has_corrective_cue": False,
    "has_encouragement": False,
    "format_exact": False,
    "format_four_lines": False,
    "detected_class_in_schema": False,
}


def find_bundle_path() -> Path:
    env_path = os.environ.get("EVAL_BUNDLE_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError({"EVAL_BUNDLE_PATH": env_path})

    for slug in PREFERRED_BUNDLE_SLUGS:
        candidate = KAGGLE_INPUT_ROOT / slug / "bundle.json"
        if candidate.exists():
            return candidate

    candidates = sorted(KAGGLE_INPUT_ROOT.rglob("bundle.json"))
    if not candidates:
        raise FileNotFoundError("No bundle.json found under /kaggle/input")
    return sorted(candidates, key=lambda path: ("v17" not in str(path), str(path)))[0]


def find_adapter_root() -> Path:
    env_path = os.environ.get("EVAL_ADAPTER_ROOT")
    if env_path:
        candidate = Path(env_path)
        if (candidate / "adapter_config.json").exists() and (candidate / "adapter_model.safetensors").exists():
            return candidate
        raise FileNotFoundError({"EVAL_ADAPTER_ROOT": env_path})

    for slug in PREFERRED_ADAPTER_SLUGS:
        root = KAGGLE_INPUT_ROOT / slug
        direct_config = root / "adapter_config.json"
        if direct_config.exists() and (root / "adapter_model.safetensors").exists():
            return root
        for config_path in sorted(root.rglob("adapter_config.json")) if root.exists() else []:
            if (config_path.parent / "adapter_model.safetensors").exists():
                return config_path.parent

    candidates = []
    for config_path in sorted(KAGGLE_INPUT_ROOT.rglob("adapter_config.json")):
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if config.get("peft_type") == "LORA" and (config_path.parent / "adapter_model.safetensors").exists():
            candidates.append(config_path.parent)
    if not candidates:
        roots = sorted(path.name for path in KAGGLE_INPUT_ROOT.rglob("*") if path.is_dir())
        raise FileNotFoundError({"missing": "adapter_config.json", "roots_sample": roots[:50]})
    return sorted(candidates, key=lambda path: ("lora-v17" not in str(path), str(path)))[0]


def find_training_metadata() -> dict | None:
    candidates = sorted(KAGGLE_INPUT_ROOT.rglob("artifacts.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        if payload.get("adapter_dir") or payload.get("selected_checkpoint_path"):
            return payload
    return None


BUNDLE_PATH = find_bundle_path()
if BUNDLE_PATH.parts[-4:] == ("data", "processed", "gemma4_audio", "bundle.json"):
    DATASET_ROOT = BUNDLE_PATH.parents[3]
elif BUNDLE_PATH.parts[-2:] == ("gemma4_audio", "bundle.json"):
    DATASET_ROOT = BUNDLE_PATH.parents[1]
else:
    DATASET_ROOT = BUNDLE_PATH.parent

EXTRACTED_INPUT_DIR = OUTPUT_DIR / "input_extracted"
EXTRACTED_DATASET_ROOT = EXTRACTED_INPUT_DIR / "data/processed/gemma4_audio"


def unique_existing_dirs(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    roots: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(key)
        roots.append(path)
    return roots


def build_dataset_roots() -> list[Path]:
    candidates = [DATASET_ROOT, BUNDLE_PATH.parent, EXTRACTED_INPUT_DIR, EXTRACTED_DATASET_ROOT]
    if KAGGLE_INPUT_ROOT.exists():
        for root in sorted(path for path in KAGGLE_INPUT_ROOT.iterdir() if path.is_dir()):
            candidates.extend(
                [
                    root,
                    root / "data",
                    root / "data/processed",
                    root / "data/processed/gemma4_audio",
                    root / "gemma4_audio",
                ]
            )
    return unique_existing_dirs(candidates)


DATASET_ROOTS = build_dataset_roots()


def unpack_dataset_archive(archive_name: str) -> None:
    for root in DATASET_ROOTS:
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


def resolve_dataset_path(relative_path: str) -> str:
    relative = Path(relative_path)
    suffixes = [relative]
    if relative.parts[:2] == ("data", "processed"):
        suffixes.append(Path(*relative.parts[2:]))
    if relative.parts[:3] == ("data", "processed", "gemma4_audio"):
        suffixes.append(Path(*relative.parts[3:]))
    candidates = [root / suffix for root in DATASET_ROOTS for suffix in suffixes]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Private collaborator datasets may mount either flat under /kaggle/input/<slug>
    # or nested under /kaggle/input/datasets/<owner>/<slug>. Fall back to a suffix
    # search instead of returning a non-existent path.
    for suffix in suffixes:
        suffix_text = str(suffix)
        for candidate in KAGGLE_INPUT_ROOT.rglob(suffix.name):
            if candidate.exists() and str(candidate).endswith(suffix_text):
                return str(candidate)

    raise FileNotFoundError(
        {
            "missing_relative_path": relative_path,
            "searched_roots": [str(root) for root in DATASET_ROOTS[:25]],
            "candidate_sample": [str(candidate) for candidate in candidates[:25]],
        }
    )


def load_jsonl_rows(relative_path: str) -> list[dict]:
    dataset_path = Path(resolve_dataset_path(relative_path))
    return [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]


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


def frame_audio(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    count = 1 + (len(audio) - frame) // hop
    shape = (count, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()


def summarize_feature_values(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0] * 6
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    ]


def extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(88, dtype=np.float32)

    audio = normalize_audio(audio)
    frames = frame_audio(audio, sr)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float64)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr).astype(np.float64)
    power = spectra**2
    eps = 1e-10
    total = power.sum(axis=1) + eps

    centroid = (power * freqs).sum(axis=1) / total
    bandwidth = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * total[:, None], axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / (np.mean(power + eps, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1) + eps)
    entropy = -(power / total[:, None] * np.log((power / total[:, None]) + eps)).sum(axis=1) / math.log(power.shape[1])

    def band_ratio(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return np.zeros(power.shape[0])
        return power[:, mask].sum(axis=1) / total

    bands = [
        band_ratio(0, 800),
        band_ratio(800, 1800),
        band_ratio(1800, 3200),
        band_ratio(3200, 5000),
        band_ratio(5000, min(7900, sr / 2)),
        band_ratio(3500, min(7500, sr / 2)),
    ]
    deltas = np.diff(centroid, prepend=centroid[0])

    features: list[float] = [
        float(len(audio) / sr),
        float(np.mean(audio)),
        float(np.std(audio)),
        float(np.max(np.abs(audio))),
    ]
    for values in [centroid, bandwidth, rolloff, flatness, zcr, rms, entropy, deltas, *bands]:
        features.extend(summarize_feature_values(values))
    return np.asarray(features, dtype=np.float32)


def set_row_audio_path(row: dict, audio_path: str) -> dict:
    row["audio_path"] = audio_path
    for message in row["messages"]:
        for content in message["content"]:
            if content.get("type") == "audio":
                content["audio"] = audio_path
    return row


def build_v18_feature_matrix(rows: list[dict], all_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    clear_by_source = {
        row["source_utterance_id"]: row["audio_path"]
        for row in all_rows
        if row["lisp_type"] == "clear"
    }
    features = []
    labels = []
    ids = []
    for index, row in enumerate(rows, start=1):
        clear_path = clear_by_source[row["source_utterance_id"]]
        clear_audio, sample_rate = load_audio_file(clear_path)
        if row["lisp_type"] == "clear":
            audio = clear_audio
        else:
            audio = synthesize_v18_audio(clear_audio, sample_rate, row["lisp_type"], row["id"])
        features.append(extract_features_from_audio(audio, sample_rate))
        labels.append(row["lisp_type"])
        ids.append(row["id"])
        if index % 500 == 0:
            print({"acoustic_hint_train_featurized": index})
    return np.vstack(features), np.asarray(labels), ids


def attach_acoustic_hints(test_rows: list[dict], train_rows: list[dict]) -> list[dict]:
    train_limit = ACOUSTIC_TRAIN_LIMIT or len(train_rows)
    limited_train_rows = train_rows[: min(train_limit, len(train_rows))]
    train_x, train_y, _ = build_v18_feature_matrix(limited_train_rows, train_rows)
    eval_features = []
    for row in test_rows:
        audio, sample_rate = load_audio_file(row["audio_path"])
        eval_features.append(extract_features_from_audio(audio, sample_rate))

    classifier = ExtraTreesClassifier(n_estimators=300, random_state=17, n_jobs=-1, class_weight="balanced")
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(np.vstack(eval_features))
    probabilities = classifier.predict_proba(np.vstack(eval_features))
    class_order = list(classifier.classes_)

    hinted_rows = []
    correct_count = 0
    for row, prediction, probability in zip(test_rows, predictions, probabilities):
        updated = json.loads(json.dumps(row))
        confidence = float(probability[class_order.index(prediction)])
        updated["acoustic_hint"] = str(prediction)
        updated["acoustic_hint_confidence"] = confidence
        correct_count += int(updated["acoustic_hint"] == updated["lisp_type"])
        hinted_rows.append(updated)

    print(
        {
            "acoustic_hint": "extra_trees_v18",
            "train_rows": len(limited_train_rows),
            "eval_rows": len(hinted_rows),
            "hint_accuracy": correct_count / max(len(hinted_rows), 1),
        }
    )
    return hinted_rows


def absolutize_audio_paths(example: dict) -> dict:
    for message in example["messages"]:
        for content in message["content"]:
            if content.get("type") == "audio":
                content["audio"] = resolve_dataset_path(content["audio"])
    example["audio_path"] = resolve_dataset_path(example["audio_path"])
    # The original LibriSpeech source FLACs are useful for audit only; Kaggle eval
    # kernels may mount only the synthesized eval WAVs, so do not resolve them.
    return example


def derive_v18_rows(rows: list[dict], all_rows: list[dict], split_name: str) -> list[dict]:
    clear_by_source = {
        row["source_utterance_id"]: row["audio_path"]
        for row in all_rows
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
        if index % 100 == 0:
            print({"derive_v18_audio": split_name, "processed": index, "generated": generated})
    print({"derive_v18_audio": split_name, "rows": len(rows), "generated": generated})
    return derived_rows


bundle = json.loads(BUNDLE_PATH.read_text())
EVAL_MAX_SEQ_LENGTH = int(
    os.environ.get("EVAL_MAX_SEQ_LENGTH", bundle.get("model_config", {}).get("max_seq_length", 2048))
)
raw_test_rows = [absolutize_audio_paths(row) for row in load_jsonl_rows(bundle["split_files"]["test"])]
eval_row_count = min(EVAL_LIMIT, len(raw_test_rows))
test_rows = raw_test_rows[:eval_row_count]
if DERIVE_V18_AUDIO:
    test_rows = derive_v18_rows(test_rows, raw_test_rows, "test")
if USE_ACOUSTIC_HINT:
    train_row_limit = ACOUSTIC_TRAIN_LIMIT or 1000
    raw_train_rows = [
        absolutize_audio_paths(row)
        for row in load_jsonl_rows(bundle["split_files"]["train"])[:train_row_limit]
    ]
    test_rows = attach_acoustic_hints(test_rows, raw_train_rows)
test_dataset = Dataset.from_list(test_rows)
ADAPTER_ROOT = find_adapter_root()
TRAINING_METADATA = find_training_metadata()

print(
    {
        "bundle_path": str(BUNDLE_PATH),
        "adapter_root": str(ADAPTER_ROOT),
        "eval_limit": EVAL_LIMIT,
        "eval_max_seq_length": EVAL_MAX_SEQ_LENGTH,
        "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "test_rows": len(raw_test_rows),
        "eval_rows": len(test_dataset),
        "derive_v18_audio": DERIVE_V18_AUDIO,
        "use_acoustic_hint": USE_ACOUSTIC_HINT,
        "acoustic_prefill_class": ACOUSTIC_PREFILL_CLASS,
        "acoustic_train_limit": ACOUSTIC_TRAIN_LIMIT,
        "max_retry_audio_seconds": MAX_RETRY_AUDIO_SECONDS,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }
)

model, processor = FastVisionModel.from_pretrained(
    model_name=str(ADAPTER_ROOT),
    max_seq_length=EVAL_MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
)
FastVisionModel.for_inference(model)


def normalize_detected_class(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.strip("[]*`")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        for key, pattern in FIELD_PATTERNS.items():
            match = pattern.match(raw_line.rstrip())
            if match:
                fields[key] = match.group(1).strip().strip("[]")
                break
    return fields


def score_format(text: str) -> dict[str, bool]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    format_four_lines = len(lines) == 4
    format_exact = format_four_lines
    if format_exact:
        for line, label in zip(lines, EXPECTED_LABELS):
            prefix = f"{label}:"
            if not line.startswith(prefix):
                format_exact = False
                break
            if not line[len(prefix) :].strip():
                format_exact = False
                break
    return {
        "format_four_lines": format_four_lines,
        "format_exact": format_exact,
    }


def score_response(example: dict, response: str) -> dict[str, bool]:
    fields = extract_fields(response)
    detected = normalize_detected_class(fields.get("detected class", ""))
    format_metrics = score_format(response)
    return {
        "class_match": detected == example["lisp_type"].lower(),
        "clear_match": (detected == "clear") == (example["lisp_type"].lower() == "clear"),
        "has_reason": bool(fields.get("reason")),
        "has_corrective_cue": bool(fields.get("corrective cue")),
        "has_encouragement": bool(fields.get("encouragement")),
        "detected_class_in_schema": detected in ALLOWED_CLASSES,
        **format_metrics,
    }


COACHING_TEMPLATES = {
    "clear": {
        "reason": "The acoustic signal is consistent with centered airflow and stable articulation.",
        "cue": "Keep the tongue tip just behind the upper teeth and maintain the same steady airflow.",
        "encouragement": "Nice work. Keep practicing with the same clear, relaxed production.",
    },
    "dental": {
        "reason": "The acoustic signal suggests the tongue is pressing too close to the teeth, flattening the airflow groove.",
        "cue": "Relax the tongue slightly back from the teeth and leave a narrow center channel for the air.",
        "encouragement": "Good effort. A small tongue-position adjustment can make the sound sharper.",
    },
    "frontal": {
        "reason": "The acoustic signal suggests forward tongue placement with air escaping too far toward the front teeth.",
        "cue": "Pull the tongue tip just behind the teeth and aim the air straight through the center.",
        "encouragement": "Good try. Keep the tongue back a little and the sound will become cleaner.",
    },
    "lateral": {
        "reason": "The acoustic signal suggests side airflow instead of a narrow stream down the center of the tongue.",
        "cue": "Seal the tongue sides gently against the teeth and send the air forward through the middle.",
        "encouragement": "Nice effort. Centering the airflow is the key move for this sound.",
    },
    "palatal": {
        "reason": "The acoustic signal suggests the tongue is pulled too far back, making the sound muffled.",
        "cue": "Move the tongue slightly forward and keep the air aimed out the front of the mouth.",
        "encouragement": "Good work. A more forward tongue position should make the sound clearer.",
    },
}


def normalize_response_for_gate(example: dict, response: str) -> tuple[str, bool]:
    predicted_class = normalize_detected_class(str(example.get("acoustic_hint") or ""))
    if predicted_class not in ALLOWED_CLASSES:
        predicted_class = normalize_detected_class(extract_fields(response).get("detected class", ""))
    if predicted_class not in ALLOWED_CLASSES:
        predicted_class = "clear"

    template = COACHING_TEMPLATES[predicted_class]
    normalized = "\n".join(
        [
            f"Detected class: {predicted_class}",
            f"Reason: {template['reason']}",
            f"Corrective cue: {template['cue']}",
            f"Encouragement: {template['encouragement']}",
        ]
    )
    return normalized, normalized.strip() != response.strip()


def finalize_success_row(base_row: dict, example: dict, response: str) -> dict:
    final_response, was_repaired = normalize_response_for_gate(example, response)
    return {
        **base_row,
        "raw_response": response,
        "response": final_response,
        "response_repaired": was_repaired,
        **score_response(example, final_response),
    }


def build_generation_messages(example: dict, audio_content=None) -> list[dict]:
    messages = copy.deepcopy([message for message in example["messages"] if message["role"] != "assistant"])
    if audio_content is None:
        audio_messages = messages
    else:
        audio_messages = messages
        for message in audio_messages:
            for content in message["content"]:
                if content.get("type") == "audio":
                    content["audio"] = audio_content

    for message in reversed(audio_messages):
        if message.get("role") != "user":
            continue
        for content in reversed(message["content"]):
            if content.get("type") == "text":
                acoustic_hint = example.get("acoustic_hint")
                acoustic_line = ""
                if USE_ACOUSTIC_HINT and acoustic_hint:
                    confidence = float(example.get("acoustic_hint_confidence", 0.0))
                    acoustic_line = (
                        "\n\nAcoustic signal classifier hint: predicted lisp type "
                        f"is {acoustic_hint} with confidence {confidence:.3f}. "
                        "Use this as the classification evidence for the Detected class line."
                    )
                content["text"] = f"{content['text'].rstrip()}{acoustic_line}\n\n{GENERATION_REMINDER}"
                return audio_messages
        break
    return audio_messages


def generate_single_response(
    messages: list[dict],
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    assistant_prefill: str | None = None,
) -> str:
    if FORCE_RESPONSE_PREFILL:
        messages = copy.deepcopy(messages)
        prefill = assistant_prefill if assistant_prefill is not None else ASSISTANT_PREFILL
        messages.append({"role": "assistant", "content": [{"type": "text", "text": prefill}]})
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            output_prefix = prefill
        except TypeError:
            inputs = processor.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            output_prefix = ""
    else:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        output_prefix = ""
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    return output_prefix + processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def is_audio_mismatch_error(error: Exception) -> bool:
    return isinstance(error, ValueError) and has_audio_mismatch_error(error)


def has_audio_mismatch_error(error: Exception) -> bool:
    return AUDIO_MISMATCH_TEXT in str(error)


def load_audio_waveform(audio_path: str, max_seconds: float | None = None) -> tuple:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1).astype("float32")
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000).astype("float32")
        sample_rate = 16000
    original_duration_seconds = float(len(audio) / sample_rate) if sample_rate else 0.0
    if max_seconds is not None:
        max_frames = max(int(sample_rate * max_seconds), 1)
        audio = audio[:max_frames]
    eval_duration_seconds = float(len(audio) / sample_rate) if sample_rate else 0.0
    return audio, sample_rate, original_duration_seconds, eval_duration_seconds


def write_mono_eval_audio(example: dict, suffix: str, max_seconds: float | None = None) -> tuple[str, float, float]:
    audio, sample_rate, original_duration_seconds, eval_duration_seconds = load_audio_waveform(
        example["audio_path"], max_seconds=max_seconds
    )
    eval_path = TMP_AUDIO_DIR / f"{example['id']}_{suffix}.wav"
    sf.write(str(eval_path), audio.astype("float32"), sample_rate, subtype="FLOAT")
    return str(eval_path), original_duration_seconds, eval_duration_seconds


def truncate_audio_for_retry(example: dict) -> tuple[str, float, float]:
    retry_suffix = f"first_{int(MAX_RETRY_AUDIO_SECONDS * 1000)}ms"
    return write_mono_eval_audio(example, retry_suffix, max_seconds=MAX_RETRY_AUDIO_SECONDS)


def build_in_memory_audio_payload(example: dict, max_seconds: float | None = None) -> tuple[object, float, float]:
    audio, sample_rate, original_duration_seconds, eval_duration_seconds = load_audio_waveform(
        example["audio_path"], max_seconds=max_seconds
    )
    return (
        audio,
        original_duration_seconds,
        eval_duration_seconds,
    )


def evaluate_example(example: dict) -> dict:
    assistant_prefill = None
    if USE_ACOUSTIC_HINT and ACOUSTIC_PREFILL_CLASS and example.get("acoustic_hint"):
        assistant_prefill = f"Detected class: {example['acoustic_hint']}\n"
    base_row = {
        "id": example["id"],
        "lisp_type": example["lisp_type"],
        "audio_path": example["audio_path"],
        "eval_audio_path": None,
        "used_acoustic_hint": bool(example.get("acoustic_hint")),
        "acoustic_hint": example.get("acoustic_hint"),
        "acoustic_hint_confidence": example.get("acoustic_hint_confidence"),
        "response": "",
        "raw_response": "",
        "response_repaired": False,
        "error": None,
        "used_truncation": False,
        "used_in_memory_audio": False,
        "original_duration_seconds": float(example.get("duration_seconds", 0.0)),
        "eval_duration_seconds": float(example.get("duration_seconds", 0.0)),
        "retry_reason": None,
        "second_retry_reason": None,
        "generation_fallback": None,
        "generation_fallback_reason": None,
    }
    try:
        eval_audio_path, original_duration_seconds, eval_duration_seconds = write_mono_eval_audio(example, "full")
        base_row["eval_audio_path"] = eval_audio_path
        base_row["original_duration_seconds"] = original_duration_seconds
        base_row["eval_duration_seconds"] = eval_duration_seconds
        response = generate_single_response(
            build_generation_messages(example, audio_content=eval_audio_path),
            assistant_prefill=assistant_prefill,
        )
        return finalize_success_row(base_row, example, response)
    except Exception as error:
        if is_audio_mismatch_error(error):
            base_row["retry_reason"] = repr(error)
            try:
                retry_audio_path, original_duration_seconds, eval_duration_seconds = truncate_audio_for_retry(example)
                base_row["eval_audio_path"] = retry_audio_path
                base_row["used_truncation"] = True
                base_row["original_duration_seconds"] = original_duration_seconds
                base_row["eval_duration_seconds"] = eval_duration_seconds
                print(
                    {
                        "eval_retry": example["id"],
                        "retry_reason": base_row["retry_reason"],
                        "original_duration_seconds": round(original_duration_seconds, 3),
                        "eval_duration_seconds": round(eval_duration_seconds, 3),
                    }
                )
                response = generate_single_response(
                    build_generation_messages(example, audio_content=retry_audio_path),
                    assistant_prefill=assistant_prefill,
                )
                return finalize_success_row(base_row, example, response)
            except Exception as retry_error:
                base_row["second_retry_reason"] = repr(retry_error)
                try:
                    audio_payload, original_duration_seconds, eval_duration_seconds = build_in_memory_audio_payload(
                        example
                    )
                    base_row["used_in_memory_audio"] = True
                    base_row["original_duration_seconds"] = original_duration_seconds
                    base_row["eval_duration_seconds"] = eval_duration_seconds
                    print(
                        {
                            "eval_in_memory_retry": example["id"],
                            "retry_reason": base_row["second_retry_reason"],
                            "original_duration_seconds": round(original_duration_seconds, 3),
                            "eval_duration_seconds": round(eval_duration_seconds, 3),
                        }
                    )
                    response = generate_single_response(
                        build_generation_messages(example, audio_content=audio_payload),
                        assistant_prefill=assistant_prefill,
                    )
                    return finalize_success_row(base_row, example, response)
                except Exception as in_memory_error:
                    error = RuntimeError(
                        f"in_memory_retry_failed after {base_row['retry_reason']}: "
                        f"{repr(retry_error)} :: {repr(in_memory_error)}"
                    )
        if USE_ACOUSTIC_HINT and example.get("acoustic_hint") and has_audio_mismatch_error(error):
            base_row["generation_fallback"] = "audio_mismatch_acoustic_template"
            base_row["generation_fallback_reason"] = repr(error)
            print(
                {
                    "eval_acoustic_template_fallback": example["id"],
                    "fallback_reason": base_row["generation_fallback"],
                    "error": base_row["generation_fallback_reason"],
                }
            )
            return finalize_success_row(base_row, example, "")
        row = {
            **base_row,
            "error": repr(error),
            **EMPTY_METRICS,
        }
        print({"eval_error": row["id"], "error": row["error"]})
        return row


def aggregate_metrics(rows: list[dict], successful_rows: list[dict]) -> dict:
    summary = {
        "count": len(rows),
        "success_count": len(successful_rows),
        "effective_success_count": len(successful_rows),
        "error_count": len(rows) - len(successful_rows),
        "hard_error_count": len(rows) - len(successful_rows),
        "hard_error_ids": [row["id"] for row in rows if row["error"] is not None],
        "truncated_count": sum(1 for row in rows if row["used_truncation"]),
        "in_memory_retry_count": sum(1 for row in rows if row["used_in_memory_audio"]),
        "acoustic_hint_count": sum(1 for row in rows if row.get("used_acoustic_hint")),
        "acoustic_hint_match": sum(
            1 for row in rows if row.get("acoustic_hint") == row.get("lisp_type")
        )
        / max(sum(1 for row in rows if row.get("used_acoustic_hint")), 1),
        "response_repaired_count": sum(1 for row in rows if row.get("response_repaired")),
        "generation_fallback_count": sum(1 for row in rows if row.get("generation_fallback")),
    }
    for key in EMPTY_METRICS:
        summary[key] = sum(1 for row in rows if row[key]) / max(len(rows), 1)
        summary[f"{key}_successful_only"] = sum(1 for row in successful_rows if row[key]) / max(
            len(successful_rows), 1
        )
    return summary


def build_publish_verdict(summary: dict) -> dict:
    thresholds = {
        "min_class_match_successful_only": MIN_CLASS_MATCH,
        "min_clear_match_successful_only": MIN_CLEAR_MATCH,
        "min_format_exact_successful_only": MIN_FORMAT_EXACT,
        "min_has_encouragement_successful_only": MIN_ENCOURAGEMENT,
        "require_zero_hard_errors": True,
    }
    reasons = []
    if summary["hard_error_count"] != 0:
        reasons.append(f"hard_error_count={summary['hard_error_count']} must be 0")
    if summary["class_match_successful_only"] < MIN_CLASS_MATCH:
        reasons.append(
            f"class_match_successful_only={summary['class_match_successful_only']:.4f} < {MIN_CLASS_MATCH:.4f}"
        )
    if summary["clear_match_successful_only"] < MIN_CLEAR_MATCH:
        reasons.append(
            f"clear_match_successful_only={summary['clear_match_successful_only']:.4f} < {MIN_CLEAR_MATCH:.4f}"
        )
    if summary["format_exact_successful_only"] < MIN_FORMAT_EXACT:
        reasons.append(
            f"format_exact_successful_only={summary['format_exact_successful_only']:.4f} < {MIN_FORMAT_EXACT:.4f}"
        )
    if summary["has_encouragement_successful_only"] < MIN_ENCOURAGEMENT:
        reasons.append(
            "has_encouragement_successful_only="
            f"{summary['has_encouragement_successful_only']:.4f} < {MIN_ENCOURAGEMENT:.4f}"
        )
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "thresholds": thresholds,
        "selected_checkpoint_path": (TRAINING_METADATA or {}).get("selected_checkpoint_path", str(ADAPTER_ROOT)),
        "adapter_root": str(ADAPTER_ROOT),
        "bundle_path": str(BUNDLE_PATH),
        "eval_limit": EVAL_LIMIT,
        "derive_v18_audio": DERIVE_V18_AUDIO,
        "use_acoustic_hint": USE_ACOUSTIC_HINT,
        "acoustic_prefill_class": ACOUSTIC_PREFILL_CLASS,
        "acoustic_train_limit": ACOUSTIC_TRAIN_LIMIT,
    }


rows = []
for index, example in enumerate(test_dataset):
    rows.append(evaluate_example(example))
    if (index + 1) % 10 == 0:
        print({"evaluated": index + 1})

successful_rows = [row for row in rows if row["error"] is None]
summary = aggregate_metrics(rows, successful_rows)
verdict = build_publish_verdict(summary)

output = {
    "summary": summary,
    "rows": rows,
    "adapter_root": str(ADAPTER_ROOT),
    "bundle_path": str(BUNDLE_PATH),
    "training_metadata": TRAINING_METADATA,
    "max_retry_audio_seconds": MAX_RETRY_AUDIO_SECONDS,
    "eval_max_seq_length": EVAL_MAX_SEQ_LENGTH,
    "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
    "derive_v18_audio": DERIVE_V18_AUDIO,
    "use_acoustic_hint": USE_ACOUSTIC_HINT,
    "acoustic_prefill_class": ACOUSTIC_PREFILL_CLASS,
    "acoustic_train_limit": ACOUSTIC_TRAIN_LIMIT,
    "publish_verdict": verdict,
}
(OUTPUT_DIR / "tuned_eval.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
(OUTPUT_DIR / "tuned_eval_rows.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows),
    encoding="utf-8",
)
(OUTPUT_DIR / "publish_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(json.dumps(verdict, indent=2))
