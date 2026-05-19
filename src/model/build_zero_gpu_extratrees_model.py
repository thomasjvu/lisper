#!/usr/bin/env python3
"""Build the v18 ExtraTrees acoustic hint model for the ZeroGPU Space.

This mirrors the acoustic sidecar used by `notebooks/kaggle_gemma4_audio_eval.py`:
the first 1,000 train rows are featurized, clear rows use the source audio, and
non-clear rows are deterministically synthesized from the matching clear clip.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_JSONL = REPO_ROOT / "data" / "processed" / "gemma4_audio" / "messages" / "train.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "spaces" / "lisper-zerogpu" / "acoustic_extratrees_v18.joblib"
TARGET_SAMPLE_RATE = 16000
CLASSES = ["clear", "dental", "frontal", "lateral", "palatal"]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def safe_filter(audio: np.ndarray, sample_rate: int, cutoff: float | tuple[float, float], btype: str, order: int = 3) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if isinstance(cutoff, tuple):
        low = max(40.0, min(float(cutoff[0]), nyquist * 0.75))
        high = min(float(cutoff[1]), nyquist * 0.95)
        if low >= high:
            return audio.astype(np.float32)
        cutoff_value: float | tuple[float, float] = (low, high)
    else:
        cutoff_value = min(float(cutoff), nyquist * 0.95)
    sos = signal.butter(order, cutoff_value, btype=btype, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


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


def load_jsonl_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def audio_basename(audio_path: str) -> str:
    return Path(audio_path).name


def load_clear_audio(row: dict[str, Any], audio_dir: Path) -> tuple[np.ndarray, int]:
    path = audio_dir / audio_basename(row["audio_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing clear source audio: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate}: {path}")
    return normalize_audio(audio), int(sample_rate)


def build_feature_matrix(rows: list[dict[str, Any]], audio_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clear_by_source = {row["source_utterance_id"]: row for row in rows if row["lisp_type"] == "clear"}
    features = []
    labels = []
    groups = []
    for index, row in enumerate(rows, start=1):
        clear_row = clear_by_source.get(row["source_utterance_id"])
        if clear_row is None:
            raise KeyError(f"No clear row for source {row['source_utterance_id']}")
        clear_audio, sample_rate = load_clear_audio(clear_row, audio_dir)
        if row["lisp_type"] == "clear":
            audio = clear_audio
        else:
            audio = synthesize_v18_audio(clear_audio, sample_rate, row["lisp_type"], row["id"])
        features.append(extract_features_from_audio(audio, sample_rate))
        labels.append(row["lisp_type"])
        groups.append(row["source_utterance_id"])
        if index % 250 == 0:
            print(json.dumps({"featurized": index}))
    return np.vstack(features), np.asarray(labels), np.asarray(groups)


def build_classifier() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=300, random_state=17, n_jobs=-1, class_weight="balanced")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-limit", type=int, default=1000)
    args = parser.parse_args()

    rows = load_jsonl_rows(args.train_jsonl, args.train_limit)
    if len(rows) != args.train_limit:
        raise RuntimeError(f"Expected {args.train_limit} rows, loaded {len(rows)}")

    matrix, labels, groups = build_feature_matrix(rows, args.audio_dir)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
    train_idx, test_idx = next(split.split(matrix, labels, groups=groups))
    audit_classifier = build_classifier()
    audit_classifier.fit(matrix[train_idx], labels[train_idx])
    audit_predictions = audit_classifier.predict(matrix[test_idx])
    holdout_accuracy = float(accuracy_score(labels[test_idx], audit_predictions))
    report = classification_report(labels[test_idx], audit_predictions, labels=CLASSES, output_dict=True, zero_division=0)

    classifier = build_classifier()
    classifier.fit(matrix, labels)

    payload = {
        "name": "lisper_v18_extratrees_acoustic_hint",
        "classifier": classifier,
        "classes": list(classifier.classes_),
        "sample_rate": TARGET_SAMPLE_RATE,
        "feature_count": int(matrix.shape[1]),
        "train_rows": int(matrix.shape[0]),
        "source_groups": int(len(set(groups))),
        "train_limit": args.train_limit,
        "holdout_accuracy": holdout_accuracy,
        "holdout_report": report,
        "notes": [
            "Mirrors the v18 acoustic sidecar: first 1,000 train rows, deterministic non-clear synthesis from clear clips.",
            "Used as a live ZeroGPU acoustic hint before Gemma formats the final coaching response.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output, compress=3)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_rows": payload["train_rows"],
                "source_groups": payload["source_groups"],
                "feature_count": payload["feature_count"],
                "classes": payload["classes"],
                "holdout_accuracy": holdout_accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
