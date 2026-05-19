#!/usr/bin/env python3
"""Build the lightweight acoustic hint model used by the ZeroGPU Space."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path("/Users/area/repos/lisper")
SYNTHETIC_ROOT = REPO_ROOT / "data" / "synthetic"
OUTPUT_PATH = REPO_ROOT / "spaces" / "lisper-zerogpu" / "acoustic_model.json"
CLASSES = ["dental", "frontal", "lateral", "palatal"]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


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
    entropy = -(power / total[:, None] * np.log((power / total[:, None]) + eps)).sum(axis=1) / math.log(
        power.shape[1]
    )

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


def source_id_from_path(path: Path) -> str:
    return re.sub(r"_(dental|frontal|lateral|palatal)$", "", path.stem)


def main() -> None:
    rows = []
    for label in CLASSES:
        for audio_path in sorted((SYNTHETIC_ROOT / label).glob("*.wav")):
            audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sample_rate != 16000:
                raise RuntimeError(f"Expected 16 kHz audio, got {sample_rate}: {audio_path}")
            rows.append(
                {
                    "label": label,
                    "source_id": source_id_from_path(audio_path),
                    "path": str(audio_path.relative_to(REPO_ROOT)),
                    "features": extract_features_from_audio(audio, sample_rate),
                }
            )

    if not rows:
        raise RuntimeError(f"No synthetic WAV files found under {SYNTHETIC_ROOT}")

    matrix = np.vstack([row["features"] for row in rows])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    exemplars = []
    for row in rows:
        normalized = (row["features"] - mean) / std
        exemplars.append(
            {
                "label": row["label"],
                "source_id": row["source_id"],
                "features": [round(float(value), 6) for value in normalized],
            }
        )

    payload = {
        "name": "lisper_zero_gpu_synthetic_knn_v1",
        "sample_rate": 16000,
        "classes": CLASSES,
        "feature_count": int(matrix.shape[1]),
        "training_examples": len(rows),
        "mean": [round(float(value), 6) for value in mean],
        "std": [round(float(value), 6) for value in std],
        "exemplars": exemplars,
        "notes": [
            "Built from local synthetic non-clear lisp examples for live demo acoustic hints.",
            "This is a lightweight deployment hint model, not the full v18 ExtraTrees evaluation classifier.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "examples": len(rows), "feature_count": int(matrix.shape[1])}))


if __name__ == "__main__":
    main()
