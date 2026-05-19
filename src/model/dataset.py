#!/usr/bin/env python3
"""Build Lisper raw-audio training data for Gemma 4 E2B."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

REPO_ROOT = Path("/Users/area/repos/lisper")
DATA_DIR = REPO_ROOT / "data"
RAW_SPEECH_ROOT = DATA_DIR / "raw"
LIBRISPEECH_ROOT = RAW_SPEECH_ROOT / "LibriSpeech"
PROCESSED_DIR = DATA_DIR / "processed"
MULTIMODAL_DIR = PROCESSED_DIR / "gemma4_audio"
TRAINING_AUDIO_DIR = MULTIMODAL_DIR / "audio"
MANIFEST_FILE = MULTIMODAL_DIR / "manifest.jsonl"
SUMMARY_FILE = MULTIMODAL_DIR / "summary.json"
BUILD_CONFIG_FILE = MULTIMODAL_DIR / "build_config.json"
TEXT_OUTPUT_FILE = DATA_DIR / "training_data.json"
LEGACY_SYNTHETIC_DIR = DATA_DIR / "synthetic"

TARGET_SAMPLE_RATE = 16000
DEFAULT_SEED = 3407
DEFAULT_PROFILE = "hackathon"
LISP_TYPES = ("clear", "frontal", "lateral", "dental", "palatal")
FOCUS_SOUND_TOKEN_PATTERN = re.compile(r"sh|ch|j|s|z", re.IGNORECASE)

TARGET_PROFILES = {
    "smoke": {
        "target_splits": {"train": 200, "val": 24, "test": 24},
        "min_speakers": 20,
        "preview_per_type": 12,
        "description": "Fast local validation profile.",
    },
    "hackathon": {
        "target_splits": {"train": 3200, "val": 400, "test": 400},
        "min_speakers": 150,
        "preview_per_type": 20,
        "description": "Real training profile targeting 4,000 unique source utterances and 20,000 total examples.",
    },
}

LISP_LABELS = {
    "clear": {
        "reasons": [
            "The airflow stays centered and the tongue appears to stay just behind the upper teeth.",
            "The sibilant sounds focused, with a clean center groove and steady forward airflow.",
        ],
        "cues": [
            "Keep the same tongue position and steady forward airflow on the next repetition.",
            "Repeat it the same way, keeping the tongue tip just behind the upper teeth.",
        ],
        "encouragements": [
            "Excellent work. That production is clear and stable.",
            "Nice job. Keep that exact placement and airflow.",
        ],
    },
    "frontal": {
        "reasons": [
            "The airflow sounds too far forward, which gives the consonant a th-like quality.",
            "The tongue appears to push between or against the teeth, which softens the sibilant.",
        ],
        "cues": [
            "Keep the tongue tip just behind the upper teeth and send the air straight forward.",
            "Pull the tongue back slightly so it stays behind the teeth instead of between them.",
        ],
        "encouragements": [
            "Good effort. Try it again with the tongue kept back.",
            "Nice try. A smaller tongue movement should clean up the sound.",
        ],
    },
    "lateral": {
        "reasons": [
            "Air is leaking over the sides of the tongue, which creates a wet or slushy quality.",
            "The sound suggests side airflow instead of a narrow stream down the center of the tongue.",
        ],
        "cues": [
            "Lift the sides of the tongue toward the upper molars and aim the air down the middle.",
            "Start from a held /t/ position, keep the sides sealed, and let the air move forward.",
        ],
        "encouragements": [
            "Good attempt. Narrowing the center airflow should help immediately.",
            "You are close. Focus on stopping the side airflow on the next try.",
        ],
    },
    "dental": {
        "reasons": [
            "The tongue seems to press too much against the teeth, which reduces the groove for airflow.",
            "The consonant sounds dentalized, with the tongue resting on the teeth instead of just behind them.",
        ],
        "cues": [
            "Make a tiny gap between the tongue and the teeth so the air can pass through cleanly.",
            "Relax the tongue off the teeth a little and keep the airflow moving over the tip.",
        ],
        "encouragements": [
            "Nice effort. A slightly freer groove should sharpen the sound.",
            "Keep going. A smaller tongue-to-teeth contact will help.",
        ],
    },
    "palatal": {
        "reasons": [
            "The tongue seems pulled too far back, which makes the sound muffled and less focused.",
            "The sibilant sounds backed up toward the palate instead of staying forward.",
        ],
        "cues": [
            "Bring the tongue tip forward so it rests just behind the upper teeth.",
            "Move the tongue slightly forward and keep the airflow aimed out the front of the mouth.",
        ],
        "encouragements": [
            "Good try. Bringing the tongue forward should make the sound clearer.",
            "Keep practicing. A more forward placement should brighten the consonant.",
        ],
    },
}


@dataclass(frozen=True)
class SourceUtterance:
    """Metadata for a single LibriSpeech utterance."""

    utterance_id: str
    subset_name: str
    speaker_id: str
    chapter_id: str
    transcript: str
    audio_path: Path
    duration_seconds: float
    focus_score: int


def repo_relative(path: Path) -> str:
    """Return a repo-relative path string."""

    return str(path.relative_to(REPO_ROOT))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def get_profile(profile_name: str) -> dict:
    """Return a profile definition."""

    if profile_name not in TARGET_PROFILES:
        raise ValueError(f"Unknown profile '{profile_name}'. Choose from {sorted(TARGET_PROFILES)}.")
    return TARGET_PROFILES[profile_name]


def discover_librispeech_subset_dirs(subset_names: list[str] | None = None) -> list[Path]:
    """Find LibriSpeech subset directories that contain audio."""

    search_roots = [LIBRISPEECH_ROOT]
    nested_root = LIBRISPEECH_ROOT / "LibriSpeech"
    if nested_root.exists():
        search_roots.append(nested_root)

    if subset_names:
        subset_dirs: list[Path] = []
        missing_subset_names: list[str] = []
        for subset_name in subset_names:
            found_dir = None
            for root in search_roots:
                candidate = root / subset_name
                if candidate.exists():
                    found_dir = candidate
                    break
            if found_dir is None:
                missing_subset_names.append(subset_name)
            else:
                subset_dirs.append(found_dir)
        if missing_subset_names:
            raise RuntimeError(
                "Missing LibriSpeech subsets: "
                + ", ".join(sorted(missing_subset_names))
                + f". Add them under {LIBRISPEECH_ROOT}."
            )
    else:
        seen_subset_names: set[str] = set()
        subset_dirs = []
        for root in search_roots:
            for path in sorted(root.iterdir()):
                if not path.is_dir() or path.name in seen_subset_names:
                    continue
                if any(path.glob("*/*/*.flac")):
                    subset_dirs.append(path)
                    seen_subset_names.add(path.name)

    valid_subset_dirs: list[Path] = []
    for subset_dir in subset_dirs:
        if any(subset_dir.glob("*/*/*.flac")):
            valid_subset_dirs.append(subset_dir)
    if not valid_subset_dirs:
        raise RuntimeError(f"No LibriSpeech audio subsets found under {LIBRISPEECH_ROOT}")
    return valid_subset_dirs


def count_focus_sounds(transcript: str) -> int:
    """Count likely sibilant targets in a transcript."""

    return len(FOCUS_SOUND_TOKEN_PATTERN.findall(transcript.lower()))


def load_transcripts(subset_dirs: list[Path]) -> dict[str, str]:
    """Parse LibriSpeech transcript files."""

    transcripts: dict[str, str] = {}
    for subset_dir in subset_dirs:
        for transcript_file in sorted(subset_dir.glob("*/*/*.trans.txt")):
            with transcript_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    utterance_id, transcript = line.split(" ", 1)
                    transcripts[utterance_id] = transcript.strip()
    return transcripts


def classify_practice_level(transcript: str) -> str:
    """Infer practice level from transcript length."""

    words = transcript.split()
    if len(words) == 1:
        return "words"
    if len(words) <= 4:
        return "phrases"
    return "sentences"


def scan_source_utterances(
    subset_names: list[str] | None = None,
    min_duration_seconds: float = 1.0,
    max_duration_seconds: float = 12.0,
) -> list[SourceUtterance]:
    """Scan available LibriSpeech subsets and return filtered utterances."""

    subset_dirs = discover_librispeech_subset_dirs(subset_names)
    transcripts = load_transcripts(subset_dirs)
    utterances: list[SourceUtterance] = []

    for subset_dir in subset_dirs:
        subset_name = subset_dir.name
        for audio_path in sorted(subset_dir.glob("*/*/*.flac")):
            relative = audio_path.relative_to(subset_dir)
            if len(relative.parts) != 3:
                continue

            utterance_id = audio_path.stem
            transcript = transcripts.get(utterance_id)
            if transcript is None:
                continue

            focus_score = count_focus_sounds(transcript)
            if focus_score == 0:
                continue

            info = sf.info(str(audio_path))
            if info.duration < min_duration_seconds or info.duration > max_duration_seconds:
                continue

            utterances.append(
                SourceUtterance(
                    utterance_id=utterance_id,
                    subset_name=subset_name,
                    speaker_id=relative.parts[0],
                    chapter_id=relative.parts[1],
                    transcript=transcript,
                    audio_path=audio_path,
                    duration_seconds=float(info.duration),
                    focus_score=focus_score,
                )
            )

    if not utterances:
        raise RuntimeError(
            f"No usable audio files found under {LIBRISPEECH_ROOT}. "
            "Check that the subsets are extracted correctly."
        )

    return utterances


def assign_speakers_to_splits(
    utterances: list[SourceUtterance],
    target_splits: dict[str, int],
    seed: int,
) -> dict[str, set[str]]:
    """Create deterministic speaker-disjoint splits."""

    by_speaker: dict[str, list[SourceUtterance]] = defaultdict(list)
    for utterance in utterances:
        by_speaker[utterance.speaker_id].append(utterance)

    speakers = sorted(by_speaker)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    train_speakers = max(1, round(len(speakers) * 0.70))
    val_speakers = max(1, round(len(speakers) * 0.15))
    test_speakers = max(1, len(speakers) - train_speakers - val_speakers)

    while train_speakers + val_speakers + test_speakers > len(speakers):
        train_speakers -= 1

    assignments = {
        "train": set(speakers[:train_speakers]),
        "val": set(speakers[train_speakers:train_speakers + val_speakers]),
        "test": set(speakers[train_speakers + val_speakers:train_speakers + val_speakers + test_speakers]),
    }

    for split_name, target_count in target_splits.items():
        available = sum(1 for utterance in utterances if utterance.speaker_id in assignments[split_name])
        if available < target_count:
            raise RuntimeError(
                f"Split {split_name} has only {available} usable utterances for target {target_count}. "
                "Add more raw subsets before building this profile."
            )

    return assignments


def balanced_sample_utterances(
    pool: list[SourceUtterance],
    target_count: int,
    seed: int,
) -> list[SourceUtterance]:
    """Sample utterances while spreading examples across speakers."""

    by_speaker: dict[str, list[SourceUtterance]] = defaultdict(list)
    for utterance in pool:
        by_speaker[utterance.speaker_id].append(utterance)

    rng = random.Random(seed)
    speaker_order = list(by_speaker)
    rng.shuffle(speaker_order)

    for speaker_id, items in by_speaker.items():
        rng.shuffle(items)
        items.sort(key=lambda utterance: (-utterance.focus_score, utterance.duration_seconds, utterance.utterance_id))

    selected: list[SourceUtterance] = []
    while len(selected) < target_count:
        progress_made = False
        for speaker_id in speaker_order:
            items = by_speaker[speaker_id]
            if not items:
                continue
            selected.append(items.pop(0))
            progress_made = True
            if len(selected) == target_count:
                break
        if not progress_made:
            break

    if len(selected) != target_count:
        raise RuntimeError(f"Balanced sampler produced {len(selected)} utterances for target {target_count}.")

    selected.sort(key=lambda utterance: utterance.utterance_id)
    return selected


def sample_source_utterances(
    utterances: list[SourceUtterance],
    target_splits: dict[str, int],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[SourceUtterance]]:
    """Sample exact source utterance counts for each split."""

    split_speakers = assign_speakers_to_splits(utterances, target_splits=target_splits, seed=seed)
    sampled: dict[str, list[SourceUtterance]] = {}

    for offset, (split_name, target_count) in enumerate(target_splits.items()):
        pool = [utterance for utterance in utterances if utterance.speaker_id in split_speakers[split_name]]
        sampled[split_name] = balanced_sample_utterances(pool, target_count, seed=seed + offset + 1)

    return sampled


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Center and peak-normalize audio."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio if needed."""

    if sample_rate == target_sample_rate:
        return audio.astype(np.float32)

    divisor = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = sample_rate // divisor
    resampled = signal.resample_poly(audio, up=up, down=down)
    return resampled.astype(np.float32)


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32."""

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return normalize_audio(resample_audio(audio, sample_rate)), TARGET_SAMPLE_RATE


def safe_bandpass(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    """Apply a bandpass filter without crossing Nyquist."""

    nyquist = sample_rate / 2.0
    low_hz = max(40.0, min(low_hz, nyquist * 0.7))
    high_hz = min(high_hz, nyquist * 0.95)

    if low_hz >= high_hz:
        return audio.astype(np.float32)

    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def choose_severity(lisp_type: str, rng: random.Random) -> int:
    """Choose severity for a lisp example."""

    if lisp_type == "clear":
        return 0
    ranges = {
        "frontal": (4, 8),
        "lateral": (5, 9),
        "dental": (3, 7),
        "palatal": (4, 8),
    }
    start, end = ranges[lisp_type]
    return rng.randint(start, end)


def choose_augmentation(lisp_type: str, severity: int, rng: random.Random) -> dict:
    """Choose a deterministic synthetic recipe description."""

    strength = severity / 10.0
    if lisp_type == "clear":
        return {
            "name": "clear_reference",
            "parameters": {"dry_mix": 1.0},
        }

    if lisp_type == "frontal":
        variant = rng.choice(("forward_leak", "breathy_th"))
        if variant == "forward_leak":
            return {
                "name": variant,
                "parameters": {
                    "presence_low": round(rng.uniform(1600, 2100), 1),
                    "presence_high": round(rng.uniform(3600, 4400), 1),
                    "breath_low": round(rng.uniform(2200, 2800), 1),
                    "breath_high": round(rng.uniform(4800, 6000), 1),
                    "presence_mix": round(0.20 + 0.18 * strength, 4),
                    "breath_mix": round(0.015 + 0.03 * strength, 4),
                    "dry_mix": round(0.84 - 0.08 * strength, 4),
                },
            }
        return {
            "name": variant,
            "parameters": {
                "presence_low": round(rng.uniform(1800, 2400), 1),
                "presence_high": round(rng.uniform(3300, 4200), 1),
                "breath_low": round(rng.uniform(2600, 3200), 1),
                "breath_high": round(rng.uniform(5200, 6800), 1),
                "presence_mix": round(0.16 + 0.12 * strength, 4),
                "breath_mix": round(0.03 + 0.035 * strength, 4),
                "dry_mix": round(0.88 - 0.10 * strength, 4),
            },
        }

    if lisp_type == "lateral":
        variant = rng.choice(("side_leak", "slushy_band"))
        if variant == "side_leak":
            return {
                "name": variant,
                "parameters": {
                    "wet_low": round(rng.uniform(2400, 3100), 1),
                    "wet_high": round(rng.uniform(5400, 6800), 1),
                    "noise_low": round(rng.uniform(2600, 3400), 1),
                    "noise_high": round(rng.uniform(5800, 7200), 1),
                    "wet_mix": round(0.18 + 0.16 * strength, 4),
                    "noise_mix": round(0.028 + 0.032 * strength, 4),
                    "dry_mix": round(0.90 - 0.08 * strength, 4),
                },
            }
        return {
            "name": variant,
            "parameters": {
                "wet_low": round(rng.uniform(2200, 2800), 1),
                "wet_high": round(rng.uniform(4800, 6200), 1),
                "noise_low": round(rng.uniform(3000, 3600), 1),
                "noise_high": round(rng.uniform(6200, 7400), 1),
                "wet_mix": round(0.22 + 0.14 * strength, 4),
                "noise_mix": round(0.02 + 0.028 * strength, 4),
                "dry_mix": round(0.88 - 0.06 * strength, 4),
            },
        }

    if lisp_type == "dental":
        variant = rng.choice(("teeth_contact", "flattened_groove"))
        if variant == "teeth_contact":
            return {
                "name": variant,
                "parameters": {
                    "dental_low": round(rng.uniform(1500, 2100), 1),
                    "dental_high": round(rng.uniform(3600, 4700), 1),
                    "band_mix": round(0.14 + 0.10 * strength, 4),
                    "dry_mix": 0.92,
                },
            }
        return {
            "name": variant,
            "parameters": {
                "dental_low": round(rng.uniform(1700, 2300), 1),
                "dental_high": round(rng.uniform(3200, 4300), 1),
                "band_mix": round(0.18 + 0.08 * strength, 4),
                "dry_mix": 0.90,
            },
        }

    if lisp_type == "palatal":
        variant = rng.choice(("backed_placement", "muffled_focus"))
        if variant == "backed_placement":
            return {
                "name": variant,
                "parameters": {
                    "muffle_low": round(rng.uniform(800, 1200), 1),
                    "muffle_high": round(rng.uniform(2400, 3200), 1),
                    "band_mix": round(0.22 + 0.12 * strength, 4),
                    "dry_mix": round(0.80 - 0.08 * strength, 4),
                },
            }
        return {
            "name": variant,
            "parameters": {
                "muffle_low": round(rng.uniform(900, 1400), 1),
                "muffle_high": round(rng.uniform(2200, 3000), 1),
                "band_mix": round(0.26 + 0.10 * strength, 4),
                "dry_mix": round(0.78 - 0.06 * strength, 4),
            },
        }

    raise ValueError(f"Unsupported lisp type: {lisp_type}")


def synthesize_lisp_audio(
    audio: np.ndarray,
    sample_rate: int,
    lisp_type: str,
    severity: int,
    augmentation: dict,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a lisp effect on audio."""

    if lisp_type == "clear":
        return normalize_audio(audio)

    params = augmentation["parameters"]
    turbulence = np_rng.normal(0.0, 1.0, len(audio)).astype(np.float32)

    if lisp_type == "frontal":
        presence = safe_bandpass(audio, sample_rate, params["presence_low"], params["presence_high"], order=3)
        breath = safe_bandpass(turbulence, sample_rate, params["breath_low"], params["breath_high"], order=2)
        result = (
            audio * params["dry_mix"]
            + presence * params["presence_mix"]
            + breath * params["breath_mix"]
        )
    elif lisp_type == "lateral":
        wet_band = safe_bandpass(audio, sample_rate, params["wet_low"], params["wet_high"], order=2)
        side_noise = safe_bandpass(turbulence, sample_rate, params["noise_low"], params["noise_high"], order=2)
        result = (
            audio * params["dry_mix"]
            + wet_band * params["wet_mix"]
            + side_noise * params["noise_mix"]
        )
    elif lisp_type == "dental":
        dental_band = safe_bandpass(audio, sample_rate, params["dental_low"], params["dental_high"], order=2)
        result = audio * params["dry_mix"] + dental_band * params["band_mix"]
    elif lisp_type == "palatal":
        muffled = safe_bandpass(audio, sample_rate, params["muffle_low"], params["muffle_high"], order=2)
        result = audio * params["dry_mix"] + muffled * params["band_mix"]
    else:
        raise ValueError(f"Unsupported lisp type: {lisp_type}")

    return normalize_audio(result)


def build_instruction(target_text: str, practice_level: str) -> str:
    """Build the user instruction for multimodal SFT."""

    return (
        "Analyze this pronunciation attempt for lisp type and give concise corrective coaching. "
        f"Expected text: '{target_text}'. "
        f"Practice level: {practice_level}. "
        "Use the audio as the primary evidence and respond with exactly four labeled lines: "
        "Detected class, Reason, Corrective cue, Encouragement. "
        "The Detected class value must be exactly one of: clear, frontal, lateral, dental, palatal. "
        "Do not use any other diagnostic category."
    )


def build_expected_feedback(
    lisp_type: str,
    target_text: str,
    practice_level: str,
    severity: int,
    rng: random.Random,
) -> str:
    """Build the expected assistant answer."""

    profile = LISP_LABELS[lisp_type]
    reason = rng.choice(profile["reasons"])
    cue = rng.choice(profile["cues"])
    encouragement = rng.choice(profile["encouragements"])

    if lisp_type == "clear":
        cue = f"{cue} Keep the same shape when you repeat '{target_text}'."
    elif practice_level == "sentences":
        cue = f"{cue} Keep that correction consistent across the full sentence."
    elif practice_level == "phrases":
        cue = f"{cue} Keep it steady across the whole phrase."

    if severity >= 8 and lisp_type != "clear":
        encouragement = "Good effort. Slow it down and reset the tongue placement before the next repetition."

    return "\n".join(
        [
            f"Detected class: {lisp_type}",
            f"Reason: {reason}",
            f"Corrective cue: {cue}",
            f"Encouragement: {encouragement}",
        ]
    )


def write_audio(path: Path, audio: np.ndarray) -> None:
    """Write mono float32 audio."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), TARGET_SAMPLE_RATE, subtype="FLOAT")


def write_preview_audio(rows: list[dict], preview_per_type: int) -> None:
    """Refresh the legacy preview folder under data/synthetic."""

    if LEGACY_SYNTHETIC_DIR.exists():
        shutil.rmtree(LEGACY_SYNTHETIC_DIR)

    written = Counter()
    for row in rows:
        lisp_type = row["lisp_type"]
        if lisp_type == "clear" or written[lisp_type] >= preview_per_type:
            continue

        source_audio = REPO_ROOT / row["audio_path"]
        target_dir = LEGACY_SYNTHETIC_DIR / lisp_type
        target_dir.mkdir(parents=True, exist_ok=True)
        preview_path = target_dir / f"{row['source_utterance_id']}_{lisp_type}.wav"
        shutil.copy2(source_audio, preview_path)
        written[lisp_type] += 1


def build_example_row(
    split_name: str,
    utterance: SourceUtterance,
    lisp_type: str,
    audio_path: Path,
    severity: int,
    augmentation: dict,
    profile_name: str,
    rng: random.Random,
) -> dict:
    """Create one manifest row."""

    practice_level = classify_practice_level(utterance.transcript)
    return {
        "id": f"{profile_name}_{split_name}_{utterance.utterance_id}_{lisp_type}",
        "profile": profile_name,
        "split": split_name,
        "subset_name": utterance.subset_name,
        "speaker_id": utterance.speaker_id,
        "chapter_id": utterance.chapter_id,
        "source_utterance_id": utterance.utterance_id,
        "source_audio_path": repo_relative(utterance.audio_path),
        "audio_path": repo_relative(audio_path),
        "target_text": utterance.transcript,
        "practice_level": practice_level,
        "focus_score": utterance.focus_score,
        "lisp_type": lisp_type,
        "severity": severity,
        "duration_seconds": round(utterance.duration_seconds, 3),
        "augmentation_recipe": augmentation["name"],
        "augmentation_parameters": augmentation["parameters"],
        "instruction": build_instruction(utterance.transcript, practice_level),
        "expected_feedback": build_expected_feedback(
            lisp_type=lisp_type,
            target_text=utterance.transcript,
            practice_level=practice_level,
            severity=severity,
            rng=rng,
        ),
    }


def summarize_available_sources(utterances: list[SourceUtterance]) -> dict:
    """Summarize the candidate raw-audio pool."""

    by_subset_counts = Counter(utterance.subset_name for utterance in utterances)
    by_subset_speakers: dict[str, set[str]] = defaultdict(set)
    for utterance in utterances:
        by_subset_speakers[utterance.subset_name].add(utterance.speaker_id)

    return {
        "available_source_utterances": len(utterances),
        "available_speakers": len({utterance.speaker_id for utterance in utterances}),
        "available_subsets": sorted(by_subset_counts),
        "subset_utterance_counts": dict(by_subset_counts),
        "subset_speaker_counts": {
            subset_name: len(speakers)
            for subset_name, speakers in sorted(by_subset_speakers.items())
        },
    }


def write_build_config(
    profile_name: str,
    target_splits: dict[str, int],
    subset_names: list[str],
    available_summary: dict,
    seed: int,
) -> dict:
    """Write build metadata so audits can validate expectations."""

    config = {
        "profile": profile_name,
        "target_splits": target_splits,
        "subset_names": subset_names,
        "seed": seed,
        "lisp_types": list(LISP_TYPES),
        "available_summary": available_summary,
    }
    BUILD_CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def load_build_config(build_config_path: Path = BUILD_CONFIG_FILE) -> dict | None:
    """Load build metadata if present."""

    if not build_config_path.exists():
        return None
    return json.loads(build_config_path.read_text(encoding="utf-8"))


def build_multimodal_dataset(
    profile_name: str = DEFAULT_PROFILE,
    seed: int = DEFAULT_SEED,
    clean: bool = True,
    subset_names: list[str] | None = None,
) -> dict:
    """Build the full raw-audio dataset and manifest."""

    profile = get_profile(profile_name)
    target_splits = profile["target_splits"]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    source_utterances = scan_source_utterances(subset_names=subset_names)
    available_summary = summarize_available_sources(source_utterances)

    if available_summary["available_speakers"] < profile["min_speakers"]:
        raise RuntimeError(
            f"Profile '{profile_name}' requires at least {profile['min_speakers']} speakers, "
            f"but only {available_summary['available_speakers']} are available. "
            f"Add more raw subsets under {LIBRISPEECH_ROOT}."
        )

    required_sources = sum(target_splits.values())
    if available_summary["available_source_utterances"] < required_sources:
        raise RuntimeError(
            f"Profile '{profile_name}' requires {required_sources} source utterances, "
            f"but only {available_summary['available_source_utterances']} are available. "
            "Add more raw subsets before rebuilding."
        )

    sampled = sample_source_utterances(source_utterances, target_splits=target_splits, seed=seed)

    if clean and MULTIMODAL_DIR.exists():
        shutil.rmtree(MULTIMODAL_DIR)

    rows: list[dict] = []
    for split_name, utterances in sampled.items():
        print(f"Building {split_name} split with {len(utterances)} source utterances...")
        for index, utterance in enumerate(utterances, start=1):
            clear_audio, sample_rate = load_audio(utterance.audio_path)

            for lisp_type in LISP_TYPES:
                severity = choose_severity(lisp_type, rng)
                augmentation = choose_augmentation(lisp_type, severity=severity, rng=rng)
                if lisp_type == "clear":
                    output_audio = clear_audio
                else:
                    output_audio = synthesize_lisp_audio(
                        audio=clear_audio,
                        sample_rate=sample_rate,
                        lisp_type=lisp_type,
                        severity=severity,
                        augmentation=augmentation,
                        np_rng=np_rng,
                    )

                output_path = (
                    TRAINING_AUDIO_DIR
                    / profile_name
                    / split_name
                    / lisp_type
                    / f"{utterance.utterance_id}_{lisp_type}.wav"
                )
                write_audio(output_path, output_audio)
                rows.append(
                    build_example_row(
                        split_name=split_name,
                        utterance=utterance,
                        lisp_type=lisp_type,
                        audio_path=output_path,
                        severity=severity,
                        augmentation=augmentation,
                        profile_name=profile_name,
                        rng=rng,
                    )
                )

            if index % 100 == 0 or index == len(utterances):
                print(f"  processed {index}/{len(utterances)} source utterances")

    write_jsonl(MANIFEST_FILE, rows)
    TEXT_OUTPUT_FILE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_preview_audio(rows, preview_per_type=profile["preview_per_type"])

    subset_names = sorted({utterance.subset_name for utterance in source_utterances})
    build_config = write_build_config(
        profile_name=profile_name,
        target_splits=target_splits,
        subset_names=subset_names,
        available_summary=available_summary,
        seed=seed,
    )

    summary = audit_manifest(manifest_path=MANIFEST_FILE, build_config=build_config)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_manifest_rows(manifest_path: Path = MANIFEST_FILE) -> list[dict]:
    """Load manifest rows from JSONL."""

    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit_manifest(
    manifest_path: Path = MANIFEST_FILE,
    build_config: dict | None = None,
) -> dict:
    """Audit manifest integrity and counts."""

    rows = load_manifest_rows(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows found in {manifest_path}")

    if build_config is None:
        build_config = load_build_config()

    target_splits = build_config["target_splits"] if build_config else None
    lisp_types = tuple(build_config["lisp_types"]) if build_config else LISP_TYPES

    split_label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    split_subset_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_counts = Counter()
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    source_ids_by_split: dict[str, set[str]] = defaultdict(set)
    missing_audio: list[str] = []

    for row in rows:
        split_name = row["split"]
        lisp_type = row["lisp_type"]
        split_label_counts[split_name][lisp_type] += 1
        split_subset_counts[split_name][row["subset_name"]] += 1
        source_counts[(split_name, row["source_utterance_id"])] += 1
        speakers_by_split[split_name].add(row["speaker_id"])
        source_ids_by_split[split_name].add(row["source_utterance_id"])
        if not (REPO_ROOT / row["audio_path"]).exists():
            missing_audio.append(row["audio_path"])

    split_totals = {split_name: sum(labels.values()) for split_name, labels in split_label_counts.items()}
    speaker_leakage = {
        "train_val": sorted(speakers_by_split["train"] & speakers_by_split["val"]),
        "train_test": sorted(speakers_by_split["train"] & speakers_by_split["test"]),
        "val_test": sorted(speakers_by_split["val"] & speakers_by_split["test"]),
    }

    bad_source_counts = {
        f"{split_name}:{source_id}": count
        for (split_name, source_id), count in source_counts.items()
        if count != len(lisp_types)
    }

    expected_row_count = None
    expected_split_totals = None
    has_expected_rows = None
    has_exact_split_totals = None
    has_exact_label_totals = None
    if target_splits:
        expected_row_count = sum(target_splits.values()) * len(lisp_types)
        expected_split_totals = {
            split_name: target_splits[split_name] * len(lisp_types)
            for split_name in target_splits
        }
        has_expected_rows = len(rows) == expected_row_count
        has_exact_split_totals = split_totals == expected_split_totals
        has_exact_label_totals = all(
            split_label_counts.get(split_name, {}).get(lisp_type, 0) == target_count
            for split_name, target_count in target_splits.items()
            for lisp_type in lisp_types
        )

    summary = {
        "manifest_path": repo_relative(manifest_path),
        "row_count": len(rows),
        "expected_row_count": expected_row_count,
        "split_totals": split_totals,
        "expected_split_totals": expected_split_totals,
        "split_label_counts": split_label_counts,
        "split_subset_counts": split_subset_counts,
        "speaker_counts": {split_name: len(speakers) for split_name, speakers in speakers_by_split.items()},
        "source_utterance_counts": {split_name: len(source_ids) for split_name, source_ids in source_ids_by_split.items()},
        "speaker_leakage": speaker_leakage,
        "bad_source_counts": bad_source_counts,
        "missing_audio": missing_audio,
        "build_config": build_config,
        "checks": {
            "has_expected_rows": has_expected_rows,
            "has_exact_split_totals": has_exact_split_totals,
            "has_exact_label_totals": has_exact_label_totals,
            "has_no_speaker_leakage": not any(speaker_leakage.values()),
            "has_no_missing_audio": not missing_audio,
            "has_complete_variant_sets": not bad_source_counts,
            "lateral_is_non_empty": split_label_counts.get("train", {}).get("lateral", 0) > 0,
        },
    }
    return summary


def main() -> None:
    """Command-line entrypoint."""

    parser = argparse.ArgumentParser(description="Build Lisper Gemma 4 audio dataset")
    parser.add_argument("--build-multimodal", action="store_true", help="Build the full multimodal dataset")
    parser.add_argument("--audit", action="store_true", help="Audit an existing manifest")
    parser.add_argument("--list-subsets", action="store_true", help="List available LibriSpeech subsets under data/raw")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE, help="Manifest path for audit")
    parser.add_argument(
        "--profile",
        choices=sorted(TARGET_PROFILES),
        default=DEFAULT_PROFILE,
        help="Dataset size profile to build",
    )
    parser.add_argument(
        "--subset",
        action="append",
        dest="subset_names",
        help="LibriSpeech subset to include, e.g. test-clean or train-clean-100. Repeat for multiple subsets.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove previous processed outputs first")
    args = parser.parse_args()

    print("=" * 60)
    print("Lisper Gemma 4 Audio Dataset")
    print("=" * 60)

    if args.list_subsets:
        subset_dirs = discover_librispeech_subset_dirs(args.subset_names)
        for subset_dir in subset_dirs:
            flac_count = sum(1 for _ in subset_dir.glob("*/*/*.flac"))
            speakers = {
                path.relative_to(subset_dir).parts[0]
                for path in subset_dir.glob("*/*/*.flac")
            }
            print(f"{subset_dir.name}: {flac_count} utterances, {len(speakers)} speakers")
    elif args.build_multimodal:
        summary = build_multimodal_dataset(
            profile_name=args.profile,
            seed=args.seed,
            clean=not args.no_clean,
            subset_names=args.subset_names,
        )
        print(json.dumps(summary, indent=2))
    elif args.audit:
        summary = audit_manifest(args.manifest)
        print(json.dumps(summary, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
