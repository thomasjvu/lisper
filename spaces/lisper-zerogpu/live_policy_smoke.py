from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from live_audio_policy import (
    LiveAudioPolicy,
    compute_live_audio_diagnostics,
    decide_live_analysis,
    validate_live_audio_diagnostics,
)


SR = 16000
POLICY = LiveAudioPolicy()


def fake_scores(**scores: float) -> dict:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    return {
        "detected_class": top_label,
        "raw_class": top_label,
        "confidence": top_score,
        "class_scores": scores,
        "model_name": "smoke_fake_classifier",
    }


def load_audio(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SR:
        raise RuntimeError(f"{path} has sample rate {sample_rate}, expected {SR}")
    return waveform.astype(np.float32)


def assert_status(name: str, waveform: np.ndarray, acoustic_result: dict | None, expected: set[str]) -> dict:
    diagnostics = compute_live_audio_diagnostics(waveform, SR, POLICY)
    preflight = validate_live_audio_diagnostics(diagnostics, POLICY)
    decision = (
        {"status": "rejected_audio", "detected_class": "inconclusive", "decision_reason": preflight["reason"]}
        if preflight["status"] != "accepted"
        else decide_live_analysis(acoustic_result, diagnostics, POLICY)
    )
    status = str(decision["status"])
    if status not in expected:
        raise AssertionError(
            f"{name}: expected status in {sorted(expected)}, got {status}; "
            f"decision={decision}; diagnostics={diagnostics}"
        )
    if name in {"silence", "low_noise", "broadband_noise"} and decision.get("detected_class") in {
        "clear",
        "frontal",
        "lateral",
        "dental",
        "palatal",
    }:
        raise AssertionError(f"{name}: unsafe false class returned: {decision}")
    return {
        "name": name,
        "status": status,
        "detected_class": decision.get("detected_class"),
        "decision_reason": decision.get("decision_reason"),
        "diagnostics": {
            key: diagnostics[key]
            for key in (
                "duration_seconds",
                "peak",
                "rms",
                "speech_frame_ratio",
                "tonal_frame_ratio",
                "sibilant_frame_ratio",
                "mean_active_flatness",
            )
        },
    }


def main() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(SR * 3, dtype=np.float32) / SR
    cases: list[tuple[str, np.ndarray, dict | None, set[str]]] = [
        ("silence", np.zeros(SR * 3, dtype=np.float32), fake_scores(palatal=0.92, clear=0.04), {"rejected_audio"}),
        (
            "low_noise",
            rng.normal(0.0, 0.0008, SR * 3).astype(np.float32),
            fake_scores(palatal=0.92, clear=0.04),
            {"rejected_audio"},
        ),
        (
            "broadband_noise",
            rng.normal(0.0, 0.01, SR * 3).astype(np.float32),
            fake_scores(palatal=0.92, clear=0.04),
            {"rejected_audio", "inconclusive"},
        ),
        (
            "non_sibilant_tone",
            (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
            fake_scores(lateral=0.88, clear=0.05),
            {"inconclusive"},
        ),
        (
            "uncertain_clear",
            load_audio(Path("/private/tmp/lisper-v18-dental-smoke.wav"))
            if Path("/private/tmp/lisper-v18-dental-smoke.wav").exists()
            else (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
            fake_scores(clear=0.73, palatal=0.25, dental=0.02),
            {"inconclusive"},
        ),
    ]

    fixture = Path("/private/tmp/lisper-v18-dental-smoke.wav")
    if fixture.exists():
        cases.append(
            (
                "known_dental_fixture",
                load_audio(fixture),
                fake_scores(dental=0.82, clear=0.08, palatal=0.05, lateral=0.03, frontal=0.02),
                {"detected"},
            )
        )

    results = [assert_status(*case) for case in cases]
    print(json.dumps({"ok": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
