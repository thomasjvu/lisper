from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


ALLOWED_CLASSES = {"clear", "frontal", "lateral", "dental", "palatal"}


@dataclass(frozen=True)
class LiveAudioPolicy:
    min_audio_seconds: float = 0.45
    min_peak: float = 0.012
    min_rms: float = 0.0015
    min_voiced_ratio: float = 0.002
    min_speech_frame_ratio: float = 0.04
    min_tonal_frame_ratio: float = 0.04
    min_sibilant_frame_ratio: float = 0.015
    max_noise_flatness: float = 0.40
    max_clipping_ratio: float = 0.08
    clear_min_confidence: float = 0.85
    clear_min_margin: float = 0.25
    nonclear_min_confidence: float = 0.55
    nonclear_min_margin: float = 0.12


def frame_audio(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    count = 1 + (len(audio) - frame) // hop
    shape = (count, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()


def _round_float(value: float, digits: int = 6) -> float:
    if not np.isfinite(value):
        return 0.0
    return round(float(value), digits)


def _band_ratio(power: np.ndarray, freqs: np.ndarray, total: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.zeros(power.shape[0], dtype=np.float64)
    return power[:, mask].sum(axis=1) / total


def compute_live_audio_diagnostics(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    policy: LiveAudioPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or LiveAudioPolicy()
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration_seconds = float(audio.shape[0] / sample_rate) if sample_rate else 0.0
    abs_audio = np.abs(audio) if audio.size else np.asarray([], dtype=np.float32)
    peak = float(np.max(abs_audio)) if abs_audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    voiced_ratio = float(np.mean(abs_audio > policy.min_peak)) if abs_audio.size else 0.0
    clipping_ratio = float(np.mean(abs_audio >= 0.98)) if abs_audio.size else 0.0

    frame_metrics = {
        "active_frame_ratio": 0.0,
        "speech_frame_ratio": 0.0,
        "tonal_frame_ratio": 0.0,
        "sibilant_frame_ratio": 0.0,
        "mean_active_flatness": 0.0,
        "mean_active_voice_band_ratio": 0.0,
        "mean_active_sibilant_band_ratio": 0.0,
    }
    if audio.size and sample_rate > 0:
        centered = audio - float(np.mean(audio))
        frames = frame_audio(centered, sample_rate)
        window = np.hanning(frames.shape[1]).astype(np.float32)
        spectra = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float64)
        freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate).astype(np.float64)
        power = spectra**2
        eps = 1e-12
        total = power.sum(axis=1) + eps
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + eps)
        active_threshold = max(policy.min_rms * 2.0, rms * 0.35)
        active = frame_rms >= active_threshold

        voice_band = _band_ratio(power, freqs, total, 300, 3200)
        sibilant_band = _band_ratio(power, freqs, total, 3500, min(7500, sample_rate / 2))
        flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / (np.mean(power + eps, axis=1) + eps)

        speech_like = active & (voice_band >= 0.18) & (flatness <= 0.85)
        tonal = active & (voice_band >= 0.35) & (flatness <= 0.25)
        sibilant = active & (sibilant_band >= 0.08)
        if active.any():
            frame_metrics = {
                "active_frame_ratio": float(np.mean(active)),
                "speech_frame_ratio": float(np.mean(speech_like)),
                "tonal_frame_ratio": float(np.mean(tonal)),
                "sibilant_frame_ratio": float(np.mean(sibilant)),
                "mean_active_flatness": float(np.mean(flatness[active])),
                "mean_active_voice_band_ratio": float(np.mean(voice_band[active])),
                "mean_active_sibilant_band_ratio": float(np.mean(sibilant_band[active])),
            }

    diagnostics = {
        "duration_seconds": round(duration_seconds, 3),
        "sample_count": int(audio.shape[0]),
        "peak": _round_float(peak),
        "rms": _round_float(rms),
        "voiced_ratio": _round_float(voiced_ratio),
        "clipping_ratio": _round_float(clipping_ratio),
        **{key: _round_float(value) for key, value in frame_metrics.items()},
    }
    diagnostics.update(
        {
            "min_audio_seconds": policy.min_audio_seconds,
            "min_peak": policy.min_peak,
            "min_rms": policy.min_rms,
            "min_voiced_ratio": policy.min_voiced_ratio,
            "min_speech_frame_ratio": policy.min_speech_frame_ratio,
            "min_tonal_frame_ratio": policy.min_tonal_frame_ratio,
            "min_sibilant_frame_ratio": policy.min_sibilant_frame_ratio,
            "max_noise_flatness": policy.max_noise_flatness,
            "max_clipping_ratio": policy.max_clipping_ratio,
        }
    )
    return diagnostics


def validate_live_audio_diagnostics(diagnostics: dict[str, Any], policy: LiveAudioPolicy | None = None) -> dict[str, Any]:
    policy = policy or LiveAudioPolicy()
    reason = ""
    if diagnostics["sample_count"] <= 0:
        reason = "No audio samples were recorded."
    elif diagnostics["duration_seconds"] < policy.min_audio_seconds:
        reason = "The recording is too short to analyze."
    elif diagnostics["peak"] < policy.min_peak or diagnostics["rms"] < policy.min_rms:
        reason = "The recording is too quiet or silent to analyze."
    elif diagnostics["voiced_ratio"] < policy.min_voiced_ratio:
        reason = "No usable speech-like audio was detected."
    elif diagnostics["clipping_ratio"] > policy.max_clipping_ratio:
        reason = "The recording is clipped or distorted. Try again farther from the microphone."
    elif (
        diagnostics["mean_active_flatness"] > policy.max_noise_flatness
        and diagnostics["tonal_frame_ratio"] < policy.min_tonal_frame_ratio
    ):
        reason = "The recording looks more like broadband noise than speech."
    elif (
        diagnostics["speech_frame_ratio"] < policy.min_speech_frame_ratio
        and diagnostics["tonal_frame_ratio"] < policy.min_tonal_frame_ratio
    ):
        reason = "No usable speech-like audio was detected."

    if not reason:
        return {"status": "accepted", "reason": "audio_preflight_passed", "thresholds": asdict(policy)}
    return {"status": "rejected_audio", "reason": reason, "thresholds": asdict(policy)}


def _normalized_scores(class_scores: Any) -> dict[str, float]:
    if not isinstance(class_scores, dict):
        return {}
    cleaned = {
        str(label): max(0.0, float(score))
        for label, score in class_scores.items()
        if str(label) in ALLOWED_CLASSES
    }
    total = sum(cleaned.values())
    if total <= 0:
        return cleaned
    if total > 1.2 or total < 0.8:
        return {label: score / total for label, score in cleaned.items()}
    return cleaned


def ranked_class_scores(acoustic_result: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not acoustic_result:
        return []
    scores = _normalized_scores(acoustic_result.get("class_scores"))
    if not scores:
        detected = str(acoustic_result.get("detected_class") or "")
        if detected in ALLOWED_CLASSES:
            scores[detected] = float(acoustic_result.get("confidence") or 0.0)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def decide_live_analysis(
    acoustic_result: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    policy: LiveAudioPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or LiveAudioPolicy()
    preflight = validate_live_audio_diagnostics(diagnostics, policy)
    if preflight["status"] != "accepted":
        return {
            "status": "rejected_audio",
            "detected_class": "inconclusive",
            "decision_reason": preflight["reason"],
            "thresholds": asdict(policy),
            "audio_diagnostics": diagnostics,
        }

    ranked = ranked_class_scores(acoustic_result)
    classifier_summary = {
        "available": acoustic_result is not None,
        "model_name": acoustic_result.get("model_name") if acoustic_result else None,
        "raw_class": acoustic_result.get("raw_class") if acoustic_result else None,
        "reported_class": acoustic_result.get("detected_class") if acoustic_result else None,
        "class_scores": {label: round(score, 6) for label, score in ranked},
    }
    if not acoustic_result or not ranked:
        return {
            "status": "error",
            "detected_class": "inconclusive",
            "decision_reason": "The acoustic model artifact is unavailable, so live analysis is disabled.",
            "thresholds": asdict(policy),
            "audio_diagnostics": diagnostics,
            "classifier": classifier_summary,
        }

    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    classifier_summary.update(
        {
            "top_class": top_label,
            "top_score": round(top_score, 6),
            "second_score": round(second_score, 6),
            "margin": round(margin, 6),
        }
    )

    if top_label == "clear":
        if top_score >= policy.clear_min_confidence and margin >= policy.clear_min_margin:
            status = "detected"
            reason = "clear_confidence_gate_passed"
        else:
            status = "inconclusive"
            reason = "The classifier was not confident enough to call this clear."
    elif diagnostics["sibilant_frame_ratio"] < policy.min_sibilant_frame_ratio:
        status = "inconclusive"
        reason = "The clip has speech energy, but not enough usable /s/ or /z/ airflow evidence."
    elif top_score >= policy.nonclear_min_confidence and margin >= policy.nonclear_min_margin:
        status = "detected"
        reason = "nonclear_confidence_gate_passed"
    else:
        status = "inconclusive"
        reason = "The classifier was not confident enough to label this lisp pattern."

    return {
        "status": status,
        "detected_class": top_label if status == "detected" else "inconclusive",
        "candidate_class": top_label,
        "decision_reason": reason,
        "thresholds": asdict(policy),
        "audio_diagnostics": diagnostics,
        "classifier": classifier_summary,
    }
