from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import tempfile
import traceback
from typing import Any
from pathlib import Path

# Unsloth's compiled Gemma 4 audio path can trip TorchDynamo on ZeroGPU's
# runtime torch build. Keep inference eager for reliability.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import spaces
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from live_audio_policy import (
    LiveAudioPolicy,
    compute_live_audio_diagnostics,
    decide_live_analysis,
    validate_live_audio_diagnostics,
)


ALLOWED_CLASSES = {"clear", "frontal", "lateral", "dental", "palatal"}
DEFAULT_MODEL_ID = "thomasjvu/lisper-gemma4-e2b-audio-full"
DEFAULT_ADAPTER_ID = ""
SPACE_ROOT = Path(__file__).resolve().parent
ACOUSTIC_MODEL_PATH = SPACE_ROOT / "acoustic_model.json"
ACOUSTIC_EXTRATREES_MODEL_PATH = SPACE_ROOT / "acoustic_extratrees_v18.joblib"
ACOUSTIC_K = 5
ACOUSTIC_MIN_CONFIDENCE = 0.42
KNN_OVERRIDE_MAX_DISTANCE = 0.25
KNN_OVERRIDE_MIN_CONFIDENCE = 0.90
LIVE_CLEAR_MIN_CONFIDENCE = 0.85
LIVE_CLEAR_MIN_MARGIN = 0.25
LIVE_NONCLEAR_MIN_CONFIDENCE = 0.55
LIVE_NONCLEAR_MIN_MARGIN = 0.12
MIN_AUDIO_SECONDS = 0.45
MIN_AUDIO_RMS = 0.0015
MIN_AUDIO_PEAK = 0.012
MIN_VOICED_RATIO = 0.002
MIN_SPEECH_FRAME_RATIO = 0.04
MIN_TONAL_FRAME_RATIO = 0.04
MIN_SIBILANT_FRAME_RATIO = 0.015
MAX_NOISE_FLATNESS = 0.40
MAX_CLIPPING_RATIO = 0.08
DEFAULT_PROMPT = """Analyze this pronunciation attempt for lisp type and give concise corrective coaching.

Return exactly four labeled lines in this order:
Detected class: clear|frontal|lateral|dental|palatal
Reason: one brief reason tied to tongue placement or airflow
Corrective cue: one concrete next-step cue
Encouragement: one brief supportive line"""

CLASS_TEMPLATES = {
    "clear": {
        "reason": "The acoustic pattern did not strongly match the trained lisp-pattern examples, so this is treated as a tentative clear result.",
        "cue": "Repeat once at a relaxed pace and keep the airflow centered through the front of the mouth.",
    },
    "dental": {
        "reason": "The acoustic pattern is closest to the dental examples, where tongue contact near the teeth can narrow the /s/ groove.",
        "cue": "Relax the tongue slightly off the teeth and keep a narrow stream of air moving forward.",
    },
    "frontal": {
        "reason": "The acoustic pattern is closest to the frontal examples, which often sound like the airflow is too far forward.",
        "cue": "Keep the tongue tip just behind the upper teeth and send the air straight forward through a small groove.",
    },
    "lateral": {
        "reason": "The acoustic pattern is closest to the lateral examples, where air may be escaping around the sides of the tongue.",
        "cue": "Start from a light /t/ position, seal the tongue sides, and let the air move forward through the center.",
    },
    "palatal": {
        "reason": "The acoustic pattern is closest to the palatal examples, where the tongue can sit too far back and muffle the sound.",
        "cue": "Bring the tongue tip slightly forward behind the upper teeth and brighten the airflow.",
    },
}

GUARDED_CLASS_TEMPLATES = {
    "dental": {
        "reason": "The acoustic model was not confident enough to call this clear; the nearest non-clear pattern is dental, which can happen when the tongue presses too close to the teeth.",
        "cue": "Try one slower repetition with the tongue relaxed just behind the teeth and the air moving forward through a narrow center groove.",
    },
    "frontal": {
        "reason": "The acoustic model was not confident enough to call this clear; the nearest non-clear pattern is frontal, which can happen when the tongue or airflow moves too far forward.",
        "cue": "Keep the tongue tip behind the upper teeth and avoid letting it push between the teeth during /s/ sounds.",
    },
    "lateral": {
        "reason": "The acoustic model was not confident enough to call this clear; the nearest non-clear pattern is lateral, where air may be leaking around the tongue sides.",
        "cue": "Seal the tongue sides lightly against the upper molars and aim the air straight down the middle.",
    },
    "palatal": {
        "reason": "The acoustic model was not confident enough to call this clear; the nearest non-clear pattern is palatal, where the tongue may be sitting too far back.",
        "cue": "Bring the tongue tip forward just behind the upper teeth and brighten the /s/ airflow.",
    },
}

BROWSER_RECORDER_START_JS = r"""
async (payload) => {
  const state = window.__lisperRecorder || {};
  if (state.recording) {
    return [payload || "", "Recording is already active.", ""];
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return [payload || "", "This browser cannot access microphone recording.", ""];
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: true,
    },
    video: false,
  });

  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextCtor({ sampleRate: 16000 });
  await audioContext.resume();

  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;

  const chunks = [];
  let peak = 0;
  let sumSquares = 0;
  let sampleCount = 0;

  processor.onaudioprocess = (event) => {
    if (!window.__lisperRecorder?.recording) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input.length);
    copy.set(input);
    chunks.push(copy);
    for (let i = 0; i < copy.length; i += 1) {
      const value = copy[i];
      const absValue = Math.abs(value);
      if (absValue > peak) peak = absValue;
      sumSquares += value * value;
    }
    sampleCount += copy.length;
  };

  source.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(audioContext.destination);

  window.__lisperRecorder = {
    recording: true,
    stream,
    audioContext,
    source,
    processor,
    silentGain,
    chunks,
    startedAt: Date.now(),
    getStats: () => ({
      peak,
      rms: sampleCount ? Math.sqrt(sumSquares / sampleCount) : 0,
      sampleCount,
      sampleRate: audioContext.sampleRate,
    }),
  };

  return ["", "Recording through Web Audio... press Stop when finished.", ""];
}
"""

BROWSER_RECORDER_STOP_JS = r"""
async () => {
  const state = window.__lisperRecorder;
  if (!state || !state.recording) {
    return ["", "No active browser recording. Press Record first.", ""];
  }

  state.recording = false;
  try { state.processor.disconnect(); } catch (_) {}
  try { state.source.disconnect(); } catch (_) {}
  try { state.silentGain.disconnect(); } catch (_) {}
  for (const track of state.stream.getTracks()) {
    track.stop();
  }

  const stats = state.getStats();
  const sampleRate = stats.sampleRate || 16000;
  const totalLength = state.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const samples = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of state.chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  await state.audioContext.close().catch(() => undefined);
  window.__lisperRecorder = null;

  function writeString(view, byteOffset, string) {
    for (let i = 0; i < string.length; i += 1) {
      view.setUint8(byteOffset + i, string.charCodeAt(i));
    }
  }

  function encodeWav(floatSamples, wavSampleRate) {
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;
    const buffer = new ArrayBuffer(44 + floatSamples.length * bytesPerSample);
    const view = new DataView(buffer);
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + floatSamples.length * bytesPerSample, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, wavSampleRate, true);
    view.setUint32(28, wavSampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, floatSamples.length * bytesPerSample, true);
    let byteOffset = 44;
    for (let i = 0; i < floatSamples.length; i += 1, byteOffset += 2) {
      const clamped = Math.max(-1, Math.min(1, floatSamples[i]));
      view.setInt16(byteOffset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    }
    return new Blob([view], { type: "audio/wav" });
  }

  const blob = encodeWav(samples, sampleRate);
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });

  const durationSeconds = sampleRate ? samples.length / sampleRate : 0;
  const payload = JSON.stringify({
    source: "browser-web-audio-wav",
    data_url: dataUrl,
    mime_type: "audio/wav",
    sample_rate: sampleRate,
    sample_count: samples.length,
    duration_seconds: Number(durationSeconds.toFixed(3)),
    peak: Number(stats.peak.toFixed(6)),
    rms: Number(stats.rms.toFixed(6)),
    created_at: new Date().toISOString(),
  });

  const status = stats.peak < 0.003
    ? `Clip captured but appears very quiet. peak=${stats.peak.toFixed(6)} rms=${stats.rms.toFixed(6)}. Check browser microphone permission/input.`
    : `Clip ready: ${durationSeconds.toFixed(1)}s, peak=${stats.peak.toFixed(3)}. Playback should contain your voice.`;
  const playback = `<audio controls src="${dataUrl}" style="width:100%;"></audio>`;

  return [payload, status, playback];
}
"""

BROWSER_RECORDER_CLEAR_JS = r"""
async () => {
  const state = window.__lisperRecorder;
  if (state?.recording) {
    state.recording = false;
    try { state.processor.disconnect(); } catch (_) {}
    try { state.source.disconnect(); } catch (_) {}
    try { state.silentGain.disconnect(); } catch (_) {}
    for (const track of state.stream.getTracks()) {
      track.stop();
    }
    await state.audioContext.close().catch(() => undefined);
  }
  window.__lisperRecorder = null;
  return ["", "No browser recording ready.", ""];
}
"""


class InvalidAudioError(ValueError):
    """Raised when a clip is too short or too quiet to analyze honestly."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def model_id() -> str:
    return os.environ.get("LISPER_ZERO_GPU_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def adapter_id() -> str:
    return os.environ.get("LISPER_ZERO_GPU_ADAPTER_ID", DEFAULT_ADAPTER_ID).strip()


def max_new_tokens() -> int:
    return env_int("LISPER_ZERO_GPU_MAX_NEW_TOKENS", 96)


def max_seq_length() -> int:
    return env_int("LISPER_ZERO_GPU_MAX_SEQ_LENGTH", 2048)


def zero_gpu_size() -> str:
    requested = os.environ.get("LISPER_ZERO_GPU_SIZE", "large").strip().lower()
    return "xlarge" if requested == "xlarge" else "large"


def eager_load_enabled() -> bool:
    return os.environ.get("LISPER_ZERO_GPU_EAGER_LOAD", "0").strip() != "0"


def load_in_4bit_enabled() -> bool:
    default = "1" if adapter_id() else "0"
    return os.environ.get("LISPER_ZERO_GPU_LOAD_IN_4BIT", default).strip() != "0"


def acoustic_hint_enabled() -> bool:
    return os.environ.get("LISPER_ZERO_GPU_ACOUSTIC_HINT", "1").strip() != "0"


def acoustic_model_preference() -> str:
    requested = os.environ.get("LISPER_ZERO_GPU_ACOUSTIC_MODEL", "auto").strip().lower()
    if requested in {"extratrees", "knn"}:
        return requested
    return "auto"


def live_clear_min_confidence() -> float:
    return env_float("LISPER_ZERO_GPU_LIVE_CLEAR_MIN_CONFIDENCE", LIVE_CLEAR_MIN_CONFIDENCE)


def live_clear_min_margin() -> float:
    return env_float("LISPER_ZERO_GPU_LIVE_CLEAR_MIN_MARGIN", LIVE_CLEAR_MIN_MARGIN)


def live_nonclear_min_confidence() -> float:
    return env_float(
        "LISPER_ZERO_GPU_LIVE_NONCLEAR_MIN_CONFIDENCE",
        env_float("LISPER_ZERO_GPU_LIVE_NONCLEAR_MIN_SCORE", LIVE_NONCLEAR_MIN_CONFIDENCE),
    )


def live_nonclear_min_margin() -> float:
    return env_float("LISPER_ZERO_GPU_LIVE_NONCLEAR_MIN_MARGIN", LIVE_NONCLEAR_MIN_MARGIN)


def knn_override_max_distance() -> float:
    return env_float("LISPER_ZERO_GPU_KNN_OVERRIDE_MAX_DISTANCE", KNN_OVERRIDE_MAX_DISTANCE)


def knn_override_min_confidence() -> float:
    return env_float("LISPER_ZERO_GPU_KNN_OVERRIDE_MIN_CONFIDENCE", KNN_OVERRIDE_MIN_CONFIDENCE)


def live_audio_policy() -> LiveAudioPolicy:
    return LiveAudioPolicy(
        min_audio_seconds=env_float("LISPER_ZERO_GPU_MIN_AUDIO_SECONDS", MIN_AUDIO_SECONDS),
        min_peak=env_float("LISPER_ZERO_GPU_MIN_AUDIO_PEAK", MIN_AUDIO_PEAK),
        min_rms=env_float("LISPER_ZERO_GPU_MIN_AUDIO_RMS", MIN_AUDIO_RMS),
        min_voiced_ratio=env_float("LISPER_ZERO_GPU_MIN_VOICED_RATIO", MIN_VOICED_RATIO),
        min_speech_frame_ratio=env_float("LISPER_ZERO_GPU_MIN_SPEECH_FRAME_RATIO", MIN_SPEECH_FRAME_RATIO),
        min_tonal_frame_ratio=env_float("LISPER_ZERO_GPU_MIN_TONAL_FRAME_RATIO", MIN_TONAL_FRAME_RATIO),
        min_sibilant_frame_ratio=env_float("LISPER_ZERO_GPU_MIN_SIBILANT_FRAME_RATIO", MIN_SIBILANT_FRAME_RATIO),
        max_noise_flatness=env_float("LISPER_ZERO_GPU_MAX_NOISE_FLATNESS", MAX_NOISE_FLATNESS),
        max_clipping_ratio=env_float("LISPER_ZERO_GPU_MAX_CLIPPING_RATIO", MAX_CLIPPING_RATIO),
        clear_min_confidence=live_clear_min_confidence(),
        clear_min_margin=live_clear_min_margin(),
        nonclear_min_confidence=live_nonclear_min_confidence(),
        nonclear_min_margin=live_nonclear_min_margin(),
    )


def audio_alignment_enabled() -> bool:
    default = "0" if adapter_id() else "1"
    return os.environ.get("LISPER_ZERO_GPU_ALIGN_AUDIO_TOKENS", default).strip() != "0"


def gemma_generation_enabled() -> bool:
    return os.environ.get("LISPER_ZERO_GPU_USE_GEMMA_GENERATION", "0").strip() != "0"


def torch_dtype() -> torch.dtype:
    requested = os.environ.get("LISPER_ZERO_GPU_DTYPE", "float16").strip().lower()
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    return torch.float16


def auth_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    return token or None


def decode_browser_audio_payload(payload: str) -> np.ndarray:
    try:
        parsed = json.loads(payload)
        data_url = str(parsed.get("data_url") or "")
        if "," not in data_url:
            raise ValueError("Browser recorder payload is missing audio data.")
        _, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as exc:
        raise InvalidAudioError(
            f"Browser recording could not be decoded: {type(exc).__name__}: {exc}",
            {"status": "invalid_browser_audio_payload"},
        ) from exc

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    return waveform.astype(np.float32)


def normalize_audio(audio_value: str | tuple[int, np.ndarray] | None) -> np.ndarray:
    if audio_value is None:
        raise gr.Error("Record or upload a short audio clip first.")

    if isinstance(audio_value, str):
        if audio_value.strip().startswith("{"):
            waveform = decode_browser_audio_payload(audio_value)
            sample_rate = 16000
        else:
            waveform, sample_rate = sf.read(audio_value, dtype="float32", always_2d=False)
    else:
        sample_rate, waveform = audio_value
        waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    # Keep ZeroGPU requests bounded.
    max_samples = 12 * 16000
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = waveform / peak

    return waveform.astype(np.float32)


def audio_diagnostics(waveform: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
    return compute_live_audio_diagnostics(waveform, sample_rate=sample_rate, policy=live_audio_policy())


def validate_audio_for_analysis(waveform: np.ndarray) -> dict[str, Any]:
    diagnostics = audio_diagnostics(waveform)
    decision = validate_live_audio_diagnostics(diagnostics, live_audio_policy())
    if decision["status"] != "accepted":
        diagnostics["live_audio_gate"] = decision
        raise InvalidAudioError(str(decision["reason"]), diagnostics)
    return diagnostics


def write_temp_audio(waveform: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(prefix="lisper-zero-gpu-", suffix=".wav")
    os.close(fd)
    sf.write(path, waveform, 16000)
    return path


def build_messages(target_text: str, audio_url: str, acoustic_result: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    instruction = DEFAULT_PROMPT
    if target_text.strip():
        instruction += f'\n\nTarget text: "{target_text.strip()}"'
    if acoustic_result:
        instruction += (
            "\n\nAcoustic pre-analysis from the waveform: "
            f"class={acoustic_result['detected_class']}, "
            f"confidence={acoustic_result['confidence']:.3f}. "
            "Use this exact class for the Detected class line. Do not override it."
        )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Lisper, a supportive speech-therapy assistant focused on concise lisp coaching.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": audio_url},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def build_runtime() -> tuple[Any, Any]:
    repo_id = model_id()
    adapter_repo_id = adapter_id()
    token = auth_token()
    if adapter_repo_id:
        import unsloth  # noqa: F401
        from unsloth import FastVisionModel

        kwargs = {
            "model_name": adapter_repo_id,
            "max_seq_length": max_seq_length(),
            "load_in_4bit": load_in_4bit_enabled(),
            "full_finetuning": False,
        }
        if token:
            kwargs["token"] = token
        model, processor = FastVisionModel.from_pretrained(**kwargs)
        FastVisionModel.for_inference(model)
        model.eval()
        return processor, model

    processor_source = adapter_repo_id or repo_id
    processor = AutoProcessor.from_pretrained(processor_source, token=token, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        repo_id,
        token=token,
        torch_dtype=torch_dtype(),
        device_map={"": "cuda"},
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


RUNTIME: tuple[Any, Any] | None = build_runtime() if eager_load_enabled() else None
LAST_INPUT_SUMMARY: dict[str, Any] = {}


def load_runtime() -> tuple[Any, Any]:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = build_runtime()
    return RUNTIME


def strip_generation_artifacts(text: str) -> str:
    return text.replace("```", "").replace("<bos>", "").strip()


def extract_line(label: str, text: str) -> str:
    match = re.search(rf"^{label}:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def normalize_label(text: str) -> str:
    value = text.strip().lower()
    if "inconclusive" in value or "unclear" in value:
        return "inconclusive"
    for candidate in ALLOWED_CLASSES:
        if candidate in value:
            return candidate
    return "inconclusive"


def parse_response(response: str) -> dict[str, Any]:
    detected = normalize_label(extract_line("Detected class", response))
    return {
        "detected_class": detected,
        "reason": extract_line("Reason", response),
        "corrective_cue": extract_line("Corrective cue", response),
        "encouragement": extract_line("Encouragement", response),
        "raw_response": response,
        "model_id": model_id(),
        "adapter_id": adapter_id() or None,
    }


def acoustic_normalize_audio(audio: np.ndarray) -> np.ndarray:
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


def extract_acoustic_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(88, dtype=np.float32)

    audio = acoustic_normalize_audio(audio)
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


ACOUSTIC_MODEL: dict[str, Any] | None = None
ACOUSTIC_EXTRATREES_MODEL: dict[str, Any] | None = None


def load_acoustic_model() -> dict[str, Any] | None:
    global ACOUSTIC_MODEL
    if not acoustic_hint_enabled():
        return None
    if ACOUSTIC_MODEL is None:
        if not ACOUSTIC_MODEL_PATH.exists():
            return None
        ACOUSTIC_MODEL = json.loads(ACOUSTIC_MODEL_PATH.read_text(encoding="utf-8"))
    return ACOUSTIC_MODEL


def load_acoustic_extratrees_model() -> dict[str, Any] | None:
    global ACOUSTIC_EXTRATREES_MODEL
    if not acoustic_hint_enabled():
        return None
    if not ACOUSTIC_EXTRATREES_MODEL_PATH.exists():
        return None
    if ACOUSTIC_EXTRATREES_MODEL is None:
        import joblib

        ACOUSTIC_EXTRATREES_MODEL = joblib.load(ACOUSTIC_EXTRATREES_MODEL_PATH)
    return ACOUSTIC_EXTRATREES_MODEL


def classify_acoustic_extratrees(waveform: np.ndarray) -> dict[str, Any] | None:
    model = load_acoustic_extratrees_model()
    if model is None:
        return None

    features = extract_acoustic_features(waveform, sr=int(model.get("sample_rate", 16000))).reshape(1, -1)
    classifier = model["classifier"]
    prediction = str(classifier.predict(features)[0])
    confidence = 1.0
    class_scores: dict[str, float] = {}
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        classes = [str(label) for label in classifier.classes_]
        class_scores = {
            label: round(float(probability), 6)
            for label, probability in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        }
        confidence = float(class_scores.get(prediction, 0.0))

    return {
        "detected_class": prediction,
        "raw_class": prediction,
        "confidence": confidence,
        "class_scores": class_scores,
        "model_name": model.get("name", "lisper_v18_extratrees_acoustic_hint"),
        "train_rows": model.get("train_rows"),
        "feature_count": model.get("feature_count"),
        "holdout_accuracy": model.get("holdout_accuracy"),
        "low_confidence_defaulted_to_clear": False,
    }


def apply_live_clear_guard(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is not None:
        result["live_clear_guard_applied"] = False
    return result


def classify_acoustic_knn(waveform: np.ndarray) -> dict[str, Any] | None:
    model = load_acoustic_model()
    if model is None:
        return None

    features = extract_acoustic_features(waveform, sr=int(model.get("sample_rate", 16000)))
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    normalized = (features - mean) / np.where(std < 1e-6, 1.0, std)

    distances = []
    for exemplar in model["exemplars"]:
        exemplar_features = np.asarray(exemplar["features"], dtype=np.float32)
        distance = float(np.linalg.norm(normalized - exemplar_features))
        distances.append((distance, exemplar["label"], exemplar.get("source_id", "")))
    distances.sort(key=lambda item: item[0])

    class_scores: dict[str, float] = {label: 0.0 for label in model["classes"]}
    for distance, label, _source_id in distances[:ACOUSTIC_K]:
        class_scores[label] += 1.0 / max(distance, 1e-4)
    ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    total_score = sum(class_scores.values()) or 1.0
    confidence = float(top_score / total_score)
    detected_class = top_label if confidence >= ACOUSTIC_MIN_CONFIDENCE else "clear"

    return {
        "detected_class": detected_class,
        "raw_class": top_label,
        "confidence": confidence,
        "nearest_distance": round(distances[0][0], 4),
        "nearest_source_id": distances[0][2],
        "class_scores": {label: round(float(score), 6) for label, score in ranked},
        "model_name": model.get("name"),
        "low_confidence_defaulted_to_clear": detected_class == "clear" and top_label != "clear",
    }


def _compact_acoustic_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    keys = (
        "detected_class",
        "raw_class",
        "confidence",
        "nearest_distance",
        "nearest_source_id",
        "class_scores",
        "model_name",
        "low_confidence_defaulted_to_clear",
    )
    return {key: result[key] for key in keys if key in result}


def maybe_apply_knn_override(
    extratrees_result: dict[str, Any] | None,
    knn_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if extratrees_result is None or knn_result is None:
        return extratrees_result

    knn_label = normalize_label(str(knn_result.get("raw_class") or knn_result.get("detected_class") or ""))
    if knn_label == "clear":
        return extratrees_result

    confidence = float(knn_result.get("confidence") or 0.0)
    nearest_distance = float(knn_result.get("nearest_distance") or math.inf)
    max_distance = knn_override_max_distance()
    min_confidence = knn_override_min_confidence()
    if confidence < min_confidence or nearest_distance > max_distance:
        return {
            **extratrees_result,
            "hybrid_override_applied": False,
            "hybrid_override_reason": "knn_not_close_enough",
            "hybrid_override_thresholds": {
                "max_distance": max_distance,
                "min_confidence": min_confidence,
            },
            "knn_result": _compact_acoustic_result(knn_result),
        }

    class_scores = {
        label: float(score)
        for label, score in (knn_result.get("class_scores") or {}).items()
        if normalize_label(str(label)) in ALLOWED_CLASSES
    }
    return {
        **extratrees_result,
        "detected_class": knn_label,
        "raw_class": knn_label,
        "confidence": confidence,
        "class_scores": class_scores,
        "model_name": "lisper_hybrid_extratrees_knn_synthetic_override",
        "low_confidence_defaulted_to_clear": False,
        "hybrid_override_applied": True,
        "hybrid_override_reason": "knn_close_synthetic_exemplar",
        "hybrid_override_thresholds": {
            "max_distance": max_distance,
            "min_confidence": min_confidence,
        },
        "extratrees_result": _compact_acoustic_result(extratrees_result),
        "knn_result": _compact_acoustic_result(knn_result),
    }


def classify_acoustic(waveform: np.ndarray) -> dict[str, Any] | None:
    preference = acoustic_model_preference()
    if preference == "extratrees":
        return classify_acoustic_extratrees(waveform)
    if preference == "knn":
        return classify_acoustic_knn(waveform)

    extratrees_result = classify_acoustic_extratrees(waveform)
    knn_result = classify_acoustic_knn(waveform)
    if extratrees_result is not None:
        return maybe_apply_knn_override(extratrees_result, knn_result)
    return knn_result


def enforce_acoustic_response(response: str, acoustic_result: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    parsed = parse_response(response)
    if not acoustic_result:
        return response, parsed

    detected_class = normalize_label(str(acoustic_result["detected_class"]))
    if acoustic_result.get("live_clear_guard_applied"):
        template = GUARDED_CLASS_TEMPLATES.get(detected_class, CLASS_TEMPLATES[detected_class])
    else:
        template = CLASS_TEMPLATES[detected_class]
    encouragement = parsed.get("encouragement") or "Good effort. One focused repetition is enough for the next try."
    final_response = "\n".join(
        [
            f"Detected class: {detected_class}",
            f"Reason: {template['reason']}",
            f"Corrective cue: {template['cue']}",
            f"Encouragement: {encouragement}",
        ]
    )
    final_parsed = parse_response(final_response)
    final_parsed["raw_model_response"] = response
    final_parsed["acoustic_hint_enforced"] = True
    return final_response, final_parsed


def build_inconclusive_response(
    decision: dict[str, Any],
    acoustic_result: dict[str, Any] | None,
    clip_diagnostics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    reason = str(decision.get("decision_reason") or "The clip was not reliable enough to classify.")
    if decision.get("status") == "error":
        response = "Analysis unavailable. The acoustic model is not loaded, so Lisper will not guess a class."
    else:
        response = "\n".join(
            [
                "Detected class: inconclusive",
                f"Reason: {reason}",
                "Corrective cue: Record one clear phrase with /s/ or /z/ sounds, close to the microphone, then try again.",
                "Encouragement: The clip was captured; we just need a cleaner attempt before giving a label.",
            ]
        )
    parsed = parse_response(response)
    parsed["status"] = str(decision.get("status") or "inconclusive")
    parsed["raw_model_response"] = None
    parsed["acoustic_hint_enforced"] = False
    parsed["audio_diagnostics"] = clip_diagnostics
    parsed["acoustic_analysis"] = acoustic_result
    parsed["live_audio_gate"] = decision
    return response, parsed


def build_detected_acoustic_response(
    acoustic_result: dict[str, Any],
    decision: dict[str, Any],
    clip_diagnostics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    detected_class = normalize_label(str(decision.get("detected_class") or acoustic_result.get("detected_class") or ""))
    if detected_class not in CLASS_TEMPLATES:
        detected_class = "inconclusive"
    if detected_class == "inconclusive":
        return build_inconclusive_response(
            {
                **decision,
                "status": "inconclusive",
                "decision_reason": "The live gate did not produce a valid class label.",
            },
            acoustic_result,
            clip_diagnostics,
        )

    template = CLASS_TEMPLATES[detected_class]
    response = "\n".join(
        [
            f"Detected class: {detected_class}",
            f"Reason: {template['reason']}",
            f"Corrective cue: {template['cue']}",
            "Encouragement: Nice work getting a usable recording. Try one focused repetition next.",
        ]
    )
    parsed = parse_response(response)
    parsed["status"] = "detected"
    parsed["raw_model_response"] = None
    parsed["acoustic_hint_enforced"] = True
    parsed["gemma_generation_skipped"] = True
    parsed["audio_diagnostics"] = clip_diagnostics
    parsed["acoustic_analysis"] = acoustic_result
    parsed["live_audio_gate"] = decision
    return response, parsed


def build_audio_only_inconclusive_decision(clip_diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    policy = live_audio_policy()
    if clip_diagnostics["sibilant_frame_ratio"] >= policy.min_sibilant_frame_ratio:
        return None
    return {
        "status": "inconclusive",
        "detected_class": "inconclusive",
        "candidate_class": None,
        "decision_reason": "The clip has speech energy, but not enough usable /s/ or /z/ airflow evidence.",
        "thresholds": {
            "min_audio_seconds": policy.min_audio_seconds,
            "min_peak": policy.min_peak,
            "min_rms": policy.min_rms,
            "min_voiced_ratio": policy.min_voiced_ratio,
            "min_speech_frame_ratio": policy.min_speech_frame_ratio,
            "min_tonal_frame_ratio": policy.min_tonal_frame_ratio,
            "min_sibilant_frame_ratio": policy.min_sibilant_frame_ratio,
            "max_noise_flatness": policy.max_noise_flatness,
            "max_clipping_ratio": policy.max_clipping_ratio,
            "clear_min_confidence": policy.clear_min_confidence,
            "clear_min_margin": policy.clear_min_margin,
            "nonclear_min_confidence": policy.nonclear_min_confidence,
            "nonclear_min_margin": policy.nonclear_min_margin,
        },
        "audio_diagnostics": clip_diagnostics,
        "classifier": {"available": False, "skipped": "insufficient_sibilant_evidence"},
    }


def audio_token_id(processor: Any) -> int | None:
    value = getattr(processor, "audio_token_id", None)
    if value is not None:
        return int(value)
    tokenizer = getattr(processor, "tokenizer", None)
    token = getattr(processor, "audio_token", None) or getattr(tokenizer, "audio_token", None)
    if tokenizer is not None and token is not None:
        return int(tokenizer.convert_tokens_to_ids(token))
    return None


def replace_audio_token_run(input_ids: torch.Tensor, token_id: int, count: int) -> tuple[torch.Tensor, dict[str, int]]:
    positions = (input_ids == token_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return input_ids, {"original_audio_tokens": 0, "aligned_audio_tokens": count}

    start = int(positions[0].item())
    end = start
    while end < input_ids.shape[0] and int(input_ids[end].item()) == token_id:
        end += 1

    replacement = torch.full((count,), token_id, dtype=input_ids.dtype, device=input_ids.device)
    aligned = torch.cat([input_ids[:start], replacement, input_ids[end:]], dim=0)
    return aligned, {"original_audio_tokens": end - start, "aligned_audio_tokens": count}


def model_inference_dtype(model: Any) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    base_model = getattr(model, "base_model", None)
    dtype = getattr(base_model, "dtype", None)
    return dtype or torch_dtype()


def module_parameter_dtype(module: Any) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return model_inference_dtype(module)


def audio_input_dtype(model: Any) -> torch.dtype:
    requested = os.environ.get("LISPER_ZERO_GPU_AUDIO_DTYPE", "").strip().lower()
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16" or adapter_id():
        return torch.bfloat16
    return module_parameter_dtype(audio_feature_module(model))


def audio_feature_module(model: Any) -> Any:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "get_audio_features"):
            return candidate
    raise AttributeError("Could not locate Gemma audio feature module on loaded model.")


def summarize_inputs(inputs: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            summary[key] = {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "device": str(getattr(value, "device", "")),
            }
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def align_audio_placeholders(inputs: Any, processor: Any, model: Any) -> tuple[Any, dict[str, int]]:
    if not audio_alignment_enabled():
        return inputs, {"audio_alignment_skipped": 1}

    if "input_features" not in inputs or "input_features_mask" not in inputs:
        return inputs, {}

    token_id = audio_token_id(processor)
    if token_id is None:
        return inputs, {}

    with torch.inference_mode():
        audio_output = audio_feature_module(model).get_audio_features(
            inputs["input_features"],
            inputs["input_features_mask"],
            return_dict=True,
        )

    encoded_count = int(audio_output.attention_mask.sum().item())
    if encoded_count <= 0 or inputs["input_ids"].shape[0] != 1:
        return inputs, {"encoded_audio_tokens": encoded_count}

    new_input_ids, metadata = replace_audio_token_run(inputs["input_ids"][0], token_id, encoded_count)
    metadata["encoded_audio_tokens"] = encoded_count
    if metadata["original_audio_tokens"] == encoded_count:
        return inputs, metadata

    inputs["input_ids"] = new_input_ids.unsqueeze(0)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    if "mm_token_type_ids" in inputs and hasattr(processor, "create_mm_token_type_ids"):
        mm_token_type_ids = processor.create_mm_token_type_ids(inputs["input_ids"].detach().cpu())
        inputs["mm_token_type_ids"] = torch.as_tensor(
            mm_token_type_ids,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )

    return inputs, metadata


def _analyze_impl(audio: str | tuple[int, np.ndarray] | None, target_text: str) -> tuple[str, str]:
    global LAST_INPUT_SUMMARY

    waveform = normalize_audio(audio)
    clip_diagnostics = validate_audio_for_analysis(waveform)
    audio_only_decision = build_audio_only_inconclusive_decision(clip_diagnostics)
    if audio_only_decision is not None:
        response, parsed = build_inconclusive_response(audio_only_decision, None, clip_diagnostics)
        return response, json.dumps(parsed, indent=2)

    acoustic_result = classify_acoustic(waveform)
    live_decision = decide_live_analysis(acoustic_result, clip_diagnostics, live_audio_policy())
    if live_decision["status"] != "detected":
        response, parsed = build_inconclusive_response(live_decision, acoustic_result, clip_diagnostics)
        return response, json.dumps(parsed, indent=2)

    if not gemma_generation_enabled():
        response, parsed = build_detected_acoustic_response(acoustic_result, live_decision, clip_diagnostics)
        return response, json.dumps(parsed, indent=2)

    audio_url = write_temp_audio(waveform)
    processor, model = load_runtime()
    messages = build_messages(target_text, audio_url, acoustic_result)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    for float_key in ("input_features", "pixel_values", "pixel_values_videos"):
        if float_key in inputs and hasattr(inputs[float_key], "to"):
            inputs[float_key] = inputs[float_key].to(dtype=model_inference_dtype(model))
    if "input_features" in inputs:
        audio_dtype = audio_input_dtype(model)
        inputs["input_features"] = inputs["input_features"].to(dtype=audio_dtype)
        if "input_features_mask" in inputs and hasattr(inputs["input_features_mask"], "to"):
            inputs["input_features_mask"] = inputs["input_features_mask"].to(device=device)
    inputs, alignment = align_audio_placeholders(inputs, processor, model)
    LAST_INPUT_SUMMARY = summarize_inputs(inputs)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens(),
            do_sample=False,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    decoded = processor.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    raw_response = strip_generation_artifacts(decoded)
    response, parsed = enforce_acoustic_response(raw_response, acoustic_result)
    parsed["status"] = "detected"
    parsed["audio_token_alignment"] = alignment
    parsed["audio_diagnostics"] = clip_diagnostics
    parsed["acoustic_analysis"] = acoustic_result
    parsed["live_audio_gate"] = live_decision
    return response, json.dumps(parsed, indent=2)


def analyze_with_errors(audio: str | tuple[int, np.ndarray] | None, target_text: str) -> tuple[str, str]:
    try:
        return _analyze_impl(audio, target_text)
    except InvalidAudioError as exc:
        payload = {
            "status": "rejected_audio",
            "reason": str(exc),
            "audio_diagnostics": exc.diagnostics,
        }
        return (
            "Recording not usable yet. Please record a clear speech clip before analysis.",
            json.dumps(payload, indent=2),
        )
    except Exception as exc:
        payload = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "model_id": model_id(),
            "adapter_id": adapter_id() or None,
            "dtype": os.environ.get("LISPER_ZERO_GPU_DTYPE", "float16"),
            "load_in_4bit": load_in_4bit_enabled(),
            "acoustic_hint_enabled": acoustic_hint_enabled(),
            "audio_alignment_enabled": audio_alignment_enabled(),
            "zero_gpu_size": zero_gpu_size(),
            "input_summary": LAST_INPUT_SUMMARY,
            "traceback": traceback.format_exc(limit=8),
        }
        return f"ZeroGPU inference failed: {type(exc).__name__}: {exc}", json.dumps(payload, indent=2)


def analyze(audio: str | tuple[int, np.ndarray] | None, target_text: str) -> tuple[str, str]:
    return analyze_with_errors(audio, target_text)


@spaces.GPU(duration=5, size=zero_gpu_size())
def zero_gpu_healthcheck() -> str:
    return "ok"


def analysis_started(browser_recording_payload: str, uploaded_audio: str | tuple[int, np.ndarray] | None) -> tuple[str, str]:
    if not browser_recording_payload.strip() and uploaded_audio is None:
        return (
            "No clip ready yet. Use the browser recorder or upload a short speech clip first.",
            json.dumps({"status": "waiting_for_audio"}, indent=2),
        )
    return (
        "Checking the recording quality and acoustic evidence...",
        json.dumps({"status": "running", "stage": "audio_preflight_then_acoustic_gate"}, indent=2),
    )


def analyze_ui(
    browser_recording_payload: str,
    uploaded_audio: str | tuple[int, np.ndarray] | None,
    target_text: str,
) -> tuple[str, str]:
    selected_audio: str | tuple[int, np.ndarray] | None = (
        browser_recording_payload.strip() if browser_recording_payload.strip() else uploaded_audio
    )
    if selected_audio is None:
        return (
            "No clip ready yet. Use the browser recorder or upload a short speech clip first.",
            json.dumps({"status": "waiting_for_audio"}, indent=2),
        )
    return analyze_with_errors(selected_audio, target_text)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Lisper ZeroGPU", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Lisper ZeroGPU

            Server-side Gemma 4 audio analysis for users whose browser cannot comfortably run the WebGPU model.

            The currently validated fine-tuned Lisper model is Gemma 4 E2B. E4B and 31B are future model targets and should be deployed as separate revisions after training/eval.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Browser recorder")
                browser_recording_payload = gr.Textbox(
                    label="Browser recorder payload",
                    visible=False,
                    elem_id="lisper-browser-recorder-payload",
                )
                browser_recording_status = gr.Markdown(
                    "No browser recording ready. This path bypasses Gradio's microphone recorder."
                )
                browser_recording_playback = gr.HTML("")
                with gr.Row():
                    browser_record = gr.Button("Record", variant="primary")
                    browser_stop = gr.Button("Stop")
                    browser_clear = gr.Button("Clear")
                gr.Markdown("### Upload fallback")
                audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Speech clip upload",
                    editable=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                )
                audio_status = gr.Markdown(
                    "No uploaded clip ready. Use the browser recorder above, or upload an audio file here."
                )
                target_text = gr.Textbox(
                    label="Expected text",
                    placeholder="Example: Sally sells seashells.",
                    lines=2,
                )
                run = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                output = gr.Textbox(label="Gemma response", lines=8)
                parsed = gr.Code(label="Parsed JSON", language="json")

        gr.Markdown(
            f"""
            **Configured model:** `{model_id()}`

            **Configured adapter:** `{adapter_id() or "none"}`

            **Adapter 4-bit load:** `{load_in_4bit_enabled()}`

            **Acoustic hint:** `{acoustic_hint_enabled()}`

            **Audio token alignment:** `{audio_alignment_enabled()}`

            **ZeroGPU size:** `{zero_gpu_size()}`

            If this Space errors on private or gated models, add `HF_TOKEN` as a Space secret. For local development without downloading the model, set `LISPER_ZERO_GPU_EAGER_LOAD=0`.
            """
        )
        browser_record.click(
            None,
            inputs=[browser_recording_payload],
            outputs=[browser_recording_payload, browser_recording_status, browser_recording_playback],
            js=BROWSER_RECORDER_START_JS,
            queue=False,
            show_progress="hidden",
        )
        browser_stop.click(
            None,
            inputs=[],
            outputs=[browser_recording_payload, browser_recording_status, browser_recording_playback],
            js=BROWSER_RECORDER_STOP_JS,
            queue=False,
            show_progress="hidden",
        )
        browser_clear.click(
            None,
            inputs=[],
            outputs=[browser_recording_payload, browser_recording_status, browser_recording_playback],
            js=BROWSER_RECORDER_CLEAR_JS,
            queue=False,
            show_progress="hidden",
        )
        audio.change(
            lambda: "Uploaded clip ready. Analyze is available.",
            inputs=[],
            outputs=[audio_status],
            queue=False,
            show_progress="hidden",
        )
        audio.clear(
            lambda: "No uploaded clip ready. Use the browser recorder above, or upload an audio file here.",
            inputs=[],
            outputs=[audio_status],
            queue=False,
            show_progress="hidden",
        )
        run.click(
            analysis_started,
            inputs=[browser_recording_payload, audio],
            outputs=[output, parsed],
            queue=False,
        ).then(
            analyze_ui,
            inputs=[browser_recording_payload, audio, target_text],
            outputs=[output, parsed],
            api_name="analyze",
        )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(show_error=True)
