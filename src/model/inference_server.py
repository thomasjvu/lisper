#!/usr/bin/env python3
"""Serve the trained Lisper Gemma 4 adapter to the web app."""

from __future__ import annotations

import io
import json
import os
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path("/Users/area/repos/lisper")
DEFAULT_ADAPTER_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "full_run_v16"
    / "lisper-gemma4-audio"
    / "adapter"
)
DEFAULT_MERGED_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "full_run_v16"
    / "lisper-gemma4-audio"
    / "merged_model"
)
DEFAULT_PROMPT = (
    "Analyze this pronunciation attempt for lisp type and give concise corrective coaching.\n\n"
    "Return exactly four labeled lines in this order:\n"
    "Detected class: clear|frontal|lateral|dental|palatal\n"
    "Reason: one brief reason tied to tongue placement or airflow\n"
    "Corrective cue: one concrete next-step cue\n"
    "Encouragement: one brief supportive line"
)
ALLOWED_CLASSES = {"clear", "frontal", "lateral", "dental", "palatal"}

app = FastAPI(title="Lisper Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _artifact_kind() -> str:
    requested = os.environ.get("LISPER_MODEL_ARTIFACT_KIND", "auto").strip().lower() or "auto"
    if requested in {"adapter", "merged"}:
        return requested
    return "auto"


def _adapter_dir() -> Path:
    return Path(os.environ.get("LISPER_ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR))).expanduser()


def _merged_dir() -> Path:
    return Path(os.environ.get("LISPER_MERGED_MODEL_DIR", str(DEFAULT_MERGED_DIR))).expanduser()


def _merged_model_path() -> Path:
    return _merged_dir() / "model.safetensors"


def _validate_safetensors_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "model.safetensors is missing"
    if path.stat().st_size < 16:
        return False, "model.safetensors is too small"

    try:
        with path.open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
    except Exception as error:
        return False, f"could not read safetensors header: {error}"

    max_offset = 0
    for tensor in header.values():
        if not isinstance(tensor, dict):
            continue
        offsets = tensor.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            max_offset = max(max_offset, int(offsets[1]))

    expected_size = 8 + header_length + max_offset
    actual_size = path.stat().st_size
    if actual_size < expected_size:
        return False, f"model.safetensors is incomplete: {actual_size} bytes, expected at least {expected_size}"
    return True, "ok"


def _merged_model_valid() -> bool:
    valid, _reason = _validate_safetensors_file(_merged_model_path())
    return valid


def _merged_model_status() -> dict[str, Any]:
    path = _merged_model_path()
    valid, reason = _validate_safetensors_file(path)
    return {
        "path": str(path),
        "valid": valid,
        "reason": reason,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _load_base_model_name() -> str:
    adapter_config_path = _adapter_dir() / "adapter_config.json"
    if adapter_config_path.exists():
        payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        value = payload.get("base_model_name_or_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return os.environ.get("LISPER_BASE_MODEL", "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit")


def _max_seq_length() -> int:
    return int(os.environ.get("LISPER_MAX_SEQ_LENGTH", "2048"))


def _max_new_tokens() -> int:
    return int(os.environ.get("LISPER_MAX_NEW_TOKENS", "96"))


def _requested_device() -> str:
    return os.environ.get("LISPER_DEVICE", "auto").strip().lower() or "auto"


def _load_waveform(upload: UploadFile) -> np.ndarray:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    try:
        waveform, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as error:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {error}") from error

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        try:
            import librosa
        except ImportError as error:  # pragma: no cover - runtime guard
            raise HTTPException(status_code=500, detail="librosa is required to resample audio to 16 kHz.") from error
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    return np.asarray(waveform, dtype=np.float32)


def _strip_generation_artifacts(text: str) -> str:
    return text.replace("```", "").replace("<bos>", "").strip()


def _extract_line(label: str, text: str) -> str:
    match = re.search(rf"^{label}:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _normalize_label(text: str) -> str:
    value = text.strip().lower()
    for candidate in ALLOWED_CLASSES:
        if candidate in value:
            return candidate
    return "frontal"


def _severity_from_response(label: str, reason: str) -> int:
    if label == "clear":
        return 2

    lowered = f"{label} {reason}".lower()
    if any(token in lowered for token in ("severe", "consistent", "strong", "heavy")):
        return 7
    if any(token in lowered for token in ("mild", "slight", "small", "subtle")):
        return 4
    return 5 if label in {"frontal", "dental", "palatal"} else 6


def _resolve_runtime_device(torch: Any) -> str:
    requested = _requested_device()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _resolve_artifact_kind(device: str) -> str:
    requested = _artifact_kind()
    if requested in {"adapter", "merged"}:
        return requested
    if device != "cuda" and _merged_model_valid():
        return "merged"
    return "adapter"


def _resolve_torch_dtype(torch: Any, device: str):
    requested = os.environ.get("LISPER_TORCH_DTYPE", "").strip().lower()
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def _build_messages(target_text: str) -> list[dict[str, Any]]:
    instruction = DEFAULT_PROMPT
    if target_text.strip():
        instruction += f'\n\nTarget text: "{target_text.strip()}"'

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
                {"type": "audio"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


@lru_cache(maxsize=1)
def _load_runtime() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    except ImportError as error:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Missing inference dependencies. Install fastapi, uvicorn, torch, transformers, peft, unsloth, soundfile, and librosa."
        ) from error

    device = _resolve_runtime_device(torch)
    artifact_kind = _resolve_artifact_kind(device)
    base_model_name = _load_base_model_name()
    torch_dtype = _resolve_torch_dtype(torch, device)

    if artifact_kind == "merged":
        merged_dir = _merged_dir()
        if not _merged_model_valid():
            raise RuntimeError(
                f"Merged model requested but invalid at {merged_dir}. model.safetensors is missing or empty."
            )
        model = Gemma4ForConditionalGeneration.from_pretrained(
            str(merged_dir),
            torch_dtype=torch_dtype,
        )
        processor = AutoProcessor.from_pretrained(str(merged_dir), trust_remote_code=True)
        backend = "transformers-merged"
    else:
        if device != "cuda":
            raise RuntimeError(
                "The LoRA adapter runtime currently requires CUDA + Unsloth. "
                "On Apple Silicon or CPU, use the merged checkpoint by setting "
                "LISPER_MODEL_ARTIFACT_KIND=merged or leaving it on auto once the merged model is repaired."
            )
        try:
            from peft import PeftModel
            import unsloth  # noqa: F401
            from unsloth import FastVisionModel
        except ImportError as error:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "Adapter inference requires peft and unsloth in addition to torch/transformers."
            ) from error
        model, processor = FastVisionModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=_max_seq_length(),
            load_in_4bit=True,
            full_finetuning=False,
        )
        model = PeftModel.from_pretrained(model, str(_adapter_dir()), is_trainable=False)
        processor = AutoProcessor.from_pretrained(str(_adapter_dir()), trust_remote_code=True)
        FastVisionModel.for_inference(model)
        backend = "unsloth-adapter"

    model.eval()

    if hasattr(model, "to"):
        model.to(device)

    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "artifact_kind": artifact_kind,
        "base_model": base_model_name,
        "device": device,
        "backend": backend,
    }


def _generate_response(waveform: np.ndarray, target_text: str) -> str:
    runtime = _load_runtime()
    torch = runtime["torch"]
    model = runtime["model"]
    processor = runtime["processor"]
    device = runtime["device"]
    messages = _build_messages(target_text)

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    inputs = processor(
        text=prompt,
        audio=[waveform],
        sampling_rate=16000,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=_max_new_tokens(),
            do_sample=False,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    decoded = processor.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    return _strip_generation_artifacts(decoded)


def _response_to_payload(response_text: str, target_text: str) -> dict[str, Any]:
    detected_class = _normalize_label(_extract_line("Detected class", response_text))
    reason = _extract_line("Reason", response_text) or "The response did not provide a clear reason, so this is a fallback summary."
    cue = _extract_line("Corrective cue", response_text) or "Try one slower repetition and keep the airflow centered."
    encouragement = _extract_line("Encouragement", response_text) or "Good try. One calmer repetition is enough."
    severity = _severity_from_response(detected_class, reason)

    return {
        "transcript": target_text.strip(),
        "assessment": {
            "lispType": detected_class if detected_class != "clear" else "frontal",
            "severity": severity,
            "notes": reason,
            "mouthShapeNotes": "Remote inference is audio-only in this build, so no frame-based mouth-shape analysis is available.",
            "confidence": 0.72,
            "sampledFrameCount": 0,
        },
        "coaching": {
            "feedback": reason,
            "encouragement": encouragement,
            "nextTryCue": cue,
        },
        "raw_response": response_text,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "loaded": _load_runtime.cache_info().currsize > 0,
        "adapter_dir": str(_adapter_dir()),
        "base_model": _load_base_model_name(),
        "model_artifact_kind": _artifact_kind(),
        "requested_device": _requested_device(),
        "merged_model_path": str(_merged_model_path()),
        "merged_model_valid": _merged_model_valid(),
        "merged_model_status": _merged_model_status(),
    }


@app.post("/warm")
def warm() -> dict[str, Any]:
    try:
        runtime = _load_runtime()
    except Exception as error:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "status": "ok",
        "loaded": True,
        "adapter_dir": str(_adapter_dir()),
        "base_model": runtime["base_model"],
        "model_artifact_kind": runtime["artifact_kind"],
        "device": runtime["device"],
        "backend": runtime["backend"],
        "merged_model_path": str(_merged_model_path()),
        "merged_model_valid": _merged_model_valid(),
        "merged_model_status": _merged_model_status(),
    }


@app.post("/analyze")
def analyze(
    audio: UploadFile = File(...),
    target_text: str = Form(""),
    duration_ms: str = Form("0"),
    frame_count: str = Form("0"),
) -> dict[str, Any]:
    del duration_ms, frame_count

    try:
        waveform = _load_waveform(audio)
        response_text = _generate_response(waveform, target_text)
        return _response_to_payload(response_text, target_text)
    except HTTPException:
        raise
    except Exception as error:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.model.inference_server:app",
        host=os.environ.get("LISPER_HOST", "127.0.0.1"),
        port=int(os.environ.get("LISPER_PORT", "8000")),
        reload=False,
    )
