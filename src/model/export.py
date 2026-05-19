#!/usr/bin/env python3
"""Export and quantization handoff helpers for Lisper training artifacts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
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
DEFAULT_EXPORT_ROOT = REPO_ROOT / "data" / "processed" / "gemma4_audio" / "artifacts" / "exports"

QUANTIZATION_TARGETS = [
    {
        "name": "q4f16",
        "bits": 4,
        "purpose": "Primary ONNX/WebGPU browser target matching the public Gemma 4 E2B reference artifact.",
        "expected_memory": "expected browser download is about 3.4 GB for E2B q4f16, before cache and runtime overhead.",
        "release_priority": 1,
    },
    {
        "name": "q8",
        "bits": 8,
        "purpose": "Conservative local baseline when quality matters more than file size.",
        "expected_memory": "roughly half of 16-bit, plus audio processor and KV cache overhead",
        "release_priority": 4,
    },
    {
        "name": "q5",
        "bits": 5,
        "purpose": "Quality-preserving consumer build for laptops with more memory.",
        "expected_memory": "usually below q8 while preserving more behavior than q4",
        "release_priority": 3,
    },
    {
        "name": "q4",
        "bits": 4,
        "purpose": "Smaller ONNX/WebGPU fallback only after q4f16 works and eval passes.",
        "expected_memory": "Unsloth reports Gemma 4 E2B around 5 GB RAM in 4-bit before app overhead.",
        "release_priority": 2,
    },
    {
        "name": "q3",
        "bits": 3,
        "purpose": "Emergency small build only if q4 cannot fit.",
        "expected_memory": "smaller than q4 but higher risk for class/cue regressions",
        "release_priority": 4,
    },
    {
        "name": "q2",
        "bits": 2,
        "purpose": "Smallest experiment; not a default release target.",
        "expected_memory": "smallest practical experimental tier, high quality risk",
        "release_priority": 5,
    },
]

WEBGPU_SIZE_ESTIMATE = {
    "source": "onnx-community/gemma-4-E2B-it-ONNX dry-run, q4/q4f16 files only",
    "base_model": "google/gemma-4-E2B-it",
    "merged_checkpoint_bytes": 10_246_621_886,
    "q4_onnx_gb_estimate": 4.0,
    "q4f16_onnx_gb_estimate": 3.4,
    "q4_and_q4f16_combined_gb": 7.3,
    "browser_download_note": "The browser should request only one dtype variant, not q4 and q4f16 together.",
}


def validate_safetensors(path: Path) -> dict:
    if not path.exists():
        return {"valid": False, "reason": "missing", "path": str(path), "size_bytes": 0}
    size = path.stat().st_size
    if size < 16:
        return {"valid": False, "reason": "too_small", "path": str(path), "size_bytes": size}

    try:
        with path.open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
    except Exception as error:
        return {
            "valid": False,
            "reason": f"invalid_safetensors_header: {error}",
            "path": str(path),
            "size_bytes": size,
        }

    max_offset = 0
    tensor_count = 0
    for tensor in header.values():
        if not isinstance(tensor, dict):
            continue
        offsets = tensor.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            tensor_count += 1
            max_offset = max(max_offset, int(offsets[1]))

    expected_size = 8 + header_length + max_offset
    return {
        "valid": size >= expected_size,
        "reason": "ok" if size >= expected_size else "incomplete",
        "path": str(path),
        "size_bytes": size,
        "expected_min_size_bytes": expected_size,
        "tensor_count": tensor_count,
    }


def write_web_notes(model_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_path = output_dir / "web-demo-notes.txt"
    notes_path.write_text(
        "\n".join(
            [
                "Lisper web demo notes",
                "=====================",
                "",
                "The current hackathon web demo loads the public ONNX Gemma 4 E2B browser model",
                "(onnx-community/gemma-4-E2B-it-ONNX) through @huggingface/transformers.",
                "",
                "Fine-tuned Lisper outputs are Python/Hugging Face artifacts until a separate",
                "quantized browser/mobile export is produced and evaluated.",
                "",
                f"Requested model path: {model_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(notes_path)}, indent=2))


def build_quantization_readme(plan: dict) -> str:
    target_lines = [
        f"- `{target['name']}`: {target['purpose']} {target['expected_memory']}."
        for target in sorted(plan["targets"], key=lambda item: item["release_priority"])
    ]
    return f"""# Lisper Quantization Plan

Source merged model:
`{plan['source_model_dir']}`

Source validation:
```json
{json.dumps(plan['source_validation'], indent=2)}
```

## Recommendation

Use `q4f16` as the first consumer-device release target because it matches the public Gemma 4 E2B WebGPU reference used by the app today. Use `q4` only if the smaller variant is needed and it passes held-out evaluation. Only test `q3` or `q2` after q4f16 passes, and do not promote them without held-out evaluation.

For the app, the preferred deployable quantized artifact is ONNX/WebGPU, not GGUF. GGUF is useful for native local/server experiments, but it is not the browser runtime target.

The public reference browser repo is `onnx-community/gemma-4-E2B-it-ONNX`, which already contains Gemma 4 audio, vision, embedding, and decoder ONNX files with `q4` / `q4f16` variants. The Lisper export should mirror that repo shape after converting the v16 merged checkpoint.

Current size estimates from the public E2B ONNX reference:

- merged v16 checkpoint: about `10.25 GB` (`9.54 GiB`)
- trained q4f16 ONNX/WebGPU target: about `3.4 GB`
- trained q4 ONNX/WebGPU fallback: about `4.0 GB`
- q4 plus q4f16 together: about `7.3 GB`

The app should request a single dtype variant, not all variants in the repo.

## Targets

{chr(10).join(target_lines)}

## Release Rules

- Keep the raw merged 16-bit checkpoint as the canonical full-model artifact.
- Publish browser quantized builds as ONNX/WebGPU repos or clearly labeled variants.
- Run held-out eval on every quantized tier.
- Treat audio support as runtime-specific; do not assume a text-only quantized runtime can serve the Lisper raw-audio path.
- Treat GGUF as optional native-local packaging, not the default web app deliverable.

## Candidate Repo Names

- `thomasjvu/lisper-gemma4-e2b-audio-full`
- `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`
- `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4`

## App Switch

After a trained q4f16 ONNX/WebGPU repo exists and passes eval:

```bash
cd lisper-app
VITE_LISPER_BROWSER_MODEL_ID=thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16 \
VITE_LISPER_BROWSER_DTYPE=q4f16 \
npm run web
```

## Local Export Helper

No paid Hugging Face Job is required by default. Use the local helper:

```bash
python src/model/export_onnx_webgpu.py --mode probe
python src/model/export_onnx_webgpu.py --mode size-estimate
python src/model/export_onnx_webgpu.py --mode copy-metadata
```

The full ONNX export still depends on a Gemma 4-compatible ONNX exporter. If released Optimum ONNX cannot load Transformers `5.x`, do not promote a partial artifact; keep the merged checkpoint as canonical and treat the trained browser q4f16 artifact as blocked on the exporter dependency.
"""


def write_quantization_plan(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_safetensors(model_dir / "model.safetensors")
    plan = {
        "source_model_dir": str(model_dir),
        "source_validation": validation,
        "recommended_release_target": "q4f16",
        "primary_browser_target": "onnx-webgpu-q4f16",
        "webgpu_size_estimate": WEBGPU_SIZE_ESTIMATE,
        "native_local_optional_target": "gguf-q4",
        "quality_preserving_target": "q5",
        "smallest_experimental_target": "q2",
        "do_not_promote_without_eval": ["q3", "q2"],
        "targets": QUANTIZATION_TARGETS,
        "notes": [
            "Quantize only after the merged base+LoRA checkpoint is complete.",
            "Keep the raw merged 16-bit checkpoint as the canonical full-model artifact.",
            "For the web app, prefer ONNX/WebGPU quantized builds over GGUF.",
            "Publish quantized builds as separate repos or clearly labeled variants.",
            "Run the held-out eval on each quantized tier before using it in the app.",
            "For raw-audio Gemma 4, verify the runtime supports the audio pathway before treating an ONNX or GGUF export as deployable.",
            "Treat GGUF as optional native-local packaging, not the browser deployment target.",
        ],
    }
    (output_dir / "quantization-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(build_quantization_readme(plan), encoding="utf-8")
    print(json.dumps({"wrote": str(output_dir), "source_valid": validation["valid"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and quantization handoff for Lisper model artifacts")
    parser.add_argument(
        "--format",
        choices=("quantization-plan", "validate-merged", "web-notes"),
        default="quantization-plan",
        help="Export helper to run",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MERGED_DIR, help="Merged model directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT_ROOT / "quantization", help="Output directory")
    args = parser.parse_args()

    if args.format == "validate-merged":
        print(json.dumps(validate_safetensors(args.model / "model.safetensors"), indent=2))
    elif args.format == "web-notes":
        write_web_notes(args.model, args.output)
    else:
        write_quantization_plan(args.model, args.output)


if __name__ == "__main__":
    main()
