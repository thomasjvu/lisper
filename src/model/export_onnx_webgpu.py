#!/usr/bin/env python3
"""Local ONNX/WebGPU export helpers for the Lisper Gemma 4 E2B checkpoint.

This script is intentionally local-only. It does not launch Hugging Face Jobs.

The expected browser layout mirrors the public reference repo:
`onnx-community/gemma-4-E2B-it-ONNX`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/area/repos/lisper")
DEFAULT_MODEL_ID = "thomasjvu/lisper-gemma4-e2b-audio-full"
DEFAULT_LOCAL_MODEL = (
    REPO_ROOT
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "full_run_v16"
    / "lisper-gemma4-audio"
    / "merged_model"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "exports"
    / "onnx-webgpu"
)

REFERENCE_SIZE_ESTIMATES = {
    "source": "onnx-community/gemma-4-E2B-it-ONNX dry-run, q4/q4f16 files only",
    "base_model": "google/gemma-4-E2B-it",
    "merged_checkpoint_bytes": 10_246_621_886,
    "merged_checkpoint_gb": 10.25,
    "merged_checkpoint_gib": 9.54,
    "q4_onnx_bytes_estimate": 4_000_000_000,
    "q4_onnx_gb_estimate": 4.0,
    "q4f16_onnx_bytes_estimate": 3_400_000_000,
    "q4f16_onnx_gb_estimate": 3.4,
    "q4_and_q4f16_combined_bytes": 7_300_000_000,
    "q4_and_q4f16_combined_gb": 7.3,
    "notes": [
        "The trained Lisper export should be the same order of size because it is the same E2B architecture.",
        "Browser first-load downloads only the requested dtype files plus tokenizer/config/processor metadata.",
        "q4 and q4f16 are separate browser variants; do not make the app download both.",
    ],
}

METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
)

QUANTIZATION_MODES = {
    "q2": {"bits": 2, "float16": False},
    "q2f16": {"bits": 2, "float16": True},
    "q4": {"bits": 4, "float16": False},
    "q4f16": {"bits": 4, "float16": True},
}

TRANSFORMERS_JS_EXTERNAL_DATA_CONFIG = {
    "use_external_data_format": {
        "audio_encoder": 1,
        "vision_encoder": 1,
        "decoder_model_merged_q2.onnx": 1,
        "decoder_model_merged_q2f16.onnx": 1,
        "decoder_model_merged_q4.onnx": 1,
        "decoder_model_merged_q4f16.onnx": 1,
        "embed_tokens_q2.onnx": 1,
        "embed_tokens_q2f16.onnx": 1,
        "embed_tokens_q4.onnx": 1,
        "embed_tokens_q4f16.onnx": 1,
    },
    "kv_cache_dtype": {
        "q2f16": "float16",
        "q4f16": "float16",
        "fp16": "float16",
    },
}


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def probe_environment() -> dict:
    status = {
        "python": sys.version.split()[0],
        "has_transformers": module_available("transformers"),
        "has_torch": module_available("torch"),
        "has_optimum": module_available("optimum"),
        "has_onnx": module_available("onnx"),
        "has_onnxruntime": module_available("onnxruntime"),
        "has_onnxruntime_q4_legacy": module_available("onnxruntime.quantization.matmul_4bits_quantizer"),
        "has_onnxruntime_q4_nbits": module_available("onnxruntime.quantization.matmul_nbits_quantizer"),
    }
    if status["has_transformers"]:
        import transformers

        status["transformers_version"] = getattr(transformers, "__version__", "unknown")
        status["has_gemma4_class"] = hasattr(transformers, "Gemma4ForConditionalGeneration")
    if status["has_torch"]:
        import torch

        status["torch_version"] = getattr(torch, "__version__", "unknown")
        status["torch_cuda_available"] = bool(torch.cuda.is_available())

    missing = [
        name
        for name, present in {
            "transformers": status["has_transformers"],
            "torch": status["has_torch"],
            "optimum": status["has_optimum"],
            "onnx": status["has_onnx"],
            "onnxruntime": status["has_onnxruntime"],
        }.items()
        if not present
    ]
    status["ready_for_export_probe"] = not missing and status.get("has_gemma4_class", False)
    status["missing"] = missing
    status["warning"] = (
        "Released optimum-onnx currently pins transformers<4.58, while Gemma 4 E2B needs Transformers 5.x. "
        "If export import fails, use the no-deps git install path documented in src/model/README.md."
        if status["has_optimum"] and status.get("transformers_version", "").startswith("5.")
        else None
    )
    return status


def write_size_estimate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "webgpu-size-estimate.json"
    path.write_text(json.dumps(REFERENCE_SIZE_ESTIMATES, indent=2), encoding="utf-8")
    return {"wrote": str(path), **REFERENCE_SIZE_ESTIMATES}


def copy_metadata(model_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for file_name in METADATA_FILES:
        source = model_dir / file_name
        if source.exists():
            shutil.copy2(source, output_dir / file_name)
            copied.append(file_name)
        else:
            missing.append(file_name)
    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["transformers.js_config"] = TRANSFORMERS_JS_EXTERNAL_DATA_CONFIG
        config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    return {"copied": copied, "missing": missing}


def run_export(model: str, output_dir: Path, task: str, device: str, dtype: str | None, skip_validation: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimum_cli = shutil.which("optimum-cli")
    if optimum_cli is None:
        raise RuntimeError("optimum-cli is not on PATH; install the ONNX export environment first")
    command = [
        optimum_cli,
        "export",
        "onnx",
        "--model",
        model,
        "--task",
        task,
        "--device",
        device,
    ]
    if dtype:
        command.extend(["--dtype", dtype])
    if skip_validation:
        command.append("--no-post-process")
    command.append(str(output_dir))

    result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return {"command": command, "returncode": result.returncode, "output_dir": str(output_dir)}


def quantize_one_onnx(source: Path, target: Path, mode: str) -> None:
    import onnx

    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"Unsupported quantization mode {mode!r}; expected one of {sorted(QUANTIZATION_MODES)}")

    quantization = QUANTIZATION_MODES[mode]
    bits = int(quantization["bits"])
    use_float16 = bool(quantization["float16"])
    model = onnx.load_model(str(source))
    if use_float16:
        try:
            from onnxconverter_common.float16 import convert_float_to_float16
        except Exception as error:  # pragma: no cover - dependency-specific
            raise RuntimeError(f"{mode} requires onnxconverter-common") from error

    try:
        if bits != 4:
            raise ModuleNotFoundError("legacy MatMul4BitsQuantizer only supports 4-bit")

        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
        quantizer = MatMul4BitsQuantizer(
            model=model,
            block_size=32,
            is_symmetric=True,
            accuracy_level=None,
            op_types_to_quantize=("MatMul", "Gather"),
        )
        quantizer.process()
        q4_model = quantizer.model.model
    except ModuleNotFoundError:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        quantizer = MatMulNBitsQuantizer(
            model,
            bits=bits,
            block_size=32,
            is_symmetric=True,
            op_types_to_quantize=("MatMul", "Gather"),
        )
        quantizer.process()
        q4_model = quantizer.model.model

    if use_float16:
        try:
            q4_model = convert_float_to_float16(q4_model, keep_io_types=True, disable_shape_infer=True)
        except ValueError as error:
            if "already converted to float16" not in str(error):
                raise

    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        q4_model,
        str(target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{target.name}_data",
        size_threshold=1024,
    )


def quantize_dir(input_dir: Path, output_dir: Path, mode: str) -> dict:
    quantized_suffixes = tuple(f"_{quant_mode}.onnx" for quant_mode in QUANTIZATION_MODES)
    onnx_files = sorted(path for path in input_dir.glob("*.onnx") if not path.name.endswith(quantized_suffixes))
    if not onnx_files:
        raise FileNotFoundError(f"No unquantized .onnx files found in {input_dir}")

    results = []
    for source in onnx_files:
        target = output_dir / f"{source.stem}_{mode}.onnx"
        quantize_one_onnx(source, target, mode)
        results.append({"source": str(source), "target": str(target)})
    return {"mode": mode, "quantized": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Gemma 4 E2B ONNX/WebGPU export helper")
    parser.add_argument("--mode", choices=("probe", "size-estimate", "copy-metadata", "export", "quantize"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model id or local model path for export")
    parser.add_argument("--local-model", type=Path, default=DEFAULT_LOCAL_MODEL, help="Local merged checkpoint for metadata")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--task", default="image-text-to-text", help="Optimum ONNX task")
    parser.add_argument("--device", default="cpu", help="Export device")
    parser.add_argument("--dtype", default=None, help="Optional export dtype, for example fp16")
    parser.add_argument("--skip-validation", action="store_true", help="Skip costly export validation when supported")
    parser.add_argument("--input-onnx-dir", type=Path, default=None, help="Directory containing unquantized ONNX files")
    parser.add_argument("--quant-mode", choices=tuple(QUANTIZATION_MODES), default="q4f16")
    args = parser.parse_args()

    if args.mode == "probe":
        result = probe_environment()
    elif args.mode == "size-estimate":
        result = write_size_estimate(args.output)
    elif args.mode == "copy-metadata":
        result = copy_metadata(args.local_model, args.output)
    elif args.mode == "export":
        result = run_export(args.model, args.output, args.task, args.device, args.dtype, args.skip_validation)
    else:
        input_dir = args.input_onnx_dir or args.output
        result = quantize_dir(input_dir, args.output / "onnx", args.quant_mode)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
