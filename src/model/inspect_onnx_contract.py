#!/usr/bin/env python3
"""Inspect ONNX component signatures without loading external tensor data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_COMPONENTS = (
    "audio_encoder_q4f16.onnx",
    "vision_encoder_q4f16.onnx",
    "embed_tokens_q4f16.onnx",
    "decoder_model_merged_q4f16.onnx",
)


def value_info_to_dict(value_info) -> dict:
    tensor_type = value_info.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        elif dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append(None)
    return {
        "name": value_info.name,
        "elem_type": tensor_type.elem_type,
        "shape": shape,
    }


def inspect_model(path: Path) -> dict:
    import onnx

    model = onnx.load_model(str(path), load_external_data=False)
    graph = model.graph
    external_initializers = []
    for initializer in graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                external_initializers.append({"initializer": initializer.name, "location": entry.value})
                break

    return {
        "file": str(path),
        "ir_version": model.ir_version,
        "opsets": [{"domain": opset.domain, "version": opset.version} for opset in model.opset_import],
        "inputs": [value_info_to_dict(item) for item in graph.input],
        "outputs": [value_info_to_dict(item) for item in graph.output],
        "initializer_count": len(graph.initializer),
        "external_data_files": sorted({item["location"] for item in external_initializers}),
        "node_op_types": sorted({node.op_type for node in graph.node}),
        "node_count": len(graph.node),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Gemma 4 ONNX q4f16 component contracts")
    parser.add_argument("onnx_dir", type=Path, help="Directory containing q4f16 wrapper .onnx files")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    missing = [name for name in EXPECTED_COMPONENTS if not (args.onnx_dir / "onnx" / name).exists() and not (args.onnx_dir / name).exists()]
    components = []
    for name in EXPECTED_COMPONENTS:
        path = args.onnx_dir / "onnx" / name
        if not path.exists():
            path = args.onnx_dir / name
        if path.exists():
            components.append(inspect_model(path))

    result = {
        "source_dir": str(args.onnx_dir),
        "missing_components": missing,
        "components": components,
    }

    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
