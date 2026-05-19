#!/usr/bin/env python3
"""Export Gemma 4 ONNX components that mirror Transformers.js layout.

The public WebGPU layout for Gemma 4 is componentized:

- audio_encoder
- vision_encoder
- embed_tokens
- decoder_model_merged

Optimum does not currently provide a Gemma 4 ONNX config, so this script owns
the component wrappers we can safely express from the local merged checkpoint.
The decoder wrapper is intentionally guarded because it must match
Transformers.js cache inputs/outputs exactly before it is safe to publish.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

DEFAULT_LOCAL_MODEL = (
    Path("/Users/area/repos/lisper")
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "full_run_v16"
    / "lisper-gemma4-audio"
    / "merged_model"
)

DEFAULT_OUTPUT_DIR = (
    Path("/Users/area/repos/lisper")
    / "data"
    / "processed"
    / "gemma4_audio"
    / "artifacts"
    / "exports"
    / "onnx-webgpu"
    / "fp16-components"
)

EMBED_TOKENS_WEIGHT = "model.language_model.embed_tokens.weight"
EMBED_TOKENS_PER_LAYER_WEIGHT = "model.language_model.embed_tokens_per_layer.weight"
GEMMA4_VOCAB_SIZE = 262_144
GEMMA4_IMAGE_TOKEN_ID = 258_880
GEMMA4_AUDIO_TOKEN_ID = 258_881
GEMMA4_EMBED_SCALE = 39.25
GEMMA4_PER_LAYER_SCALE = 16.0


def load_model(model_dir: Path):
    from transformers import Gemma4ForConditionalGeneration

    return Gemma4ForConditionalGeneration.from_pretrained(
        str(model_dir),
        device_map="cpu",
        dtype="auto",
        low_cpu_mem_usage=True,
    ).eval()


def component_spec(component: str) -> dict[str, Any]:
    if component == "decoder_model_merged":
        input_names = [
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "num_logits_to_keep",
            "per_layer_inputs",
        ]
        output_names = ["logits"]
        dynamic_axes = {
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "per_layer_inputs": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "num_logits_to_keep"},
        }
        for cache_idx in range(15):
            key_name = f"past_key_values.{cache_idx}.key"
            value_name = f"past_key_values.{cache_idx}.value"
            present_key_name = f"present.{cache_idx}.key"
            present_value_name = f"present.{cache_idx}.value"
            input_names.extend([key_name, value_name])
            output_names.extend([present_key_name, present_value_name])
            dynamic_axes[key_name] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[value_name] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[present_key_name] = {0: "batch_size", 2: "total_sequence_length"}
            dynamic_axes[present_value_name] = {0: "batch_size", 2: "total_sequence_length"}

        return {
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
        }
    if component == "embed_tokens":
        return {
            "input_names": ["input_ids"],
            "output_names": ["inputs_embeds", "per_layer_inputs"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                "per_layer_inputs": {0: "batch_size", 1: "sequence_length"},
            },
        }
    if component == "audio_encoder":
        return {
            "input_names": ["input_features", "input_features_mask"],
            "output_names": ["audio_features"],
            "dynamic_axes": {
                "input_features": {0: "num_audio", 1: "audio_sequence_length"},
                "input_features_mask": {0: "num_audio", 1: "audio_sequence_length"},
                "audio_features": {0: "num_audio_tokens"},
            },
        }
    if component == "vision_encoder":
        return {
            "input_names": ["pixel_values", "pixel_position_ids"],
            "output_names": ["image_features"],
            "dynamic_axes": {
                "pixel_values": {0: "num_images", 1: "num_patches"},
                "pixel_position_ids": {0: "num_images", 1: "num_patches"},
                "image_features": {0: "num_image_tokens"},
            },
        }
    raise ValueError(f"Unsupported component for direct export: {component}")


def decoder_cache_dims(config: Any) -> list[int]:
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config.text_config
    num_kv_layers = text_config.num_hidden_layers - getattr(text_config, "num_kv_shared_layers", 0)
    dims = []
    for layer_idx in range(num_kv_layers):
        layer_type = text_config.layer_types[layer_idx]
        if layer_type == "full_attention" and getattr(text_config, "global_head_dim", None):
            dims.append(text_config.global_head_dim)
        else:
            dims.append(text_config.head_dim)
    return dims


def model_float_dtype(model: Any):
    import torch

    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def make_dummy_inputs(component: str, config: Any | None = None, dtype: Any | None = None):
    import torch

    dtype = dtype or torch.float32
    if component == "decoder_model_merged":
        if config is None:
            raise ValueError("decoder_model_merged dummy inputs require model config")
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config.text_config
        batch_size = 1
        sequence_length = 2
        past_sequence_length = 1
        inputs = [
            torch.zeros((batch_size, sequence_length, text_config.hidden_size), dtype=dtype),
            torch.ones((batch_size, past_sequence_length + sequence_length), dtype=torch.long),
            torch.arange(past_sequence_length, past_sequence_length + sequence_length, dtype=torch.long).unsqueeze(0),
            torch.tensor(1, dtype=torch.long),
            torch.zeros(
                (
                    batch_size,
                    sequence_length,
                    text_config.num_hidden_layers,
                    text_config.hidden_size_per_layer_input,
                ),
                dtype=dtype,
            ),
        ]
        for head_dim in decoder_cache_dims(config):
            cache_shape = (batch_size, text_config.num_key_value_heads, past_sequence_length, head_dim)
            inputs.extend(
                [
                    torch.zeros(cache_shape, dtype=dtype),
                    torch.zeros(cache_shape, dtype=dtype),
                ]
            )
        return tuple(inputs)
    if component == "embed_tokens":
        return (torch.tensor([[2, 106, 107, 1]], dtype=torch.long),)
    if component == "audio_encoder":
        return (
            torch.zeros((1, 32, 128), dtype=dtype),
            torch.ones((1, 32), dtype=torch.bool),
        )
    if component == "vision_encoder":
        return (
            torch.zeros((1, 252, 768), dtype=dtype),
            torch.zeros((1, 252, 2), dtype=torch.long),
        )
    raise ValueError(f"Unsupported dummy input component: {component}")


def make_wrapper(component: str, model: Any):
    import torch

    if component == "embed_tokens":
        class EmbedTokensWrapper(torch.nn.Module):
            """Exports the text embedding session expected by Transformers.js."""

            def __init__(self, source_model: Any):
                super().__init__()
                self.language_model = source_model.model.language_model

            def forward(self, input_ids):
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                per_layer_inputs = self.language_model.get_per_layer_inputs(input_ids, inputs_embeds)
                return inputs_embeds, per_layer_inputs

        return EmbedTokensWrapper(model)
    if component == "audio_encoder":
        class AudioEncoderWrapper(torch.nn.Module):
            """Exports projected, padding-stripped audio soft tokens."""

            def __init__(self, source_model: Any):
                super().__init__()
                self.model = source_model.model
                self.model.audio_tower.config._attn_implementation = "eager"

            def _make_audio_attention_mask(self, hidden_states, output_mask):
                batch_size = hidden_states.shape[0]
                sequence_length = hidden_states.shape[1]
                device = hidden_states.device
                query_positions = torch.arange(sequence_length, device=device).view(1, 1, sequence_length, 1)
                key_positions = torch.arange(sequence_length, device=device).view(1, 1, 1, sequence_length)
                distance = query_positions - key_positions
                left_context = self.model.audio_tower.config.attention_context_left - 1
                right_context = self.model.audio_tower.config.attention_context_right
                keep = ((distance >= 0) & (distance < left_context)) | (
                    (distance < 0) & ((-distance) < right_context)
                )
                valid_keys = output_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, sequence_length)
                valid_queries = output_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, sequence_length, 1)
                return keep & valid_keys & valid_queries

            def forward(self, input_features, input_features_mask):
                audio_tower = self.model.audio_tower
                hidden_states, output_mask = audio_tower.subsample_conv_projection(input_features, input_features_mask)
                position_embeddings = audio_tower.rel_pos_enc(hidden_states)
                attention_mask = self._make_audio_attention_mask(hidden_states, output_mask)
                attention_mask = audio_tower._convert_4d_mask_to_blocked_5d(attention_mask)
                for encoder_layer in audio_tower.layers[: audio_tower.config.num_hidden_layers]:
                    hidden_states = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                    )
                hidden_states = audio_tower.output_proj(hidden_states)
                audio_features = self.model.embed_audio(inputs_embeds=hidden_states)
                audio_mask = output_mask.to(dtype=bool)
                return audio_features[audio_mask]

        return AudioEncoderWrapper(model)
    if component == "vision_encoder":
        class VisionEncoderWrapper(torch.nn.Module):
            """Exports projected vision soft tokens."""

            def __init__(self, source_model: Any):
                super().__init__()
                self.model = source_model.model

            def forward(self, pixel_values, pixel_position_ids):
                image_output = self.model.get_image_features(pixel_values, pixel_position_ids, return_dict=True)
                return image_output.pooler_output

        return VisionEncoderWrapper(model)
    if component == "decoder_model_merged":
        class DecoderModelMergedWrapper(torch.nn.Module):
            """Decoder session scaffold matching the official Transformers.js cache contract."""

            def __init__(self, source_model: Any):
                super().__init__()
                self.language_model = source_model.model.language_model
                self.lm_head = source_model.lm_head
                self.text_config = source_model.config.get_text_config()
                self.text_config._attn_implementation = "eager"
                self.final_logit_softcapping = self.text_config.final_logit_softcapping
                self.num_kv_layers = self.text_config.num_hidden_layers - getattr(
                    self.text_config,
                    "num_kv_shared_layers",
                    0,
                )
                self.sliding_window = self.text_config.sliding_window

            def _make_additive_attention_mask(self, inputs_embeds, attention_mask, position_ids, sliding: bool):
                batch_size = inputs_embeds.shape[0]
                query_length = inputs_embeds.shape[1]
                key_value_length = attention_mask.shape[1]
                key_positions = torch.arange(key_value_length, device=inputs_embeds.device).view(1, 1, key_value_length)
                query_positions = position_ids.to(device=inputs_embeds.device).view(batch_size, query_length, 1)
                keep = key_positions <= query_positions
                if sliding:
                    keep = keep & (key_positions > query_positions - self.sliding_window)
                keep = keep & attention_mask.to(device=inputs_embeds.device, dtype=torch.bool).view(
                    batch_size,
                    1,
                    key_value_length,
                )
                keep = keep.unsqueeze(1)
                zero = torch.zeros((), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                masked = torch.full((), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
                return torch.where(keep, zero, masked)

            def forward(
                self,
                inputs_embeds,
                attention_mask,
                position_ids,
                num_logits_to_keep,
                per_layer_inputs,
                *flat_past_key_values,
            ):
                from transformers.cache_utils import DynamicCache

                ddp_cache_data = [
                    (flat_past_key_values[index * 2], flat_past_key_values[index * 2 + 1])
                    for index in range(self.num_kv_layers)
                ]
                past_key_values = DynamicCache(ddp_cache_data=ddp_cache_data, config=self.text_config)
                attention_masks = {
                    "full_attention": self._make_additive_attention_mask(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        sliding=False,
                    ),
                    "sliding_attention": self._make_additive_attention_mask(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        sliding=True,
                    ),
                }
                outputs = self.language_model(
                    attention_mask=attention_masks,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(hidden_states[:, -1:, :])
                if self.final_logit_softcapping is not None:
                    logits = logits / self.final_logit_softcapping
                    logits = torch.tanh(logits)
                    logits = logits * self.final_logit_softcapping

                # Keep the scalar input in the traced graph while limiting the first scaffold to one-token decode.
                logits = logits + num_logits_to_keep.to(dtype=logits.dtype).reshape(1, 1, 1) * 0
                present = []
                for layer in outputs.past_key_values.layers[: self.num_kv_layers]:
                    present.extend([layer.keys, layer.values])
                return (logits, *present)

        return DecoderModelMergedWrapper(model)
    raise ValueError(f"Unknown component: {component}")


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    return header_length, header


def _external_tensor(name: str, data_type: int, shape: list[int], location: str, offset: int, length: int):
    import onnx

    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(shape)
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in {
        "location": location,
        "offset": str(offset),
        "length": str(length),
    }.items():
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    return tensor


def _copy_bf16_tensor_as_fp16(
    source_path: Path,
    output_handle: Any,
    source_data_start: int,
    source_data_end: int,
    chunk_elements: int = 16_777_216,
) -> int:
    """Convert BF16 safetensors bytes to FP16 ONNX external-data bytes without loading everything."""

    import numpy as np

    bytes_per_bf16 = 2
    chunk_bytes = chunk_elements * bytes_per_bf16
    total_output_bytes = 0
    with source_path.open("rb") as source_handle:
        source_handle.seek(source_data_start)
        remaining = source_data_end - source_data_start
        while remaining > 0:
            read_size = min(chunk_bytes, remaining)
            if read_size % bytes_per_bf16:
                raise ValueError("BF16 tensor byte count is not aligned to 2 bytes")
            raw = source_handle.read(read_size)
            if len(raw) != read_size:
                raise EOFError(f"Expected {read_size} bytes, read {len(raw)}")
            bf16 = np.frombuffer(raw, dtype="<u2")
            fp32_bits = bf16.astype(np.uint32) << 16
            fp32 = fp32_bits.view(np.float32)
            fp16 = fp32.astype(np.float16)
            output_handle.write(fp16.tobytes())
            total_output_bytes += fp16.nbytes
            remaining -= read_size
    return total_output_bytes


def _quantize_fp16_block_asym(block: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    min_val = np.minimum(block.min(axis=2, keepdims=True), 0)
    max_val = np.maximum(block.max(axis=2, keepdims=True), 0)
    scale = ((max_val - min_val) / 15.0).astype(np.float16)
    zero_point = np.where(scale == 0, 8, -min_val / scale).round().clip(0, 15).astype(np.uint8)
    quantized = np.where(scale == 0, 8, block / scale + zero_point).round().clip(0, 15).astype(np.uint8)
    return quantized, scale.squeeze(2), zero_point.squeeze(2)


def _pack_u4(values: Any, output_shape: tuple[int, int]) -> Any:
    import numpy as np

    flat = values.reshape(values.shape[0], -1)
    if flat.shape[1] % 2:
        flat = np.pad(flat, ((0, 0), (0, 1)))
    packed = (flat[:, 0::2] & 0xF) | ((flat[:, 1::2] & 0xF) << 4)
    return packed.astype(np.uint8).reshape(output_shape)


def _copy_bf16_tensor_as_q4f16(
    source_path: Path,
    output_handle: Any,
    source_data_start: int,
    source_data_end: int,
    shape: list[int],
    chunk_rows: int = 512,
    block_size: int = 32,
    label: str = "tensor",
) -> dict[str, int]:
    """Write official GatherBlockQuantized external-data tensors for a 2D BF16 weight."""

    import numpy as np

    rows, cols = shape
    if cols % block_size:
        raise ValueError(f"{label} width {cols} is not divisible by block size {block_size}")

    row_bytes = cols * 2
    source_length = source_data_end - source_data_start
    expected_source_length = rows * row_bytes
    if source_length != expected_source_length:
        raise ValueError(f"{label} byte length mismatch: expected {expected_source_length}, got {source_length}")

    blocks = cols // block_size
    quant_offset = output_handle.tell()
    quant_length = rows * (cols // 2)
    scales_offset = quant_offset + quant_length
    scales_length = rows * blocks * 2
    zero_points_offset = scales_offset + scales_length
    zero_points_length = rows * (blocks // 2)
    output_handle.truncate(zero_points_offset + zero_points_length)

    with source_path.open("rb") as source_handle:
        for row_start in range(0, rows, chunk_rows):
            row_count = min(chunk_rows, rows - row_start)
            source_handle.seek(source_data_start + row_start * row_bytes)
            raw = source_handle.read(row_count * row_bytes)
            if len(raw) != row_count * row_bytes:
                raise EOFError(f"{label}: expected {row_count * row_bytes} bytes, read {len(raw)}")

            bf16 = np.frombuffer(raw, dtype="<u2").reshape(row_count, cols)
            fp32_bits = bf16.astype(np.uint32) << 16
            fp16 = fp32_bits.view(np.float32).astype(np.float16).reshape(row_count, cols)
            block = fp16.reshape(row_count, blocks, block_size)
            quantized, scales, zero_points = _quantize_fp16_block_asym(block)
            quant_packed = _pack_u4(quantized.reshape(row_count, cols), (row_count, cols // 2))
            zero_points_packed = _pack_u4(zero_points, (row_count, blocks // 2))

            output_handle.seek(quant_offset + row_start * (cols // 2))
            output_handle.write(quant_packed.tobytes())
            output_handle.seek(scales_offset + row_start * blocks * 2)
            output_handle.write(scales.astype(np.float16).tobytes())
            output_handle.seek(zero_points_offset + row_start * (blocks // 2))
            output_handle.write(zero_points_packed.tobytes())

            if row_start == 0 or (row_start // chunk_rows) % 32 == 0:
                print(
                    f"{label}: quantized {row_start + row_count}/{rows} rows",
                    file=sys.stderr,
                    flush=True,
                )

    output_handle.seek(zero_points_offset + zero_points_length)
    return {
        "quant_offset": quant_offset,
        "quant_length": quant_length,
        "scales_offset": scales_offset,
        "scales_length": scales_length,
        "zero_points_offset": zero_points_offset,
        "zero_points_length": zero_points_length,
    }


def _make_embed_graph(
    initializers: list[Any],
    output_name: str,
    opset: int,
    quantized: bool,
    hidden_size: int,
    per_layer_count: int,
):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    constants = [
        numpy_helper.from_array(np.array(GEMMA4_EMBED_SCALE, dtype=np.float16), name="embed_scale"),
        numpy_helper.from_array(np.array(GEMMA4_PER_LAYER_SCALE, dtype=np.float16), name="per_layer_scale"),
        numpy_helper.from_array(np.array(GEMMA4_VOCAB_SIZE, dtype=np.int64), name="vocab_size"),
        numpy_helper.from_array(np.array(GEMMA4_IMAGE_TOKEN_ID, dtype=np.int64), name="image_token_id"),
        numpy_helper.from_array(np.array(GEMMA4_AUDIO_TOKEN_ID, dtype=np.int64), name="audio_token_id"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero_token_id"),
        numpy_helper.from_array(np.array([0, 0, per_layer_count, 256], dtype=np.int64), name="per_layer_shape"),
    ]

    gather_op = "GatherBlockQuantized" if quantized else "Gather"
    gather_domain = "com.microsoft" if quantized else ""
    main_gather_inputs = (
        [
            "model_embed_tokens_weight_quant",
            "input_ids",
            "model_embed_tokens_weight_scales",
            "model_embed_tokens_weight_zp",
        ]
        if quantized
        else ["embed_tokens.weight", "input_ids"]
    )
    per_layer_gather_inputs = (
        [
            "model_embed_tokens_per_layer_weight_quant",
            "per_layer_input_ids",
            "model_embed_tokens_per_layer_weight_scales",
            "model_embed_tokens_per_layer_weight_zp",
        ]
        if quantized
        else ["embed_tokens_per_layer.weight", "per_layer_input_ids"]
    )
    gather_attrs = {"axis": 0} if not quantized else {"bits": 4, "block_size": 32, "gather_axis": 0, "quantize_axis": 1}
    nodes = [
        helper.make_node(
            gather_op,
            main_gather_inputs,
            ["inputs_embeds_unscaled"],
            domain=gather_domain,
            **gather_attrs,
        ),
        helper.make_node("Mul", ["inputs_embeds_unscaled", "embed_scale"], ["inputs_embeds_scaled"]),
        helper.make_node("Cast", ["inputs_embeds_scaled"], ["inputs_embeds"], to=TensorProto.FLOAT),
        helper.make_node("Less", ["input_ids", "vocab_size"], ["is_vocab_token"]),
        helper.make_node("Equal", ["input_ids", "image_token_id"], ["is_image_token"]),
        helper.make_node("Not", ["is_image_token"], ["is_not_image_token"]),
        helper.make_node("And", ["is_vocab_token", "is_not_image_token"], ["is_vocab_not_image"]),
        helper.make_node("Equal", ["input_ids", "audio_token_id"], ["is_audio_token"]),
        helper.make_node("Not", ["is_audio_token"], ["is_not_audio_token"]),
        helper.make_node("And", ["is_vocab_not_image", "is_not_audio_token"], ["has_per_layer_input"]),
        helper.make_node("Where", ["has_per_layer_input", "input_ids", "zero_token_id"], ["per_layer_input_ids"]),
        helper.make_node(
            gather_op,
            per_layer_gather_inputs,
            ["per_layer_inputs_flat_unscaled"],
            domain=gather_domain,
            **gather_attrs,
        ),
        helper.make_node(
            "Mul",
            ["per_layer_inputs_flat_unscaled", "per_layer_scale"],
            ["per_layer_inputs_flat_scaled"],
        ),
        helper.make_node("Reshape", ["per_layer_inputs_flat_scaled", "per_layer_shape"], ["per_layer_inputs_scaled"]),
        helper.make_node("Cast", ["per_layer_inputs_scaled"], ["per_layer_inputs"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "lisper_gemma4_embed_tokens_q4f16" if quantized else "lisper_gemma4_embed_tokens",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        ],
        [
            helper.make_tensor_value_info(
                "inputs_embeds",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", hidden_size],
            ),
            helper.make_tensor_value_info(
                "per_layer_inputs",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", per_layer_count, 256],
            ),
        ],
        initializer=[*constants, *initializers],
    )
    opsets = [helper.make_opsetid("", opset)]
    if quantized:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(
        graph,
        opset_imports=opsets,
        producer_name="lisper-export-gemma4-onnx-components",
    )
    model.ir_version = 10
    onnx.save_model(model, output_name)
    return model


def export_embed_tokens_manual(model_dir: Path, output_dir: Path, opset: int) -> dict[str, Any]:
    """Export embed_tokens without tracing the 5.5 GB embedding path through Torch."""

    from onnx import TensorProto

    source_path = model_dir / "model.safetensors"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing merged safetensors checkpoint: {source_path}")

    header_length, header = _read_safetensors_header(source_path)
    data_start = 8 + header_length
    for key in (EMBED_TOKENS_WEIGHT, EMBED_TOKENS_PER_LAYER_WEIGHT):
        if key not in header:
            raise KeyError(f"Missing expected embedding tensor in checkpoint: {key}")
        if header[key]["dtype"] != "BF16":
            raise ValueError(f"Expected {key} to be BF16, got {header[key]['dtype']}")

    main_meta = header[EMBED_TOKENS_WEIGHT]
    per_layer_meta = header[EMBED_TOKENS_PER_LAYER_WEIGHT]
    vocab_size, hidden_size = main_meta["shape"]
    per_layer_vocab_size, per_layer_flat_size = per_layer_meta["shape"]
    if vocab_size != per_layer_vocab_size:
        raise ValueError("Embedding vocab sizes do not match")
    if per_layer_flat_size % 256:
        raise ValueError("Per-layer embedding width is not divisible by 256")
    per_layer_count = per_layer_flat_size // 256

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embed_tokens.onnx"
    external_name = "embed_tokens.onnx_data"
    external_path = output_dir / external_name

    tensor_offsets: dict[str, tuple[int, int]] = {}
    with external_path.open("wb") as external_handle:
        main_start = external_handle.tell()
        main_length = _copy_bf16_tensor_as_fp16(
            source_path,
            external_handle,
            data_start + main_meta["data_offsets"][0],
            data_start + main_meta["data_offsets"][1],
        )
        tensor_offsets["embed_tokens.weight"] = (main_start, main_length)

        per_layer_start = external_handle.tell()
        per_layer_length = _copy_bf16_tensor_as_fp16(
            source_path,
            external_handle,
            data_start + per_layer_meta["data_offsets"][0],
            data_start + per_layer_meta["data_offsets"][1],
        )
        tensor_offsets["embed_tokens_per_layer.weight"] = (per_layer_start, per_layer_length)

    initializers = [
        _external_tensor(
            "embed_tokens.weight",
            TensorProto.FLOAT16,
            main_meta["shape"],
            external_name,
            *tensor_offsets["embed_tokens.weight"],
        ),
        _external_tensor(
            "embed_tokens_per_layer.weight",
            TensorProto.FLOAT16,
            per_layer_meta["shape"],
            external_name,
            *tensor_offsets["embed_tokens_per_layer.weight"],
        ),
    ]
    _make_embed_graph(
        initializers,
        str(output_path),
        opset,
        quantized=False,
        hidden_size=hidden_size,
        per_layer_count=per_layer_count,
    )
    return {
        "component": "embed_tokens",
        "output": str(output_path),
        "external_data": str(external_path),
        "opset": opset,
        "weights": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "per_layer_count": per_layer_count,
            "external_data_bytes": external_path.stat().st_size,
        },
    }


def export_embed_tokens_q4f16_manual(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Directly export the official q4f16 GatherBlockQuantized embedding component."""

    from onnx import TensorProto

    source_path = model_dir / "model.safetensors"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing merged safetensors checkpoint: {source_path}")

    header_length, header = _read_safetensors_header(source_path)
    data_start = 8 + header_length
    main_meta = header[EMBED_TOKENS_WEIGHT]
    per_layer_meta = header[EMBED_TOKENS_PER_LAYER_WEIGHT]
    vocab_size, hidden_size = main_meta["shape"]
    per_layer_vocab_size, per_layer_flat_size = per_layer_meta["shape"]
    if vocab_size != per_layer_vocab_size:
        raise ValueError("Embedding vocab sizes do not match")
    per_layer_count = per_layer_flat_size // 256

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embed_tokens_q4f16.onnx"
    external_name = "embed_tokens_q4f16.onnx_data"
    external_path = output_dir / external_name

    with external_path.open("wb") as external_handle:
        main = _copy_bf16_tensor_as_q4f16(
            source_path,
            external_handle,
            data_start + main_meta["data_offsets"][0],
            data_start + main_meta["data_offsets"][1],
            main_meta["shape"],
            label="embed_tokens",
        )
        per_layer = _copy_bf16_tensor_as_q4f16(
            source_path,
            external_handle,
            data_start + per_layer_meta["data_offsets"][0],
            data_start + per_layer_meta["data_offsets"][1],
            per_layer_meta["shape"],
            label="embed_tokens_per_layer",
        )

    main_blocks = hidden_size // 32
    per_layer_blocks = per_layer_flat_size // 32
    initializers = [
        _external_tensor(
            "model_embed_tokens_weight_quant",
            TensorProto.UINT8,
            [vocab_size, hidden_size // 2],
            external_name,
            main["quant_offset"],
            main["quant_length"],
        ),
        _external_tensor(
            "model_embed_tokens_weight_scales",
            TensorProto.FLOAT16,
            [vocab_size, main_blocks],
            external_name,
            main["scales_offset"],
            main["scales_length"],
        ),
        _external_tensor(
            "model_embed_tokens_weight_zp",
            TensorProto.UINT8,
            [vocab_size, main_blocks // 2],
            external_name,
            main["zero_points_offset"],
            main["zero_points_length"],
        ),
        _external_tensor(
            "model_embed_tokens_per_layer_weight_quant",
            TensorProto.UINT8,
            [vocab_size, per_layer_flat_size // 2],
            external_name,
            per_layer["quant_offset"],
            per_layer["quant_length"],
        ),
        _external_tensor(
            "model_embed_tokens_per_layer_weight_scales",
            TensorProto.FLOAT16,
            [vocab_size, per_layer_blocks],
            external_name,
            per_layer["scales_offset"],
            per_layer["scales_length"],
        ),
        _external_tensor(
            "model_embed_tokens_per_layer_weight_zp",
            TensorProto.UINT8,
            [vocab_size, per_layer_blocks // 2],
            external_name,
            per_layer["zero_points_offset"],
            per_layer["zero_points_length"],
        ),
    ]
    _make_embed_graph(
        initializers,
        str(output_path),
        opset=21,
        quantized=True,
        hidden_size=hidden_size,
        per_layer_count=per_layer_count,
    )
    return {
        "component": "embed_tokens",
        "output": str(output_path),
        "external_data": str(external_path),
        "quant_mode": "q4f16",
        "weights": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "per_layer_count": per_layer_count,
            "external_data_bytes": external_path.stat().st_size,
        },
    }


def dry_run(component: str, model_dir: Path) -> dict[str, Any]:
    import torch

    model = load_model(model_dir)
    wrapper = make_wrapper(component, model)
    dummy_inputs = make_dummy_inputs(component, model.config, model_float_dtype(model))
    with torch.no_grad():
        outputs = wrapper(*dummy_inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    return {
        "component": component,
        "output_shapes": [list(output.shape) for output in outputs],
        "output_dtypes": [str(output.dtype) for output in outputs],
        "contract": component_spec(component),
    }


def export_component(
    component: str,
    model_dir: Path,
    output_dir: Path,
    opset: int,
    allow_decoder_export: bool = False,
    direct_q4f16: bool = False,
) -> dict[str, Any]:
    if component == "decoder_model_merged" and not allow_decoder_export:
        raise NotImplementedError(
            "decoder_model_merged export is guarded. The wrapper dry-run matches the reference cache "
            "shapes, but export must be requested explicitly with --allow-decoder-export until the "
            "resulting ONNX file is validated against Transformers.js."
        )
    if component == "embed_tokens" and direct_q4f16:
        return export_embed_tokens_q4f16_manual(model_dir, output_dir)
    if component == "embed_tokens":
        return export_embed_tokens_manual(model_dir, output_dir, opset)
    import torch

    model = load_model(model_dir)
    wrapper = make_wrapper(component, model)
    dummy_inputs = make_dummy_inputs(component, model.config, model_float_dtype(model))
    spec = component_spec(component)
    output_path = output_dir / f"{component}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        dynamic_axes=spec["dynamic_axes"],
        opset_version=opset,
        do_constant_folding=True,
        external_data=True,
    )
    return {"component": component, "output": str(output_path), "opset": opset}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gemma 4 ONNX components from the merged Lisper checkpoint")
    parser.add_argument(
        "--component",
        choices=("embed_tokens", "audio_encoder", "vision_encoder", "decoder_model_merged"),
        required=True,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="Legacy torch.onnx.export in the local Torch 2.6 environment supports opset <= 20.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--direct-q4f16",
        action="store_true",
        help="For embed_tokens, write the official q4f16 GatherBlockQuantized graph directly.",
    )
    parser.add_argument(
        "--allow-decoder-export",
        action="store_true",
        help="Explicitly permit the guarded decoder_model_merged ONNX export attempt.",
    )
    args = parser.parse_args()

    try:
        if args.dry_run:
            result = dry_run(args.component, args.model_dir)
        else:
            result = export_component(
                args.component,
                args.model_dir,
                args.output_dir,
                args.opset,
                allow_decoder_export=args.allow_decoder_export,
                direct_q4f16=args.direct_q4f16,
            )
    except NotImplementedError as error:
        result = {"component": args.component, "status": "blocked", "error": str(error)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
