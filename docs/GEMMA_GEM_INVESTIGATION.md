# Gemma Gem Investigation

## Finding

Gemma Gem does not implement a custom Gemma 4 ONNX export or a special lightweight model format. It loads the public ONNX repos directly through Transformers.js:

- `onnx-community/gemma-4-E2B-it-ONNX`
- `onnx-community/gemma-4-E4B-it-ONNX`
- `dtype: "q4f16"`
- `device: "webgpu"`

Relevant local clone paths:

- `/tmp/gemma-gem/shared/models.ts`
- `/tmp/gemma-gem/offscreen/model-host.ts`

## Size Reality

The `~500MB` and `~1.5GB` values in Gemma Gem are hardcoded UI estimates in `shared/models.ts`; they are not computed from the actual Hugging Face files.

Measured with `hf download --dry-run`:

| Model | q4f16 files | Total |
| --- | ---: | ---: |
| Gemma 4 E2B ONNX | 8 | `3.4G` |
| Gemma 4 E4B ONNX | 9 | `5.2G` |

Gemma 4 E2B q4f16 component sizes:

| Component | Size |
| --- | ---: |
| `audio_encoder_q4f16` | `171.5M` |
| `vision_encoder_q4f16` | `99.4M` |
| `decoder_model_merged_q4f16` | `1.5G` |
| `embed_tokens_q4f16` | `1.6G` |

Audio is not the main size driver. The decoder and per-layer/token embeddings dominate the browser download.

## Implication For Lisper

Keep raw audio in scope. The official ONNX repo already supports the full multimodal layout that Transformers.js expects:

- `audio_encoder_q4f16`
- `vision_encoder_q4f16`
- `embed_tokens_q4f16`
- `decoder_model_merged_q4f16`

The remaining blocker is export tooling: our merged v16 checkpoint must be converted into the same component layout. Plain Optimum ONNX cannot do this yet because it has no native `gemma4` ONNX config.

## App Takeaway

Gemma Gem's smoothness comes from:

- loading official prebuilt ONNX files
- using a Chrome offscreen document
- loading `dtype: "q4f16"` on WebGPU
- caching downloaded files after first run

Transformers.js also defines text-only sessions for Gemma-style multimodal models. A text-only path can avoid loading audio and vision encoder sessions, but it still needs:

- `embed_tokens_q4f16`
- `decoder_model_merged_q4f16`

So text-only use can avoid roughly `270 MB` of E2B audio/vision files, but it does not avoid the multi-GB decoder and embedding files. Gemma Gem is not avoiding the large `embed_tokens` or decoder files.
