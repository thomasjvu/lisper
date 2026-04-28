import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';

import {
  buildGemmaBrowserInputs,
  formatGemmaBrowserRuntimeError,
  getGemmaBrowserFloat16BridgeDiagnostics,
  getLastGemmaBrowserDiagnostics,
  getLastImageDiagnostics,
  patchGemmaBrowserFloat16Feeds,
  runGemmaOneTokenSmoke,
  runGemmaImagePipelineSelfTest,
  summarizeGemmaInputs,
  type GemmaBrowserSelfTestResult,
} from '../utils/gemmaBrowserMultimodal';

type DtypeMap = Record<string, 'q2f16' | 'q4f16'>;

export interface ModelPreset {
  id: string;
  label: string;
  description: string;
  dtype: 'q4f16' | DtypeMap;
  dtypeLabel: string;
  expectedSize: string;
}

export interface ProgressEvent {
  status?: string;
  name?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

export interface GenerateRequest {
  modelId: string;
  dtype: ModelPreset['dtype'];
  messages: any[];
  audio?: Blob | null;
  images?: Blob[];
  maxNewTokens: number;
  onProgress?: (event: ProgressEvent) => void;
}

export interface SmokeRequest extends GenerateRequest {
  prompt: string;
}

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16',
    label: 'Lisper trained q4f16',
    description: 'Canonical trained WebGPU package. Use this for hackathon browser testing.',
    dtype: 'q4f16',
    dtypeLabel: 'q4f16',
    expectedSize: '~3.15 GB',
  },
  {
    id: 'thomasjvu/lisper-gemma4-e2b-audio-onnx-q2f16-experimental',
    label: 'Lisper q2f16 experiment',
    description: 'Decoder-only q2f16 experiment; embed/audio/vision remain q4f16 for WebGPU compatibility.',
    dtype: {
      decoder_model_merged: 'q2f16',
      embed_tokens: 'q4f16',
      audio_encoder: 'q4f16',
      vision_encoder: 'q4f16',
    },
    dtypeLabel: 'q2f16 decoder + q4f16 media',
    expectedSize: '~2.61 GB',
  },
  {
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    label: 'Public base Gemma 4 E2B',
    description: 'Official ONNX reference package for comparing base-model behavior.',
    dtype: 'q4f16',
    dtypeLabel: 'q4f16',
    expectedSize: '~3.4 GB',
  },
];

let runtimeModule: any = null;
let processor: any = null;
let model: any = null;
let loadedKey: string | null = null;
let loadingPromise: Promise<void> | null = null;

function getRuntimeAssetUrls() {
  if (typeof window === 'undefined') {
    return [ortMjsUrl, ortWasmUrl];
  }

  return [new URL(ortMjsUrl, window.location.href).href, new URL(ortWasmUrl, window.location.href).href];
}

async function loadTransformerModule() {
  if (runtimeModule) {
    return runtimeModule;
  }

  runtimeModule = await import('@huggingface/transformers');
  runtimeModule.env.allowLocalModels = false;
  runtimeModule.env.allowRemoteModels = true;
  runtimeModule.env.useBrowserCache = true;
  runtimeModule.env.useWasmCache = true;

  const onnx = runtimeModule.env.backends?.onnx;
  if (onnx?.wasm) {
    onnx.wasm.wasmPaths = {
      wasm: getRuntimeAssetUrls()[1],
      mjs: getRuntimeAssetUrls()[0],
    };
  }

  if (!runtimeModule.Gemma4Processor || !runtimeModule.Gemma4ForConditionalGeneration) {
    throw new Error('Gemma 4 requires @huggingface/transformers 4.1.0 or newer.');
  }

  return runtimeModule;
}

function keyFor(modelId: string, dtype: ModelPreset['dtype']) {
  return `${modelId}::${typeof dtype === 'string' ? dtype : JSON.stringify(dtype)}`;
}

export function getLoadedModelKey() {
  return loadedKey;
}

export async function loadGemmaModel(modelId: string, dtype: ModelPreset['dtype'], onProgress?: (event: ProgressEvent) => void) {
  const nextKey = keyFor(modelId, dtype);
  if (loadedKey === nextKey && processor && model) {
    return;
  }

  if (loadingPromise) {
    await loadingPromise;
    if (loadedKey === nextKey && processor && model) {
      return;
    }
  }

  loadingPromise = (async () => {
    const module = await loadTransformerModule();
    model?.dispose?.();
    processor = null;
    model = null;
    loadedKey = null;

    processor = await module.Gemma4Processor.from_pretrained(modelId, {
      progress_callback: onProgress,
    });
    model = await module.Gemma4ForConditionalGeneration.from_pretrained(modelId, {
      dtype,
      device: 'webgpu',
      progress_callback: onProgress,
    });
    patchGemmaBrowserFloat16Feeds(module, model);
    loadedKey = nextKey;
  })();

  try {
    await loadingPromise;
  } finally {
    loadingPromise = null;
  }
}

function stripModelArtifacts(text: string) {
  return text
    .replace(/<\|channel\|>thought[\s\S]*?<\|channel\|>/g, '')
    .replace(/```json/g, '')
    .replace(/```/g, '')
    .trim();
}

export async function generateGemmaResponse(request: GenerateRequest) {
  await loadGemmaModel(request.modelId, request.dtype, request.onProgress);

  const module = await loadTransformerModule();
  request.onProgress?.({ status: 'template' });
  let prompt: string;
  try {
    prompt = processor.apply_chat_template(request.messages, {
      enable_thinking: false,
      add_generation_prompt: true,
    });
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'gemma-lab',
      stage: 'apply_chat_template',
      error,
    });
  }

  const hasMedia = Boolean(request.audio || request.images?.length);
  let inputs: any;
  try {
    inputs = hasMedia
      ? await buildGemmaBrowserInputs({
          module,
          processor,
          prompt,
          route: 'gemma-lab',
          images: request.images,
          audio: request.audio,
          onProgress: request.onProgress,
        })
      : await processor(prompt, null, null, { add_special_tokens: false });
  } catch (error) {
    throw error;
  }

  let outputs: any;
  request.onProgress?.({ status: 'generate' });
  try {
    outputs = await model.generate({
      ...inputs,
      max_new_tokens: request.maxNewTokens,
      do_sample: false,
      temperature: 1.0,
      top_p: 0.95,
      top_k: 64,
    });
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'gemma-lab',
      stage: 'model.generate',
      error,
      inputSummary: summarizeGemmaInputs(inputs),
      float16Bridge: getGemmaBrowserFloat16BridgeDiagnostics(model),
    });
  }

  request.onProgress?.({ status: 'decode output' });
  try {
    const promptLength = inputs?.input_ids?.dims?.at?.(-1) ?? null;
    const generated = promptLength ? outputs.slice(null, [promptLength, null]) : outputs;
    const decoded = processor.batch_decode(generated, { skip_special_tokens: true });
    return stripModelArtifacts(decoded?.[0] ?? '');
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'gemma-lab',
      stage: 'decode output',
      error,
      inputSummary: summarizeGemmaInputs(inputs),
      float16Bridge: getGemmaBrowserFloat16BridgeDiagnostics(model),
    });
  }
}

export async function selfTestGemmaImagePipeline(modelId: string, dtype: ModelPreset['dtype'], images?: Blob[]) {
  await loadGemmaModel(modelId, dtype);
  const module = await loadTransformerModule();
  return runGemmaImagePipelineSelfTest({
    module,
    processor,
    route: 'gemma-lab',
    images,
  }) as Promise<GemmaBrowserSelfTestResult>;
}

export async function oneTokenGemmaSmoke(request: SmokeRequest) {
  await loadGemmaModel(request.modelId, request.dtype, request.onProgress);
  const module = await loadTransformerModule();
  request.onProgress?.({ status: 'template' });
  let prompt: string;
  try {
    prompt = processor.apply_chat_template(
      [
        {
          role: 'user',
          content: request.messages[0]?.content ?? [{ type: 'text', text: request.prompt }],
        },
      ],
      {
        enable_thinking: false,
        add_generation_prompt: true,
      }
    );
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'gemma-lab',
      stage: 'apply_chat_template',
      error,
    });
  }

  return runGemmaOneTokenSmoke({
    module,
    processor,
    model,
    route: 'gemma-lab',
    prompt,
    images: request.images,
    audio: request.audio,
    onProgress: request.onProgress,
  }) as Promise<GemmaBrowserSelfTestResult>;
}

export function getGemmaLabImageDiagnostics() {
  return getLastImageDiagnostics('gemma-lab');
}

export function getGemmaLabRuntimeDiagnostics() {
  return getLastGemmaBrowserDiagnostics('gemma-lab');
}
