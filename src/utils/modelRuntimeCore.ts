import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';

import { BROWSER_MODEL_DTYPE } from './appConfig';
import {
  buildGemmaBrowserInputs,
  formatGemmaBrowserRuntimeError,
  getGemmaBrowserFloat16BridgeDiagnostics,
  getLastImageDiagnostics,
  patchGemmaBrowserFloat16Feeds,
  runGemmaImagePipelineSelfTest,
  summarizeGemmaInputs,
} from './gemmaBrowserMultimodal';
import {
  MODEL_ID,
  detectBrowserCapability,
  getCacheOrigin,
  getModelStatus,
  updateModelStatus,
} from './modelRuntimeStore';
import type {
  AssessmentResult,
  CoachResult,
  LoadSource,
  ModelStatus,
  PracticeAnalysisResult,
  PracticeCapture,
} from './modelRuntimeStore';
import { getDefaultEncouragement, getDefaultFeedback, getDefaultTip } from './speechService';

interface CacheManifest {
  origin: string;
  modelId: string;
  modelFiles: string[];
  runtimeFiles: string[];
  savedAt: string;
}

interface CacheStateInspection {
  cacheAvailable: boolean;
  cacheComplete: boolean;
  loadSource: LoadSource;
  missingFiles: string[];
  cacheOrigin: string | null;
}

const CACHE_KEY = 'transformers-cache';
const MANIFEST_KEY = `lisper:model-manifest:v2:${MODEL_ID}`;

let runtimeModule: any = null;
let processor: any = null;
let model: any = null;
let ensurePromise: Promise<ModelStatus> | null = null;

const fileProgress = new Map<string, { file: string; loaded: number; total: number; done: boolean }>();
let missingFileNames = new Set<string>();

function getRuntimeAssetUrls() {
  if (typeof window === 'undefined') {
    return [ortMjsUrl, ortWasmUrl];
  }

  return [new URL(ortMjsUrl, window.location.href).href, new URL(ortWasmUrl, window.location.href).href];
}

function getFileName(value: string | null | undefined) {
  if (!value) {
    return null;
  }

  const name = value.split('/').pop() || value;
  return name.split('?')[0] || name;
}

function shortFileLabel(file: string | null | undefined) {
  const value = getFileName(file);
  if (!value) {
    return null;
  }

  return value.length > 36 ? `${value.slice(0, 33)}...` : value;
}

function getManifestStorageKey(origin: string | null) {
  return `${MANIFEST_KEY}:${origin ?? 'server'}`;
}

function readCacheManifest(origin: string | null): CacheManifest | null {
  if (typeof window === 'undefined' || !window.localStorage || !origin) {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(getManifestStorageKey(origin));
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as CacheManifest;
    if (parsed?.origin !== origin || parsed?.modelId !== MODEL_ID) {
      return null;
    }

    return parsed;
  } catch {
    return null;
  }
}

function writeCacheManifest(manifest: CacheManifest) {
  if (typeof window === 'undefined' || !window.localStorage) {
    return;
  }

  try {
    window.localStorage.setItem(getManifestStorageKey(manifest.origin), JSON.stringify(manifest));
  } catch (error) {
    console.warn('[ModelRuntime] Failed to write cache manifest', error);
  }
}

function isManagedCacheUrl(url: string) {
  return url.includes(MODEL_ID) || getRuntimeAssetUrls().includes(url);
}

async function inspectCacheState(): Promise<CacheStateInspection> {
  const cacheOrigin = getCacheOrigin();
  const base = {
    cacheAvailable: false,
    cacheComplete: false,
    loadSource: 'cold-network' as LoadSource,
    missingFiles: [] as string[],
    cacheOrigin,
  };

  if (typeof caches === 'undefined') {
    return base;
  }

  try {
    const cache = await caches.open(CACHE_KEY);
    const requests = await cache.keys();
    const urls = requests.map((request) => request.url);
    const urlSet = new Set(urls);
    const manifest = readCacheManifest(cacheOrigin);
    const cachedEntries = urls.filter(isManagedCacheUrl);

    if (!manifest) {
      return {
        ...base,
        cacheAvailable: true,
        loadSource: cachedEntries.length ? 'partial-network' : 'cold-network',
      };
    }

    const requiredUrls = Array.from(new Set([...manifest.modelFiles, ...manifest.runtimeFiles]));
    const missingUrls = requiredUrls.filter((url) => !urlSet.has(url));

    if (!requiredUrls.length) {
      return {
        ...base,
        cacheAvailable: true,
        loadSource: cachedEntries.length ? 'partial-network' : 'cold-network',
      };
    }

    if (missingUrls.length === 0) {
      return {
        ...base,
        cacheAvailable: true,
        cacheComplete: true,
        loadSource: 'warm-cache',
      };
    }

    if (missingUrls.length < requiredUrls.length || cachedEntries.length) {
      return {
        ...base,
        cacheAvailable: true,
        loadSource: 'partial-network',
        missingFiles: missingUrls.map((url) => getFileName(url) || url),
      };
    }

    return {
      ...base,
      cacheAvailable: true,
      loadSource: 'cold-network',
    };
  } catch (error) {
    console.warn('[ModelRuntime] Failed to inspect browser cache', error);
    return base;
  }
}

async function captureCacheManifest() {
  const origin = getCacheOrigin();
  if (!origin || typeof caches === 'undefined') {
    return;
  }

  try {
    const cache = await caches.open(CACHE_KEY);
    const urls = (await cache.keys()).map((request) => request.url);
    const modelFiles = urls.filter((url) => url.includes(MODEL_ID));
    const runtimeFiles = getRuntimeAssetUrls().filter((url) => urls.includes(url));

    if (!modelFiles.length && !runtimeFiles.length) {
      return;
    }

    writeCacheManifest({
      origin,
      modelId: MODEL_ID,
      modelFiles,
      runtimeFiles,
      savedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.warn('[ModelRuntime] Failed to capture cache manifest', error);
  }
}

function getFileKey(name: string | undefined, file: string | undefined) {
  return `${name ?? MODEL_ID}::${file ?? 'resource'}`;
}

function clampSeverity(value: number) {
  return Math.max(1, Math.min(10, Math.round(value)));
}

function clampConfidence(value: number) {
  if (!Number.isFinite(value)) {
    return 0.55;
  }

  if (value > 1) {
    return Math.max(0, Math.min(1, value / 100));
  }

  return Math.max(0, Math.min(1, value));
}

function clampProgress(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return null;
  }

  return Math.max(0, Math.min(100, value));
}

function resetDownloadState(loadSource: LoadSource) {
  fileProgress.clear();
  updateModelStatus({
    totalProgress: 0,
    totalBytesLoaded: 0,
    totalBytesExpected: null,
    currentFileLabel: null,
    currentFileProgress: null,
    currentFileBytesLoaded: null,
    currentFileBytesExpected: null,
    filesCompleted: 0,
    filesTotal: 0,
    downloadActive: true,
    progress: 0,
    label:
      loadSource === 'warm-cache'
        ? 'loading Gemma 4 from browser cache'
        : loadSource === 'partial-network'
          ? 'restoring Gemma 4'
          : 'loading Gemma 4',
  });
}

function describeProgressLabel(fileName: string | null, fileLabel: string | null) {
  const status = getModelStatus();

  if (status.loadSource === 'warm-cache') {
    return fileLabel ? `loading ${fileLabel} from browser cache` : 'loading Gemma 4 from browser cache';
  }

  if (status.loadSource === 'partial-network') {
    const missing = fileName ? missingFileNames.has(fileName) : false;
    return missing
      ? fileLabel
        ? `restoring ${fileLabel}`
        : 'restoring missing model files'
      : fileLabel
        ? `loading cached ${fileLabel}`
        : 'loading cached model files';
  }

  return fileLabel ? `downloading ${fileLabel}` : 'downloading Gemma 4';
}

function syncAggregateProgress(partial: Partial<ModelStatus> = {}) {
  const status = getModelStatus();
  const values = Array.from(fileProgress.values());
  const filesTotal = values.length;
  const filesCompleted = values.filter((item) => item.done).length;
  const loaded = values.reduce((sum, item) => sum + (Number.isFinite(item.loaded) ? item.loaded : 0), 0);
  const total = values.reduce((sum, item) => sum + (Number.isFinite(item.total) ? item.total : 0), 0);
  const totalProgress =
    total > 0 ? (loaded / total) * 100 : filesTotal > 0 ? (filesCompleted / filesTotal) * 100 : (partial.totalProgress ?? 0);

  updateModelStatus({
    totalProgress: clampProgress(totalProgress),
    totalBytesLoaded: loaded || partial.totalBytesLoaded || 0,
    totalBytesExpected: total > 0 ? total : (partial.totalBytesExpected ?? null),
    filesCompleted,
    filesTotal,
    progress: clampProgress(totalProgress),
    downloadActive: partial.downloadActive ?? status.phase === 'loading',
    ...partial,
  });
}

function handleProgressInfo(info: any) {
  const fileName = getFileName(info?.file);
  const fileLabel = shortFileLabel(info?.file);
  const key = getFileKey(info?.name, info?.file);

  if (!fileProgress.has(key)) {
    fileProgress.set(key, {
      file: fileLabel || 'resource',
      loaded: 0,
      total: 0,
      done: false,
    });
  }

  const current = fileProgress.get(key)!;
  const label = describeProgressLabel(fileName, fileLabel);

  if (info?.status === 'initiate' || info?.status === 'download') {
    syncAggregateProgress({
      label,
      currentFileLabel: fileLabel,
      downloadActive: true,
    });
    return;
  }

  if (info?.status === 'progress') {
    current.loaded = Number.isFinite(info.loaded) ? info.loaded : current.loaded;
    current.total = Number.isFinite(info.total) ? info.total : current.total;

    syncAggregateProgress({
      currentFileLabel: fileLabel,
      currentFileProgress: clampProgress(info.progress ?? null),
      currentFileBytesLoaded: Number.isFinite(info.loaded) ? info.loaded : null,
      currentFileBytesExpected: Number.isFinite(info.total) ? info.total : null,
      label,
      downloadActive: true,
    });
    return;
  }

  if (info?.status === 'progress_total') {
    if (info.files && typeof info.files === 'object') {
      Object.entries(info.files).forEach(([file, data]: [string, any]) => {
        const aggregateKey = getFileKey(info?.name, file);
        const entry = fileProgress.get(aggregateKey) || {
          file: shortFileLabel(file) || file,
          loaded: 0,
          total: 0,
          done: false,
        };
        entry.loaded = Number.isFinite(data?.loaded) ? data.loaded : entry.loaded;
        entry.total = Number.isFinite(data?.total) ? data.total : entry.total;
        fileProgress.set(aggregateKey, entry);
      });
    }

    syncAggregateProgress({
      totalProgress: clampProgress(info.progress ?? null),
      totalBytesLoaded: Number.isFinite(info.loaded) ? info.loaded : null,
      totalBytesExpected: Number.isFinite(info.total) ? info.total : null,
      label,
      downloadActive: true,
    });
    return;
  }

  if (info?.status === 'done') {
    current.done = true;
    if (!current.total && current.loaded) {
      current.total = current.loaded;
    }

    if (fileName && missingFileNames.has(fileName)) {
      missingFileNames.delete(fileName);
    }

    syncAggregateProgress({
      currentFileLabel: fileLabel,
      currentFileProgress: 100,
      currentFileBytesLoaded: current.loaded || null,
      currentFileBytesExpected: current.total || null,
      label: getModelStatus().loadSource === 'warm-cache' ? 'finalizing cached model' : 'finalizing Gemma 4',
      downloadActive: true,
      missingFiles: Array.from(missingFileNames),
    });
  }
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
    throw new Error('Gemma 4 requires @huggingface/transformers 4.1.0 or newer in this app.');
  }

  return runtimeModule;
}

function stripModelArtifacts(text: string) {
  return text
    .replace(/<\|channel\|>thought[\s\S]*?<\|channel\|>/g, '')
    .replace(/```json/g, '')
    .replace(/```/g, '')
    .trim();
}

function extractJsonBlock<T>(text: string): T | null {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) {
    return null;
  }

  try {
    return JSON.parse(match[0]) as T;
  } catch {
    return null;
  }
}

function normalizeLispType(value: unknown): AssessmentResult['lispType'] {
  const input = String(value ?? '').toLowerCase();
  if (input.includes('lateral')) {
    return 'lateral';
  }
  if (input.includes('dental')) {
    return 'dental';
  }
  if (input.includes('palatal')) {
    return 'palatal';
  }
  return 'frontal';
}

async function decodeOutput(inputs: any, outputs: any) {
  const promptLength = inputs?.input_ids?.dims?.at?.(-1) ?? null;
  const generated = promptLength ? outputs.slice(null, [promptLength, null]) : outputs;
  const decoded = processor.batch_decode(generated, {
    skip_special_tokens: true,
  });

  return stripModelArtifacts(decoded?.[0] ?? '');
}

async function generateFromMessages(messages: any[], options?: { audio?: Blob; images?: Blob[]; maxNewTokens?: number }) {
  await ensureReady();

  const module = await loadTransformerModule();
  let prompt: string;
  try {
    prompt = processor.apply_chat_template(messages, {
      enable_thinking: false,
      add_generation_prompt: true,
    });
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'main-app',
      stage: 'apply_chat_template',
      error,
    });
  }

  let inputs: any;
  try {
    if (options?.audio || options?.images?.length) {
      inputs = await buildGemmaBrowserInputs({
        module,
        processor,
        prompt,
        route: 'main-app',
        images: options.images,
        audio: options.audio,
      });
    } else {
      inputs = await processor(prompt, null, null, {
        add_special_tokens: false,
      });
    }
  } catch (error) {
    throw error;
  }

  let outputs: any;
  try {
    outputs = await model.generate({
      ...inputs,
      max_new_tokens: options?.maxNewTokens ?? 180,
      do_sample: false,
      temperature: 1.0,
      top_p: 0.95,
      top_k: 64,
    });
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'main-app',
      stage: 'model.generate',
      error,
      inputSummary: summarizeGemmaInputs(inputs),
      float16Bridge: getGemmaBrowserFloat16BridgeDiagnostics(model),
    });
  }

  try {
    return decodeOutput(inputs, outputs);
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route: 'main-app',
      stage: 'decode output',
      error,
      inputSummary: summarizeGemmaInputs(inputs),
      float16Bridge: getGemmaBrowserFloat16BridgeDiagnostics(model),
    });
  }
}

export async function selfTestImagePipeline() {
  await ensureReady();
  const module = await loadTransformerModule();
  return runGemmaImagePipelineSelfTest({
    module,
    processor,
    route: 'main-app',
  });
}

export function getImagePipelineDiagnostics() {
  return getLastImageDiagnostics('main-app');
}

function buildAssessmentMessages(capture: PracticeCapture) {
  const content: Array<{ type: 'image' | 'audio' | 'text'; text?: string }> = [
    ...capture.frames.map(() => ({ type: 'image' as const })),
    { type: 'audio' as const },
    {
      type: 'text' as const,
      text: `Review this short speech practice attempt. Use the sampled mouth frames and the audio together.
Return strict JSON only with keys lispType, severity, notes, mouthShapeNotes, confidence.

Rules:
- lispType must be one of frontal, lateral, dental, palatal.
- severity must be an integer from 1 to 10.
- notes must be one short sentence about the speech result.
- mouthShapeNotes must be one short sentence about visible lip, jaw, or tongue-placement cues.
- confidence must be a number from 0 to 1.
- If visual evidence is weak or unavailable, say that plainly in mouthShapeNotes.`,
    },
  ];

  return [
    {
      role: 'system',
      content:
        'You are a speech assessment assistant for lisp practice. Be cautious, concise, and return strict JSON only.',
    },
    {
      role: 'user',
      content,
    },
  ];
}

function buildPracticeMessages(capture: PracticeCapture, targetText: string) {
  const content: Array<{ type: 'image' | 'audio' | 'text'; text?: string }> = [
    ...capture.frames.map(() => ({ type: 'image' as const })),
    { type: 'audio' as const },
    {
      type: 'text' as const,
      text: `You are reviewing one short lisp-practice attempt for the target phrase "${targetText || 'baseline speech sample'}".
Return strict JSON only with keys transcript, lispType, severity, notes, mouthShapeNotes, confidence, feedback, encouragement, nextTryCue.

Rules:
- transcript must be the spoken English transcript only, with no line breaks.
- lispType must be one of frontal, lateral, dental, palatal.
- severity must be an integer from 1 to 10.
- notes must be one short sentence about the speech result.
- mouthShapeNotes must be one short sentence about visible lip, jaw, or tongue-placement cues.
- confidence must be a number from 0 to 1.
- feedback must be one short coaching sentence tied to the attempt.
- encouragement must be one short calm sentence.
- nextTryCue must be one short actionable cue for the next repetition.
- If visual evidence is weak or unavailable, say that plainly in mouthShapeNotes.
- Output JSON only.`,
    },
  ];

  return [
    {
      role: 'system',
      content:
        'You are a low-anxiety multimodal speech coach for lisp practice. Be concise, cautious, and return strict JSON only.',
    },
    {
      role: 'user',
      content,
    },
  ];
}

function getFallbackNextTryCue(targetText: string, assessment: AssessmentResult) {
  const fallbackSound = targetText.replace(/[^a-z]/gi, '').charAt(0).toLowerCase() || 's';
  return getDefaultTip(fallbackSound, assessment.lispType);
}

function getFallbackAssessment(
  capture: PracticeCapture,
  parsed?: {
    lispType?: string;
    severity?: number;
    notes?: string;
    mouthShapeNotes?: string;
    confidence?: number;
  }
): AssessmentResult {
  return {
    lispType: normalizeLispType(parsed?.lispType),
    severity: clampSeverity(Number(parsed?.severity ?? 5)),
    notes: parsed?.notes?.trim() || 'Gemma 4 returned a partial speech assessment, so this is a best-effort interpretation.',
    mouthShapeNotes:
      parsed?.mouthShapeNotes?.trim() ||
      (capture.frames.length
        ? 'Visible articulation cues were weak, so use this as a rough mouth-shape hint only.'
        : 'Visual cues were unavailable because this attempt did not include sampled mouth frames.'),
    confidence: clampConfidence(Number(parsed?.confidence ?? 0.55)),
    sampledFrameCount: capture.frames.length,
  };
}

export async function ensureReady(): Promise<ModelStatus> {
  const capability = detectBrowserCapability();
  const cacheState = await inspectCacheState();
  missingFileNames = new Set(cacheState.missingFiles);

  if (!capability.webgpu) {
    updateModelStatus(
      {
        phase: 'unsupported',
        label: 'unsupported',
        progress: null,
        warm: false,
        capability,
        error: capability.reason,
        downloadActive: false,
        cacheAvailable: cacheState.cacheAvailable,
        cacheComplete: cacheState.cacheComplete,
        loadSource: cacheState.loadSource,
        missingFiles: cacheState.missingFiles,
        cacheOrigin: cacheState.cacheOrigin,
      },
      true
    );
    return getModelStatus();
  }

  const status = getModelStatus();
  if (status.phase === 'ready' && processor && model) {
    return status;
  }

  if (ensurePromise) {
    return ensurePromise;
  }

  resetDownloadState(cacheState.loadSource);
  updateModelStatus(
    {
      phase: 'loading',
      label:
        cacheState.loadSource === 'warm-cache'
          ? 'loading Gemma 4 from browser cache'
          : cacheState.loadSource === 'partial-network'
            ? 'restoring Gemma 4'
            : 'loading Gemma 4',
      progress: 0,
      warm: false,
      error: undefined,
      capability,
      loadSource: cacheState.loadSource,
      cacheAvailable: cacheState.cacheAvailable,
      cacheComplete: cacheState.cacheComplete,
      missingFiles: cacheState.missingFiles,
      cacheOrigin: cacheState.cacheOrigin,
    },
    true
  );

  ensurePromise = (async () => {
    try {
      const module = await loadTransformerModule();
      const progress_callback = (info: any) => handleProgressInfo(info);

      processor = await module.Gemma4Processor.from_pretrained(MODEL_ID, {
        progress_callback,
      });
      model = await module.Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
        dtype: BROWSER_MODEL_DTYPE,
        device: 'webgpu',
        progress_callback,
      });
      patchGemmaBrowserFloat16Feeds(module, model);

      await captureCacheManifest();
      missingFileNames.clear();

      updateModelStatus(
        {
          phase: 'ready',
          label: 'Gemma 4 ready',
          progress: 100,
          warm: true,
          error: undefined,
          totalProgress: 100,
          currentFileProgress: 100,
          downloadActive: false,
          cacheComplete: true,
          missingFiles: [],
        },
        true
      );

      return getModelStatus();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown Gemma 4 load error';
      updateModelStatus(
        {
          phase: 'error',
          label: 'Gemma 4 failed',
          progress: null,
          warm: false,
          error: message,
          downloadActive: false,
        },
        true
      );
      throw error;
    } finally {
      ensurePromise = null;
    }
  })();

  return ensurePromise;
}

export async function transcribe(capture: PracticeCapture): Promise<{ text: string }> {
  const text = await generateFromMessages(
    [
      {
        role: 'user',
        content: [
          { type: 'audio' },
          {
            type: 'text',
            text:
              'Transcribe the following speech segment in English into English text. Only output the transcription, with no newlines.',
          },
        ],
      },
    ],
    { audio: capture.audioBlob, maxNewTokens: 128 }
  );

  return { text: text.trim() };
}

export async function assess(capture: PracticeCapture): Promise<AssessmentResult> {
  const response = await generateFromMessages(buildAssessmentMessages(capture), {
    audio: capture.audioBlob,
    images: capture.frames,
    maxNewTokens: 180,
  });

  const parsed = extractJsonBlock<{
    lispType?: string;
    severity?: number;
    notes?: string;
    mouthShapeNotes?: string;
    confidence?: number;
  }>(response);

  return getFallbackAssessment(capture, parsed ?? undefined);
}

export async function coach(input: {
  targetText: string;
  transcript: string;
  assessment: AssessmentResult;
}): Promise<CoachResult> {
  const response = await generateFromMessages(
    [
      {
        role: 'system',
        content:
          'You are a low-anxiety speech coach. Return strict JSON only with keys feedback, encouragement, nextTryCue. Keep each value short and specific.',
      },
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Target phrase: ${input.targetText}
Transcript: ${input.transcript}
Lisp type: ${input.assessment.lispType}
Severity: ${input.assessment.severity}
Speech result: ${input.assessment.notes}
Visible articulation: ${input.assessment.mouthShapeNotes}
Confidence: ${input.assessment.confidence}

Return JSON only.`,
          },
        ],
      },
    ],
    { maxNewTokens: 160 }
  );

  const parsed = extractJsonBlock<{ feedback?: string; encouragement?: string; nextTryCue?: string }>(response);
  return {
    feedback:
      parsed?.feedback?.trim() || getDefaultFeedback(input.assessment.lispType, input.assessment.severity),
    encouragement:
      parsed?.encouragement?.trim() ||
      getDefaultEncouragement(Math.max(40, 100 - input.assessment.severity * 5), 1),
    nextTryCue: parsed?.nextTryCue?.trim() || getFallbackNextTryCue(input.targetText, input.assessment),
  };
}

export async function analyzePractice(capture: PracticeCapture, targetText: string): Promise<PracticeAnalysisResult> {
  const response = await generateFromMessages(buildPracticeMessages(capture, targetText), {
    audio: capture.audioBlob,
    images: capture.frames,
    maxNewTokens: 240,
  });

  const parsed = extractJsonBlock<{
    transcript?: string;
    lispType?: string;
    severity?: number;
    notes?: string;
    mouthShapeNotes?: string;
    confidence?: number;
    feedback?: string;
    encouragement?: string;
    nextTryCue?: string;
  }>(response);

  const assessment = getFallbackAssessment(capture, parsed ?? undefined);

  return {
    transcript: parsed?.transcript?.trim() || '',
    assessment,
    coaching: {
      feedback: parsed?.feedback?.trim() || getDefaultFeedback(assessment.lispType, assessment.severity),
      encouragement:
        parsed?.encouragement?.trim() ||
        getDefaultEncouragement(Math.max(40, 100 - assessment.severity * 5), 1),
      nextTryCue: parsed?.nextTryCue?.trim() || getFallbackNextTryCue(targetText, assessment),
    },
  };
}
