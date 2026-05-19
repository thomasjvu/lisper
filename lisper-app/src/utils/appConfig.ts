export const DOCS_URL = 'https://github.com/thomasjvu/lisper-docs';

const rawInferenceUrl = import.meta.env.VITE_LISPER_INFERENCE_URL?.trim() || '';
const rawModelLabel = import.meta.env.VITE_LISPER_MODEL_LABEL?.trim() || '';
const rawBrowserModelId = import.meta.env.VITE_LISPER_BROWSER_MODEL_ID?.trim() || '';
const rawBrowserDtype = import.meta.env.VITE_LISPER_BROWSER_DTYPE?.trim() || '';
const rawAllowUnvalidatedBrowserModel =
  import.meta.env.VITE_LISPER_ALLOW_UNVALIDATED_BROWSER_MODEL?.trim().toLowerCase() || '';

export const PUBLIC_BASE_BROWSER_MODEL_ID = 'onnx-community/gemma-4-E2B-it-ONNX';
export const TRAINED_BROWSER_MODEL_ID = 'thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16';
export const TRAINED_BROWSER_MODEL_READY = true;
export const DEFAULT_BROWSER_MODEL_ID = TRAINED_BROWSER_MODEL_ID;
export const DEFAULT_BROWSER_DTYPE = 'q4f16';
export const EXPERIMENTAL_Q2F16_BROWSER_DTYPE = 'q2f16-experimental';
export const ALLOW_UNVALIDATED_BROWSER_MODEL =
  rawAllowUnvalidatedBrowserModel === '1' || rawAllowUnvalidatedBrowserModel === 'true';

export type BrowserModelDtype = 'q4f16' | Record<string, 'q2f16' | 'q4f16'>;

const experimentalQ2F16BrowserDtype: BrowserModelDtype = {
  decoder_model_merged: 'q2f16',
  embed_tokens: 'q4f16',
  audio_encoder: 'q4f16',
  vision_encoder: 'q4f16',
};

const supportedBrowserDtypes = new Set(['q4f16', EXPERIMENTAL_Q2F16_BROWSER_DTYPE]);
const requestedBrowserModelId = rawBrowserModelId || DEFAULT_BROWSER_MODEL_ID;
const requestedTrainedBrowserModel = requestedBrowserModelId === TRAINED_BROWSER_MODEL_ID;
const browserModelBlocked =
  requestedTrainedBrowserModel && !TRAINED_BROWSER_MODEL_READY && !ALLOW_UNVALIDATED_BROWSER_MODEL;
const requestedBrowserDtype = supportedBrowserDtypes.has(rawBrowserDtype) ? rawBrowserDtype : DEFAULT_BROWSER_DTYPE;

export const INFERENCE_SERVICE_URL = rawInferenceUrl.replace(/\/+$/, '');
export const USE_REMOTE_INFERENCE = Boolean(INFERENCE_SERVICE_URL);
export const BROWSER_MODEL_OVERRIDE_BLOCKED = browserModelBlocked;
export const BROWSER_MODEL_ID = BROWSER_MODEL_OVERRIDE_BLOCKED ? PUBLIC_BASE_BROWSER_MODEL_ID : requestedBrowserModelId;
export const BROWSER_MODEL_DTYPE: BrowserModelDtype =
  requestedBrowserDtype === EXPERIMENTAL_Q2F16_BROWSER_DTYPE ? experimentalQ2F16BrowserDtype : DEFAULT_BROWSER_DTYPE;
export const BROWSER_MODEL_DTYPE_LABEL =
  requestedBrowserDtype === EXPERIMENTAL_Q2F16_BROWSER_DTYPE
    ? 'q2f16 decoder + q4f16 embed/audio/vision'
    : DEFAULT_BROWSER_DTYPE;
export const BROWSER_MODEL_IS_PUBLIC_BASE = BROWSER_MODEL_ID === PUBLIC_BASE_BROWSER_MODEL_ID;
export const BROWSER_MODEL_FALLBACK_REASON = BROWSER_MODEL_OVERRIDE_BLOCKED
  ? `The trained browser ONNX repo is not validated yet, so this build is using ${PUBLIC_BASE_BROWSER_MODEL_ID}.`
  : null;
export const INFERENCE_MODEL_LABEL =
  rawModelLabel ||
  (USE_REMOTE_INFERENCE ? 'thomasjvu/lisper-gemma4-e2b-audio-full (remote merged)' : BROWSER_MODEL_ID);
