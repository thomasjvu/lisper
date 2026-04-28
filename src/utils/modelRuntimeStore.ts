import { useSyncExternalStore } from 'react';

import { BROWSER_MODEL_ID, INFERENCE_MODEL_LABEL, INFERENCE_SERVICE_URL, USE_REMOTE_INFERENCE } from './appConfig';
import type { LispType } from './gameState';

export type ModelPhase = 'idle' | 'unsupported' | 'loading' | 'ready' | 'error';
export type LoadSource = 'cold-network' | 'partial-network' | 'warm-cache' | 'remote-server';
export type RuntimeKind = 'browser' | 'remote';

export interface BrowserCapability {
  webgpu: boolean;
  mediaDevices: boolean;
  mediaRecorder: boolean;
  camera: boolean;
  microphone: boolean;
  canvasCapture: boolean;
  fileUpload: boolean;
  reason?: string;
}

export interface ModelStatus {
  phase: ModelPhase;
  label: string;
  progress: number | null;
  error?: string;
  modelId: string;
  runtimeKind: RuntimeKind;
  serviceUrl: string | null;
  warm: boolean;
  capability: BrowserCapability;
  totalProgress: number | null;
  totalBytesLoaded: number | null;
  totalBytesExpected: number | null;
  currentFileLabel: string | null;
  currentFileProgress: number | null;
  currentFileBytesLoaded: number | null;
  currentFileBytesExpected: number | null;
  filesCompleted: number;
  filesTotal: number;
  downloadActive: boolean;
  loadSource: LoadSource;
  cacheAvailable: boolean;
  cacheComplete: boolean;
  missingFiles: string[];
  cacheOrigin: string | null;
}

export interface PracticeCapture {
  audioBlob: Blob;
  frames: Blob[];
  durationMs: number;
  source: 'recording' | 'upload';
}

export interface AssessmentResult {
  lispType: LispType;
  severity: number;
  notes: string;
  mouthShapeNotes: string;
  confidence: number;
  sampledFrameCount: number;
}

export interface CoachResult {
  feedback: string;
  encouragement: string;
  nextTryCue: string;
}

export interface PracticeAnalysisResult {
  transcript: string;
  assessment: AssessmentResult;
  coaching: CoachResult;
}

export const MODEL_ID = BROWSER_MODEL_ID;

const listeners = new Set<() => void>();
let emitTimeout: ReturnType<typeof setTimeout> | null = null;
let status: ModelStatus = {
  phase: 'idle',
  label: 'idle',
  progress: null,
  modelId: USE_REMOTE_INFERENCE ? INFERENCE_MODEL_LABEL : MODEL_ID,
  runtimeKind: USE_REMOTE_INFERENCE ? 'remote' : 'browser',
  serviceUrl: USE_REMOTE_INFERENCE ? INFERENCE_SERVICE_URL : null,
  warm: false,
  capability: detectBrowserCapability(),
  totalProgress: null,
  totalBytesLoaded: null,
  totalBytesExpected: null,
  currentFileLabel: null,
  currentFileProgress: null,
  currentFileBytesLoaded: null,
  currentFileBytesExpected: null,
  filesCompleted: 0,
  filesTotal: 0,
  downloadActive: false,
  loadSource: USE_REMOTE_INFERENCE ? 'remote-server' : 'cold-network',
  cacheAvailable: false,
  cacheComplete: false,
  missingFiles: [],
  cacheOrigin: getCacheOrigin(),
};

function emit(immediate = false) {
  if (immediate) {
    if (emitTimeout) {
      clearTimeout(emitTimeout);
      emitTimeout = null;
    }
    listeners.forEach((listener) => listener());
    return;
  }

  if (emitTimeout) {
    return;
  }

  emitTimeout = setTimeout(() => {
    emitTimeout = null;
    listeners.forEach((listener) => listener());
  }, 90);
}

export function updateModelStatus(next: Partial<ModelStatus>, immediate = false) {
  status = {
    ...status,
    ...next,
  };
  emit(immediate);
}

export function getCacheOrigin() {
  return typeof window !== 'undefined' ? window.location.origin : null;
}

export function detectBrowserCapability(): BrowserCapability {
  if (typeof navigator === 'undefined') {
    return {
      webgpu: false,
      mediaDevices: false,
      mediaRecorder: false,
      camera: false,
      microphone: false,
      canvasCapture: false,
      fileUpload: false,
      reason: 'Browser APIs are unavailable in this environment.',
    };
  }

  const mediaDevices = !!navigator.mediaDevices?.getUserMedia;
  const webgpu = 'gpu' in navigator;
  const mediaRecorder = typeof MediaRecorder !== 'undefined';
  const fileUpload = typeof window !== 'undefined';
  const canvasCapture = typeof document !== 'undefined' && !!document.createElement('canvas').getContext;

  return {
    webgpu,
    mediaDevices,
    mediaRecorder,
    camera: mediaDevices,
    microphone: mediaDevices,
    canvasCapture,
    fileUpload,
    reason: webgpu ? undefined : 'WebGPU is required to run Gemma 4 in the browser.',
  };
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useModelStatus() {
  return useSyncExternalStore(subscribe, () => status, () => status);
}

export function getModelStatus() {
  return status;
}
