export type {
  AssessmentResult,
  BrowserCapability,
  CoachResult,
  LoadSource,
  ModelPhase,
  ModelStatus,
  PracticeAnalysisResult,
  PracticeCapture,
  RuntimeKind,
} from './modelRuntimeStore';
export { detectBrowserCapability, getModelStatus, useModelStatus } from './modelRuntimeStore';

import { USE_REMOTE_INFERENCE } from './appConfig';
import type {
  AssessmentResult,
  CoachResult,
  ModelStatus,
  PracticeAnalysisResult,
  PracticeCapture,
} from './modelRuntimeStore';

let corePromise:
  | Promise<typeof import('./modelRuntimeCore')>
  | Promise<typeof import('./modelRuntimeRemote')>
  | null = null;

async function loadModelRuntimeCore() {
  if (!corePromise) {
    corePromise = USE_REMOTE_INFERENCE ? import('./modelRuntimeRemote') : import('./modelRuntimeCore');
  }

  return corePromise;
}

export function preloadModelRuntime() {
  return loadModelRuntimeCore();
}

export async function ensureReady(): Promise<ModelStatus> {
  const core = await loadModelRuntimeCore();
  return core.ensureReady();
}

export async function transcribe(capture: PracticeCapture): Promise<{ text: string }> {
  const core = await loadModelRuntimeCore();
  return core.transcribe(capture);
}

export async function assess(capture: PracticeCapture): Promise<AssessmentResult> {
  const core = await loadModelRuntimeCore();
  return core.assess(capture);
}

export async function coach(input: {
  targetText: string;
  transcript: string;
  assessment: AssessmentResult;
}): Promise<CoachResult> {
  const core = await loadModelRuntimeCore();
  return core.coach(input);
}

export async function analyzePractice(capture: PracticeCapture, targetText: string): Promise<PracticeAnalysisResult> {
  const core = await loadModelRuntimeCore();
  return core.analyzePractice(capture, targetText);
}

export async function selfTestImagePipeline(): Promise<{
  ok: boolean;
  route: string;
  details: string;
  diagnostics: unknown[];
}> {
  const core = (await loadModelRuntimeCore()) as {
    selfTestImagePipeline: () => Promise<{
      ok: boolean;
      route: string;
      details: string;
      diagnostics: unknown[];
    }>;
  };
  return core.selfTestImagePipeline();
}

export async function getImagePipelineDiagnostics(): Promise<unknown[]> {
  const core = (await loadModelRuntimeCore()) as {
    getImagePipelineDiagnostics: () => unknown[];
  };
  return core.getImagePipelineDiagnostics();
}
