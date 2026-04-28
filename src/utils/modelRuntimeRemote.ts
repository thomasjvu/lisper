import { INFERENCE_SERVICE_URL, INFERENCE_MODEL_LABEL } from './appConfig';
import { getDefaultEncouragement, getDefaultFeedback, getDefaultTip } from './speechService';
import { detectBrowserCapability, getModelStatus, updateModelStatus } from './modelRuntimeStore';
import type {
  AssessmentResult,
  CoachResult,
  ModelStatus,
  PracticeAnalysisResult,
  PracticeCapture,
} from './modelRuntimeStore';

interface RemoteHealthResponse {
  status: string;
  loaded: boolean;
  adapter_dir: string;
  base_model: string;
  model_artifact_kind: 'adapter' | 'merged';
  merged_model_path?: string | null;
  merged_model_valid?: boolean;
}

interface RemoteAnalyzeResponse {
  transcript: string;
  assessment: AssessmentResult;
  coaching: CoachResult;
  raw_response?: string;
}

let warmPromise: Promise<ModelStatus> | null = null;

function getServiceUrl(path: string) {
  if (!INFERENCE_SERVICE_URL) {
    throw new Error('Remote inference is not configured. Set VITE_LISPER_INFERENCE_URL to use the trained model.');
  }

  return `${INFERENCE_SERVICE_URL}${path}`;
}

function buildFallbackResult(targetText: string): PracticeAnalysisResult {
  const assessment: AssessmentResult = {
    lispType: 'frontal',
    severity: 5,
    confidence: 0.35,
    sampledFrameCount: 0,
    notes: 'The remote Lisper model did not return a complete response, so this is a safe fallback.',
    mouthShapeNotes: 'Remote inference is audio-only in this build, so no frame-based mouth-shape analysis is available.',
  };

  return {
    transcript: targetText.trim(),
    assessment,
    coaching: {
      feedback: getDefaultFeedback(assessment.lispType, assessment.severity),
      encouragement: getDefaultEncouragement(55, 0),
      nextTryCue: getDefaultTip(targetText.replace(/[^a-z]/gi, '').charAt(0).toLowerCase() || 's', assessment.lispType),
    },
  };
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      payload && typeof payload === 'object' && payload && 'detail' in payload && typeof payload.detail === 'string'
        ? payload.detail
        : `${response.status} ${response.statusText}`;
    throw new Error(`Remote Lisper inference failed: ${detail}`);
  }

  return payload as T;
}

export async function ensureReady(): Promise<ModelStatus> {
  if (!warmPromise) {
    warmPromise = (async () => {
      updateModelStatus(
        {
          phase: 'loading',
          label: 'connecting to trained Lisper model',
          error: undefined,
          progress: null,
          warm: false,
          runtimeKind: 'remote',
          serviceUrl: INFERENCE_SERVICE_URL || null,
          modelId: INFERENCE_MODEL_LABEL,
          loadSource: 'remote-server',
          cacheAvailable: false,
          cacheComplete: false,
          missingFiles: [],
          capability: detectBrowserCapability(),
        },
        true
      );

      try {
        const response = await fetch(getServiceUrl('/warm'), {
          method: 'POST',
        });
        const health = await readJson<RemoteHealthResponse>(response);
        const summary =
          health.model_artifact_kind === 'merged'
            ? health.merged_model_valid
              ? 'Remote merged Gemma checkpoint is ready.'
              : 'Remote service is ready, but the merged export is not currently valid.'
            : 'Remote Gemma base + Lisper adapter is ready.';

        updateModelStatus(
          {
            phase: 'ready',
            label: summary,
            error: undefined,
            warm: true,
            progress: 100,
            runtimeKind: 'remote',
            serviceUrl: INFERENCE_SERVICE_URL || null,
            modelId: INFERENCE_MODEL_LABEL,
            loadSource: 'remote-server',
            cacheAvailable: false,
            cacheComplete: false,
            missingFiles: [],
          },
          true
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Remote Lisper warmup failed.';
        updateModelStatus(
          {
            phase: 'error',
            label: 'remote model unavailable',
            error: message,
            warm: false,
            runtimeKind: 'remote',
            serviceUrl: INFERENCE_SERVICE_URL || null,
            modelId: INFERENCE_MODEL_LABEL,
            loadSource: 'remote-server',
          },
          true
        );
        warmPromise = null;
        throw error;
      }

      return getModelStatus();
    })();
  }

  return warmPromise;
}

export async function analyzePractice(capture: PracticeCapture, targetText: string): Promise<PracticeAnalysisResult> {
  await ensureReady();

  const formData = new FormData();
  formData.append('audio', capture.audioBlob, 'attempt.wav');
  formData.append('target_text', targetText);
  formData.append('duration_ms', String(capture.durationMs));
  formData.append('frame_count', String(capture.frames.length));

  const response = await fetch(getServiceUrl('/analyze'), {
    method: 'POST',
    body: formData,
  });

  const payload = await readJson<RemoteAnalyzeResponse>(response);
  if (!payload?.assessment || !payload?.coaching) {
    return buildFallbackResult(targetText);
  }

  return {
    transcript: payload.transcript?.trim() || targetText.trim(),
    assessment: {
      ...payload.assessment,
      sampledFrameCount: capture.frames.length,
      mouthShapeNotes:
        payload.assessment.mouthShapeNotes ||
        'Remote inference is audio-only in this build, so no frame-based mouth-shape analysis is available.',
    },
    coaching: payload.coaching,
  };
}

export async function transcribe(capture: PracticeCapture): Promise<{ text: string }> {
  const result = await analyzePractice(capture, '');
  return { text: result.transcript };
}

export async function assess(capture: PracticeCapture): Promise<AssessmentResult> {
  const result = await analyzePractice(capture, '');
  return result.assessment;
}

export async function coach(input: {
  targetText: string;
  transcript: string;
  assessment: AssessmentResult;
}): Promise<CoachResult> {
  return {
    feedback: getDefaultFeedback(input.assessment.lispType, input.assessment.severity),
    encouragement: getDefaultEncouragement(60, 0),
    nextTryCue: getDefaultTip(input.targetText.replace(/[^a-z]/gi, '').charAt(0).toLowerCase() || 's', input.assessment.lispType),
  };
}

export async function selfTestImagePipeline() {
  throw new Error('Image self-test is only available for the in-browser Gemma runtime.');
}

export function getImagePipelineDiagnostics() {
  return [];
}
