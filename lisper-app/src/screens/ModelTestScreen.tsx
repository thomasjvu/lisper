import { useState } from 'react';
import type { CSSProperties } from 'react';

import AudioClipInput from '../components/AudioClipInput';
import ModelDownloadPanel from '../components/ModelDownloadPanel';
import SampledFramesStrip from '../components/SampledFramesStrip';
import { BROWSER_MODEL_DTYPE_LABEL, BROWSER_MODEL_FALLBACK_REASON, BROWSER_MODEL_IS_PUBLIC_BASE } from '../utils/appConfig';
import type { AssessmentResult, CoachResult, PracticeCapture } from '../utils/modelRuntime';
import { analyzePractice, ensureReady, getImagePipelineDiagnostics, selfTestImagePipeline, useModelStatus } from '../utils/modelRuntime';

interface ModelTestScreenProps {
  onBack: () => void;
}

export default function ModelTestScreen({ onBack }: ModelTestScreenProps) {
  const modelStatus = useModelStatus();

  const [capture, setCapture] = useState<PracticeCapture | null>(null);
  const [targetText, setTargetText] = useState('sun');
  const [transcriptText, setTranscriptText] = useState('');
  const [assessmentResult, setAssessmentResult] = useState<AssessmentResult | null>(null);
  const [coaching, setCoaching] = useState<CoachResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageSelfTest, setImageSelfTest] = useState<string | null>(null);

  async function warmModel() {
    try {
      setBusy(true);
      setError(null);
      await ensureReady();
    } catch (warmError) {
      setError(warmError instanceof Error ? warmError.message : 'Gemma 4 warmup failed.');
    } finally {
      setBusy(false);
    }
  }

  async function runMultimodalFlow() {
    if (!capture) {
      setError('Attach an attempt first.');
      return;
    }

    try {
      setBusy(true);
      setError(null);

      const result = await analyzePractice(capture, targetText);
      setTranscriptText(result.transcript);
      setAssessmentResult(result.assessment);
      setCoaching(result.coaching);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : 'Gemma 4 analysis failed.');
    } finally {
      setBusy(false);
    }
  }

  async function runImageSelfTest() {
    try {
      setBusy(true);
      setError(null);
      const result = await selfTestImagePipeline();
      setImageSelfTest(`${result.details}\n${JSON.stringify(result.diagnostics, null, 2)}`);
    } catch (runError) {
      const diagnostics = await getImagePipelineDiagnostics();
      const message = runError instanceof Error ? runError.message : 'Gemma 4 image self-test failed.';
      setError(`${message}\n${JSON.stringify(diagnostics, null, 2)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section style={styles.container}>
      <div style={styles.content}>
        <header style={styles.header}>
          <button type="button" style={styles.backButton} onClick={onBack}>
            ← back
          </button>
          <h1 style={styles.title}>Gemma runtime diagnostics</h1>
        </header>

        <ModelDownloadPanel status={modelStatus} />

        <section style={styles.panel}>
          <label htmlFor="target-text" style={styles.panelLabel}>Target phrase</label>
          <input
            id="target-text"
            value={targetText}
            onChange={(event) => setTargetText(event.target.value)}
            style={styles.input}
            placeholder="sun"
          />
          <div style={styles.buttonRow}>
            <button type="button" style={styles.primaryButton} onClick={warmModel} disabled={busy}>
              <span style={styles.primaryButtonText}>{busy ? 'warming...' : 'prepare Gemma 4'}</span>
            </button>
            <button type="button" style={styles.secondaryButton} onClick={runImageSelfTest} disabled={busy}>
              <span style={styles.secondaryButtonText}>{busy ? 'testing...' : 'image self-test'}</span>
            </button>
          </div>
        </section>

        <AudioClipInput
          label="debug attempt"
          capture={capture}
          analyzing={busy}
          onCaptureReady={(nextCapture) => {
            setCapture(nextCapture);
            setTranscriptText('');
            setAssessmentResult(null);
            setCoaching(null);
            setError(null);
          }}
        />

        <button
          type="button"
          style={{ ...styles.primaryButton, ...((!capture || busy) ? styles.buttonDisabled : null) }}
          onClick={runMultimodalFlow}
          disabled={!capture || busy}
        >
          <span style={styles.primaryButtonText}>{busy ? 'processing...' : 'run transcript + lip coaching'}</span>
        </button>

        {error ? <div style={styles.error}>{error}</div> : null}

        {imageSelfTest ? (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Image pipeline</div>
            <pre style={styles.outputText}>{imageSelfTest}</pre>
          </section>
        ) : null}

        {capture ? (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Capture</div>
            <pre style={styles.outputText}>
{`source: ${capture.source}
duration: ${Math.max(1, Math.round(capture.durationMs / 1000))}s
sampled frames: ${capture.frames.length}`}
            </pre>
            <SampledFramesStrip frames={capture.frames} />
          </section>
        ) : null}

        {transcriptText ? (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Transcript</div>
            <pre style={styles.outputText}>{transcriptText}</pre>
          </section>
        ) : null}

        {assessmentResult ? (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Assessment</div>
            <pre style={styles.outputText}>
              {JSON.stringify(
                {
                  lispType: assessmentResult.lispType,
                  severity: assessmentResult.severity,
                  confidence: assessmentResult.confidence,
                  sampledFrameCount: assessmentResult.sampledFrameCount,
                  notes: assessmentResult.notes,
                  mouthShapeNotes: assessmentResult.mouthShapeNotes,
                },
                null,
                2
              )}
            </pre>
          </section>
        ) : null}

        {coaching ? (
          <section style={styles.panel}>
            <div style={styles.panelLabel}>Coach output</div>
            <pre style={styles.outputText}>
              {JSON.stringify(
                {
                  feedback: coaching.feedback,
                  encouragement: coaching.encouragement,
                  nextTryCue: coaching.nextTryCue,
                },
                null,
                2
              )}
            </pre>
          </section>
        ) : null}

        <section style={styles.panel}>
          <div style={styles.panelLabel}>Cache diagnostics</div>
          <pre style={styles.outputText}>
{`runtime: ${modelStatus.runtimeKind}
service url: ${modelStatus.serviceUrl || 'n/a'}
model id: ${modelStatus.modelId}
browser dtype: ${BROWSER_MODEL_DTYPE_LABEL}
browser baseline: ${BROWSER_MODEL_IS_PUBLIC_BASE ? 'public Gemma 4 E2B ONNX' : 'custom browser model'}
fallback reason: ${BROWSER_MODEL_FALLBACK_REASON || 'none'}
load source: ${modelStatus.loadSource}
cache available: ${String(modelStatus.cacheAvailable)}
cache complete: ${String(modelStatus.cacheComplete)}
cache origin: ${modelStatus.cacheOrigin || 'unavailable'}
missing files: ${modelStatus.missingFiles.length ? modelStatus.missingFiles.join(', ') : 'none'}`}
          </pre>
        </section>
      </div>
    </section>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    height: '100%',
    overflowY: 'auto',
  },
  content: {
    padding: 24,
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  backButton: {
    alignSelf: 'flex-start',
    fontSize: 14,
    color: 'var(--color-text-subtle)',
  },
  title: {
    margin: 0,
    fontSize: 30,
    lineHeight: '34px',
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -0.9,
  },
  panel: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 18,
    border: '1px solid var(--color-border)',
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  panelLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  input: {
    backgroundColor: 'var(--color-surface-alt)',
    borderRadius: 12,
    border: '1px solid var(--color-border)',
    padding: '14px',
    fontSize: 15,
    color: 'var(--color-text)',
  },
  buttonRow: {
    display: 'flex',
    gap: 10,
  },
  primaryButton: {
    alignSelf: 'flex-start',
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '14px 16px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonText: {
    color: 'var(--color-bg)',
    fontSize: 14,
    fontWeight: 700,
  },
  buttonDisabled: {
    opacity: 0.55,
  },
  error: {
    fontSize: 13,
    color: 'var(--color-danger)',
  },
  outputText: {
    margin: 0,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '20px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
};
