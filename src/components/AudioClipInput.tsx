import { type ChangeEvent, useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';

import type { PracticeCapture } from '../utils/modelRuntime';
import { useMediaSession } from '../utils/mediaSession';

interface AudioClipInputProps {
  label: string;
  capture: PracticeCapture | null;
  analyzing?: boolean;
  onCaptureReady: (capture: PracticeCapture) => void;
}

const MAX_CAPTURE_DURATION_MS = 10000;

function formatDuration(durationMs: number) {
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

async function readAudioDurationMs(file: Blob) {
  if (typeof Audio === 'undefined') {
    return 0;
  }

  const url = URL.createObjectURL(file);
  try {
    const audio = new Audio();
    audio.preload = 'metadata';
    audio.src = url;

    const duration = await new Promise<number>((resolve) => {
      const cleanup = () => {
        audio.onloadedmetadata = null;
        audio.onerror = null;
      };

      audio.onloadedmetadata = () => {
        cleanup();
        resolve(Number.isFinite(audio.duration) ? audio.duration : 0);
      };
      audio.onerror = () => {
        cleanup();
        resolve(0);
      };
    });

    return Math.max(0, Math.round(duration * 1000));
  } finally {
    URL.revokeObjectURL(url);
  }
}

export default function AudioClipInput({ label, capture, analyzing = false, onCaptureReady }: AudioClipInputProps) {
  const {
    captureState,
    deviceMode,
    streamReady,
    visualSupport,
    message,
    durationMs,
    prepareCameraAndMic,
    prepareMicOnly,
    attachPreview,
    attachMeter,
    startRecording,
    stopRecording,
  } = useMediaSession();

  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const meterCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    attachPreview(videoRef.current);
    attachMeter(meterCanvasRef.current);

    return () => {
      attachPreview(null);
      attachMeter(null);
    };
  }, [attachMeter, attachPreview]);

  useEffect(() => {
    if (!capture?.audioBlob) {
      setAudioPreviewUrl((current) => {
        if (current) {
          URL.revokeObjectURL(current);
        }
        return null;
      });
      return;
    }

    const nextUrl = URL.createObjectURL(capture.audioBlob);
    setAudioPreviewUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return nextUrl;
    });

    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [capture]);

  async function handleUploadChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file) {
      return;
    }

    const duration = await readAudioDurationMs(file);
    onCaptureReady({
      audioBlob: file,
      frames: [],
      durationMs: duration,
      source: 'upload',
    });
  }

  const preparing = captureState === 'requesting';
  const recording = captureState === 'recording';
  const canRecord = streamReady || recording;
  const captureSummary = capture
    ? `${capture.source} • ${formatDuration(capture.durationMs)} • ${capture.frames.length} frame${capture.frames.length === 1 ? '' : 's'}`
    : `Max clip length ${Math.round(MAX_CAPTURE_DURATION_MS / 1000)} seconds.`;
  const helperText =
    capture?.source === 'upload' ? 'Uploaded audio is ready. Visual lip cues are unavailable for uploaded clips.' : message;

  return (
    <section style={styles.card}>
      <div style={styles.label}>{label}</div>
      <p style={styles.helperText}>{helperText}</p>

      <div style={{ ...styles.notice, ...(captureState === 'blocked' ? styles.noticeBlocked : null) }}>
        <div style={styles.noticeText}>
          {captureState === 'mic-only'
            ? 'Microphone-only mode is active.'
            : captureState === 'ready'
              ? 'Full camera + microphone mode is active.'
              : captureState === 'blocked'
                ? 'Device access needs attention.'
                : captureState === 'unsupported'
                  ? 'This browser cannot support the capture flow.'
                  : 'Prepare devices before you record.'}
        </div>
      </div>

      <div style={styles.deviceActions}>
        <button
          type="button"
          style={{ ...styles.primaryDeviceButton, ...((preparing || analyzing || recording) ? styles.buttonDisabled : null) }}
          onClick={() => {
            void prepareCameraAndMic();
          }}
          disabled={preparing || analyzing || recording}
        >
          <span style={styles.primaryDeviceButtonText}>{preparing ? 'requesting access...' : 'prepare camera + mic'}</span>
        </button>

        <div style={styles.secondaryActionRow}>
          <button
            type="button"
            style={styles.secondaryButton}
            onClick={() => {
              void prepareMicOnly();
            }}
            disabled={preparing || analyzing || recording}
          >
            <span style={styles.secondaryButtonText}>continue with mic only</span>
          </button>
          {(captureState === 'blocked' || captureState === 'unsupported' || captureState === 'mic-only') && (
            <button
              type="button"
              style={styles.secondaryButton}
              onClick={() => {
                void prepareCameraAndMic();
              }}
              disabled={preparing || analyzing || recording}
            >
              <span style={styles.secondaryButtonText}>retry full devices</span>
            </button>
          )}
        </div>
      </div>

      <div style={styles.previewCard}>
        {visualSupport ? (
          <video ref={videoRef} autoPlay muted playsInline style={styles.video} />
        ) : (
          <div style={styles.previewFallback}>
            <div style={styles.previewFallbackTitle}>{deviceMode === 'mic-only' ? 'mic-only mode' : 'camera preview unavailable'}</div>
            <div style={styles.previewFallbackText}>
              {deviceMode === 'mic-only'
                ? 'You can still record and analyze audio while keeping the lesson moving.'
                : 'Prepare your camera to unlock sampled mouth frames.'}
            </div>
          </div>
        )}

        <canvas ref={meterCanvasRef} width={640} height={160} style={styles.meterCanvas} />
      </div>

      <div style={styles.stateRow}>
        <div style={styles.stateLabel}>capture state</div>
        <div style={styles.stateValue}>
          {analyzing ? 'analyzing' : captureState}
          {captureState === 'recording' ? ` • ${formatDuration(durationMs)}` : ''}
        </div>
      </div>

      <div style={styles.recordActions}>
        <button
          type="button"
          style={{ ...styles.recordButton, ...(((!canRecord || analyzing) ? styles.buttonDisabled : null)) }}
          onClick={() => {
            if (recording) {
              stopRecording();
              return;
            }
            void startRecording(onCaptureReady);
          }}
          disabled={(!canRecord && !recording) || analyzing}
        >
          <span style={styles.recordButtonText}>
            {recording ? 'stop capture' : capture ? 'record a retake' : 'record attempt'}
          </span>
        </button>

        <button type="button" style={styles.secondaryButton} onClick={() => fileInputRef.current?.click()} disabled={analyzing || recording}>
          <span style={styles.secondaryButtonText}>upload audio</span>
        </button>
      </div>

      <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleUploadChange} style={styles.hiddenInput} />

      <div style={styles.captureMeta}>
        <div style={styles.metaLabel}>attempt</div>
        <div style={styles.metaValue}>{captureSummary}</div>
      </div>

      {audioPreviewUrl ? <audio controls src={audioPreviewUrl} style={styles.audioPreview} /> : null}
    </section>
  );
}

const styles: Record<string, CSSProperties> = {
  card: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 20,
    padding: 18,
    border: '1px solid var(--color-border)',
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  label: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.9,
  },
  helperText: {
    margin: 0,
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '20px',
  },
  notice: {
    borderRadius: 14,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: '12px 14px',
  },
  noticeBlocked: {
    borderColor: 'var(--color-danger)',
  },
  noticeText: {
    fontSize: 13,
    color: 'var(--color-text)',
    lineHeight: '18px',
  },
  deviceActions: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  secondaryActionRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 10,
  },
  previewCard: {
    backgroundColor: 'var(--color-surface-alt)',
    borderRadius: 18,
    padding: 14,
    border: '1px solid var(--color-border)',
  },
  video: {
    width: '100%',
    borderRadius: 16,
    backgroundColor: 'var(--color-bg)',
    aspectRatio: '16 / 9',
    objectFit: 'cover',
    transform: 'scaleX(-1)',
  },
  previewFallback: {
    borderRadius: 14,
    backgroundColor: 'var(--color-bg)',
    minHeight: 220,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0 18px',
    flexDirection: 'column',
  },
  previewFallbackTitle: {
    color: 'var(--color-text)',
    fontSize: 16,
    fontWeight: 600,
    marginBottom: 6,
    textTransform: 'lowercase',
  },
  previewFallbackText: {
    color: 'var(--color-text-subtle)',
    fontSize: 13,
    lineHeight: '18px',
    textAlign: 'center',
    maxWidth: 280,
  },
  meterCanvas: {
    width: '100%',
    height: 86,
    borderRadius: 12,
    marginTop: 14,
    background: 'var(--color-bg)',
  },
  stateRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
  },
  stateLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  stateValue: {
    fontSize: 14,
    color: 'var(--color-text)',
    fontWeight: 600,
    textTransform: 'lowercase',
  },
  recordActions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 10,
  },
  primaryDeviceButton: {
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '14px 18px',
    display: 'inline-flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  primaryDeviceButtonText: {
    color: 'var(--color-bg)',
    fontSize: 14,
    fontWeight: 700,
  },
  recordButton: {
    flexGrow: 1,
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '14px 18px',
    minWidth: 180,
    display: 'inline-flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordButtonText: {
    color: 'var(--color-bg)',
    fontSize: 14,
    fontWeight: 700,
  },
  secondaryButton: {
    borderRadius: 12,
    border: '1px solid var(--color-border-strong)',
    padding: '12px 16px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'var(--color-surface)',
  },
  secondaryButtonText: {
    color: 'var(--color-text)',
    fontSize: 13,
    fontWeight: 600,
  },
  buttonDisabled: {
    opacity: 0.55,
  },
  hiddenInput: {
    display: 'none',
  },
  captureMeta: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  metaLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  metaValue: {
    fontSize: 13,
    color: 'var(--color-text-muted)',
  },
  audioPreview: {
    width: '100%',
    marginTop: 10,
    filter: 'invert(0.92) hue-rotate(160deg) saturate(0.5)',
  },
};
