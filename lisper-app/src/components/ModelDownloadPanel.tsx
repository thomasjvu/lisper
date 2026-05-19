import type { CSSProperties } from 'react';

import { BROWSER_MODEL_DTYPE_LABEL, BROWSER_MODEL_FALLBACK_REASON, BROWSER_MODEL_IS_PUBLIC_BASE } from '../utils/appConfig';
import type { ModelStatus } from '../utils/modelRuntime';
import ProgressBar from './ProgressBar';

interface ModelDownloadPanelProps {
  status: ModelStatus;
}

function formatBytes(value: number | null) {
  if (!value || value <= 0) {
    return null;
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let current = value;
  let index = 0;

  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }

  const precision = current >= 100 || index === 0 ? 0 : current >= 10 ? 1 : 2;
  return `${current.toFixed(precision)} ${units[index]}`;
}

function formatRatio(loaded: number | null, total: number | null) {
  const left = formatBytes(loaded);
  const right = formatBytes(total);
  if (!left || !right) {
    return null;
  }

  return `${left} / ${right}`;
}

export default function ModelDownloadPanel({ status }: ModelDownloadPanelProps) {
  const remoteMode = status.runtimeKind === 'remote';
  const totalHint =
    typeof status.totalProgress === 'number'
      ? `${Math.round(status.totalProgress)}%`
      : status.filesTotal
        ? `${status.filesCompleted}/${status.filesTotal} files`
        : remoteMode
          ? 'service check'
          : 'starting';

  const totalRatio = formatRatio(status.totalBytesLoaded, status.totalBytesExpected);
  const currentHint =
    typeof status.currentFileProgress === 'number' ? `${Math.round(status.currentFileProgress)}%` : 'estimating';
  const currentRatio = formatRatio(status.currentFileBytesLoaded, status.currentFileBytesExpected);
  const summary =
    remoteMode
      ? status.phase === 'ready'
        ? `Connected to ${status.serviceUrl?.replace(/^https?:\/\//, '') || 'the Lisper inference service'}.`
        : status.phase === 'idle'
          ? 'Uses the trained Kaggle adapter through a Python inference service instead of the browser ONNX demo.'
          : status.error || status.label
      : status.phase === 'ready'
      ? status.cacheComplete
        ? `${BROWSER_MODEL_IS_PUBLIC_BASE ? 'Public base Gemma 4 E2B' : 'Browser Gemma 4'} ${BROWSER_MODEL_DTYPE_LABEL} cached for ${
            status.cacheOrigin?.replace(/^https?:\/\//, '') || 'this origin'
          }.`
        : `${BROWSER_MODEL_IS_PUBLIC_BASE ? 'Public base Gemma 4 E2B' : 'Browser Gemma 4'} ${BROWSER_MODEL_DTYPE_LABEL} ready.`
      : status.phase === 'idle'
        ? BROWSER_MODEL_FALLBACK_REASON ||
          `${BROWSER_MODEL_IS_PUBLIC_BASE ? 'Public base Gemma 4 E2B' : 'Browser Gemma 4'} ${BROWSER_MODEL_DTYPE_LABEL} loads when you open assessment, training, or lab.`
        : status.error ||
          (status.loadSource === 'warm-cache'
            ? 'Loading from browser cache.'
            : status.loadSource === 'partial-network'
              ? status.missingFiles.length
                ? `Restoring ${status.missingFiles.length} missing file${status.missingFiles.length === 1 ? '' : 's'}.`
                : 'Restoring missing assets.'
              : status.label);
  const currentFileText = status.currentFileLabel || 'model file';

  return (
    <section style={styles.card}>
      <div style={styles.summaryBlock}>
        <h3 style={styles.title}>
          {remoteMode
            ? status.phase === 'ready'
              ? 'Remote model ready'
              : status.phase === 'error'
                ? 'Remote model unavailable'
                : status.phase === 'loading'
                  ? 'Connecting to trained model'
                  : 'Remote model standby'
            : status.phase === 'ready'
            ? 'Gemma is ready'
            : status.phase === 'idle'
              ? 'Gemma on standby'
              : status.phase === 'unsupported'
                ? 'WebGPU required'
                : status.phase === 'error'
                  ? 'Load failed'
                  : status.loadSource === 'warm-cache'
                    ? 'Loading from browser cache'
                    : status.loadSource === 'partial-network'
                      ? 'Restoring cached model'
                      : 'Preparing model'}
        </h3>
        <p style={styles.body}>{summary}</p>
      </div>

      {!remoteMode && (status.phase === 'loading' || status.downloadActive) ? (
        <div style={styles.progressGroup}>
          <ProgressBar
            label={status.loadSource === 'warm-cache' ? 'Total load' : status.loadSource === 'partial-network' ? 'Total restore' : 'Total download'}
            value={status.totalProgress}
            hint={totalHint}
          />
          <div style={styles.meta}>{totalRatio || ' '}</div>

          <div style={styles.fileLabel}>{currentFileText}</div>

          <ProgressBar label="Current file" value={status.currentFileProgress} hint={currentHint} size="sm" />
          <div style={styles.meta}>{currentRatio || ' '}</div>

          {status.missingFiles.length && status.loadSource === 'partial-network' ? (
            <div style={styles.meta}>Missing: {status.missingFiles.join(', ')}</div>
          ) : null}
        </div>
      ) : (
        <div style={styles.readyRow}>
          <span style={styles.readyLabel}>Status</span>
          <span style={styles.readyValue}>
            {remoteMode
              ? status.phase === 'ready'
                ? 'connected'
                : status.phase
              : status.phase === 'ready'
              ? status.cacheComplete
                ? 'warm cache'
                : 'ready'
              : status.phase === 'unsupported'
                ? 'unsupported'
                : status.phase}
          </span>
        </div>
      )}
    </section>
  );
}

const styles: Record<string, CSSProperties> = {
  card: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 18,
    border: '1px solid var(--color-border)',
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    minHeight: 212,
  },
  summaryBlock: {
    minHeight: 58,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  title: {
    margin: 0,
    fontSize: 18,
    color: 'var(--color-text)',
    fontWeight: 700,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  body: {
    margin: 0,
    fontSize: 13,
    color: 'var(--color-text-muted)',
    lineHeight: '19px',
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
  },
  progressGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  fileLabel: {
    minHeight: 16,
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  meta: {
    marginTop: -4,
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    minHeight: 16,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  readyRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  readyLabel: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  readyValue: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 600,
  },
};
