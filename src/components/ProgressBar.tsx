import type { CSSProperties } from 'react';

interface ProgressBarProps {
  label?: string;
  value: number | null;
  hint?: string;
  size?: 'sm' | 'md';
}

function clamp(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return 36;
  }

  return Math.max(0, Math.min(100, value));
}

export default function ProgressBar({ label, value, hint, size = 'md' }: ProgressBarProps) {
  const progress = clamp(value);
  const indeterminate = value === null || !Number.isFinite(value);

  return (
    <div style={styles.wrapper}>
      {label || hint ? (
        <div style={styles.header}>
          {label ? <div style={styles.label}>{label}</div> : <div />}
          {hint ? <div style={styles.hint}>{hint}</div> : null}
        </div>
      ) : null}

      <div style={{ ...styles.track, ...(size === 'sm' ? styles.trackSmall : null) }}>
        <div
          style={{
            ...styles.fill,
            ...(size === 'sm' ? styles.fillSmall : null),
            ...(indeterminate ? styles.fillIndeterminate : null),
            width: `${progress}%`,
          }}
        />
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  label: {
    flex: 1,
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.9,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  hint: {
    fontSize: 12,
    color: 'var(--color-text)',
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  track: {
    height: 10,
    borderRadius: 999,
    backgroundColor: 'var(--color-surface-alt)',
    overflow: 'hidden',
  },
  trackSmall: {
    height: 7,
  },
  fill: {
    height: '100%',
    borderRadius: 999,
    backgroundColor: 'var(--color-accent)',
    transition: 'width 180ms ease',
  },
  fillSmall: {
    backgroundColor: 'var(--color-accent-strong)',
  },
  fillIndeterminate: {
    opacity: 0.45,
  },
};
