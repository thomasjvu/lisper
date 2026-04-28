import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

import { usePreferences } from '../utils/preferences';

interface MonochromeMascotProps {
  size?: 'sm' | 'md' | 'lg';
  caption?: string;
}

type MascotFrame = 'idle' | 'blink' | 'chirp' | 'wing-lift' | 'tilt';

const HERO_SEQUENCE: ReadonlyArray<{ frame: MascotFrame; duration: number }> = [
  { frame: 'idle', duration: 2000 },
  { frame: 'blink', duration: 140 },
  { frame: 'idle', duration: 220 },
  { frame: 'chirp', duration: 240 },
  { frame: 'idle', duration: 200 },
  { frame: 'wing-lift', duration: 260 },
  { frame: 'idle', duration: 220 },
  { frame: 'tilt', duration: 260 },
];

const INLINE_SEQUENCE: ReadonlyArray<{ frame: MascotFrame; duration: number }> = [
  { frame: 'idle', duration: 2600 },
  { frame: 'blink', duration: 120 },
  { frame: 'idle', duration: 180 },
  { frame: 'chirp', duration: 220 },
];

export default function MonochromeMascot({ size = 'md', caption }: MonochromeMascotProps) {
  const [frameIndex, setFrameIndex] = useState(0);
  const { reducedMotion } = usePreferences();

  const large = size === 'lg';
  const small = size === 'sm';
  const sequence = small ? INLINE_SEQUENCE : HERO_SEQUENCE;
  const frame = reducedMotion ? 'idle' : sequence[frameIndex % sequence.length].frame;
  const scale = large ? 1.2 : small ? 0.76 : 1;

  useEffect(() => {
    if (reducedMotion) {
      return;
    }

    const step = sequence[frameIndex % sequence.length];
    const timeoutId = window.setTimeout(() => {
      setFrameIndex((current) => (current + 1) % sequence.length);
    }, step.duration);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [frameIndex, reducedMotion, sequence]);

  const blink = frame === 'blink';
  const chirp = frame === 'chirp';
  const wingLift = frame === 'wing-lift';
  const tilt = frame === 'tilt';

  return (
    <div style={{ ...styles.wrapper, ...(large ? styles.wrapperLarge : null), ...(small ? styles.wrapperSmall : null) }}>
      <div style={{ ...styles.halo, ...(large ? styles.haloLarge : null), ...(small ? styles.haloSmall : null) }} />
      <div style={{ ...styles.shadow, ...(large ? styles.shadowLarge : null), ...(small ? styles.shadowSmall : null) }} />

      {chirp ? (
        <div style={{ ...styles.noteCluster, ...(large ? styles.noteClusterLarge : null) }}>
          <div style={{ ...styles.noteDot, ...styles.noteDotLarge }} />
          <div style={styles.noteDot} />
          <div style={{ ...styles.noteDot, ...styles.noteDotTiny }} />
        </div>
      ) : null}

      <div
        style={{
          ...styles.scene,
          transform: `scale(${scale}) translateY(${wingLift ? -6 : chirp ? -3 : 0}px) rotate(${tilt ? '-5deg' : '0deg'})`,
        }}
      >
        <div style={{ ...styles.tail, transform: `rotate(${tilt ? '-18deg' : wingLift ? '-8deg' : '-10deg'})` }} />
        <div
          style={{
            ...styles.wing,
            ...styles.wingLeft,
            transform: `rotate(${wingLift ? '-48deg' : chirp ? '-32deg' : '-24deg'}) translateY(${wingLift ? -7 : 0}px)`,
          }}
        />
        <div
          style={{
            ...styles.wing,
            ...styles.wingRight,
            transform: `rotate(${wingLift ? '48deg' : chirp ? '30deg' : '22deg'}) translateY(${wingLift ? -7 : 0}px)`,
          }}
        />

        <div style={styles.body}>
          <div style={styles.belly} />
          <div style={styles.bellyStripe} />
        </div>

        <div style={styles.neck} />

        <div style={{ ...styles.head, transform: `rotate(${chirp ? '5deg' : tilt ? '-7deg' : '0deg'})` }}>
          <div style={{ ...styles.crest, ...styles.crestLeft }} />
          <div style={{ ...styles.crest, ...styles.crestCenter }} />
          <div style={{ ...styles.crest, ...styles.crestRight }} />

          <div style={styles.facePatch} />

          <div style={styles.eyeRow}>
            <div style={{ ...styles.eye, ...(blink ? styles.eyeBlink : null) }} />
            <div style={{ ...styles.eye, ...(blink ? styles.eyeBlink : null) }} />
          </div>

          <div style={styles.cheekRow}>
            <div style={styles.cheek} />
            <div style={styles.cheek} />
          </div>

          <div style={styles.beakWrap}>
            <div
              style={{
                ...styles.beakUpper,
                transform: `rotate(${chirp ? '-10deg' : '0deg'}) translateY(${chirp ? -2 : 0}px)`,
              }}
            />
            <div
              style={{
                ...styles.beakLower,
                transform: `rotate(${chirp ? '11deg' : '0deg'}) translateY(${chirp ? 2 : 0}px)`,
              }}
            />
          </div>
        </div>

        <div style={styles.footRow}>
          <div style={styles.foot} />
          <div style={styles.foot} />
        </div>

        <div style={styles.perch} />
      </div>

      {caption ? <p style={styles.caption}>{caption}</p> : null}
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  wrapper: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  wrapperLarge: {
    gap: 16,
  },
  wrapperSmall: {
    gap: 6,
  },
  halo: {
    position: 'absolute',
    top: 8,
    width: 148,
    height: 148,
    borderRadius: 999,
    backgroundColor: 'rgba(25, 84, 102, 0.24)',
  },
  haloLarge: {
    width: 196,
    height: 196,
  },
  haloSmall: {
    width: 104,
    height: 104,
  },
  shadow: {
    position: 'absolute',
    bottom: 10,
    width: 102,
    height: 20,
    borderRadius: 999,
    backgroundColor: 'rgba(10, 55, 73, 0.82)',
    opacity: 0.8,
  },
  shadowLarge: {
    width: 142,
    height: 24,
  },
  shadowSmall: {
    width: 74,
    height: 14,
  },
  noteCluster: {
    position: 'absolute',
    right: -2,
    top: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    alignItems: 'center',
  },
  noteClusterLarge: {
    right: -12,
    top: 28,
  },
  noteDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-warning)',
    border: '1px solid var(--color-bg)',
  },
  noteDotLarge: {
    width: 10,
    height: 10,
  },
  noteDotTiny: {
    width: 6,
    height: 6,
  },
  scene: {
    position: 'relative',
    width: 164,
    height: 190,
    transition: 'transform 180ms ease',
  },
  tail: {
    position: 'absolute',
    left: 18,
    bottom: 58,
    width: 42,
    height: 72,
    borderRadius: 24,
    backgroundColor: 'var(--color-accent-strong)',
    border: '2px solid var(--color-bg)',
    transformOrigin: 'bottom center',
  },
  wing: {
    position: 'absolute',
    top: 62,
    width: 46,
    height: 78,
    backgroundColor: 'var(--color-accent)',
    border: '2px solid var(--color-bg)',
    borderRadius: 26,
    transformOrigin: 'top center',
  },
  wingLeft: {
    left: 22,
  },
  wingRight: {
    right: 20,
  },
  body: {
    position: 'absolute',
    top: 54,
    left: 40,
    width: 84,
    height: 98,
    borderRadius: 44,
    backgroundColor: 'var(--color-success)',
    border: '2px solid var(--color-bg)',
    overflow: 'hidden',
  },
  belly: {
    position: 'absolute',
    inset: '18px 16px 12px 16px',
    borderRadius: 32,
    backgroundColor: 'var(--color-surface)',
  },
  bellyStripe: {
    position: 'absolute',
    left: 26,
    right: 26,
    bottom: 18,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-border)',
  },
  neck: {
    position: 'absolute',
    top: 40,
    left: 66,
    width: 34,
    height: 28,
    borderRadius: 18,
    backgroundColor: 'var(--color-accent-strong)',
    border: '2px solid var(--color-bg)',
  },
  head: {
    position: 'absolute',
    top: 8,
    left: 46,
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'var(--color-accent)',
    border: '2px solid var(--color-bg)',
    transformOrigin: 'bottom center',
  },
  crest: {
    position: 'absolute',
    top: -9,
    width: 16,
    height: 22,
    backgroundColor: 'var(--color-success)',
    border: '2px solid var(--color-bg)',
    borderRadius: 12,
  },
  crestLeft: {
    left: 12,
    transform: 'rotate(-18deg)',
  },
  crestCenter: {
    left: 27,
  },
  crestRight: {
    right: 11,
    transform: 'rotate(18deg)',
  },
  facePatch: {
    position: 'absolute',
    left: 11,
    right: 11,
    bottom: 12,
    height: 30,
    borderRadius: 18,
    backgroundColor: 'var(--color-surface)',
  },
  eyeRow: {
    position: 'absolute',
    top: 22,
    left: 16,
    right: 16,
    display: 'flex',
    justifyContent: 'space-between',
  },
  eye: {
    width: 10,
    height: 10,
    borderRadius: 999,
    backgroundColor: 'var(--color-bg)',
  },
  eyeBlink: {
    height: 2,
    marginTop: 4,
  },
  cheekRow: {
    position: 'absolute',
    top: 38,
    left: 14,
    right: 14,
    display: 'flex',
    justifyContent: 'space-between',
  },
  cheek: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-warning)',
    opacity: 0.9,
  },
  beakWrap: {
    position: 'absolute',
    left: 22,
    right: 22,
    bottom: 12,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 2,
  },
  beakUpper: {
    width: 28,
    height: 12,
    borderRadius: 12,
    backgroundColor: 'var(--color-warning)',
    border: '2px solid var(--color-bg)',
  },
  beakLower: {
    width: 22,
    height: 8,
    borderRadius: 10,
    backgroundColor: 'var(--color-danger)',
    border: '2px solid var(--color-bg)',
  },
  footRow: {
    position: 'absolute',
    bottom: 18,
    left: 56,
    right: 56,
    display: 'flex',
    justifyContent: 'space-between',
  },
  foot: {
    width: 18,
    height: 12,
    borderRadius: 999,
    backgroundColor: 'var(--color-warning)',
    border: '2px solid var(--color-bg)',
  },
  perch: {
    position: 'absolute',
    bottom: 12,
    left: 34,
    right: 34,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-border-strong)',
  },
  caption: {
    margin: 0,
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textAlign: 'center',
  },
};
