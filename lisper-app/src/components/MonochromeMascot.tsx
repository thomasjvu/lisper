import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

import mascotSpriteUrl from '../../assets/snake-talking-spritesheet.png';
import { usePreferences } from '../utils/preferences';

interface MonochromeMascotProps {
  size?: 'sm' | 'md' | 'lg';
  caption?: string;
  talking?: boolean;
}

type MascotFrame = 'idle' | 'blink' | 'talk-1' | 'talk-2' | 'talk-3' | 'talk-4' | 'coil-lift' | 'tilt';

const HERO_SEQUENCE: ReadonlyArray<{ frame: MascotFrame; duration: number }> = [
  { frame: 'idle', duration: 2200 },
  { frame: 'blink', duration: 140 },
  { frame: 'idle', duration: 280 },
  { frame: 'talk-1', duration: 90 },
  { frame: 'talk-2', duration: 90 },
  { frame: 'talk-3', duration: 110 },
  { frame: 'talk-4', duration: 90 },
  { frame: 'idle', duration: 260 },
  { frame: 'coil-lift', duration: 380 },
  { frame: 'idle', duration: 260 },
  { frame: 'tilt', duration: 380 },
];

const INLINE_SEQUENCE: ReadonlyArray<{ frame: MascotFrame; duration: number }> = [
  { frame: 'idle', duration: 2800 },
  { frame: 'blink', duration: 140 },
  { frame: 'idle', duration: 220 },
  { frame: 'talk-1', duration: 90 },
  { frame: 'talk-2', duration: 90 },
  { frame: 'talk-3', duration: 100 },
  { frame: 'talk-4', duration: 80 },
];

const TALK_SEQUENCE: ReadonlyArray<{ frame: MascotFrame; duration: number }> = [
  { frame: 'talk-1', duration: 90 },
  { frame: 'talk-2', duration: 90 },
  { frame: 'talk-3', duration: 110 },
  { frame: 'talk-4', duration: 90 },
];

const SIZE_STYLES: Record<NonNullable<MonochromeMascotProps['size']>, CSSProperties> = {
  sm: { width: 104 },
  md: { width: 154 },
  lg: { width: 232 },
};

export default function MonochromeMascot({ size = 'md', caption, talking: forceTalking = false }: MonochromeMascotProps) {
  const [frameIndex, setFrameIndex] = useState(0);
  const { reducedMotion } = usePreferences();

  const sequence = forceTalking ? TALK_SEQUENCE : size === 'sm' ? INLINE_SEQUENCE : HERO_SEQUENCE;
  const frame = reducedMotion ? 'idle' : sequence[frameIndex % sequence.length].frame;
  const talkFrame = frame.startsWith('talk-') ? Number(frame.slice(-1)) - 1 : 0;
  const talking = frame.startsWith('talk-');
  const coilLift = frame === 'coil-lift';
  const tilt = frame === 'tilt';
  const blink = frame === 'blink';

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

  return (
    <div style={{ ...styles.wrapper, ...(size === 'lg' ? styles.wrapperLarge : null) }}>
      <div style={{ ...styles.halo, ...(size === 'sm' ? styles.haloSmall : null), ...(size === 'lg' ? styles.haloLarge : null) }} />
      <div style={{ ...styles.shadow, ...(size === 'sm' ? styles.shadowSmall : null), ...(size === 'lg' ? styles.shadowLarge : null) }} />

      {talking ? (
        <div style={{ ...styles.noteCluster, ...(size === 'lg' ? styles.noteClusterLarge : null) }}>
          <div style={{ ...styles.noteDot, ...styles.noteDotLarge }} />
          <div style={styles.noteDot} />
          <div style={{ ...styles.noteDot, ...styles.noteDotTiny }} />
        </div>
      ) : null}

      {blink ? <div style={{ ...styles.sparkle, ...(size === 'sm' ? styles.sparkleSmall : null) }} /> : null}

      <div
        aria-hidden="true"
        style={{
          ...styles.sprite,
          ...SIZE_STYLES[size],
          backgroundImage: `url(${mascotSpriteUrl})`,
          backgroundPosition: `${(talking ? talkFrame : 0) * (100 / 3)}% 0%`,
          transform: `translateY(${coilLift ? -7 : talking ? -4 : 0}px) rotate(${tilt ? '-4deg' : talking ? '2deg' : '0deg'})`,
        }}
      />

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
    minWidth: 118,
    minHeight: 118,
    isolation: 'isolate',
  },
  wrapperLarge: {
    gap: 16,
    minWidth: 246,
    minHeight: 246,
  },
  halo: {
    position: 'absolute',
    top: '10%',
    width: 124,
    height: 124,
    borderRadius: 999,
    background: 'radial-gradient(circle at 50% 40%, rgba(88, 255, 140, 0.32), rgba(17, 73, 52, 0.18) 62%, transparent 70%)',
    zIndex: -2,
  },
  haloLarge: {
    width: 220,
    height: 220,
  },
  haloSmall: {
    width: 96,
    height: 96,
  },
  shadow: {
    position: 'absolute',
    bottom: 9,
    width: 82,
    height: 16,
    borderRadius: 999,
    backgroundColor: 'rgba(6, 32, 25, 0.24)',
    filter: 'blur(2px)',
    zIndex: -1,
  },
  shadowLarge: {
    bottom: 4,
    width: 148,
    height: 24,
  },
  shadowSmall: {
    width: 60,
    height: 12,
  },
  sprite: {
    display: 'block',
    aspectRatio: '1 / 1',
    backgroundRepeat: 'no-repeat',
    backgroundSize: '400% 100%',
    userSelect: 'none',
    filter: 'drop-shadow(0 12px 18px rgba(4, 30, 24, 0.28))',
    transformOrigin: '50% 86%',
    transition: 'transform 220ms ease, filter 220ms ease',
  },
  noteCluster: {
    position: 'absolute',
    right: 2,
    top: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    alignItems: 'center',
    zIndex: 2,
  },
  noteClusterLarge: {
    right: -2,
    top: 34,
  },
  noteDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: '#63ff8a',
    border: '1px solid rgba(255, 255, 255, 0.78)',
    boxShadow: '0 3px 10px rgba(28, 198, 96, 0.24)',
  },
  noteDotLarge: {
    width: 11,
    height: 11,
  },
  noteDotTiny: {
    width: 5,
    height: 5,
  },
  sparkle: {
    position: 'absolute',
    right: 18,
    top: 16,
    width: 11,
    height: 11,
    borderRadius: 999,
    backgroundColor: '#c8ffd4',
    boxShadow: '0 0 0 5px rgba(94, 255, 136, 0.2), 0 0 20px rgba(47, 228, 99, 0.28)',
    zIndex: 2,
  },
  sparkleSmall: {
    right: 8,
    top: 9,
    width: 8,
    height: 8,
  },
  caption: {
    margin: 0,
    maxWidth: 180,
    color: 'var(--color-muted)',
    fontSize: 12,
    lineHeight: 1.45,
    textAlign: 'center',
  },
};
