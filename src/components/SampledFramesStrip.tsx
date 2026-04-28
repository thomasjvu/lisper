import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

interface SampledFramesStripProps {
  frames: Blob[];
  title?: string;
}

export default function SampledFramesStrip({
  frames,
  title = 'what Gemma saw',
}: SampledFramesStripProps) {
  const [urls, setUrls] = useState<string[]>([]);

  useEffect(() => {
    const nextUrls = frames.map((frame) => URL.createObjectURL(frame));
    setUrls(nextUrls);

    return () => {
      nextUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [frames]);

  if (!frames.length || !urls.length) {
    return null;
  }

  return (
    <div style={styles.card}>
      <div style={styles.label}>{title}</div>
      <div style={styles.row}>
        {urls.map((url, index) => (
          <img key={url} src={url} alt={`sampled frame ${index + 1}`} style={styles.image} />
        ))}
      </div>
      <div style={styles.caption}>
        {frames.length} sampled frame{frames.length === 1 ? '' : 's'} from the last attempt
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  card: {
    marginTop: 14,
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  label: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  row: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  image: {
    width: 96,
    height: 96,
    borderRadius: 14,
    backgroundColor: 'var(--color-surface-alt)',
    border: '1px solid var(--color-border)',
    objectFit: 'cover',
  },
  caption: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
  },
};
