import { useEffect, useState } from 'react';

interface ViewportSize {
  width: number;
  height: number;
}

function readViewport(): ViewportSize {
  if (typeof window === 'undefined') {
    return {
      width: 1440,
      height: 900,
    };
  }

  return {
    width: window.innerWidth,
    height: window.innerHeight,
  };
}

export default function useViewport() {
  const [viewport, setViewport] = useState<ViewportSize>(() => readViewport());

  useEffect(() => {
    const sync = () => {
      setViewport(readViewport());
    };

    window.addEventListener('resize', sync);
    return () => {
      window.removeEventListener('resize', sync);
    };
  }, []);

  return viewport;
}
