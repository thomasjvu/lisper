import { createContext, type ReactNode, useContext, useEffect, useMemo, useState } from 'react';

import { cancelSpeech } from './webSpeech';

export interface AppPreferences {
  soundEnabled: boolean;
  reducedMotion: boolean;
}

interface PreferencesContextValue extends AppPreferences {
  setSoundEnabled: (value: boolean) => void;
  setReducedMotion: (value: boolean) => void;
}

const STORAGE_KEY = 'lisper:web-preferences:v1';

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

function readSystemReducedMotion() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }

  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function createDefaultPreferences(): AppPreferences {
  return {
    soundEnabled: true,
    reducedMotion: readSystemReducedMotion(),
  };
}

function readStoredPreferences(): AppPreferences | null {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as Partial<AppPreferences>;
    const defaults = createDefaultPreferences();

    return {
      soundEnabled: typeof parsed.soundEnabled === 'boolean' ? parsed.soundEnabled : defaults.soundEnabled,
      reducedMotion: typeof parsed.reducedMotion === 'boolean' ? parsed.reducedMotion : defaults.reducedMotion,
    };
  } catch (error) {
    console.warn('[Preferences] Failed to read saved preferences', error);
    return null;
  }
}

function saveStoredPreferences(preferences: AppPreferences) {
  if (typeof window === 'undefined' || !window.localStorage) {
    return;
  }

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch (error) {
    console.warn('[Preferences] Failed to persist preferences', error);
  }
}

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [preferences, setPreferences] = useState<AppPreferences>(() => readStoredPreferences() || createDefaultPreferences());

  useEffect(() => {
    saveStoredPreferences(preferences);
  }, [preferences]);

  useEffect(() => {
    if (!preferences.soundEnabled) {
      cancelSpeech();
    }
  }, [preferences.soundEnabled]);

  const value = useMemo<PreferencesContextValue>(
    () => ({
      ...preferences,
      setSoundEnabled: (soundEnabled: boolean) => {
        setPreferences((current) => ({
          ...current,
          soundEnabled,
        }));
      },
      setReducedMotion: (reducedMotion: boolean) => {
        setPreferences((current) => ({
          ...current,
          reducedMotion,
        }));
      },
    }),
    [preferences]
  );

  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences() {
  const context = useContext(PreferencesContext);
  if (!context) {
    throw new Error('usePreferences must be used within PreferencesProvider');
  }

  return context;
}
