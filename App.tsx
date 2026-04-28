import { lazy, startTransition, Suspense, useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

import wallpaperUrl from './assets/title-wallpaper.svg';
import ModelDownloadPanel from './src/components/ModelDownloadPanel';
import MonochromeMascot from './src/components/MonochromeMascot';
import ProgressBar from './src/components/ProgressBar';
import HomeScreen from './src/screens/HomeScreen';
import { DOCS_URL } from './src/utils/appConfig';
import { ACHIEVEMENTS, GameProvider, LEVEL_XP, RANK_NAMES, useGame } from './src/utils/gameState';
import { MediaSessionProvider, useMediaSession } from './src/utils/mediaSession';
import { ensureReady, useModelStatus } from './src/utils/modelRuntime';
import type { ModelStatus } from './src/utils/modelRuntime';
import { PreferencesProvider, usePreferences } from './src/utils/preferences';
import useViewport from './src/utils/useViewport';

const loadAssessmentScreen = () => import('./src/screens/AssessmentScreen');
const loadTrainingScreen = () => import('./src/screens/TrainingScreen');
const loadProgressScreen = () => import('./src/screens/ProgressScreen');
const loadModelTestScreen = () => import('./src/screens/ModelTestScreen');

const AssessmentScreen = lazy(loadAssessmentScreen);
const TrainingScreen = lazy(loadTrainingScreen);
const ProgressScreen = lazy(loadProgressScreen);
const ModelTestScreen = lazy(loadModelTestScreen);

type Tab = 'home' | 'assessment' | 'training' | 'progress' | 'lab';
type LaunchPhase = 'splash' | 'boot' | 'app';
type BootPhase = 'booting' | 'ready' | 'error' | 'unsupported';

interface BootState {
  phase: BootPhase;
  storageReady: boolean;
  routesReady: boolean;
  modelStatus: ModelStatus;
  message: string;
}

let routePreloadPromise: Promise<void> | null = null;

function preloadAppRoutes() {
  if (!routePreloadPromise) {
    routePreloadPromise = Promise.all([
      loadAssessmentScreen(),
      loadTrainingScreen(),
      loadProgressScreen(),
      loadModelTestScreen(),
    ])
      .then(() => undefined)
      .catch((error) => {
        routePreloadPromise = null;
        throw error;
      });
  }

  return routePreloadPromise;
}

function ScreenFallback() {
  return (
    <div style={styles.loadingScreen}>
      <div className="app-spinner" />
      <div style={styles.loadingText}>loading screen...</div>
    </div>
  );
}

function getLevelProgress(level: number, xp: number) {
  const currentThreshold = LEVEL_XP[level - 1] || 0;
  const nextThreshold = LEVEL_XP[level] || LEVEL_XP[LEVEL_XP.length - 1];
  const span = Math.max(1, nextThreshold - currentThreshold);
  return Math.min(100, Math.max(0, ((xp - currentThreshold) / span) * 100));
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

function getCacheSourceLabel(modelStatus: ModelStatus) {
  if (modelStatus.loadSource === 'warm-cache') {
    return 'warm cache';
  }

  if (modelStatus.loadSource === 'partial-network') {
    return 'restoring cache';
  }

  return 'cold download';
}

function createBootState(
  storageReady: boolean,
  routesReady: boolean,
  routeError: string | null,
  modelStatus: ModelStatus
): BootState {
  if (modelStatus.phase === 'unsupported') {
    return {
      phase: 'unsupported',
      storageReady,
      routesReady,
      modelStatus,
      message: modelStatus.error || modelStatus.capability.reason || 'WebGPU is required to run Gemma 4 in the browser.',
    };
  }

  if (routeError) {
    return {
      phase: 'error',
      storageReady,
      routesReady,
      modelStatus,
      message: routeError,
    };
  }

  if (modelStatus.phase === 'error') {
    return {
      phase: 'error',
      storageReady,
      routesReady,
      modelStatus,
      message: modelStatus.error || 'Startup failed while preparing Gemma 4.',
    };
  }

  if (storageReady && routesReady && modelStatus.phase === 'ready') {
    return {
      phase: 'ready',
      storageReady,
      routesReady,
      modelStatus,
      message: 'Startup complete.',
    };
  }

  let message = 'Preparing startup.';
  if (!storageReady) {
    message = 'Restoring your saved progress.';
  } else if (!routesReady) {
    message = 'Preloading lessons and diagnostics.';
  } else if (modelStatus.phase === 'loading' || modelStatus.downloadActive) {
    message = modelStatus.label;
  } else if (modelStatus.phase === 'idle') {
    message = 'Starting Gemma 4.';
  }

  return {
    phase: 'booting',
    storageReady,
    routesReady,
    modelStatus,
    message,
  };
}

function ToggleRow({
  label,
  value,
  onToggle,
}: {
  label: string;
  value: boolean;
  onToggle: () => void;
}) {
  return (
    <button type="button" style={styles.settingsRow} onClick={onToggle}>
      <span style={styles.settingsLabel}>{label}</span>
      <span style={{ ...styles.settingsPill, ...(value ? styles.settingsPillActive : null) }}>{value ? 'on' : 'off'}</span>
    </button>
  );
}

function TitleScreen({ onStart }: { onStart: () => void }) {
  const { soundEnabled, reducedMotion, setSoundEnabled, setReducedMotion } = usePreferences();
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div
      style={{
        ...styles.titleRoot,
        backgroundImage: `linear-gradient(180deg, rgba(17, 17, 17, 0.42) 0%, rgba(17, 17, 17, 0.6) 100%), url(${wallpaperUrl})`,
      }}
    >
      <div style={styles.titleBackdrop} />
      <div style={styles.titleShell}>
        <div style={styles.titleHero}>
          <div style={styles.titleCopy}>
            <div style={styles.titleEyebrow}>Lisper</div>
            <h1 style={styles.titleHeading}>Speak clearly, one sound at a time.</h1>
            <p style={styles.titleBody}>
              Start the session when you are ready. Gemma 4 will download and prepare after you press start, then the app will enter automatically.
            </p>
          </div>

          <div style={styles.titleMascotWrap}>
            <MonochromeMascot size="lg" caption="placeholder title wallpaper + menu" />
          </div>
        </div>

        <div style={styles.menuStack}>
          <button type="button" style={styles.startButton} onClick={onStart}>
            <span style={styles.startButtonText}>Start</span>
          </button>

          <div style={styles.menuRow}>
            <button
              type="button"
              style={styles.menuButton}
              onClick={() => {
                window.open(DOCS_URL, '_blank', 'noopener,noreferrer');
              }}
            >
              <span style={styles.menuButtonText}>Read Docs</span>
            </button>

            <button type="button" style={styles.menuButton} onClick={() => setShowSettings((current) => !current)}>
              <span style={styles.menuButtonText}>Settings</span>
            </button>
          </div>
        </div>

        {showSettings ? (
          <div style={styles.settingsOverlay} onClick={() => setShowSettings(false)}>
            <div style={styles.settingsModal} onClick={(event) => event.stopPropagation()}>
              <div style={styles.settingsHeader}>
                <div style={styles.settingsEyebrow}>Settings</div>
                <button type="button" style={styles.settingsClose} onClick={() => setShowSettings(false)}>
                  close
                </button>
              </div>

              <ToggleRow label="Sound" value={soundEnabled} onToggle={() => setSoundEnabled(!soundEnabled)} />
              <ToggleRow label="Reduced Motion" value={reducedMotion} onToggle={() => setReducedMotion(!reducedMotion)} />

              <p style={styles.settingsHint}>
                Sound controls speech playback. Reduced motion dials down decorative animation on the splash screen, boot gate, and mascot.
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function StartupGate({ bootState, onRetry }: { bootState: BootState; onRetry: () => void }) {
  const { reducedMotion } = usePreferences();
  const totalHint =
    typeof bootState.modelStatus.totalProgress === 'number'
      ? `${Math.round(bootState.modelStatus.totalProgress)}%`
      : bootState.modelStatus.filesTotal
        ? `${bootState.modelStatus.filesCompleted}/${bootState.modelStatus.filesTotal} files`
        : 'starting';
  const totalRatio = formatRatio(bootState.modelStatus.totalBytesLoaded, bootState.modelStatus.totalBytesExpected);
  const currentHint =
    typeof bootState.modelStatus.currentFileProgress === 'number'
      ? `${Math.round(bootState.modelStatus.currentFileProgress)}%`
      : 'estimating';
  const currentRatio = formatRatio(
    bootState.modelStatus.currentFileBytesLoaded,
    bootState.modelStatus.currentFileBytesExpected
  );
  const currentFile = bootState.modelStatus.currentFileLabel || 'awaiting model file';
  const blocked = bootState.phase === 'error' || bootState.phase === 'unsupported';
  const title =
    bootState.phase === 'unsupported'
      ? 'This browser cannot run Lisper'
      : bootState.phase === 'error'
        ? 'Startup was blocked'
        : 'Preparing Gemma 4 and lessons';
  const summary =
    bootState.phase === 'unsupported'
      ? 'WebGPU support is required for the in-browser Gemma runtime.'
      : bootState.phase === 'error'
        ? 'The app is staying on this screen until startup succeeds.'
        : 'Stay here while the model, routes, and saved progress finish loading.';

  return (
    <div style={styles.startupRoot}>
      <div style={styles.startupShell}>
        <div style={styles.startupHero}>
          <div style={styles.startupMascotWrap}>
            <MonochromeMascot size="lg" />
          </div>

          <div style={styles.startupCopy}>
            <div style={styles.startupEyebrow}>
              {blocked ? 'startup blocked' : bootState.modelStatus.phase === 'ready' ? 'startup ready' : 'startup'}
            </div>
            <h1 style={styles.startupTitle}>{title}</h1>
            <p style={styles.startupBody}>{summary}</p>
            <div
              style={{
                ...styles.startupStatus,
                ...(blocked ? styles.startupStatusBlocked : null),
              }}
            >
              {!blocked && !reducedMotion ? <div className="app-spinner" style={styles.startupSpinner} /> : null}
              <span style={styles.startupStatusText}>{bootState.message}</span>
            </div>
          </div>
        </div>

        <div style={styles.startupGrid}>
          <section style={styles.startupPanel}>
            <ProgressBar label="Total progress" value={bootState.modelStatus.totalProgress} hint={totalHint} />
            <div style={styles.startupMeta}>{totalRatio || ' '}</div>

            <div style={styles.startupFileSlot}>
              <div style={styles.startupPanelLabel}>Current file</div>
              <div style={styles.startupFileValue}>{currentFile}</div>
            </div>

            <ProgressBar label="Current file" value={bootState.modelStatus.currentFileProgress} hint={currentHint} size="sm" />
            <div style={styles.startupMeta}>{currentRatio || ' '}</div>
          </section>

          <section style={styles.startupPanel}>
            <div style={styles.startupInfoRow}>
              <span style={styles.startupInfoLabel}>Cache source</span>
              <span style={styles.startupInfoValue}>{getCacheSourceLabel(bootState.modelStatus)}</span>
            </div>
            <div style={styles.startupInfoRow}>
              <span style={styles.startupInfoLabel}>App state</span>
              <span style={styles.startupInfoValue}>{bootState.storageReady ? 'ready' : 'loading'}</span>
            </div>
            <div style={styles.startupInfoRow}>
              <span style={styles.startupInfoLabel}>Routes</span>
              <span style={styles.startupInfoValue}>{bootState.routesReady ? 'ready' : 'preloading'}</span>
            </div>
            <div style={styles.startupInfoRow}>
              <span style={styles.startupInfoLabel}>Model</span>
              <span style={styles.startupInfoValue}>
                {bootState.modelStatus.phase === 'loading' ? 'loading' : bootState.modelStatus.phase}
              </span>
            </div>
            <div style={styles.startupInfoRow}>
              <span style={styles.startupInfoLabel}>Origin</span>
              <span style={styles.startupInfoValue}>
                {bootState.modelStatus.cacheOrigin?.replace(/^https?:\/\//, '') || 'unavailable'}
              </span>
            </div>
            <div style={styles.startupInfoRow}>
              <span style={styles.startupInfoLabel}>Missing files</span>
              <span style={styles.startupInfoValue}>
                {bootState.modelStatus.missingFiles.length ? bootState.modelStatus.missingFiles.length : 'none'}
              </span>
            </div>
          </section>
        </div>

        {blocked ? (
          <div style={styles.startupActions}>
            <button type="button" style={styles.startupPrimaryAction} onClick={onRetry}>
              <span style={styles.startupPrimaryActionText}>retry startup</span>
            </button>
            <button
              type="button"
              style={styles.startupSecondaryAction}
              onClick={() => {
                window.location.reload();
              }}
            >
              <span style={styles.startupSecondaryActionText}>reload page</span>
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function UtilityRail({ navigate, compact = false, activeTab }: { navigate: (tab: Tab) => void; compact?: boolean; activeTab: Tab }) {
  const { stats } = useGame();
  const modelStatus = useModelStatus();
  const nextAction = !stats.assessmentComplete
    ? {
        title: 'Take the assessment',
        body: 'Create the starting speech profile before the first lesson.',
        tab: 'assessment' as Tab,
      }
    : !stats.introTutorialComplete
      ? {
          title: 'Start the guided tutorial',
          body: 'Use the saved baseline to move through three coached first reps.',
          tab: 'training' as Tab,
        }
    : {
        title: 'Continue training',
        body: 'Run one short attempt and focus on a single correction.',
        tab: 'training' as Tab,
      };

  const recentAchievements = stats.achievements
    .slice(-2)
    .map((id) => ACHIEVEMENTS[id as keyof typeof ACHIEVEMENTS]?.name || id);
  const levelProgress = getLevelProgress(stats.level, stats.xp);
  const rankName = RANK_NAMES[Math.min(stats.level - 1, RANK_NAMES.length - 1)] || RANK_NAMES[0];
  const challengeProgress = `${Math.min(stats.dailyChallenge.progress, stats.dailyChallenge.goal)}/${stats.dailyChallenge.goal}`;
  const showProgressCard = activeTab !== 'home';
  const modelStatusBody = modelStatus.downloadActive
    ? modelStatus.loadSource === 'warm-cache'
      ? `${Math.round(modelStatus.totalProgress ?? 0)}% · loading from cache`
      : modelStatus.loadSource === 'partial-network'
        ? `${Math.round(modelStatus.totalProgress ?? 0)}% · restoring ${modelStatus.missingFiles.length || modelStatus.filesTotal || 1} file${(modelStatus.missingFiles.length || modelStatus.filesTotal || 1) === 1 ? '' : 's'}`
        : `${Math.round(modelStatus.totalProgress ?? 0)}% · ${modelStatus.currentFileLabel || 'loading files'}`
    : modelStatus.phase === 'ready'
      ? modelStatus.cacheComplete
        ? `ready on ${modelStatus.cacheOrigin?.replace(/^https?:\/\//, '') || 'this origin'}`
        : 'cached where available for faster reloads'
      : modelStatus.capability.reason || modelStatus.label;

  if (compact) {
    return (
      <div style={styles.compactUtilityRow}>
        <div style={styles.compactUtilityCard}>
          <div style={styles.compactUtilityTitle}>
            {modelStatus.phase === 'ready'
              ? 'Gemma ready'
              : modelStatus.loadSource === 'warm-cache'
                ? 'Loading cached Gemma'
                : modelStatus.downloadActive
                  ? 'Preparing Gemma'
                  : modelStatus.phase}
          </div>
          <div style={styles.compactUtilityBody}>{modelStatusBody}</div>
        </div>

        <button type="button" style={styles.compactAction} onClick={() => navigate(nextAction.tab)}>
          <div style={styles.compactActionTitle}>{nextAction.title}</div>
          <div style={styles.compactActionBody}>{nextAction.body}</div>
        </button>
      </div>
    );
  }

  return (
    <aside style={styles.utilityRail}>
      <ModelDownloadPanel status={modelStatus} />

      <section style={styles.utilityCard}>
        <div style={styles.utilityLead}>
          <MonochromeMascot size="sm" />
          <div style={styles.utilityLeadCopy}>
            <div style={styles.utilityTitle}>{nextAction.title}</div>
            <div style={styles.utilityBody}>{nextAction.body}</div>
          </div>
        </div>
        <button type="button" style={styles.utilityButton} onClick={() => navigate(nextAction.tab)}>
          <span style={styles.utilityButtonText}>Open next step</span>
        </button>
      </section>

      {showProgressCard ? (
        <section style={styles.utilityCard}>
          <div style={styles.utilityTitle}>Progress at a glance</div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>Rank</span>
            <span style={styles.metaValue}>{rankName}</span>
          </div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>Level</span>
            <span style={styles.metaValue}>{stats.level}</span>
          </div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>XP</span>
            <span style={styles.metaValue}>{Math.floor(stats.xp)}</span>
          </div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>Next level</span>
            <span style={styles.metaValue}>{Math.round(levelProgress)}%</span>
          </div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>Streak</span>
            <span style={styles.metaValue}>{stats.streak} days</span>
          </div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>Combo</span>
            <span style={styles.metaValue}>{stats.currentCombo}x</span>
          </div>
          <div style={styles.metaRow}>
            <span style={styles.metaLabel}>Daily</span>
            <span style={styles.metaValue}>{challengeProgress}</span>
          </div>

          <div style={styles.utilityBody}>{stats.dailyChallenge.description}</div>

          {recentAchievements.length ? (
            <div style={styles.achievementList}>
              {recentAchievements.map((item) => (
                <div key={item} style={styles.achievementRow}>
                  <div style={styles.achievementDot} />
                  <div style={styles.achievementText}>{item}</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={styles.utilityBody}>Unlock the first milestones by finishing the assessment and first training loop.</div>
          )}
        </section>
      ) : null}
    </aside>
  );
}

function AppShell({ tab, navigate, desktop, wide }: { tab: Tab; navigate: (tab: Tab) => void; desktop: boolean; wide: boolean }) {
  function renderScreen() {
    if (tab === 'assessment') {
      return (
        <Suspense fallback={<ScreenFallback />}>
          <AssessmentScreen
            onComplete={() => {
              startTransition(() => {
                navigate('training');
              });
            }}
          />
        </Suspense>
      );
    }

    if (tab === 'training') {
      return (
        <Suspense fallback={<ScreenFallback />}>
          <TrainingScreen />
        </Suspense>
      );
    }

    if (tab === 'progress') {
      return (
        <Suspense fallback={<ScreenFallback />}>
          <ProgressScreen />
        </Suspense>
      );
    }

    if (tab === 'lab') {
      return (
        <Suspense fallback={<ScreenFallback />}>
          <ModelTestScreen onBack={() => navigate('home')} />
        </Suspense>
      );
    }

    return <HomeScreen navigation={{ navigate }} />;
  }

  const navItems: Array<{ key: Tab; title: string; secondary?: boolean }> = [
    { key: 'home', title: 'Home' },
    { key: 'assessment', title: 'Assessment' },
    { key: 'training', title: 'Train' },
    { key: 'progress', title: 'Progress' },
    { key: 'lab', title: 'Lab', secondary: true },
  ];

  return (
    <div style={styles.container}>
      {desktop ? (
        <aside style={styles.sideRail}>
          <div style={styles.brandBlock}>
            <MonochromeMascot size="sm" />
            <div style={styles.brandName}>Lisper</div>
          </div>

          <nav style={styles.navSection}>
            {navItems.map((item) => {
              const active = tab === item.key;
              return (
                <button
                  key={item.key}
                  type="button"
                  style={{
                    ...styles.navItem,
                    ...(active ? styles.navItemActive : null),
                    ...(item.secondary ? styles.navItemSecondary : null),
                  }}
                  onClick={() => navigate(item.key)}
                >
                  <span style={{ ...styles.navItemText, ...(active ? styles.navItemTextActive : null) }}>{item.title}</span>
                </button>
              );
            })}
          </nav>
        </aside>
      ) : (
        <div style={styles.topNav}>
          <div style={styles.topNavHeader}>
            <div style={styles.brandName}>Lisper</div>
          </div>
          <div style={styles.topNavItems}>
            {navItems
              .filter((item) => !item.secondary)
              .map((item) => {
                const active = tab === item.key;
                return (
                  <button
                    key={item.key}
                    type="button"
                    style={{ ...styles.topNavItem, ...(active ? styles.topNavItemActive : null) }}
                    onClick={() => navigate(item.key)}
                  >
                    <span style={{ ...styles.topNavText, ...(active ? styles.topNavTextActive : null) }}>{item.title}</span>
                  </button>
                );
              })}
          </div>
        </div>
      )}

      <main style={{ ...styles.mainShell, ...(!desktop ? styles.mainShellCompact : null) }}>
        <div style={styles.centerPane}>
          {!wide ? <UtilityRail navigate={navigate} compact activeTab={tab} /> : null}
          <div style={styles.screenPane}>{renderScreen()}</div>
        </div>
        {wide ? <UtilityRail navigate={navigate} activeTab={tab} /> : null}
      </main>
    </div>
  );
}

function CaptureRouteController({ active }: { active: boolean }) {
  const { setCaptureRouteActive } = useMediaSession();

  useEffect(() => {
    setCaptureRouteActive(active);
  }, [active, setCaptureRouteActive]);

  return null;
}

function AppContent() {
  const { loading } = useGame();
  const modelStatus = useModelStatus();
  const [tab, setTab] = useState<Tab>('home');
  const [launchPhase, setLaunchPhase] = useState<LaunchPhase>('splash');
  const [routesReady, setRoutesReady] = useState(false);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [bootAttempt, setBootAttempt] = useState(0);
  const { width } = useViewport();

  const desktop = width >= 920;
  const wide = width >= 1320;
  const bootState = createBootState(!loading, routesReady, routeError, modelStatus);

  useEffect(() => {
    if (launchPhase !== 'boot') {
      return;
    }

    let cancelled = false;
    setRoutesReady(false);
    setRouteError(null);

    void preloadAppRoutes()
      .then(() => {
        if (!cancelled) {
          setRoutesReady(true);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setRouteError(error instanceof Error ? error.message : 'Failed to preload app routes.');
        }
      });

    void ensureReady().catch(() => undefined);

    return () => {
      cancelled = true;
    };
  }, [launchPhase, bootAttempt]);

  useEffect(() => {
    if (launchPhase === 'boot' && bootState.phase === 'ready') {
      startTransition(() => {
        setLaunchPhase('app');
      });
    }
  }, [bootState.phase, launchPhase]);

  function navigate(nextTab: Tab) {
    startTransition(() => {
      setTab(nextTab);
    });
  }

  if (launchPhase === 'splash') {
    return (
      <TitleScreen
        onStart={() => {
          startTransition(() => {
            setLaunchPhase('boot');
          });
        }}
      />
    );
  }

  if (launchPhase === 'boot') {
    return (
      <StartupGate
        bootState={bootState}
        onRetry={() => {
          setBootAttempt((current) => current + 1);
        }}
      />
    );
  }

  return (
    <MediaSessionProvider>
      <CaptureRouteController active={tab === 'assessment' || tab === 'training' || tab === 'lab'} />
      <AppShell tab={tab} navigate={navigate} desktop={desktop} wide={wide} />
    </MediaSessionProvider>
  );
}

export default function App() {
  return (
    <PreferencesProvider>
      <GameProvider>
        <AppContent />
      </GameProvider>
    </PreferencesProvider>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    minHeight: '100vh',
    backgroundColor: 'var(--color-bg)',
    display: 'flex',
  },
  sideRail: {
    width: 220,
    backgroundColor: 'var(--color-surface)',
    borderRight: '1px solid var(--color-border)',
    padding: '24px 18px',
    display: 'flex',
    flexDirection: 'column',
    gap: 28,
  },
  brandBlock: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  brandName: {
    fontSize: 24,
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -0.6,
  },
  navSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  navItem: {
    borderRadius: 14,
    border: '1px solid var(--color-border)',
    padding: '14px 16px',
    backgroundColor: 'var(--color-surface-alt)',
    textAlign: 'left',
  },
  navItemActive: {
    backgroundColor: 'var(--color-accent-soft)',
    borderColor: 'var(--color-accent-strong)',
  },
  navItemSecondary: {
    marginTop: 8,
  },
  navItemText: {
    fontSize: 14,
    fontWeight: 600,
    color: 'var(--color-text)',
  },
  navItemTextActive: {
    color: 'var(--color-text)',
  },
  topNav: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 10,
    backgroundColor: 'var(--color-surface)',
    borderBottom: '1px solid var(--color-border)',
    padding: '18px 18px 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  topNavHeader: {
    display: 'flex',
  },
  topNavItems: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
  },
  topNavItem: {
    borderRadius: 999,
    border: '1px solid var(--color-border)',
    padding: '10px 14px',
    backgroundColor: 'var(--color-surface-alt)',
  },
  topNavItemActive: {
    backgroundColor: 'var(--color-accent-soft)',
    borderColor: 'var(--color-accent-strong)',
  },
  topNavText: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 600,
  },
  topNavTextActive: {
    color: 'var(--color-text)',
  },
  mainShell: {
    flex: 1,
    display: 'flex',
    gap: 18,
    padding: 18,
    minWidth: 0,
  },
  mainShellCompact: {
    paddingTop: 112,
  },
  centerPane: {
    flex: 1,
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  screenPane: {
    flex: 1,
    minHeight: 0,
  },
  compactUtilityRow: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
  },
  compactUtilityCard: {
    flexGrow: 1,
    minWidth: 240,
    backgroundColor: 'var(--color-surface)',
    borderRadius: 16,
    border: '1px solid var(--color-border)',
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  compactUtilityTitle: {
    fontSize: 16,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  compactUtilityBody: {
    fontSize: 13,
    color: 'var(--color-text-muted)',
    lineHeight: '18px',
  },
  compactAction: {
    flexGrow: 1,
    minWidth: 240,
    backgroundColor: 'var(--color-accent)',
    borderRadius: 16,
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    textAlign: 'left',
  },
  compactActionTitle: {
    fontSize: 16,
    color: 'var(--color-bg)',
    fontWeight: 700,
  },
  compactActionBody: {
    fontSize: 13,
    color: 'var(--color-text-muted)',
    lineHeight: '18px',
  },
  utilityRail: {
    width: 320,
    display: 'flex',
    flexDirection: 'column',
    gap: 18,
    minWidth: 320,
  },
  utilityCard: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 18,
    border: '1px solid var(--color-border)',
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  utilityLead: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
  },
  utilityLeadCopy: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  utilityTitle: {
    fontSize: 18,
    fontWeight: 700,
    color: 'var(--color-text)',
  },
  utilityBody: {
    fontSize: 13,
    color: 'var(--color-text-muted)',
    lineHeight: '19px',
  },
  utilityButton: {
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '12px 14px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  utilityButtonText: {
    color: 'var(--color-bg)',
    fontSize: 14,
    fontWeight: 700,
  },
  metaRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
  },
  metaLabel: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.7,
  },
  metaValue: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  achievementList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    marginTop: 4,
  },
  achievementRow: {
    display: 'flex',
    gap: 10,
    alignItems: 'center',
  },
  achievementDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-success)',
    flexShrink: 0,
  },
  achievementText: {
    fontSize: 13,
    color: 'var(--color-text)',
  },
  loadingScreen: {
    height: '100%',
    minHeight: 360,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    backgroundColor: 'var(--color-surface)',
    borderRadius: 24,
    border: '1px solid var(--color-border)',
  },
  loadingText: {
    fontSize: 13,
    color: 'var(--color-text-subtle)',
  },
  titleRoot: {
    minHeight: '100vh',
    backgroundColor: 'var(--color-bg)',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  titleBackdrop: {
    position: 'absolute',
    inset: 0,
    background: 'linear-gradient(180deg, rgba(12, 16, 20, 0.22) 0%, rgba(12, 16, 20, 0.58) 100%)',
  },
  titleShell: {
    position: 'relative',
    zIndex: 1,
    width: '100%',
    maxWidth: 1180,
    minHeight: 680,
    borderRadius: 32,
    border: '1px solid rgba(51, 133, 158, 0.34)',
    backgroundColor: 'rgba(12, 16, 20, 0.72)',
    backdropFilter: 'blur(12px)',
    padding: 32,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    gap: 24,
    color: 'var(--color-text)',
    boxShadow: '0 28px 80px rgba(0, 0, 0, 0.28)',
  },
  titleHero: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 24,
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  titleCopy: {
    flex: 1,
    minWidth: 280,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  titleEyebrow: {
    fontSize: 12,
    letterSpacing: 1,
    textTransform: 'uppercase',
    color: 'rgba(153, 209, 206, 0.72)',
  },
  titleHeading: {
    margin: 0,
    fontSize: 56,
    lineHeight: '58px',
    letterSpacing: -1.8,
    fontWeight: 800,
    maxWidth: 640,
  },
  titleBody: {
    margin: 0,
    fontSize: 16,
    lineHeight: '24px',
    color: 'rgba(153, 209, 206, 0.82)',
    maxWidth: 540,
  },
  titleMascotWrap: {
    minWidth: 280,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  menuStack: {
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    maxWidth: 420,
  },
  startButton: {
    minHeight: 74,
    borderRadius: 18,
    backgroundColor: 'var(--color-accent)',
    color: 'var(--color-bg)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 16px 34px rgba(0, 0, 0, 0.22)',
  },
  startButtonText: {
    fontSize: 28,
    fontWeight: 800,
    letterSpacing: -0.8,
  },
  menuRow: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
  },
  menuButton: {
    flex: 1,
    minWidth: 160,
    minHeight: 54,
    borderRadius: 14,
    border: '1px solid rgba(51, 133, 158, 0.34)',
    backgroundColor: 'rgba(25, 84, 102, 0.18)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  menuButtonText: {
    color: 'var(--color-text)',
    fontSize: 15,
    fontWeight: 700,
  },
  settingsOverlay: {
    position: 'fixed',
    inset: 0,
    backgroundColor: 'rgba(12, 16, 20, 0.62)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    zIndex: 30,
  },
  settingsModal: {
    width: '100%',
    maxWidth: 420,
    borderRadius: 24,
    backgroundColor: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    padding: 22,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    boxShadow: '0 24px 70px rgba(0, 0, 0, 0.3)',
  },
  settingsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  settingsEyebrow: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.9,
    fontWeight: 700,
  },
  settingsClose: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 600,
    textTransform: 'lowercase',
  },
  settingsRow: {
    borderRadius: 16,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: '14px 16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  settingsLabel: {
    fontSize: 15,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  settingsPill: {
    minWidth: 58,
    padding: '8px 10px',
    borderRadius: 999,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface)',
    textAlign: 'center',
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: 0.7,
  },
  settingsPillActive: {
    backgroundColor: 'var(--color-accent)',
    borderColor: 'var(--color-accent)',
    color: 'var(--color-bg)',
  },
  settingsHint: {
    margin: 0,
    fontSize: 13,
    color: 'var(--color-text-muted)',
    lineHeight: '19px',
  },
  startupRoot: {
    minHeight: '100vh',
    backgroundColor: 'var(--color-bg)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  startupShell: {
    width: '100%',
    maxWidth: 1080,
    backgroundColor: 'var(--color-surface)',
    borderRadius: 30,
    border: '1px solid var(--color-border)',
    padding: 28,
    display: 'flex',
    flexDirection: 'column',
    gap: 24,
    boxShadow: '0 22px 70px rgba(0, 0, 0, 0.26)',
  },
  startupHero: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 24,
    alignItems: 'center',
  },
  startupMascotWrap: {
    minWidth: 220,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  startupCopy: {
    flex: 1,
    minWidth: 280,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  startupEyebrow: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.9,
  },
  startupTitle: {
    margin: 0,
    fontSize: 36,
    lineHeight: '40px',
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -1.2,
    maxWidth: 620,
  },
  startupBody: {
    margin: 0,
    fontSize: 15,
    lineHeight: '22px',
    color: 'var(--color-text-muted)',
    maxWidth: 580,
  },
  startupStatus: {
    minHeight: 48,
    borderRadius: 16,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: '12px 14px',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  startupStatusBlocked: {
    borderColor: 'var(--color-danger)',
    backgroundColor: 'rgba(194, 49, 39, 0.12)',
  },
  startupSpinner: {
    flexShrink: 0,
  },
  startupStatusText: {
    fontSize: 14,
    fontWeight: 600,
    color: 'var(--color-text)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  startupGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 18,
  },
  startupPanel: {
    minHeight: 232,
    backgroundColor: 'var(--color-surface-alt)',
    borderRadius: 20,
    border: '1px solid var(--color-border)',
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  startupMeta: {
    minHeight: 16,
    marginTop: -4,
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  startupFileSlot: {
    minHeight: 42,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  startupPanelLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  startupFileValue: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 600,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  startupInfoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
    minHeight: 26,
  },
  startupInfoLabel: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.7,
  },
  startupInfoValue: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 700,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  startupActions: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
  },
  startupPrimaryAction: {
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '14px 18px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  startupPrimaryActionText: {
    color: 'var(--color-bg)',
    fontSize: 14,
    fontWeight: 700,
  },
  startupSecondaryAction: {
    borderRadius: 12,
    border: '1px solid var(--color-border-strong)',
    padding: '12px 16px',
    backgroundColor: 'var(--color-surface)',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  startupSecondaryActionText: {
    color: 'var(--color-text)',
    fontSize: 13,
    fontWeight: 600,
  },
};
