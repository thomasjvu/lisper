import type { CSSProperties } from 'react';

import MonochromeMascot from '../components/MonochromeMascot';
import ProgressBar from '../components/ProgressBar';
import { LEVEL_XP, RANK_NAMES, useGame } from '../utils/gameState';
import useViewport from '../utils/useViewport';

interface HomeScreenProps {
  navigation: {
    navigate: (page: 'home' | 'assessment' | 'training' | 'progress' | 'lab') => void;
  };
}

function getProgressToNextLevel(level: number, xp: number) {
  const currentThreshold = LEVEL_XP[level - 1] || 0;
  const nextThreshold = LEVEL_XP[level] || LEVEL_XP[LEVEL_XP.length - 1];
  const span = Math.max(1, nextThreshold - currentThreshold);
  return Math.min(Math.max(((xp - currentThreshold) / span) * 100, 0), 100);
}

function getGreeting() {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

export default function HomeScreen({ navigation }: HomeScreenProps) {
  const { stats, loading } = useGame();
  const { width, height } = useViewport();
  const wide = width >= 980;
  const keepOnePage = width >= 1080 && height >= 760;
  const progress = getProgressToNextLevel(stats.level, stats.xp);
  const rankName = RANK_NAMES[Math.min(stats.level - 1, RANK_NAMES.length - 1)] || RANK_NAMES[0];

  if (loading) {
    return (
      <div style={styles.loading}>
        <div style={styles.loadingText}>loading...</div>
      </div>
    );
  }

  const primaryLabel = !stats.assessmentComplete
    ? 'Take the assessment'
    : !stats.introTutorialComplete
      ? 'Start guided tutorial'
      : 'Start today’s lesson';
  const primaryRoute = stats.assessmentComplete ? 'training' : 'assessment';
  const recentWins = stats.achievements.slice(-2).map((id) => id.replace(/_/g, ' '));
  const challengeValue = Math.min(stats.dailyChallenge.progress, stats.dailyChallenge.goal);
  const challengeProgress = (challengeValue / Math.max(1, stats.dailyChallenge.goal)) * 100;
  const nextStepText = stats.assessmentComplete
    ? !stats.introTutorialComplete
      ? 'Open training to move through the three guided first reps built from your saved baseline.'
      : 'Open training, record one short attempt, and use the next-try cue immediately.'
    : 'Run the assessment once so the app can save your starting profile.';

  return (
    <section style={{ ...styles.container, overflowY: keepOnePage ? 'hidden' : 'auto' }}>
      <div style={{ ...styles.content, ...(keepOnePage ? styles.contentFixed : null) }}>
        <section style={{ ...styles.hero, ...(wide ? styles.heroWide : null) }}>
          <div style={styles.heroCopy}>
            <div style={styles.greeting}>{getGreeting()}</div>
            <h1 style={styles.title}>Small reps. Clearer speech. Less clutter.</h1>
            <p style={styles.body}>
              {!stats.assessmentComplete
                ? 'Start with one quick baseline, then move straight into guided practice.'
                : !stats.introTutorialComplete
                  ? 'Your baseline is saved. The next step is a short guided tutorial before free training.'
                  : 'You already have the shell. Use the next lesson to get one cleaner repeat.'}
            </p>

            <div style={styles.heroStats}>
              <div style={styles.statPill}>
                <div style={styles.statPillValue}>{stats.streak}</div>
                <div style={styles.statPillLabel}>day streak</div>
              </div>
              <div style={styles.statPill}>
                <div style={styles.statPillValue}>{stats.level}</div>
                <div style={styles.statPillLabel}>level</div>
              </div>
              <div style={styles.statPill}>
                <div style={styles.statPillValue}>{stats.totalSessions}</div>
                <div style={styles.statPillLabel}>sessions</div>
              </div>
            </div>

            <div style={styles.ctaRow}>
              <button type="button" style={styles.primaryButton} onClick={() => navigation.navigate(primaryRoute)}>
                <span style={styles.primaryButtonText}>{primaryLabel}</span>
              </button>
              <button type="button" style={styles.secondaryButton} onClick={() => navigation.navigate('progress')}>
                <span style={styles.secondaryButtonText}>View progress</span>
              </button>
            </div>
          </div>

          <div style={styles.heroMascot}>
            <MonochromeMascot size="lg" />
          </div>
        </section>

        <div style={styles.dashboardRow}>
          <section style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>Today</h2>
              <div style={styles.panelMeta}>
                {challengeValue}/{stats.dailyChallenge.goal}
              </div>
            </div>
            <p style={styles.panelBody}>{stats.dailyChallenge.description}</p>
            <ProgressBar value={challengeProgress} hint={`${Math.round(challengeProgress)}%`} size="sm" />

            <div style={styles.metricRow}>
              <div style={styles.metricItem}>
                <div style={styles.metricValue}>{stats.currentCombo}x</div>
                <div style={styles.metricLabel}>combo</div>
              </div>
              <div style={styles.metricItem}>
                <div style={styles.metricValue}>{stats.dailyChallenge.rewardXP}</div>
                <div style={styles.metricLabel}>reward XP</div>
              </div>
              <div style={styles.metricItem}>
                <div style={styles.metricValue}>{Math.floor(stats.totalMinutes)}</div>
                <div style={styles.metricLabel}>minutes</div>
              </div>
            </div>

            <div style={styles.inlineDivider} />

            <h2 style={styles.panelTitle}>{stats.assessmentComplete ? 'Next step' : 'Start here'}</h2>
            <p style={styles.panelBody}>{nextStepText}</p>
          </section>

          <section style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>Progress</h2>
              <div style={styles.panelMeta}>{rankName}</div>
            </div>
            <ProgressBar value={progress} hint={`${Math.round(progress)}%`} />
            <p style={styles.panelBody}>{Math.floor(stats.xp)} XP earned so far. Keep sessions short and stack cleaner repeats.</p>

            <div style={styles.metricRow}>
              <div style={styles.metricItem}>
                <div style={styles.metricValue}>{stats.level}</div>
                <div style={styles.metricLabel}>level</div>
              </div>
              <div style={styles.metricItem}>
                <div style={styles.metricValue}>{stats.bestCombo}x</div>
                <div style={styles.metricLabel}>best combo</div>
              </div>
              <div style={styles.metricItem}>
                <div style={styles.metricValue}>{stats.soundsMastered.length}</div>
                <div style={styles.metricLabel}>mastered</div>
              </div>
            </div>

            <div style={styles.inlineDivider} />

            <h2 style={styles.panelTitle}>{recentWins.length ? 'Recent wins' : 'Milestones'}</h2>
            {recentWins.length ? (
              <div style={styles.winRow}>
                {recentWins.map((item) => (
                  <div key={item} style={styles.winChip}>
                    <span style={styles.winChipText}>{item}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={styles.panelBody}>Complete the assessment and first lesson to unlock the early milestones.</p>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    height: '100%',
  },
  content: {
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    minHeight: '100%',
  },
  contentFixed: {
    justifyContent: 'space-between',
  },
  loading: {
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    fontSize: 14,
    color: 'var(--color-text-subtle)',
  },
  hero: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 24,
    border: '1px solid var(--color-border)',
    padding: 22,
    display: 'flex',
    flexDirection: 'column',
    gap: 18,
  },
  heroWide: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  heroCopy: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    minWidth: 0,
  },
  heroMascot: {
    minWidth: 210,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  greeting: {
    fontSize: 14,
    color: 'var(--color-text-subtle)',
    fontWeight: 600,
  },
  title: {
    margin: 0,
    fontSize: 30,
    lineHeight: '34px',
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -1,
    maxWidth: 540,
  },
  body: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '20px',
    maxWidth: 520,
  },
  heroStats: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  statPill: {
    minWidth: 104,
    borderRadius: 14,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: '10px 12px',
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
  },
  statPillValue: {
    fontSize: 20,
    color: 'var(--color-text)',
    fontWeight: 800,
  },
  statPillLabel: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
  },
  ctaRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  primaryButton: {
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '14px 18px',
  },
  primaryButtonText: {
    color: 'var(--color-bg)',
    fontSize: 14,
    fontWeight: 700,
  },
  secondaryButton: {
    borderRadius: 12,
    border: '1px solid var(--color-border-strong)',
    padding: '12px 16px',
    backgroundColor: 'var(--color-surface)',
  },
  secondaryButtonText: {
    color: 'var(--color-text)',
    fontSize: 13,
    fontWeight: 600,
  },
  dashboardRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: 14,
  },
  panel: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 18,
    border: '1px solid var(--color-border)',
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    minHeight: 0,
  },
  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    gap: 12,
  },
  panelTitle: {
    margin: 0,
    fontSize: 20,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  panelMeta: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    whiteSpace: 'nowrap',
  },
  panelBody: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '21px',
  },
  metricRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  metricItem: {
    flex: '1 1 110px',
    minWidth: 0,
    borderRadius: 14,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: 12,
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 800,
    color: 'var(--color-text)',
  },
  metricLabel: {
    fontSize: 12,
    color: 'var(--color-text-subtle)',
    textTransform: 'lowercase',
  },
  inlineDivider: {
    height: 1,
    backgroundColor: 'var(--color-border)',
    margin: '2px 0',
  },
  winRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  winChip: {
    borderRadius: 999,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-accent-soft)',
    padding: '10px 14px',
  },
  winChipText: {
    fontSize: 13,
    color: 'var(--color-text)',
    fontWeight: 600,
    textTransform: 'capitalize',
  },
};
