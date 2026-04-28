import type { CSSProperties } from 'react';

import ProgressBar from '../components/ProgressBar';
import { ACHIEVEMENTS, LEVEL_XP, RANK_NAMES, useGame } from '../utils/gameState';
import useViewport from '../utils/useViewport';

function getProgressToNextLevel(level: number, xp: number) {
  const currentThreshold = LEVEL_XP[level - 1] || 0;
  const nextThreshold = LEVEL_XP[level] || LEVEL_XP[LEVEL_XP.length - 1];
  const span = Math.max(1, nextThreshold - currentThreshold);
  return Math.min(Math.max(((xp - currentThreshold) / span) * 100, 0), 100);
}

export default function ProgressScreen() {
  const { stats, reset } = useGame();
  const { width, height } = useViewport();
  const keepOnePage = width >= 1100 && height >= 780;
  const earnedCount = stats.achievements.length;
  const totalCount = Object.keys(ACHIEVEMENTS).length;
  const progress = getProgressToNextLevel(stats.level, stats.xp);
  const rankName = RANK_NAMES[Math.min(stats.level - 1, RANK_NAMES.length - 1)] || RANK_NAMES[0];

  return (
    <section style={{ ...styles.container, overflowY: keepOnePage ? 'hidden' : 'auto' }}>
      <div style={{ ...styles.content, ...(keepOnePage ? styles.contentFixed : null) }}>
        <header style={styles.header}>
          <h1 style={styles.title}>Streaks, XP, mastery, and milestones in one place.</h1>
        </header>

        <div style={styles.statsRow}>
          <div style={styles.statCard}>
            <div style={styles.statValue}>{stats.streak}</div>
            <div style={styles.statLabel}>day streak</div>
          </div>
          <div style={styles.statCard}>
            <div style={styles.statValue}>{stats.level}</div>
            <div style={styles.statLabel}>level</div>
          </div>
          <div style={styles.statCard}>
            <div style={styles.statValue}>{stats.totalSessions}</div>
            <div style={styles.statLabel}>sessions</div>
          </div>
          <div style={styles.statCard}>
            <div style={styles.statValue}>{Math.floor(stats.totalMinutes)}</div>
            <div style={styles.statLabel}>minutes</div>
          </div>
        </div>

        <div style={styles.grid}>
          <section style={styles.panel}>
            <h2 style={styles.panelTitle}>{Math.floor(stats.xp)} XP · {rankName}</h2>
            <ProgressBar value={progress} hint={`${Math.round(progress)}%`} />
            <p style={styles.panelBody}>You are moving from level {stats.level} toward level {stats.level + 1}.</p>

            <div style={styles.inlineDivider} />

            <h2 style={styles.panelTitle}>{stats.lispType ? `${stats.lispType} profile` : 'Assessment pending'}</h2>
            <p style={styles.panelBody}>
              {stats.lispType ? `Saved baseline severity: ${stats.severity}/10.` : 'Run the assessment to anchor the practice plan.'}
            </p>
            {stats.assessmentProfile ? (
              <p style={styles.panelBody}>Baseline phrase: “{stats.assessmentProfile.baselinePhrase}” · tutorial {stats.introTutorialComplete ? 'complete' : 'pending'}.</p>
            ) : null}

            <div style={styles.inlineDivider} />

            <h2 style={styles.panelTitle}>Sounds mastered</h2>
            {stats.soundsMastered.length ? (
              <div style={styles.soundRow}>
                {stats.soundsMastered.map((sound) => (
                  <div key={sound} style={styles.soundBadge}>
                    <span style={styles.soundText}>/{sound}/</span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={styles.panelBody}>No mastered sounds yet. Keep practicing clean repetitions.</p>
            )}

            <div style={styles.inlineDivider} />

            <h2 style={styles.panelTitle}>{stats.weeklyChallenge.title}</h2>
            <p style={styles.panelBody}>{stats.weeklyChallenge.description}</p>
            <ProgressBar
              value={(Math.min(stats.weeklyChallenge.progress, stats.weeklyChallenge.goal) / Math.max(1, stats.weeklyChallenge.goal)) * 100}
              hint={`${Math.min(stats.weeklyChallenge.progress, stats.weeklyChallenge.goal)}/${stats.weeklyChallenge.goal}`}
              size="sm"
            />
            <p style={styles.panelBody}>Reward: {stats.weeklyChallenge.rewardXP} XP</p>
          </section>

          <section style={styles.panel}>
            <h2 style={styles.panelTitle}>
              {earnedCount} of {totalCount} achievements earned
            </h2>
            <div style={styles.achievementList}>
              {Object.entries(ACHIEVEMENTS).map(([id, value]) => {
                const earned = stats.achievements.includes(id);
                return (
                  <div key={id} style={styles.achievementRow}>
                    <div style={{ ...styles.achievementDot, ...(earned ? styles.achievementDotEarned : styles.achievementDotLocked) }} />
                    <div style={styles.achievementCopy}>
                      <div style={{ ...styles.achievementName, ...(!earned ? styles.achievementLocked : null) }}>{value.name}</div>
                      <div style={{ ...styles.achievementDesc, ...(!earned ? styles.achievementLocked : null) }}>{value.desc}</div>
                    </div>
                  </div>
                );
              })}
            </div>

            <button type="button" style={styles.resetButton} onClick={reset}>
              <span style={styles.resetButtonText}>reset progress</span>
            </button>
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
    padding: 20,
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    minHeight: '100%',
  },
  contentFixed: {
    justifyContent: 'space-between',
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  title: {
    margin: 0,
    fontSize: 28,
    lineHeight: '32px',
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -0.9,
    maxWidth: 760,
  },
  statsRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
    gap: 14,
  },
  statCard: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 18,
    border: '1px solid var(--color-border)',
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  statValue: {
    fontSize: 28,
    color: 'var(--color-text)',
    fontWeight: 800,
  },
  statLabel: {
    fontSize: 13,
    color: 'var(--color-text-subtle)',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
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
  panelTitle: {
    margin: 0,
    fontSize: 20,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  panelBody: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '21px',
  },
  inlineDivider: {
    height: 1,
    backgroundColor: 'var(--color-border)',
    margin: '2px 0',
  },
  soundRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  soundBadge: {
    backgroundColor: 'var(--color-surface-alt)',
    borderRadius: 999,
    border: '1px solid var(--color-border)',
    padding: '10px 14px',
  },
  soundText: {
    fontSize: 14,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  achievementList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  achievementRow: {
    display: 'flex',
    gap: 12,
    alignItems: 'flex-start',
  },
  achievementDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
    marginTop: 6,
    flexShrink: 0,
  },
  achievementDotEarned: {
    backgroundColor: 'var(--color-success)',
  },
  achievementDotLocked: {
    backgroundColor: 'var(--color-border)',
  },
  achievementCopy: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  achievementName: {
    fontSize: 14,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
  achievementDesc: {
    fontSize: 13,
    color: 'var(--color-text-muted)',
    lineHeight: '18px',
  },
  achievementLocked: {
    color: 'var(--color-text-subtle)',
  },
  resetButton: {
    alignSelf: 'flex-start',
    borderRadius: 12,
    border: '1px solid var(--color-border-strong)',
    padding: '12px 16px',
    backgroundColor: 'var(--color-surface)',
    marginTop: 8,
  },
  resetButtonText: {
    fontSize: 14,
    color: 'var(--color-text)',
    fontWeight: 700,
  },
};
