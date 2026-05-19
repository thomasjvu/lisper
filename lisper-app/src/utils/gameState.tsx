import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';

export type LispType = 'frontal' | 'lateral' | 'dental' | 'palatal';
export type ChallengeMetric = 'attempts' | 'good_attempts' | 'sessions' | 'xp' | 'minutes';

export interface AssessmentProfile {
  baselinePhrase: string;
  transcript: string;
  notes: string;
  mouthShapeNotes: string;
  completedAt: string;
}

export interface ChallengeState {
  id: string;
  title: string;
  description: string;
  metric: ChallengeMetric;
  goal: number;
  progress: number;
  periodKey: string;
  rewardXP: number;
}

interface UserStats {
  level: number;
  xp: number;
  streak: number;
  lastPractice: string;
  totalSessions: number;
  totalMinutes: number;
  soundsMastered: string[];
  achievements: string[];
  assessmentComplete: boolean;
  lispType: LispType | null;
  severity: number;
  assessmentProfile: AssessmentProfile | null;
  introTutorialComplete: boolean;
  currentCombo: number;
  bestCombo: number;
  dailyChallenge: ChallengeState;
  weeklyChallenge: ChallengeState;
  completedChallengeKeys: string[];
}

export interface GameStateValue {
  stats: UserStats;
  loading: boolean;
  addXP: (amount: number) => void;
  completeSession: (minutes: number, score: number) => void;
  unlockAchievement: (id: string) => void;
  setAssessment: (type: LispType, severity: number, profile?: AssessmentProfile | null) => void;
  completeIntroTutorial: () => void;
  masterSound: (sound: string) => void;
  recordAttempt: (score: number) => void;
  reset: () => void;
}

interface ChallengeTemplate {
  id: string;
  title: string;
  description: string;
  metric: ChallengeMetric;
  goal: number;
  rewardXP: number;
}

const STORAGE_KEY = 'lisper:web-state:v3';

export const ACHIEVEMENTS = {
  first_step: { name: 'first step', desc: 'complete assessment' },
  day_3_streak: { name: '3 day streak', desc: 'practice 3 days' },
  day_7_streak: { name: '7 day streak', desc: 'practice 7 days' },
  day_30_streak: { name: '30 day streak', desc: 'practice 30 days' },
  sound_master: { name: 'sound master', desc: 'master /s/ sound' },
  first_session: { name: 'first session', desc: 'complete first session' },
  perfect_round: { name: 'perfect round', desc: 'score 100% in a session' },
  consistent: { name: 'consistent', desc: '10 total sessions' },
  sharpshooter: { name: 'sharpshooter', desc: 'hit a 5 attempt clean combo' },
} as const;

export const LEVEL_XP = [0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700];
export const RANK_NAMES = ['Foundation', 'Steady', 'Clearer', 'Focused', 'Refined', 'Precise', 'Polished', 'Confident', 'Advanced', 'Elite'];

const DAILY_TEMPLATES: ChallengeTemplate[] = [
  {
    id: 'daily_attempts',
    title: 'Daily challenge',
    description: 'Analyze 3 focused attempts today.',
    metric: 'attempts',
    goal: 3,
    rewardXP: 20,
  },
  {
    id: 'daily_clean',
    title: 'Daily challenge',
    description: 'Land 2 clean attempts at 80% or better.',
    metric: 'good_attempts',
    goal: 2,
    rewardXP: 24,
  },
  {
    id: 'daily_session',
    title: 'Daily challenge',
    description: 'Finish 1 full session today.',
    metric: 'sessions',
    goal: 1,
    rewardXP: 18,
  },
];

const WEEKLY_TEMPLATES: ChallengeTemplate[] = [
  {
    id: 'weekly_sessions',
    title: 'Weekly milestone',
    description: 'Complete 4 sessions this week.',
    metric: 'sessions',
    goal: 4,
    rewardXP: 60,
  },
  {
    id: 'weekly_xp',
    title: 'Weekly milestone',
    description: 'Earn 250 XP this week.',
    metric: 'xp',
    goal: 250,
    rewardXP: 70,
  },
  {
    id: 'weekly_minutes',
    title: 'Weekly milestone',
    description: 'Practice for 20 minutes this week.',
    metric: 'minutes',
    goal: 20,
    rewardXP: 56,
  },
];

function toDayKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function toWeekKey(date: Date) {
  const value = new Date(date);
  const day = (value.getDay() + 6) % 7;
  value.setHours(0, 0, 0, 0);
  value.setDate(value.getDate() - day);
  return toDayKey(value);
}

function chooseTemplate<T>(items: T[], periodKey: string) {
  const seed = periodKey.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return items[seed % items.length];
}

function createChallenge(templates: ChallengeTemplate[], periodKey: string): ChallengeState {
  const template = chooseTemplate(templates, periodKey);
  return {
    ...template,
    progress: 0,
    periodKey,
  };
}

function createDefaultStats(now = new Date()): UserStats {
  const todayKey = toDayKey(now);
  const weekKey = toWeekKey(now);

  return {
    level: 1,
    xp: 0,
    streak: 0,
    lastPractice: '',
    totalSessions: 0,
    totalMinutes: 0,
    soundsMastered: [],
    achievements: [],
    assessmentComplete: false,
    lispType: null,
    severity: 0,
    assessmentProfile: null,
    introTutorialComplete: false,
    currentCombo: 0,
    bestCombo: 0,
    dailyChallenge: createChallenge(DAILY_TEMPLATES, todayKey),
    weeklyChallenge: createChallenge(WEEKLY_TEMPLATES, weekKey),
    completedChallengeKeys: [],
  };
}

function readStoredState(): UserStats | null {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as Partial<UserStats>;
    const defaults = createDefaultStats();
    return {
      ...defaults,
      ...parsed,
      lispType: parsed.lispType ?? null,
      severity: Number.isFinite(parsed.severity) ? Number(parsed.severity) : 0,
      assessmentProfile:
        parsed.assessmentProfile &&
        typeof parsed.assessmentProfile.baselinePhrase === 'string' &&
        typeof parsed.assessmentProfile.transcript === 'string'
          ? {
              baselinePhrase: parsed.assessmentProfile.baselinePhrase,
              transcript: parsed.assessmentProfile.transcript,
              notes: typeof parsed.assessmentProfile.notes === 'string' ? parsed.assessmentProfile.notes : '',
              mouthShapeNotes:
                typeof parsed.assessmentProfile.mouthShapeNotes === 'string' ? parsed.assessmentProfile.mouthShapeNotes : '',
              completedAt: typeof parsed.assessmentProfile.completedAt === 'string' ? parsed.assessmentProfile.completedAt : '',
            }
          : defaults.assessmentProfile,
      introTutorialComplete: typeof parsed.introTutorialComplete === 'boolean' ? parsed.introTutorialComplete : false,
      currentCombo: Number.isFinite(parsed.currentCombo) ? Number(parsed.currentCombo) : 0,
      bestCombo: Number.isFinite(parsed.bestCombo) ? Number(parsed.bestCombo) : 0,
      soundsMastered: Array.isArray(parsed.soundsMastered) ? parsed.soundsMastered : [],
      achievements: Array.isArray(parsed.achievements) ? parsed.achievements : [],
      completedChallengeKeys: Array.isArray(parsed.completedChallengeKeys) ? parsed.completedChallengeKeys : [],
      dailyChallenge: parsed.dailyChallenge
        ? { ...defaults.dailyChallenge, ...parsed.dailyChallenge }
        : defaults.dailyChallenge,
      weeklyChallenge: parsed.weeklyChallenge
        ? { ...defaults.weeklyChallenge, ...parsed.weeklyChallenge }
        : defaults.weeklyChallenge,
    };
  } catch (error) {
    console.warn('[GameState] Failed to read saved state', error);
    return null;
  }
}

function saveStoredState(stats: UserStats) {
  if (typeof window === 'undefined' || !window.localStorage) {
    return;
  }

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stats));
  } catch (error) {
    console.warn('[GameState] Failed to persist state', error);
  }
}

function clearStoredState() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return;
  }

  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.warn('[GameState] Failed to clear state', error);
  }
}

function getLevelForXP(xp: number): number {
  for (let i = LEVEL_XP.length - 1; i >= 0; i -= 1) {
    if (xp >= LEVEL_XP[i]) {
      return i + 1;
    }
  }

  return 1;
}

function computeStreak(lastPractice: string, currentStreak: number): number {
  if (!lastPractice) {
    return 0;
  }

  const today = toDayKey(new Date());
  const lastDay = lastPractice.split('T')[0];

  if (lastDay === today) {
    return currentStreak;
  }

  const lastDate = new Date(lastPractice);
  const diffDays = Math.floor((Date.now() - lastDate.getTime()) / (1000 * 60 * 60 * 24));
  return diffDays <= 1 ? currentStreak : 0;
}

function withAchievements(stats: UserStats, score: number): string[] {
  const achievements = new Set(stats.achievements);

  if (stats.assessmentComplete) {
    achievements.add('first_step');
  }
  if (stats.totalSessions >= 1) {
    achievements.add('first_session');
  }
  if (stats.totalSessions >= 10) {
    achievements.add('consistent');
  }
  if (stats.streak >= 3) {
    achievements.add('day_3_streak');
  }
  if (stats.streak >= 7) {
    achievements.add('day_7_streak');
  }
  if (stats.streak >= 30) {
    achievements.add('day_30_streak');
  }
  if (score >= 100) {
    achievements.add('perfect_round');
  }
  if (stats.soundsMastered.includes('s') || stats.soundsMastered.includes('z')) {
    achievements.add('sound_master');
  }
  if (stats.bestCombo >= 5) {
    achievements.add('sharpshooter');
  }

  return Array.from(achievements);
}

function ensureCurrentChallenges(stats: UserStats, now = new Date()): UserStats {
  const todayKey = toDayKey(now);
  const weekKey = toWeekKey(now);

  return {
    ...stats,
    dailyChallenge:
      stats.dailyChallenge?.periodKey === todayKey ? stats.dailyChallenge : createChallenge(DAILY_TEMPLATES, todayKey),
    weeklyChallenge:
      stats.weeklyChallenge?.periodKey === weekKey ? stats.weeklyChallenge : createChallenge(WEEKLY_TEMPLATES, weekKey),
    currentCombo: stats.dailyChallenge?.periodKey === todayKey ? stats.currentCombo : 0,
  };
}

function updateChallenge(challenge: ChallengeState, metric: ChallengeMetric, amount: number) {
  if (challenge.metric !== metric || amount <= 0) {
    return challenge;
  }

  return {
    ...challenge,
    progress: Math.min(challenge.goal, challenge.progress + amount),
  };
}

function completeChallengeKey(challenge: ChallengeState) {
  return `${challenge.id}:${challenge.periodKey}`;
}

function finalizeChallengeRewards(stats: UserStats) {
  let next = stats;

  [stats.dailyChallenge, stats.weeklyChallenge].forEach((challenge) => {
    const key = completeChallengeKey(challenge);
    if (challenge.progress < challenge.goal || next.completedChallengeKeys.includes(key)) {
      return;
    }

    const xp = next.xp + challenge.rewardXP;
    next = {
      ...next,
      xp,
      level: getLevelForXP(xp),
      completedChallengeKeys: [...next.completedChallengeKeys, key],
    };
  });

  return next;
}

const GameContext = createContext<GameStateValue | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
  const [stats, setStats] = useState<UserStats>(() => createDefaultStats());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = readStoredState();
    if (stored) {
      const next = ensureCurrentChallenges({
        ...stored,
        streak: computeStreak(stored.lastPractice, stored.streak),
      });
      setStats(next);
    } else {
      setStats(createDefaultStats());
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (!loading) {
      saveStoredState(stats);
    }
  }, [loading, stats]);

  const value = useMemo<GameStateValue>(() => {
    function addXP(amount: number) {
      if (amount <= 0) {
        return;
      }

      setStats((current) => {
        let next = ensureCurrentChallenges(current);
        const xp = next.xp + amount;
        next = {
          ...next,
          xp,
          level: getLevelForXP(xp),
          weeklyChallenge: updateChallenge(next.weeklyChallenge, 'xp', amount),
        };

        next = finalizeChallengeRewards(next);
        return {
          ...next,
          achievements: withAchievements(next, 0),
        };
      });
    }

    function unlockAchievement(id: string) {
      setStats((current) => {
        if (current.achievements.includes(id)) {
          return current;
        }

        return {
          ...current,
          achievements: [...current.achievements, id],
        };
      });
    }

    function setAssessment(type: LispType, severity: number, profile?: AssessmentProfile | null) {
      setStats((current) => {
        const next = ensureCurrentChallenges({
          ...current,
          assessmentComplete: true,
          lispType: type,
          severity,
          assessmentProfile: profile ?? current.assessmentProfile,
          introTutorialComplete: false,
        });

        return {
          ...next,
          achievements: withAchievements(next, 0),
        };
      });
    }

    function completeIntroTutorial() {
      setStats((current) => ({
        ...current,
        introTutorialComplete: true,
      }));
    }

    function masterSound(sound: string) {
      setStats((current) => {
        if (current.soundsMastered.includes(sound)) {
          return current;
        }

        const next = ensureCurrentChallenges({
          ...current,
          soundsMastered: [...current.soundsMastered, sound],
        });

        return {
          ...next,
          achievements: withAchievements(next, 0),
        };
      });
    }

    function recordAttempt(score: number) {
      setStats((current) => {
        let next = ensureCurrentChallenges(current);
        const goodAttempt = score >= 80;
        const currentCombo = goodAttempt ? next.currentCombo + 1 : 0;
        const comboBonusXP = goodAttempt && currentCombo > 1 ? Math.min(18, 4 * (currentCombo - 1)) : 0;
        const xp = next.xp + comboBonusXP;

        next = {
          ...next,
          currentCombo,
          bestCombo: Math.max(next.bestCombo, currentCombo),
          xp,
          level: getLevelForXP(xp),
          dailyChallenge: updateChallenge(next.dailyChallenge, 'attempts', 1),
          weeklyChallenge: updateChallenge(next.weeklyChallenge, 'xp', comboBonusXP),
        };

        if (goodAttempt) {
          next = {
            ...next,
            dailyChallenge: updateChallenge(next.dailyChallenge, 'good_attempts', 1),
          };
        }

        next = finalizeChallengeRewards(next);
        return {
          ...next,
          achievements: withAchievements(next, score),
        };
      });
    }

    function completeSession(minutes: number, score: number) {
      setStats((current) => {
        let next = ensureCurrentChallenges(current);
        const now = new Date();
        const nowIso = now.toISOString();
        const today = toDayKey(now);
        const practicedToday = next.lastPractice.split('T')[0] === today;
        const lastDate = next.lastPractice ? new Date(next.lastPractice) : null;
        const dayDiff = lastDate
          ? Math.floor((Date.now() - lastDate.getTime()) / (1000 * 60 * 60 * 24))
          : 0;

        let streak = next.streak;
        if (!practicedToday) {
          streak = next.lastPractice && dayDiff > 1 ? 1 : next.streak + 1;
        }

        const earnedXP = Math.max(10, Math.floor(minutes * 10 + score * 0.5));
        const xp = next.xp + earnedXP;

        next = {
          ...next,
          xp,
          level: getLevelForXP(xp),
          streak,
          lastPractice: nowIso,
          totalSessions: next.totalSessions + 1,
          totalMinutes: next.totalMinutes + minutes,
          currentCombo: 0,
          dailyChallenge: updateChallenge(next.dailyChallenge, 'sessions', 1),
          weeklyChallenge: updateChallenge(updateChallenge(next.weeklyChallenge, 'sessions', 1), 'minutes', minutes),
        };

        next = {
          ...next,
          weeklyChallenge: updateChallenge(next.weeklyChallenge, 'xp', earnedXP),
        };

        next = finalizeChallengeRewards(next);
        return {
          ...next,
          achievements: withAchievements(next, score),
        };
      });
    }

    function reset() {
      clearStoredState();
      setStats(createDefaultStats());
    }

    return {
      stats,
      loading,
      addXP,
      completeSession,
      unlockAchievement,
      setAssessment,
      completeIntroTutorial,
      masterSound,
      recordAttempt,
      reset,
    };
  }, [loading, stats]);

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame() {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error('useGame must be used within GameProvider');
  }

  return context;
}
