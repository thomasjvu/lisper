import type { LispType } from './gameState';

export const SOUND_TARGETS = [
  { text: '/s/', level: 'isolation', hint: 'keep tongue back' },
  { text: '/z/', level: 'isolation', hint: 'add voice to /s/' },
  { text: 'sa', level: 'syllables', hint: 's + a' },
  { text: 'se', level: 'syllables', hint: 's + e' },
  { text: 'si', level: 'syllables', hint: 's + i' },
  { text: 'so', level: 'syllables', hint: 's + o' },
  { text: 'sun', level: 'words', hint: 's + uh + n' },
  { text: 'sad', level: 'words', hint: 's + a + d' },
  { text: 'sea', level: 'words', hint: 's + ee' },
  { text: 'zip', level: 'words', hint: 'z + i + p' },
  { text: 'zoo', level: 'words', hint: 'z + oo' },
  { text: 'seashells', level: 'phrases', hint: 'see + shell + s' },
  { text: 'Sally sells seashells', level: 'phrases', hint: 'slow and steady' },
] as const;

export function getDefaultFeedback(lispType: LispType, severity: number): string {
  if (lispType === 'frontal') {
    return severity >= 6
      ? 'Pull your tongue back behind your upper teeth and send the air straight forward.'
      : 'Keep your tongue just behind your upper teeth and hold a narrow stream of air.';
  }
  if (lispType === 'lateral') {
    return 'Seal the sides of your tongue against your upper teeth so the air moves down the middle.';
  }
  if (lispType === 'dental') {
    return 'Leave a tiny gap between your tongue and your teeth so the sound does not flatten out.';
  }
  return 'Bring your tongue a little farther forward and keep the tip near the ridge behind your teeth.';
}

export function getDefaultEncouragement(score: number, streak: number): string {
  if (score >= 90) {
    return streak >= 3 ? 'Locked in. Keep the streak alive.' : 'Clear sound. Repeat that pattern.';
  }
  if (score >= 70) {
    return 'Close. One more clean repetition.';
  }
  if (score >= 50) {
    return 'You are shaping it. Slow down and try again.';
  }
  return 'Start small. One clean sound is enough.';
}

export function getDefaultTip(sound: string, lispType: LispType): string {
  if (lispType === 'lateral') {
    return `For ${sound}, keep the tongue edges lifted and the airflow centered.`;
  }
  if (lispType === 'frontal') {
    return `For ${sound}, keep your tongue behind your front teeth and avoid pushing it forward.`;
  }
  if (lispType === 'dental') {
    return `For ${sound}, ease your tongue off the teeth and let the air pass through a narrow gap.`;
  }
  return `For ${sound}, bring the tongue forward and aim the air straight out.`;
}

export function normalizePracticeText(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9/ ]+/g, '').trim();
}

export function scoreTranscript(targetText: string, transcript: string, severity: number): number {
  const normalizedTarget = normalizePracticeText(targetText);
  const normalizedTranscript = normalizePracticeText(transcript);

  if (!normalizedTranscript) {
    return 0;
  }

  if (normalizedTranscript.includes(normalizedTarget) || normalizedTarget.includes(normalizedTranscript)) {
    return Math.max(70, 100 - severity * 4);
  }

  const shared = normalizedTarget
    .split(' ')
    .filter((token) => token && normalizedTranscript.includes(token)).length;

  if (shared > 0) {
    return Math.max(45, 75 - severity * 3);
  }

  return Math.max(20, 55 - severity * 3);
}
