import { useMemo, useState } from 'react';
import type { CSSProperties } from 'react';

import AudioClipInput from '../components/AudioClipInput';
import MonochromeMascot from '../components/MonochromeMascot';
import SampledFramesStrip from '../components/SampledFramesStrip';
import type { AssessmentResult, CoachResult, PracticeCapture } from '../utils/modelRuntime';
import { analyzePractice } from '../utils/modelRuntime';
import { usePreferences } from '../utils/preferences';
import { getDefaultFeedback, getDefaultTip, scoreTranscript, SOUND_TARGETS } from '../utils/speechService';
import useViewport from '../utils/useViewport';
import { speakText } from '../utils/webSpeech';
import { useGame } from '../utils/gameState';

const LEVEL_NAMES = ['isolation', 'syllables', 'words', 'phrases', 'sentences'];

interface TutorialTarget {
  phase: string;
  text: string;
  hint: string;
  coaching: string;
}

export default function TrainingScreen() {
  const { stats, addXP, completeSession, completeIntroTutorial, masterSound, recordAttempt } = useGame();
  const { soundEnabled } = usePreferences();
  const { width, height } = useViewport();
  const wide = width >= 1180;
  const keepOnePage = width >= 1180 && height >= 820;

  const [targetIndex, setTargetIndex] = useState(0);
  const [tutorialIndex, setTutorialIndex] = useState(0);
  const [capture, setCapture] = useState<PracticeCapture | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transcriptText, setTranscriptText] = useState('');
  const [assessmentResult, setAssessmentResult] = useState<AssessmentResult | null>(null);
  const [coaching, setCoaching] = useState<CoachResult | null>(null);
  const [score, setScore] = useState(0);
  const [totalAttempts, setTotalAttempts] = useState(0);
  const [sessionStart] = useState(Date.now());

  const tutorialActive = stats.assessmentComplete && !stats.introTutorialComplete;
  const fallbackLispType = stats.lispType ?? 'frontal';
  const tutorialTargets = useMemo<TutorialTarget[]>(
    () => [
      {
        phase: 'tutorial 1 of 3',
        text: '/s/',
        hint: 'Start small with one centered hiss.',
        coaching: getDefaultTip('s', fallbackLispType),
      },
      {
        phase: 'tutorial 2 of 3',
        text: 'sun',
        hint: 'Keep the opening sound clean before you finish the word.',
        coaching: getDefaultTip('s', fallbackLispType),
      },
      {
        phase: 'tutorial 3 of 3',
        text: 'Sally sells seashells',
        hint: 'Stay slow enough to keep the /s/ shape through the whole phrase.',
        coaching: getDefaultFeedback(fallbackLispType, stats.severity || 5),
      },
    ],
    [fallbackLispType, stats.severity]
  );

  const currentLevel = LEVEL_NAMES[Math.min(stats.level - 1, LEVEL_NAMES.length - 1)] || LEVEL_NAMES[0];
  const targets = useMemo(
    () => SOUND_TARGETS.filter((target) => target.level === currentLevel || (currentLevel === 'sentences' && target.level === 'phrases')),
    [currentLevel]
  );
  const tutorialTarget = tutorialTargets[Math.min(tutorialIndex, tutorialTargets.length - 1)];
  const practiceTarget = targets[targetIndex % Math.max(targets.length, 1)] ?? SOUND_TARGETS[0];
  const currentTarget = tutorialActive ? tutorialTarget : practiceTarget;

  function resetAttempt() {
    setCapture(null);
    setTranscriptText('');
    setAssessmentResult(null);
    setCoaching(null);
    setError(null);
  }

  function nextTarget() {
    resetAttempt();
    setTargetIndex((current) => current + 1);
  }

  function advanceTutorial() {
    if (tutorialIndex < tutorialTargets.length - 1) {
      resetAttempt();
      setTutorialIndex((current) => current + 1);
      return;
    }

    completeIntroTutorial();
    setTutorialIndex(0);
    resetAttempt();
  }

  async function analyzeClip() {
    if (!capture) {
      setError('Record or upload an attempt first.');
      return;
    }

    try {
      setBusy(true);
      setError(null);

      const result = await analyzePractice(capture, currentTarget.text);
      const nextAssessment = result.assessment;
      const nextCoaching = result.coaching;

      const sessionScore = scoreTranscript(currentTarget.text, result.transcript, nextAssessment.severity);
      const total = totalAttempts + 1;

      setTranscriptText(result.transcript);
      setAssessmentResult(nextAssessment);
      setCoaching(nextCoaching);
      setScore((current) => current + sessionScore);
      setTotalAttempts(total);
      recordAttempt(sessionScore);
      addXP(Math.max(8, Math.floor(sessionScore / 8)));

      const sound = currentTarget.text.replace(/[^a-z]/gi, '').charAt(0).toLowerCase();
      if (sessionScore >= 80 && (sound === 's' || sound === 'z')) {
        masterSound(sound);
      }
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'Gemma 4 analysis failed.');
    } finally {
      setBusy(false);
    }
  }

  function finishSession() {
    const minutes = Math.max(1, Math.floor((Date.now() - sessionStart) / 60000));
    const sessionScore = totalAttempts > 0 ? Math.round(score / totalAttempts) : 0;
    completeSession(minutes, sessionScore);
    setScore(0);
    setTotalAttempts(0);
    nextTarget();
  }

  const averageScore = totalAttempts > 0 ? Math.round(score / totalAttempts) : 0;

  return (
    <section style={{ ...styles.container, overflowY: keepOnePage ? 'hidden' : 'auto' }}>
      <div style={{ ...styles.content, ...(keepOnePage ? styles.contentFixed : null) }}>
        <header style={styles.header}>
          <h1 style={styles.title}>
            {tutorialActive ? 'Use the saved baseline to move through three coached first reps.' : 'Stay small, stay clear, then repeat it once better.'}
          </h1>
          <div style={styles.chipRow}>
            <div style={styles.chip}><span style={styles.chipText}>{tutorialActive ? tutorialTarget.phase : currentLevel}</span></div>
            <div style={styles.chip}><span style={styles.chipText}>level {stats.level}</span></div>
            <div style={styles.chip}><span style={styles.chipText}>avg {averageScore}%</span></div>
            <div style={styles.chip}><span style={styles.chipText}>combo {stats.currentCombo}x</span></div>
          </div>
        </header>

        <div style={{ ...styles.layout, ...(wide ? styles.layoutWide : null) }}>
          <div style={styles.primaryColumn}>
            <section style={styles.panel}>
              {tutorialActive ? <div style={styles.stepBadge}>{tutorialTarget.phase}</div> : null}
              <h2 style={styles.targetText}>{currentTarget.text}</h2>
              <p style={styles.targetHint}>{currentTarget.hint}</p>

              {tutorialActive ? (
                <div style={styles.callout}>
                  <div style={styles.calloutTitle}>Saved baseline cue</div>
                  <div style={styles.calloutText}>
                    {stats.assessmentProfile?.notes || `${fallbackLispType} · severity ${stats.severity}/10`}
                  </div>
                </div>
              ) : null}

              <div style={styles.inlineRow}>
                <button
                  type="button"
                  style={{ ...styles.primaryButton, ...(!soundEnabled ? styles.buttonDisabled : null) }}
                  onClick={() => {
                    if (soundEnabled) {
                      speakText(currentTarget.text);
                    }
                  }}
                  disabled={!soundEnabled}
                >
                  <span style={styles.primaryButtonText}>{soundEnabled ? 'listen' : 'sound off'}</span>
                </button>
                {!tutorialActive ? (
                  <button type="button" style={styles.secondaryButton} onClick={nextTarget}>
                    <span style={styles.secondaryButtonText}>next target</span>
                  </button>
                ) : null}
              </div>
            </section>

            <AudioClipInput
              label={tutorialActive ? 'guided tutorial attempt' : 'practice attempt'}
              capture={capture}
              analyzing={busy}
              onCaptureReady={(nextCapture) => {
                setCapture(nextCapture);
                setTranscriptText('');
                setAssessmentResult(null);
                setCoaching(null);
                setError(null);
              }}
            />

            <div style={styles.actionRow}>
              <button
                type="button"
                style={{ ...styles.primaryButton, ...((!capture || busy) ? styles.buttonDisabled : null) }}
                onClick={analyzeClip}
                disabled={!capture || busy}
              >
                <span style={styles.primaryButtonText}>{busy ? 'running analysis...' : tutorialActive ? 'analyze guided rep' : 'analyze attempt'}</span>
              </button>

              {tutorialActive ? (
                <button
                  type="button"
                  style={{ ...styles.secondaryButton, ...((!assessmentResult || !coaching) ? styles.buttonDisabled : null) }}
                  onClick={advanceTutorial}
                  disabled={!assessmentResult || !coaching}
                >
                  <span style={styles.secondaryButtonText}>
                    {tutorialIndex === tutorialTargets.length - 1 ? 'finish tutorial' : 'next tutorial step'}
                  </span>
                </button>
              ) : (
                <button type="button" style={styles.secondaryButton} onClick={finishSession}>
                  <span style={styles.secondaryButtonText}>end session</span>
                </button>
              )}
            </div>

            {error ? <div style={styles.error}>{error}</div> : null}
          </div>

          <div style={styles.sideColumn}>
            <section style={styles.panel}>
              <div style={styles.metricGrid}>
                <div style={styles.metricCard}>
                  <div style={styles.metricValue}>{totalAttempts}</div>
                  <div style={styles.metricLabel}>attempts</div>
                </div>
                <div style={styles.metricCard}>
                  <div style={styles.metricValue}>{averageScore}%</div>
                  <div style={styles.metricLabel}>average</div>
                </div>
                <div style={styles.metricCard}>
                  <div style={styles.metricValue}>{capture ? Math.max(1, Math.round(capture.durationMs / 1000)) : 0}s</div>
                  <div style={styles.metricLabel}>last clip</div>
                </div>
                <div style={styles.metricCard}>
                  <div style={styles.metricValue}>
                    {Math.min(stats.dailyChallenge.progress, stats.dailyChallenge.goal)}/{stats.dailyChallenge.goal}
                  </div>
                  <div style={styles.metricLabel}>daily</div>
                </div>
              </div>

              {assessmentResult && coaching ? (
                <>
                  <SampledFramesStrip frames={capture?.frames ?? []} />

                  <div style={styles.sectionLabel}>Visible cue</div>
                  <p style={styles.bodyText}>{assessmentResult.mouthShapeNotes}</p>

                  <div style={styles.sectionLabel}>Transcript</div>
                  <p style={styles.bodyText}>{transcriptText || 'No transcript returned.'}</p>

                  <div style={styles.sectionLabel}>Result</div>
                  <p style={styles.bodyText}>
                    {assessmentResult.lispType} · severity {assessmentResult.severity}/10 · confidence {Math.round(assessmentResult.confidence * 100)}%
                  </p>

                  <div style={styles.callout}>
                    <div style={styles.calloutTitle}>{tutorialActive ? 'Next tutorial cue' : 'Next try cue'}</div>
                    <div style={styles.calloutText}>{coaching.nextTryCue}</div>
                  </div>

                  {tutorialActive ? (
                    <>
                      <div style={styles.sectionLabel}>Saved baseline</div>
                      <p style={styles.bodyText}>
                        {stats.assessmentProfile?.baselinePhrase || 'Baseline profile saved.'} · {stats.assessmentProfile?.notes || `${fallbackLispType} cue`}
                      </p>
                    </>
                  ) : (
                    <>
                      <div style={styles.sectionLabel}>Daily challenge</div>
                      <p style={styles.bodyText}>{stats.dailyChallenge.description}</p>
                    </>
                  )}

                  <div style={styles.coachBubble}>
                    <div style={styles.coachMascot}>
                      <MonochromeMascot size="sm" talking />
                    </div>
                    <div style={styles.coachCopy}>
                      <div style={styles.sectionLabel}>Coach</div>
                      <p style={styles.bodyText}>{coaching.feedback}</p>
                    </div>
                  </div>
                </>
              ) : (
                <div style={styles.emptyState}>
                  <MonochromeMascot size="sm" />
                  <h2 style={styles.emptyTitle}>
                    {tutorialActive ? 'The first three reps stay guided and tied to the saved baseline.' : 'Feedback lands here after each attempt.'}
                  </h2>
                  <p style={styles.emptyBody}>
                    {tutorialActive
                      ? 'Follow the current cue, analyze the rep, then move to the next card.'
                      : 'Keep the phrase short and aim for one cleaner repeat, not a perfect run.'}
                  </p>
                </div>
              )}
            </section>
          </div>
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
    gap: 10,
  },
  title: {
    margin: 0,
    fontSize: 28,
    lineHeight: '32px',
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -0.9,
    maxWidth: 820,
  },
  chipRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  chip: {
    borderRadius: 999,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: '8px 12px',
  },
  chipText: {
    fontSize: 12,
    color: 'var(--color-text)',
    fontWeight: 600,
    textTransform: 'lowercase',
  },
  layout: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    minHeight: 0,
  },
  layoutWide: {
    flexDirection: 'row',
    alignItems: 'stretch',
  },
  primaryColumn: {
    flex: 1.06,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    minWidth: 0,
  },
  sideColumn: {
    flex: 0.94,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    minWidth: 0,
  },
  panel: {
    backgroundColor: 'var(--color-surface)',
    borderRadius: 18,
    border: '1px solid var(--color-border)',
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  coachBubble: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    borderRadius: 16,
    border: '1px solid rgba(88, 255, 140, 0.22)',
    backgroundColor: 'rgba(10, 30, 23, 0.42)',
    padding: 12,
  },
  coachMascot: {
    flex: '0 0 auto',
    transform: 'scale(0.82)',
    transformOrigin: 'center',
    margin: -10,
  },
  coachCopy: {
    minWidth: 0,
    flex: 1,
  },
  stepBadge: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  targetText: {
    margin: 0,
    fontSize: 32,
    lineHeight: '34px',
    fontWeight: 800,
    color: 'var(--color-text)',
    letterSpacing: -0.9,
  },
  targetHint: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '20px',
  },
  inlineRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  actionRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  primaryButton: {
    backgroundColor: 'var(--color-accent)',
    borderRadius: 12,
    padding: '14px 16px',
    minWidth: 148,
    display: 'inline-flex',
    justifyContent: 'center',
    alignItems: 'center',
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
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'var(--color-surface)',
  },
  secondaryButtonText: {
    color: 'var(--color-text)',
    fontSize: 13,
    fontWeight: 600,
  },
  buttonDisabled: {
    opacity: 0.55,
  },
  error: {
    fontSize: 13,
    color: 'var(--color-danger)',
  },
  metricGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
    gap: 10,
  },
  metricCard: {
    borderRadius: 14,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-surface-alt)',
    padding: '12px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  metricValue: {
    fontSize: 22,
    color: 'var(--color-text)',
    fontWeight: 800,
  },
  metricLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  sectionLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  bodyText: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '21px',
  },
  callout: {
    padding: '12px 14px',
    borderRadius: 14,
    backgroundColor: 'var(--color-accent-soft)',
    border: '1px solid var(--color-border-strong)',
  },
  calloutTitle: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 6,
  },
  calloutText: {
    fontSize: 13,
    color: 'var(--color-text)',
    lineHeight: '18px',
  },
  emptyState: {
    minHeight: 360,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
  },
  emptyTitle: {
    fontSize: 20,
    color: 'var(--color-text)',
    fontWeight: 700,
    maxWidth: 360,
  },
  emptyBody: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '20px',
    maxWidth: 360,
  },
};
