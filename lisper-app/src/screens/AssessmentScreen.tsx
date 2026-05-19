import { useState } from 'react';
import type { CSSProperties } from 'react';

import AudioClipInput from '../components/AudioClipInput';
import MonochromeMascot from '../components/MonochromeMascot';
import SampledFramesStrip from '../components/SampledFramesStrip';
import type { PracticeAnalysisResult, PracticeCapture } from '../utils/modelRuntime';
import { analyzePractice } from '../utils/modelRuntime';
import { type LispType, useGame } from '../utils/gameState';
import { getDefaultFeedback, getDefaultTip } from '../utils/speechService';
import useViewport from '../utils/useViewport';
import { usePreferences } from '../utils/preferences';
import { speakText } from '../utils/webSpeech';

interface AssessmentScreenProps {
  onComplete: () => void;
}

type AssessmentStep = 'setup' | 'baseline' | 'guided-sounds' | 'guided-phrase';

const LISP_TYPES: LispType[] = ['frontal', 'lateral', 'dental', 'palatal'];
const BASELINE_PHRASE = 'I see the sun outside today.';
const GUIDED_SOUND_PROMPT = 'Say each one once with a small pause: /s/, /z/, sun, zoo.';
const GUIDED_SOUND_TARGET = 's z sun zoo';
const GUIDED_PHRASE_PROMPT = 'Say “Sally sells seashells.” twice in one clip. Keep the pace relaxed, not rushed.';
const GUIDED_PHRASE_TARGET = 'Sally sells seashells. Sally sells seashells.';

const STEP_ORDER: AssessmentStep[] = ['setup', 'baseline', 'guided-sounds', 'guided-phrase'];

export default function AssessmentScreen({ onComplete }: AssessmentScreenProps) {
  const { stats, setAssessment } = useGame();
  const { soundEnabled } = usePreferences();
  const { width, height } = useViewport();
  const wide = width >= 1160;
  const keepOnePage = width >= 1160 && height >= 820;

  const [stepIndex, setStepIndex] = useState(0);
  const [rerun, setRerun] = useState(false);
  const [baselineCapture, setBaselineCapture] = useState<PracticeCapture | null>(null);
  const [guidedSoundsCapture, setGuidedSoundsCapture] = useState<PracticeCapture | null>(null);
  const [guidedPhraseCapture, setGuidedPhraseCapture] = useState<PracticeCapture | null>(null);
  const [baselineResult, setBaselineResult] = useState<PracticeAnalysisResult | null>(null);
  const [guidedSoundsResult, setGuidedSoundsResult] = useState<PracticeAnalysisResult | null>(null);
  const [guidedPhraseResult, setGuidedPhraseResult] = useState<PracticeAnalysisResult | null>(null);
  const [selectedType, setSelectedType] = useState<LispType | null>(stats.lispType);
  const [severity, setSeverity] = useState(stats.severity || 5);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentStep = STEP_ORDER[stepIndex];

  function stepLabel(step: AssessmentStep) {
    if (step === 'setup') return 'setup';
    if (step === 'baseline') return 'baseline';
    if (step === 'guided-sounds') return 'guided sounds';
    return 'guided phrase';
  }

  function resetWizard() {
    setStepIndex(0);
    setBaselineCapture(null);
    setGuidedSoundsCapture(null);
    setGuidedPhraseCapture(null);
    setBaselineResult(null);
    setGuidedSoundsResult(null);
    setGuidedPhraseResult(null);
    setSelectedType(stats.lispType);
    setSeverity(stats.severity || 5);
    setError(null);
  }

  async function runAnalysis(capture: PracticeCapture, targetText: string, onSuccess: (result: PracticeAnalysisResult) => void) {
    try {
      setAnalyzing(true);
      setError(null);
      const result = await analyzePractice(capture, targetText);
      onSuccess(result);
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'Assessment failed.');
    } finally {
      setAnalyzing(false);
    }
  }

  function saveAssessmentProfile() {
    if (!baselineResult) {
      setError('Complete the baseline first so the app can save your starting profile.');
      return;
    }

    const nextType = selectedType ?? baselineResult.assessment.lispType;
    const nextSeverity = severity || baselineResult.assessment.severity;

    setAssessment(nextType, nextSeverity, {
      baselinePhrase: BASELINE_PHRASE,
      transcript: baselineResult.transcript,
      notes: baselineResult.assessment.notes,
      mouthShapeNotes: baselineResult.assessment.mouthShapeNotes,
      completedAt: new Date().toISOString(),
    });
    onComplete();
  }

  if (stats.assessmentComplete && !rerun) {
    return (
      <section style={{ ...styles.container, overflowY: keepOnePage ? 'hidden' : 'auto' }}>
        <div style={{ ...styles.content, ...(keepOnePage ? styles.contentFixed : null) }}>
          <header style={styles.header}>
            <h1 style={styles.title}>Your baseline profile is already saved. Reuse it for training or run the intake again.</h1>
          </header>

          <div style={{ ...styles.layout, ...(wide ? styles.layoutWide : null) }}>
            <section style={styles.primaryColumn}>
              <div style={styles.panel}>
                <div style={styles.stepBadge}>Saved assessment</div>
                <h2 style={styles.stepTitle}>{stats.lispType ? `${stats.lispType} · severity ${stats.severity}/10` : 'Assessment complete'}</h2>
                <p style={styles.bodyText}>
                  {stats.assessmentProfile
                    ? `Baseline phrase: “${stats.assessmentProfile.baselinePhrase}”`
                    : 'The app will keep using the saved profile until you rerun the guided intake.'}
                </p>
                {stats.assessmentProfile ? (
                  <>
                    <div style={styles.sectionLabel}>Transcript</div>
                    <p style={styles.bodyText}>{stats.assessmentProfile.transcript}</p>
                    <div style={styles.sectionLabel}>Speech note</div>
                    <p style={styles.bodyText}>{stats.assessmentProfile.notes}</p>
                    <div style={styles.sectionLabel}>Visible cue</div>
                    <p style={styles.bodyText}>{stats.assessmentProfile.mouthShapeNotes}</p>
                  </>
                ) : null}

                <div style={styles.actionRow}>
                  <button type="button" style={styles.primaryButton} onClick={onComplete}>
                    <span style={styles.primaryButtonText}>continue to guided training</span>
                  </button>
                  <button
                    type="button"
                    style={styles.secondaryButton}
                    onClick={() => {
                      setRerun(true);
                      resetWizard();
                    }}
                  >
                    <span style={styles.secondaryButtonText}>rerun guided assessment</span>
                  </button>
                </div>
              </div>
            </section>

            <section style={styles.sideColumn}>
              <div style={styles.emptyState}>
                <MonochromeMascot size="sm" />
                <h2 style={styles.emptyTitle}>The saved baseline stays separate from guided practice.</h2>
                <p style={styles.emptyBody}>Rerunning the wizard replaces the saved profile and resets the first-session tutorial.</p>
              </div>
            </section>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section style={{ ...styles.container, overflowY: keepOnePage ? 'hidden' : 'auto' }}>
      <div style={{ ...styles.content, ...(keepOnePage ? styles.contentFixed : null) }}>
        <header style={styles.header}>
          <h1 style={styles.title}>Start with one natural baseline, then move through guided checks before training.</h1>
        </header>

        <div style={styles.stepper}>
          {STEP_ORDER.map((step, index) => {
            const active = currentStep === step;
            const completed = index < stepIndex || (step === 'baseline' && !!baselineResult) || (step === 'guided-sounds' && !!guidedSoundsResult);
            return (
              <div key={step} style={{ ...styles.stepChip, ...(active ? styles.stepChipActive : null), ...(completed ? styles.stepChipDone : null) }}>
                <span style={styles.stepChipIndex}>{index + 1}</span>
                <span style={styles.stepChipText}>{stepLabel(step)}</span>
              </div>
            );
          })}
        </div>

        <div style={{ ...styles.layout, ...(wide ? styles.layoutWide : null) }}>
          <div style={styles.primaryColumn}>
            {currentStep === 'setup' ? (
              <section style={styles.panel}>
                <div style={styles.stepBadge}>Step 1 · setup</div>
                <h2 style={styles.stepTitle}>Frame your face, keep your jaw relaxed, and plan to speak once in a normal voice.</h2>
                <div style={styles.guidanceList}>
                  <p style={styles.guidanceItem}>Sit centered in frame with your lips fully visible.</p>
                  <p style={styles.guidanceItem}>Use your natural speaking voice first. Do not try to fix anything yet.</p>
                  <p style={styles.guidanceItem}>Keep the first clip short. The baseline should sound like your everyday speech.</p>
                </div>
                <button type="button" style={styles.primaryButton} onClick={() => setStepIndex(1)}>
                  <span style={styles.primaryButtonText}>begin baseline</span>
                </button>
              </section>
            ) : null}

            {currentStep === 'baseline' ? (
              <>
                <section style={styles.panel}>
                  <div style={styles.stepBadge}>Step 2 · baseline</div>
                  <h2 style={styles.stepTitle}>Say this once in your normal voice.</h2>
                  <div style={styles.targetText}>{BASELINE_PHRASE}</div>
                  <p style={styles.bodyText}>Goal: natural pace, one clean take, no deliberate correction. This clip becomes the saved baseline profile.</p>
                  <div style={styles.inlineRow}>
                    <button
                      type="button"
                      style={{ ...styles.secondaryButton, ...(!soundEnabled ? styles.buttonDisabled : null) }}
                      onClick={() => {
                        if (soundEnabled) {
                          speakText(BASELINE_PHRASE);
                        }
                      }}
                      disabled={!soundEnabled}
                    >
                      <span style={styles.secondaryButtonText}>{soundEnabled ? 'listen to prompt' : 'sound off'}</span>
                    </button>
                  </div>
                </section>

                <AudioClipInput
                  label="baseline capture"
                  capture={baselineCapture}
                  analyzing={analyzing}
                  onCaptureReady={(nextCapture) => {
                    setBaselineCapture(nextCapture);
                    setBaselineResult(null);
                    setError(null);
                  }}
                />

                <div style={styles.actionRow}>
                  <button
                    type="button"
                    style={{ ...styles.primaryButton, ...((!baselineCapture || analyzing) ? styles.buttonDisabled : null) }}
                    disabled={!baselineCapture || analyzing}
                    onClick={() => {
                      if (!baselineCapture) {
                        setError('Capture the baseline phrase first.');
                        return;
                      }
                      void runAnalysis(baselineCapture, BASELINE_PHRASE, (result) => {
                        setBaselineResult(result);
                        setSelectedType(result.assessment.lispType);
                        setSeverity(result.assessment.severity);
                      });
                    }}
                  >
                    <span style={styles.primaryButtonText}>{analyzing ? 'analyzing baseline...' : 'analyze baseline'}</span>
                  </button>
                  {baselineResult ? (
                    <button type="button" style={styles.secondaryButton} onClick={() => setStepIndex(2)}>
                      <span style={styles.secondaryButtonText}>continue to guided sounds</span>
                    </button>
                  ) : null}
                </div>
              </>
            ) : null}

            {currentStep === 'guided-sounds' ? (
              <>
                <section style={styles.panel}>
                  <div style={styles.stepBadge}>Step 3 · guided sounds</div>
                  <h2 style={styles.stepTitle}>Now try one coached check using the provisional cue from your baseline.</h2>
                  <div style={styles.targetText}>{GUIDED_SOUND_PROMPT}</div>
                  <p style={styles.bodyText}>
                    {selectedType
                      ? getDefaultTip('s', selectedType)
                      : 'Keep the airflow centered, and give each item a small pause so Gemma can compare the sounds.'}
                  </p>
                </section>

                <AudioClipInput
                  label="guided sounds"
                  capture={guidedSoundsCapture}
                  analyzing={analyzing}
                  onCaptureReady={(nextCapture) => {
                    setGuidedSoundsCapture(nextCapture);
                    setGuidedSoundsResult(null);
                    setError(null);
                  }}
                />

                <div style={styles.actionRow}>
                  <button
                    type="button"
                    style={{ ...styles.primaryButton, ...((!guidedSoundsCapture || analyzing) ? styles.buttonDisabled : null) }}
                    disabled={!guidedSoundsCapture || analyzing}
                    onClick={() => {
                      if (!guidedSoundsCapture) {
                        setError('Capture the guided sound check first.');
                        return;
                      }
                      void runAnalysis(guidedSoundsCapture, GUIDED_SOUND_TARGET, (result) => {
                        setGuidedSoundsResult(result);
                        setSelectedType(result.assessment.lispType);
                        setSeverity(result.assessment.severity);
                      });
                    }}
                  >
                    <span style={styles.primaryButtonText}>{analyzing ? 'checking sounds...' : 'analyze guided sounds'}</span>
                  </button>
                  {guidedSoundsResult ? (
                    <button type="button" style={styles.secondaryButton} onClick={() => setStepIndex(3)}>
                      <span style={styles.secondaryButtonText}>continue to guided phrase</span>
                    </button>
                  ) : null}
                </div>
              </>
            ) : null}

            {currentStep === 'guided-phrase' ? (
              <>
                <section style={styles.panel}>
                  <div style={styles.stepBadge}>Step 4 · guided phrase</div>
                  <h2 style={styles.stepTitle}>Finish with one coached phrase clip, then save the baseline profile.</h2>
                  <div style={styles.targetText}>{GUIDED_PHRASE_PROMPT}</div>
                  <p style={styles.bodyText}>
                    {selectedType ? getDefaultFeedback(selectedType, severity) : 'Stay relaxed and repeat the phrase twice in one clip.'}
                  </p>
                </section>

                <AudioClipInput
                  label="guided phrase"
                  capture={guidedPhraseCapture}
                  analyzing={analyzing}
                  onCaptureReady={(nextCapture) => {
                    setGuidedPhraseCapture(nextCapture);
                    setGuidedPhraseResult(null);
                    setError(null);
                  }}
                />

                <div style={styles.actionRow}>
                  <button
                    type="button"
                    style={{ ...styles.primaryButton, ...((!guidedPhraseCapture || analyzing) ? styles.buttonDisabled : null) }}
                    disabled={!guidedPhraseCapture || analyzing}
                    onClick={() => {
                      if (!guidedPhraseCapture) {
                        setError('Capture the guided phrase first.');
                        return;
                      }
                      void runAnalysis(guidedPhraseCapture, GUIDED_PHRASE_TARGET, (result) => {
                        setGuidedPhraseResult(result);
                      });
                    }}
                  >
                    <span style={styles.primaryButtonText}>{analyzing ? 'reviewing phrase...' : 'analyze guided phrase'}</span>
                  </button>
                  {guidedPhraseResult ? (
                    <button type="button" style={styles.secondaryButton} onClick={saveAssessmentProfile}>
                      <span style={styles.secondaryButtonText}>save assessment and train</span>
                    </button>
                  ) : null}
                </div>
              </>
            ) : null}

            {error ? <div style={styles.error}>{error}</div> : null}
          </div>

          <div style={styles.sideColumn}>
            <section style={styles.panel}>
              {currentStep === 'setup' ? (
                <div style={styles.emptyState}>
                  <MonochromeMascot size="sm" />
                  <h2 style={styles.emptyTitle}>The wizard will guide what to say and how to say it.</h2>
                  <p style={styles.emptyBody}>The first clip is natural baseline only. The next two are coached checks that prepare the training tutorial.</p>
                </div>
              ) : null}

              {currentStep === 'baseline' && baselineResult ? (
                <>
                  <SampledFramesStrip frames={baselineCapture?.frames ?? []} />
                  <div style={styles.sectionLabel}>Transcript</div>
                  <p style={styles.bodyText}>{baselineResult.transcript}</p>
                  <div style={styles.sectionLabel}>Visible cue</div>
                  <p style={styles.bodyText}>{baselineResult.assessment.mouthShapeNotes}</p>
                  <div style={styles.sectionLabel}>Speech result</div>
                  <p style={styles.bodyText}>{baselineResult.assessment.notes}</p>
                  <div style={styles.callout}>
                    <div style={styles.calloutText}>Provisional cue: {getDefaultTip('s', baselineResult.assessment.lispType)}</div>
                  </div>
                </>
              ) : null}

              {currentStep === 'guided-sounds' && guidedSoundsResult ? (
                <>
                  <SampledFramesStrip frames={guidedSoundsCapture?.frames ?? []} />
                  <div style={styles.sectionLabel}>Visible cue</div>
                  <p style={styles.bodyText}>{guidedSoundsResult.assessment.mouthShapeNotes}</p>
                  <div style={styles.sectionLabel}>Speech result</div>
                  <p style={styles.bodyText}>{guidedSoundsResult.assessment.notes}</p>
                  <div style={styles.sectionLabel}>Next try</div>
                  <p style={styles.bodyText}>{guidedSoundsResult.coaching.nextTryCue}</p>
                </>
              ) : null}

              {currentStep === 'guided-phrase' && baselineResult ? (
                <>
                  {guidedPhraseResult ? <SampledFramesStrip frames={guidedPhraseCapture?.frames ?? []} /> : null}
                  <div style={styles.sectionLabel}>Baseline profile</div>
                  <p style={styles.bodyText}>
                    {selectedType ?? baselineResult.assessment.lispType} · severity {severity}/10
                  </p>
                  <div style={styles.sectionLabel}>Baseline transcript</div>
                  <p style={styles.bodyText}>{baselineResult.transcript}</p>
                  <div style={styles.sectionLabel}>Baseline cue</div>
                  <p style={styles.bodyText}>{baselineResult.assessment.mouthShapeNotes}</p>

                  {guidedPhraseResult ? (
                    <>
                      <div style={styles.inlineDivider} />
                      <div style={styles.sectionLabel}>Guided phrase result</div>
                      <p style={styles.bodyText}>{guidedPhraseResult.assessment.notes}</p>
                      <div style={styles.sectionLabel}>Next try</div>
                      <p style={styles.bodyText}>{guidedPhraseResult.coaching.nextTryCue}</p>
                    </>
                  ) : null}

                  <div style={styles.choiceGrid}>
                    {LISP_TYPES.map((type) => (
                      <button
                        key={type}
                        type="button"
                        style={{ ...styles.choiceButton, ...(selectedType === type ? styles.choiceButtonActive : null) }}
                        onClick={() => setSelectedType(type)}
                      >
                        <span style={{ ...styles.choiceText, ...(selectedType === type ? styles.choiceTextActive : null) }}>{type}</span>
                      </button>
                    ))}
                  </div>

                  <div style={styles.sliderRow}>
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((value) => (
                      <button
                        key={value}
                        type="button"
                        style={{ ...styles.sliderDot, ...(severity >= value ? styles.sliderDotSelected : null) }}
                        onClick={() => setSeverity(value)}
                      />
                    ))}
                  </div>
                  <p style={styles.bodyText}>saved severity {severity}/10</p>
                </>
              ) : null}
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
  stepper: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  stepChip: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 12px',
    borderRadius: 999,
    backgroundColor: 'var(--color-surface-alt)',
    border: '1px solid var(--color-border)',
  },
  stepChipActive: {
    borderColor: 'var(--color-accent-strong)',
    backgroundColor: 'var(--color-accent-soft)',
  },
  stepChipDone: {
    borderColor: 'var(--color-success)',
  },
  stepChipIndex: {
    width: 22,
    height: 22,
    borderRadius: 999,
    backgroundColor: 'var(--color-accent)',
    color: 'var(--color-bg)',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 12,
    fontWeight: 700,
  },
  stepChipText: {
    fontSize: 12,
    fontWeight: 700,
    color: 'var(--color-text)',
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
    flex: 1.08,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    minWidth: 0,
  },
  sideColumn: {
    flex: 0.92,
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
  stepBadge: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  stepTitle: {
    margin: 0,
    fontSize: 26,
    lineHeight: '30px',
    color: 'var(--color-text)',
    fontWeight: 800,
    letterSpacing: -0.8,
  },
  targetText: {
    fontSize: 24,
    lineHeight: '28px',
    color: 'var(--color-text)',
    fontWeight: 800,
    letterSpacing: -0.8,
  },
  bodyText: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '21px',
  },
  guidanceList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  guidanceItem: {
    fontSize: 14,
    color: 'var(--color-text-muted)',
    lineHeight: '21px',
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
    minWidth: 180,
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
  sectionLabel: {
    fontSize: 11,
    color: 'var(--color-text-subtle)',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  callout: {
    padding: '12px 14px',
    borderRadius: 14,
    backgroundColor: 'var(--color-accent-soft)',
    border: '1px solid var(--color-border-strong)',
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
  choiceGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 10,
  },
  choiceButton: {
    borderRadius: 12,
    border: '1px solid var(--color-border)',
    padding: '12px 14px',
    backgroundColor: 'var(--color-surface-alt)',
  },
  choiceButtonActive: {
    backgroundColor: 'var(--color-accent-soft)',
    borderColor: 'var(--color-accent-strong)',
  },
  choiceText: {
    fontSize: 13,
    color: 'var(--color-text-muted)',
    fontWeight: 700,
    textTransform: 'lowercase',
  },
  choiceTextActive: {
    color: 'var(--color-text)',
  },
  sliderRow: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
  },
  sliderDot: {
    width: 18,
    height: 18,
    borderRadius: 999,
    border: '1px solid var(--color-border-strong)',
    backgroundColor: 'var(--color-surface-alt)',
  },
  sliderDotSelected: {
    backgroundColor: 'var(--color-warning)',
    borderColor: 'var(--color-warning)',
  },
  inlineDivider: {
    height: 1,
    backgroundColor: 'var(--color-border)',
    margin: '2px 0',
  },
};
