# SPEC.md - Lisper: AI-Powered Lisp Vocal Trainer

> Version: 1.0 | Last Updated: 2026-04-21

---

## 1. Project Overview

**Project Name:** Lisper
**Tagline:** "Speak Confidently, One Sound at a Time"

### Problem Statement

- **50M+** people in the US have speech sound disorders, including lisps
- Traditional speech therapy costs $150-300/session and requires in-person visits
- **No AI app specifically targets lisps** - existing tools treat lisps as one of many conditions
- Children and adults with lisps often avoid seeking help due to anxiety and stigma

### Solution

An AI-powered speech training app specifically designed for lisps that provides:
1. **Detection** - Identifies and classifies lisp type (frontal, lateral, dental, palatal)
2. **Training** - Guided exercises with progressive difficulty
3. **Feedback** - Real-time pronunciation assessment
4. **Gamification** - Low-anxiety learning environment with achievements and progress tracking

### Unique Value Proposition

**First AI app dedicated specifically to lisps** - all competitors treat lisps as secondary to stuttering or other disorders.

---

## 2. Target Audience

| Segment | Age Range | Use Case |
|---------|-----------|----------|
| Children | 6-12 years | Parents supervise, gamified exercises |
| Teens | 13-17 years | Self-directed practice with peer motivation |
| Adults | 18+ years | Professional practice, anxiety-free environment |
| Caregivers | Parents/Guardians | Monitor progress, assist with exercises |

---

## 3. Competitor Analysis

| App | Focus | Lisp-Specific? | Gamification | AI-Powered |
|-----|-------|-----------------|---------------|------------|
| **Better Speech** | Multiple | Partial | Limited | AI assistant |
| **Bubu Speech** | Multiple | Partial | Games (40+ courses) | ASR-based |
| **Stutter Stars** | Stuttering only | No | Game-based | No |
| **Eloquent** | Stuttering | No | Limited | AI coaching |
| **Lisper** | **Lisp ONLY** | **Yes** | **Full** | **Gemma 4** |

### Key Differentiator

All existing solutions treat lisps as **one of many conditions**. Lisper is the **first app dedicated specifically to lisps** with AI-powered feedback and gamification.

---

## 4. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│   Vite + React Native Web (Browser/WebGPU)                      │
├─────────────────────────────────────────────────────────────────┤
│                      INFERENCE LAYER                              │
│   Fine-tuned Gemma 4 (E2B via Unsloth)                          │
│   - Speech analysis    - Feedback generation   - Progress       │
├─────────────────────────────────────────────────────────────────┤
│                       SPEECH API                                 │
│   Microsoft Speech SDK / Speechace API                          │
│   - STT conversion   - Pronunciation scoring                   │
├─────────────────────────────────────────────────────────────────┤
│                     OFFLINE CAPABLE                              │
│   Local-first: runs without internet after initial setup        │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack (100% Free)

| Layer | Technology | Justification |
|-------|------------|--------------|
| LLM | Gemma 4 E2B | Multimodal (text + audio), offline capable |
| Fine-tuning | Unsloth | 2x faster, 70% less VRAM |
| Backend | FastAPI | Lightweight |
| Frontend | Vite + React Native Web | Browser-first demo with React Native primitives |
| Speech STT | browser-whisper (WebGPU) | Free, runs in browser, no API |
| Pronunciation | OpenPronounce / Wav2Vec2 | Free, self-hosted scoring |
| Storage | AsyncStorage / SQLite | Offline-first |

---

## 5. Core Features

### 5.1 Assessment Module

- **Initial Assessment**: 2-minute diagnostic with voice samples
- **Lisp Classification**: Detect frontal, lateral, dental, palatal
- **Severity Scoring**: 1-10 scale for baseline

### 5.2 Training Module

| Level | Focus | Example Exercises |
|-------|-------|------------------|
| **Level 1** | Isolation | /s/ and /z/ in isolation |
| **Level 2** | Syllables | sa, se, si, so; za, ze, zi, zo |
| **Level 3** | Words | "sun", "sad", "zebra", "zoo" |
| **Level 4** | Phrases | "Sally sells seashells" |
| **Level 5** | Sentences | "The snake slithers smoothly" |
| **Level 6** | Conversation | Free speech with feedback |

### 5.3 Feedback System

- **Real-time**: Immediate feedback on each sound
- **Detailed**: Tongue placement, airflow direction tips
- **Encouraging**: Positive reinforcement, no negative correction

### 5.4 Gamification

- **Achievements**: "Sound Master", "7-Day Streak", "Perfect Session"
- **Progress Dashboard**: Visual progress charts
- **Avatar**: Animated speech therapist companion
- **Levels**: Unlocked exercises and themes

---

## 6. Hackathon Alignment

### Tracks Supported

1. **Main Track** - Gemma 4 multimodal for speech analysis
2. **Impact Track** - Health & Sciences (democratizing speech therapy access)
3. **Unsloth Track** - Fine-tuned Gemma 4 model using Unsloth

### Deployment Constraints (All Met)

| Constraint | Implementation |
|------------|----------------|
| **Low Bandwidth** | Lightweight model, cached content |
| **Limited Compute** | Unsloth optimization, 4-bit quantization |
| **Offline Capability** | Local-first architecture |

---

## 7. Evaluation Criteria

| Criterion | Weight | How Addressed |
|-----------|--------|--------------|
| **Innovation** | 30% | First lisp-specific AI app with gamification |
| **Impact Potential** | 30% | Democratizing access, reducing therapy costs |
| **Technical Execution** | 25% | Gemma 4 fine-tuning, offline architecture |
| **Accessibility** | 15% | Cross-platform (Web + Mobile), free tier |

---

## 8. Deliverables

1. **Working Demo**: Interactive Vite web app demonstrating assessment and training
2. **Code Repository**: Public GitHub repo with fine-tuning pipeline
3. **Video**: 2-3 minute demo showing features
4. **Technical Writeup**: Architecture, fine-tuning approach, deployment

---

## 9. Milestones

| Phase | Deliverable | Timeline |
|-------|------------|----------|
| 1 | Dataset compilation | Week 1 |
| 2 | Fine-tuning Gemma 4 | Week 2 |
| 3 | Speech API integration | Week 3 |
| 4 | Frontend/UI development | Week 4 |
| 5 | Demo and polish | Week 5 |

---

## 10. Dataset Strategy

### Research Findings

| Dataset | Language | Focus | Availability |
|---------|----------|-------|--------------|
| PAVSig | Polish | Sigmatism (lisp) | DUA required |
| PERCEPT-R | English | /ɹ/ sounds | Via PhonBank |
| UCL Dysfluency | English | Stuttering | Physical DVD |

### Approach

**Primary**: Synthesize lisp audio data using audio manipulation techniques
- Use existing speech datasets (LibriSpeech, common voice)
- Apply signal processing to simulate lisp patterns

**Secondary**: Leverage PAVSig research methodology for proof-of-concept

---

## 11. Lisp Types Reference

### Frontal Lisp
- Tongue protrudes between front teeth
- /s/ sounds like "th"
- Most common in young children

### Lateral Lisp
- Air flows over sides of tongue
- /s/ sounds "wet" or "slushy"
- Often requires therapy to correct

### Dental Lisp
- Tongue pushes against teeth
- Similar to frontal lisp

### Palatal Lisp
- Tongue touches roof of mouth too far back
- Rare, typically requires therapy

### Treatment Progression

1. **Auditory discrimination** - Recognize correct vs incorrect
2. **Tongue placement awareness** - Visual/tactile cues
3. **Phonetic placement** - Verbal instructions, straw technique
4. **Sound isolation** - Practice /s/ and /z/ alone
5. **Carryover** - Words → Phrases → Sentences → Conversation

---

## 12. Differentiation Summary

| Feature | Lisper | Competitors |
|---------|--------|-------------|
| **Focus** | Lisp ONLY | Multiple conditions |
| **AI Model** | Gemma 4 fine-tuned | Generic ASR |
| **Platform** | Browser/WebGPU first | Usually one platform |
| **Gamification** | Full game mechanics | Limited |
| **Offline** | Local-first | Cloud-dependent |

---

*End of SPEC.md*
