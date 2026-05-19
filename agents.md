# AGENTS.md - Lisper Development Guide

> This file evolves throughout the project. Update as decisions are made and architecture solidifies.
> Last Updated: 2026-05-04

---

## Project Context

- **Hackathon**: Kaggle Gemma 4 Good Hackathon
- **Tracks**: Main + Impact + Unsloth
- **Focus**: Lisp vocal trainer (not stuttering)
- **Goal**: Gamified, low-anxiety speech training with Gemma 4

---

## Current Status

**Phase**: Hackathon Closeout - Model Gate Complete

**Next Priority**:
1. Validate Gemma Lab and main app demo flows in a WebGPU browser
2. Polish the app/demo path and record hackathon footage
3. Package public submission links and final writeup

**Model Gate**:
- v18 hybrid held-out eval passed on `2000` rows with `0` hard errors
- `publish_verdict.json` status is `pass`
- current browser demo model is `thomasjvu/lisper-gemma4-e2b-audio-onnx-q4f16`

---

## Architecture Decisions

### ✅ Model Selection
- **Base Model**: Gemma 4 E2B (2-billion parameters, supports text + audio)
- **Fine-tuning**: Unsloth with LoRA adapters; final quality gate uses the v18 hybrid acoustic+Gemma eval path
- **Quantization**: ONNX/WebGPU `q4f16` for browser demo; q2f16 remains experimental

### ✅ Frontend Stack
- **Framework**: Vite + React Native Web
- **Justification**: Browser/WebGPU is the hackathon demo target; the legacy mobile bundler was removed after ONNX Runtime Web bundling moved to Vite

### ✅ Speech API (Free, No API Keys)
- **STT**: browser-whisper (WebGPU Whisper in browser)
- **Pronunciation Scoring**: OpenPronounce (Wav2Vec2-based)
- **Rationale**: 100% free, runs locally, no cloud costs

---

## Dataset Strategy

### Research Completed

| Dataset | Language | Focus | Availability |
|---------|----------|-------|--------------|
| PAVSig | Polish | Sigmatism (lisp) | DUA required |
| PERCEPT-R | English | /ɹ/ sounds | Via PhonBank |
| UCL Dysfluency | English | Stuttering | Physical DVD |

### Decision: Synthesize + Use Generic Speech

**Approach**:
1. Use generic speech datasets (LibriSpeech, Common Voice)
2. Apply audio signal processing to simulate lisp patterns
3. Create training pairs: [normal audio] → [lisp audio] + [feedback]

**Audio Manipulation Techniques**:
- Spectral shaping to simulate lateral/frontal lisps
- Add turbulence noise for "wet" lateral lisp sounds
- Dentalize by adding formant shifts

---

## File Structure

```
/Users/area/repos/lisper/
├── SPEC.md                                   # Project specification
├── agents.md                                 # This file
├── README.md                                 # Project README
├── data/
│   ├── raw/                                 # Raw speech data
│   ├── processed/                           # Processed datasets
│   └── synthetic/                           # Generated lisp data
├── src/
│   ├── model/                              # Fine-tuning scripts
│   │   ├── finetune.py                     # Main fine-tuning script
│   │   ├── dataset.py                      # Dataset preparation
│   │   └── export.py                       # Model export
│   └── app/
│       └── lisper-app/                      # Vite web app
│           ├── App.tsx                      # Main component
│           ├── screens/                    # App screens
│           │   ├── HomeScreen.tsx
│           │   ├── AssessmentScreen.tsx
│           │   ├── TrainingScreen.tsx
│           │   └── ProgressScreen.tsx
│           ├── components/                # Reusable components
│           │   ├── SoundRecorder.tsx
│           │   ├── FeedbackCard.tsx
│           │   └── ProgressChart.tsx
│           └── utils/                     # Helper functions
├── notebooks/
│   └── exploration/                         # Research notebooks
└── tests/                                 # Test files
```

---

## Commands

### Environment Setup
```bash
# Create environment
conda create -n lisper python=3.11
conda activate lisper

# Install Unsloth and deps
pip install unsloth fastapi uvicorn

cd lisper-app
npm install
```

### Fine-tuning
```bash
# Add larger LibriSpeech subset for the real run
curl -L -o data/raw/LibriSpeech/train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf data/raw/LibriSpeech/train-clean-100.tar.gz -C data/raw/LibriSpeech

# Build raw-audio dataset + audit manifest
python src/model/dataset.py --build-multimodal --profile hackathon
python src/model/dataset.py --audit

# Export Kaggle / Unsloth training bundles
python src/model/finetune.py --prepare

# Run Kaggle notebook
# notebooks/kaggle_gemma4_audio_unsloth.ipynb

# Export model
python src/model/export.py --format gguf
```

### Run App
```bash
# Start Vite dev server
cd lisper-app
npm run start

# Web build
npm run build:web
```

---

## Research Notes

### Lisp Types (Clinical Reference)

| Type | Characteristics | Sound Example |
|------|----------------|----------------|
| Frontal | Tongue between teeth | /s/ → "th" |
| Lateral | Air over sides | "wet" /s/ |
| Dental | Tongue against teeth | Dentalized /s/ |
| Palatal | Tongue touches roof | Palatalized |

### Treatment Progression (SLP Standard)
1. Auditory discrimination
2. Tongue placement awareness
3. Sound isolation (/s/, /z/)
4. Syllables (sa, se, si...)
5. Words (sun, sad, zip...)
6. Phrases (Sally sells...)
7. Sentences (The snake...)
8. Conversation (free speech)

### Competitors Analyzed
- **Better Speech**: Includes lisps but not focused
- **Bubu Speech**: Kids-focused, games
- **Stutter Stars**: Stuttering only
- **Eloquent**: Stuttering only
- ** Lisper**: FIRST lisp-specific

---

## Todo List

- [x] Research hackathon requirements
- [x] Research competitors
- [x] Research datasets
- [x] Finalize SPEC.md
- [x] Write SPEC.md
- [x] Write agents.md
- [x] Set up Vite web project
- [x] Build demo UI screens
- [x] Create dataset generation script
- [x] Set up Unsloth environments
- [x] Fine-tune Gemma 4 baseline model
- [x] Run full held-out model eval and produce passing publish verdict
- [x] Build and test web export
- [ ] Validate browser audio/image/video demo flows on target WebGPU machine
- [ ] Record demo video
- [ ] Create technical writeup
- [ ] Submit to Kaggle

---

## Open Questions

1. **Demo Validation**: Do Gemma Lab and the main app both complete text/audio/image/video flows with the q4f16 browser package on the demo machine?
2. **Runtime Claim**: For submission, clearly distinguish the q4f16 browser demo from the v18 hybrid eval path.
3. **Public Release Timing**: Which Hugging Face, Kaggle, and GitHub artifacts should be made public only at final submission time?

---

## API Keys Needed

**NONE** - 100% free stack:
- browser-whisper: WebGPU-based STT (free, no keys)
- OpenPronounce: Self-hosted pronunciation scoring
- Gemma 4: Downloadable model (needs HuggingFace token for download)

---

## Updated Log

### 2026-04-21
- Created initial SPEC.md with full project specification
- Completed competitor research: found no lisp-specific AI apps
- Completed dataset research: PAVSig (Polish, DUA), PERCEPT-R (English)
- Decided on dataset approach: synthesize from generic speech + audio manipulation
- Frontend: Vite + React Native Web

### 2026-05-04
- Completed model training/eval gate with v18 hybrid held-out eval: `2000` rows, `0` hard errors, publish verdict `pass`
- Confirmed app builds pass for Gemma Lab and web
- Set q4f16 ONNX/WebGPU package as the browser demo target

---

## Notes for Future Updates

> When adding new sections:
> - Keep commands tested and working
> - Document any environment changes
> - Note API key requirements
> - Update todo list with checkmarks as completed
