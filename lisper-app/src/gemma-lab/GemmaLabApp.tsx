import { useEffect, useRef, useState } from 'react';

import {
  generateGemmaResponse,
  getGemmaLabImageDiagnostics,
  getGemmaLabRuntimeDiagnostics,
  getLoadedModelKey,
  loadGemmaModel,
  MODEL_PRESETS,
  oneTokenGemmaSmoke,
  selfTestGemmaImagePipeline,
  type ModelPreset,
  type ProgressEvent,
} from './gemmaLabRuntime';

type LabMode = 'chat' | 'image' | 'audio' | 'video' | 'combined';

interface VideoFrame {
  blob: Blob;
  url: string;
  time: number;
}

const MODE_COPY: Record<LabMode, { label: string; icon: string; hint: string; defaultPrompt: string }> = {
  chat: {
    icon: 'T',
    label: 'Chat',
    hint: 'Text-only Gemma 4 generation.',
    defaultPrompt: 'Explain in two short bullets how to practice a clean /s/ sound.',
  },
  image: {
    icon: 'I',
    label: 'Image',
    hint: 'Uploads one image to the Gemma 4 vision encoder.',
    defaultPrompt: 'Describe the visible mouth posture and any speech-practice cues you can infer.',
  },
  audio: {
    icon: 'A',
    label: 'Audio',
    hint: 'Uploads or records one audio clip to the Gemma 4 audio encoder.',
    defaultPrompt: 'Transcribe the clip, then give one concise lisp-practice coaching cue.',
  },
  video: {
    icon: 'V',
    label: 'Video Frames',
    hint: 'Samples video frames and sends them as images. Gemma 4 has vision, not a temporal video encoder.',
    defaultPrompt: 'Review these sampled mouth frames and describe any useful placement cues.',
  },
  combined: {
    icon: '+',
    label: 'Combined',
    hint: 'Sends available audio plus image/video frames in one multimodal prompt.',
    defaultPrompt:
      'Use the audio and visual evidence together. Return JSON with transcript, lispType, severity, feedback, and nextTryCue.',
  },
};

function formatBytes(bytes?: number) {
  if (!bytes || !Number.isFinite(bytes)) {
    return null;
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = units[0];
  for (let index = 0; index < units.length - 1 && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index + 1];
  }

  return `${value.toFixed(value >= 10 || unit === 'B' ? 0 : 1)} ${unit}`;
}

function progressLabel(event: ProgressEvent) {
  const file = event.file || event.name || null;
  const progress = typeof event.progress === 'number' ? `${event.progress.toFixed(1)}%` : null;
  const loaded = formatBytes(event.loaded);
  const total = formatBytes(event.total);
  const byteLabel = loaded && total ? `${loaded}/${total}` : loaded;

  return [event.status || 'progress', file, progress, byteLabel].filter(Boolean).join(' | ');
}

async function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Could not encode video frame.'));
        }
      },
      'image/jpeg',
      0.84
    );
  });
}

async function extractVideoFrames(file: File, count: number) {
  const videoUrl = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.preload = 'metadata';
  video.muted = true;
  video.playsInline = true;
  video.src = videoUrl;

  try {
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Could not read video metadata.'));
    });

    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
    const frameCount = Math.max(1, Math.min(count, 8));
    const times = Array.from({ length: frameCount }, (_, index) => {
      if (frameCount === 1) {
        return Math.min(0.5, duration);
      }

      const padding = Math.min(0.35, duration / 8);
      return padding + ((duration - padding * 2) * index) / (frameCount - 1);
    });

    const canvas = document.createElement('canvas');
    const sourceWidth = video.videoWidth || 640;
    const sourceHeight = video.videoHeight || 360;
    const scale = Math.min(1, 640 / Math.max(sourceWidth, sourceHeight));
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));

    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Could not create canvas context for video frames.');
    }

    const frames: VideoFrame[] = [];
    for (const time of times) {
      await new Promise<void>((resolve, reject) => {
        video.onseeked = () => resolve();
        video.onerror = () => reject(new Error(`Could not seek video to ${time.toFixed(2)}s.`));
        video.currentTime = Math.min(Math.max(time, 0), duration);
      });

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await canvasToBlob(canvas);
      frames.push({
        blob,
        url: URL.createObjectURL(blob),
        time,
      });
    }

    return frames;
  } finally {
    URL.revokeObjectURL(videoUrl);
  }
}

function getSelectedPreset(index: number): ModelPreset {
  return MODEL_PRESETS[Math.max(0, Math.min(index, MODEL_PRESETS.length - 1))];
}

export function GemmaLabApp() {
  const [presetIndex, setPresetIndex] = useState(0);
  const [mode, setMode] = useState<LabMode>('chat');
  const [prompt, setPrompt] = useState(MODE_COPY.chat.defaultPrompt);
  const [maxNewTokens, setMaxNewTokens] = useState(180);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoFrames, setVideoFrames] = useState<VideoFrame[]>([]);
  const [frameCount, setFrameCount] = useState(4);
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('idle');
  const [events, setEvents] = useState<string[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExtractingFrames, setIsExtractingFrames] = useState(false);
  const [isRunningSelfTest, setIsRunningSelfTest] = useState(false);
  const [isRunningSmoke, setIsRunningSmoke] = useState(false);
  const [gpuAvailable] = useState(() => typeof navigator !== 'undefined' && 'gpu' in navigator);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<BlobPart[]>([]);
  const preset = getSelectedPreset(presetIndex);
  const isBusy = isLoadingModel || isGenerating || isExtractingFrames || isRecording || isRunningSelfTest || isRunningSmoke;

  useEffect(() => {
    return () => {
      videoFrames.forEach((frame) => URL.revokeObjectURL(frame.url));
    };
  }, [videoFrames]);

  function appendEvent(message: string) {
    setEvents((current) => [message, ...current].slice(0, 20));
  }

  function handleProgress(event: ProgressEvent) {
    const label = progressLabel(event);
    if (label) {
      setStatus(label);
    }
  }

  function formatFailureOutput(message: string) {
    const diagnostics = getGemmaLabRuntimeDiagnostics();
    return `${message}\n\n${JSON.stringify(diagnostics, null, 2)}`;
  }

  function handleRunProgress(event: ProgressEvent) {
    handleProgress(event);
    if (event.status && ['template', 'decode image', 'decode images', 'decode audio', 'processor', 'generate', 'decode output'].includes(event.status)) {
      setOutput(`Stage: ${event.status}`);
    }
  }

  function handleModeChange(nextMode: LabMode) {
    setMode(nextMode);
    setPrompt(MODE_COPY[nextMode].defaultPrompt);
    setOutput('');
  }

  async function handleLoadModel() {
    setIsLoadingModel(true);
    setStatus(`loading ${preset.label}`);
    setOutput('');
    try {
      await loadGemmaModel(preset.id, preset.dtype, handleProgress);
      setStatus(`ready: ${preset.label}`);
      appendEvent(`ready | ${preset.id}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus('model load failed');
      setOutput(message);
      appendEvent(`error | ${message}`);
    } finally {
      setIsLoadingModel(false);
    }
  }

  async function handleClearCache() {
    let deletedCache = false;
    if ('caches' in window) {
      deletedCache = await window.caches.delete('transformers-cache');
    }

    const removedKeys: string[] = [];
    for (let index = window.localStorage.length - 1; index >= 0; index -= 1) {
      const key = window.localStorage.key(index);
      if (key?.includes('gemma') || key?.includes('transformers') || key?.includes('lisper:model-manifest')) {
        window.localStorage.removeItem(key);
        removedKeys.push(key);
      }
    }

    const message = `cleared cache: transformers-cache=${deletedCache ? 'yes' : 'not present'}, localStorage keys=${removedKeys.length}`;
    setStatus(message);
    appendEvent(message);
  }

  async function handleExtractFrames(nextFile = videoFile) {
    if (!nextFile) {
      return;
    }

    setIsExtractingFrames(true);
    setStatus('extracting video frames');
    try {
      videoFrames.forEach((frame) => URL.revokeObjectURL(frame.url));
      const frames = await extractVideoFrames(nextFile, frameCount);
      setVideoFrames(frames);
      setStatus(`extracted ${frames.length} frames`);
      appendEvent(`frames | ${frames.length}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus('video frame extraction failed');
      setOutput(message);
      appendEvent(`error | ${message}`);
    } finally {
      setIsExtractingFrames(false);
    }
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setOutput('This browser does not expose microphone recording APIs.');
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    recordingChunksRef.current = [];
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordingChunksRef.current.push(event.data);
      }
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(recordingChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
      const file = new File([blob], `gemma-lab-recording-${Date.now()}.webm`, { type: blob.type });
      setAudioFile(file);
      setIsRecording(false);
      setStatus(`recording ready: ${file.name}`);
    };

    recorder.start();
    setIsRecording(true);
    setStatus('recording audio');
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
  }

  function buildMessages() {
    const content: Array<{ type: 'text' | 'audio' | 'image'; text?: string }> = [];
    const images: Blob[] = [];
    let audio: Blob | null = null;

    if ((mode === 'image' || mode === 'combined') && imageFile) {
      content.push({ type: 'image' });
      images.push(imageFile);
    }

    if ((mode === 'video' || mode === 'combined') && videoFrames.length) {
      videoFrames.forEach(() => content.push({ type: 'image' }));
      images.push(...videoFrames.map((frame) => frame.blob));
    }

    if ((mode === 'audio' || mode === 'combined') && audioFile) {
      content.push({ type: 'audio' });
      audio = audioFile;
    }

    content.push({ type: 'text', text: prompt });

    return {
      messages: [
        {
          role: 'user',
          content,
        },
      ],
      images,
      audio,
    };
  }

  async function handleRun() {
    const startedAt = performance.now();
    setIsGenerating(true);
    setOutput('');
    setStatus(`generating with ${preset.label}`);

    try {
      const { messages, images, audio } = buildMessages();
      const response = await generateGemmaResponse({
        modelId: preset.id,
        dtype: preset.dtype,
        messages,
        images,
        audio,
        maxNewTokens,
        onProgress: handleRunProgress,
      });
      const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      setOutput(response || '(empty response)');
      setStatus(`done in ${elapsedSeconds}s`);
      appendEvent(`generated | ${elapsedSeconds}s | ${preset.label}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setOutput(formatFailureOutput(message));
      setStatus('generation failed');
      appendEvent(`error | generate | ${message}`);
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleImageSelfTest() {
    setIsRunningSelfTest(true);
    setOutput('');
    setStatus(`testing image pipeline with ${preset.label}`);

    try {
      const { images } = buildMessages();
      const result = await selfTestGemmaImagePipeline(preset.id, preset.dtype, images);
      setOutput(`${result.details}\n${JSON.stringify(result.diagnostics, null, 2)}`);
      setStatus('image pipeline ready');
      appendEvent(`image-test | ok | ${preset.label}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const diagnostics = getGemmaLabImageDiagnostics();
      const detail = diagnostics.length ? diagnostics : getGemmaLabRuntimeDiagnostics();
      setOutput(`${message}\n${JSON.stringify(detail, null, 2)}`);
      setStatus('image pipeline failed');
      appendEvent(`image-test | fail | ${message}`);
    } finally {
      setIsRunningSelfTest(false);
    }
  }

  async function handleOneTokenSmoke() {
    setIsRunningSmoke(true);
    setOutput('Stage: template');
    setStatus(`1-token smoke with ${preset.label}`);

    try {
      const { messages, images, audio } = buildMessages();
      const result = await oneTokenGemmaSmoke({
        modelId: preset.id,
        dtype: preset.dtype,
        messages,
        images,
        audio,
        prompt,
        maxNewTokens: 1,
        onProgress: handleRunProgress,
      });
      setOutput(`${result.details}\n${JSON.stringify(getGemmaLabRuntimeDiagnostics(), null, 2)}`);
      setStatus('1-token smoke passed');
      appendEvent(`1tok | ok | ${preset.label}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setOutput(formatFailureOutput(message));
      setStatus('1-token smoke failed');
      appendEvent(`1tok | fail | ${message}`);
    } finally {
      setIsRunningSmoke(false);
    }
  }

  return (
    <main className="lab-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Lisper Gemma Lab</p>
          <h1>Gemma Lab</h1>
        </div>
        <div className="status-card">
          <span className={gpuAvailable ? 'pill good' : 'pill bad'}>
            WebGPU {gpuAvailable ? 'available' : 'not detected'}
          </span>
          <strong>
            {isGenerating
              ? 'thinking'
              : isLoadingModel
                ? 'loading model'
                : isRunningSelfTest
                  ? 'testing image path'
                  : isRunningSmoke
                    ? '1-token smoke'
                    : isExtractingFrames
                      ? 'extracting'
                      : isRecording
                        ? 'recording'
                        : status}
          </strong>
          <small>{preset.label}</small>
          <small>Loaded key: {getLoadedModelKey() || 'none'}</small>
        </div>
      </section>

      <section className="control-grid">
        <article className="panel">
          <h2>Model</h2>
          <label>
            Preset
            <select value={presetIndex} onChange={(event) => setPresetIndex(Number(event.target.value))}>
              {MODEL_PRESETS.map((item, index) => (
                <option key={item.id} value={index}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <div className="model-meta">
            <code>{preset.id}</code>
            <span>{preset.dtypeLabel}</span>
            <span>{preset.expectedSize}</span>
          </div>
          <button className="primary" onClick={handleLoadModel} disabled={isLoadingModel || isGenerating}>
            {isLoadingModel ? 'Loading...' : 'Load'}
          </button>
          <button onClick={handleImageSelfTest} disabled={isBusy}>
            {isRunningSelfTest ? 'Img...' : 'Img Test'}
          </button>
          <button onClick={handleOneTokenSmoke} disabled={isBusy}>
            {isRunningSmoke ? '1 Tok...' : '1 Tok'}
          </button>
          <button onClick={handleClearCache} disabled={isBusy}>
            Clear Cache
          </button>
        </article>

        <article className="panel">
          <h2>Mode</h2>
          <div className="mode-list">
            {Object.entries(MODE_COPY).map(([key, value]) => (
              <button
                key={key}
                className={mode === key ? 'mode active' : 'mode'}
                onClick={() => handleModeChange(key as LabMode)}
                title={value.hint}
              >
                <em>{value.icon}</em>
                <strong>{value.label}</strong>
              </button>
            ))}
          </div>
        </article>
      </section>

      <section className="workspace">
        <article className="panel prompt-panel">
          <h2>Prompt</h2>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          <label className="tokens">
            Tokens
            <input
              type="number"
              min={16}
              max={512}
              step={16}
              value={maxNewTokens}
              onChange={(event) => setMaxNewTokens(Number(event.target.value))}
            />
          </label>

          <div className="media-grid">
            <label>
              Img
              <input
                type="file"
                accept="image/*"
                onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <label>
              Aud
              <input
                type="file"
                accept="audio/*"
                onChange={(event) => setAudioFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <label>
              Vid
              <input
                type="file"
                accept="video/*"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setVideoFile(file);
                  if (file) {
                    void handleExtractFrames(file);
                  }
                }}
              />
            </label>
            <label>
              Frames
              <input
                type="number"
                min={1}
                max={8}
                value={frameCount}
                onChange={(event) => setFrameCount(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="button-row">
            <button onClick={isRecording ? stopRecording : startRecording} disabled={isGenerating || isLoadingModel || isRunningSelfTest}>
              {isRecording ? 'Stop' : 'Rec'}
            </button>
            <button onClick={() => handleExtractFrames()} disabled={!videoFile || isExtractingFrames || isGenerating || isRunningSelfTest}>
              {isExtractingFrames ? 'Sampling...' : 'Sample'}
            </button>
            <button className="primary" onClick={handleRun} disabled={isGenerating || isLoadingModel || isRunningSelfTest}>
              {isGenerating ? 'Thinking...' : 'Send'}
            </button>
          </div>

          <div className="media-summary">
            <span>I {imageFile?.name || '-'}</span>
            <span>A {audioFile?.name || '-'}</span>
            <span>V {videoFile?.name || '-'}</span>
            <span>F {videoFrames.length || 0}</span>
          </div>
        </article>

        <article className="panel output-panel">
          <h2>Output</h2>
          <pre>{output || (isGenerating ? 'Generating...' : 'Ready.')}</pre>
        </article>
      </section>

      {videoFrames.length > 0 ? (
        <section className="panel frames-panel">
          <h2>Sampled Video Frames</h2>
          <div className="frames">
            {videoFrames.map((frame) => (
              <figure key={`${frame.time}-${frame.url}`}>
                <img src={frame.url} alt={`Video frame at ${frame.time.toFixed(1)} seconds`} />
                <figcaption>{frame.time.toFixed(1)}s</figcaption>
              </figure>
            ))}
          </div>
        </section>
      ) : null}

      <section className="panel events-panel">
        <h2>Log</h2>
        <ol>
          {events.length ? events.map((event, index) => <li key={`${event}-${index}`}>{event}</li>) : <li>idle</li>}
        </ol>
      </section>
    </main>
  );
}
