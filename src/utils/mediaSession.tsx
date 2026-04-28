import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import type { PracticeCapture } from './modelRuntime';

export type CaptureState = 'idle' | 'requesting' | 'ready' | 'mic-only' | 'recording' | 'blocked' | 'unsupported';
export type DeviceMode = 'camera+mic' | 'mic-only';

export interface MediaSessionState {
  deviceMode: DeviceMode | null;
  captureState: CaptureState;
  streamReady: boolean;
  visualSupport: boolean;
  message: string;
  lastError: string | null;
  durationMs: number;
  prepareCameraAndMic: () => Promise<boolean>;
  prepareMicOnly: () => Promise<boolean>;
  attachPreview: (node: HTMLVideoElement | null) => void;
  attachMeter: (node: HTMLCanvasElement | null) => void;
  startRecording: (onCaptureReady: (capture: PracticeCapture) => void) => Promise<void>;
  stopRecording: () => void;
  release: () => void;
  setCaptureRouteActive: (active: boolean) => void;
}

const MAX_CAPTURE_DURATION_MS = 10000;
const SAMPLE_FRAME_LIMIT = 3;
const SAMPLE_INTERVAL_MS = 2200;
const RELEASE_DELAY_MS = 1400;

const DEFAULT_MESSAGE = 'Prepare your camera and microphone, or continue in microphone-only mode.';

const MediaSessionContext = createContext<MediaSessionState | null>(null);

function stopMediaStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function describeMediaError(error: unknown, cameraRequested: boolean) {
  if (!(error instanceof Error)) {
    return 'Camera or microphone access failed.';
  }

  if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
    return 'Permission was denied. Allow camera and microphone access in the browser and retry.';
  }
  if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    return cameraRequested
      ? 'No compatible camera was found. You can continue with microphone only.'
      : 'No compatible microphone was found on this device.';
  }
  if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
    return 'Your camera or microphone is already in use by another app or tab.';
  }
  if (error.name === 'OverconstrainedError') {
    return 'The requested camera setup is unavailable. Retry or continue with microphone only.';
  }

  return error.message || 'Camera or microphone access failed.';
}

function getRecorderOptions(): MediaRecorderOptions | undefined {
  if (typeof MediaRecorder === 'undefined') {
    return undefined;
  }

  const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
  for (const mimeType of mimeTypes) {
    if (typeof MediaRecorder.isTypeSupported !== 'function' || MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType };
    }
  }

  return undefined;
}

export function MediaSessionProvider({ children }: { children: ReactNode }) {
  const [deviceMode, setDeviceMode] = useState<DeviceMode | null>(null);
  const [captureState, setCaptureState] = useState<CaptureState>('idle');
  const [message, setMessage] = useState(DEFAULT_MESSAGE);
  const [lastError, setLastError] = useState<string | null>(null);
  const [durationMs, setDurationMs] = useState(0);

  const previewRef = useRef<HTMLVideoElement | null>(null);
  const meterRef = useRef<HTMLCanvasElement | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const framesRef = useRef<Blob[]>([]);
  const samplePromisesRef = useRef<Promise<void>[]>([]);
  const pendingCaptureHandlerRef = useRef<((capture: PracticeCapture) => void) | null>(null);
  const dropCaptureOnStopRef = useRef(false);
  const lastSuccessfulModeRef = useRef<DeviceMode | null>(null);
  const routeActiveRef = useRef(false);
  const animationFrameRef = useRef<number | null>(null);
  const sampleIntervalRef = useRef<number | null>(null);
  const durationIntervalRef = useRef<number | null>(null);
  const autoStopTimeoutRef = useRef<number | null>(null);
  const releaseTimeoutRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const streamReady = !!streamRef.current;
  const visualSupport = !!streamRef.current?.getVideoTracks().length;

  const clearReleaseTimeout = useCallback(() => {
    if (releaseTimeoutRef.current) {
      window.clearTimeout(releaseTimeoutRef.current);
      releaseTimeoutRef.current = null;
    }
  }, []);

  const attachPreview = useCallback((node: HTMLVideoElement | null) => {
    previewRef.current = node;
    if (!node) {
      return;
    }

    node.srcObject = streamRef.current;
    node.muted = true;

    if (streamRef.current?.getVideoTracks().length && document.visibilityState === 'visible') {
      void node.play().catch(() => undefined);
    }
  }, []);

  function drawIdleMeter() {
    const canvas = meterRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#11151c';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#195466';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(0, canvas.height / 2);
    context.lineTo(canvas.width, canvas.height / 2);
    context.stroke();
  }

  const attachMeter = useCallback((node: HTMLCanvasElement | null) => {
    meterRef.current = node;
    if (node) {
      drawIdleMeter();
    }
  }, []);

  function clearRecordingTimers() {
    if (sampleIntervalRef.current) {
      window.clearInterval(sampleIntervalRef.current);
      sampleIntervalRef.current = null;
    }
    if (durationIntervalRef.current) {
      window.clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }
    if (autoStopTimeoutRef.current) {
      window.clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
  }

  function stopVisualization() {
    if (animationFrameRef.current) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    analyserRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    analyserRef.current = null;
    sourceNodeRef.current = null;

    const audioContext = audioContextRef.current;
    audioContextRef.current = null;
    if (audioContext) {
      void audioContext.close().catch(() => undefined);
    }

    drawIdleMeter();
  }

  function startVisualization(stream: MediaStream) {
    stopVisualization();

    if (!stream.getAudioTracks().length || typeof AudioContext === 'undefined') {
      drawIdleMeter();
      return;
    }

    const canvas = meterRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    const source = audioContext.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
    source.connect(analyser);

    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    sourceNodeRef.current = source;
    void audioContext.resume().catch(() => undefined);

    const waveform = new Uint8Array(analyser.frequencyBinCount);

    const draw = () => {
      const currentCanvas = meterRef.current;
      const currentContext = currentCanvas?.getContext('2d');
      if (!currentCanvas || !currentContext || !analyserRef.current) {
        return;
      }

      analyserRef.current.getByteTimeDomainData(waveform);

      currentContext.fillStyle = '#11151c';
      currentContext.fillRect(0, 0, currentCanvas.width, currentCanvas.height);
      currentContext.lineWidth = 2;
      currentContext.strokeStyle = '#99d1ce';
      currentContext.beginPath();

      const sliceWidth = currentCanvas.width / waveform.length;
      let x = 0;

      for (let i = 0; i < waveform.length; i += 1) {
        const value = waveform[i] / 128.0;
        const y = (value * currentCanvas.height) / 2;

        if (i === 0) {
          currentContext.moveTo(x, y);
        } else {
          currentContext.lineTo(x, y);
        }

        x += sliceWidth;
      }

      currentContext.lineTo(currentCanvas.width, currentCanvas.height / 2);
      currentContext.stroke();

      animationFrameRef.current = window.requestAnimationFrame(draw);
    };

    draw();
  }

  function ensureSupport() {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setCaptureState('unsupported');
      setMessage('This browser cannot access camera and microphone devices.');
      setLastError('This browser cannot access camera and microphone devices.');
      return false;
    }

    if (typeof MediaRecorder === 'undefined') {
      setCaptureState('unsupported');
      setMessage('This browser does not support in-browser recording.');
      setLastError('This browser does not support in-browser recording.');
      return false;
    }

    return true;
  }

  async function syncPreview(stream: MediaStream | null) {
    const preview = previewRef.current;
    if (!preview) {
      return;
    }

    preview.srcObject = stream;
    preview.muted = true;

    if (stream?.getVideoTracks().length && document.visibilityState === 'visible') {
      await preview.play().catch(() => undefined);
    }
  }

  async function applyStream(nextStream: MediaStream, nextMode: DeviceMode, nextState: 'ready' | 'mic-only', nextMessage: string) {
    stopMediaStream(streamRef.current);
    streamRef.current = nextStream;
    setDeviceMode(nextMode);
    setCaptureState(nextState);
    setMessage(nextMessage);
    setLastError(null);
    setDurationMs(0);
    await syncPreview(nextStream);
    drawIdleMeter();
  }

  const prepareCameraAndMic = useCallback(async () => {
    clearReleaseTimeout();
    if (!ensureSupport()) {
      return false;
    }

    setCaptureState('requesting');
    setMessage('Requesting camera and microphone access...');
    setLastError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: {
          facingMode: 'user',
          width: { ideal: 640, max: 960 },
          height: { ideal: 360, max: 540 },
          frameRate: { ideal: 24, max: 24 },
        },
      });

      lastSuccessfulModeRef.current = 'camera+mic';
      await applyStream(stream, 'camera+mic', 'ready', 'Camera and microphone are ready. Keep your face in frame and start a short attempt.');
      return true;
    } catch (error) {
      const nextError = describeMediaError(error, true);
      setCaptureState('blocked');
      setMessage(nextError);
      setLastError(nextError);
      return false;
    }
  }, [clearReleaseTimeout]);

  const prepareMicOnly = useCallback(async () => {
    clearReleaseTimeout();
    if (!ensureSupport()) {
      return false;
    }

    setCaptureState('requesting');
    setMessage('Requesting microphone access...');
    setLastError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });

      lastSuccessfulModeRef.current = 'mic-only';
      await applyStream(stream, 'mic-only', 'mic-only', 'Microphone is ready. Camera is off, so analysis will use audio only.');
      return true;
    } catch (error) {
      const nextError = describeMediaError(error, false);
      setCaptureState('blocked');
      setMessage(nextError);
      setLastError(nextError);
      return false;
    }
  }, [clearReleaseTimeout]);

  function captureVideoFrame() {
    const stream = streamRef.current;
    if (!stream?.getVideoTracks().length || framesRef.current.length >= SAMPLE_FRAME_LIMIT) {
      return;
    }

    const video = previewRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    if (!frameCanvasRef.current && typeof document !== 'undefined') {
      frameCanvasRef.current = document.createElement('canvas');
    }

    const canvas = frameCanvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    canvas.width = Math.min(320, video.videoWidth);
    canvas.height = Math.round((canvas.width / video.videoWidth) * video.videoHeight);

    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.restore();

    const promise = new Promise<void>((resolve) => {
      canvas.toBlob(
        (blob) => {
          if (blob && framesRef.current.length < SAMPLE_FRAME_LIMIT) {
            framesRef.current.push(blob);
          }
          resolve();
        },
        'image/jpeg',
        0.76
      );
    });

    samplePromisesRef.current.push(promise);
  }

  const stopRecording = useCallback(() => {
    if (!recorderRef.current || recorderRef.current.state === 'inactive') {
      return;
    }

    captureVideoFrame();
    clearRecordingTimers();
    recorderRef.current.stop();
  }, []);

  const release = useCallback(() => {
    clearReleaseTimeout();
    dropCaptureOnStopRef.current = true;
    pendingCaptureHandlerRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      clearRecordingTimers();
      recorderRef.current.stop();
    }

    recorderRef.current = null;
    stopVisualization();
    stopMediaStream(streamRef.current);
    streamRef.current = null;
    void syncPreview(null);
    setCaptureState('idle');
    setDeviceMode(lastSuccessfulModeRef.current);
    setMessage(DEFAULT_MESSAGE);
    setLastError(null);
    setDurationMs(0);
  }, [clearReleaseTimeout]);

  const startRecording = useCallback(async (onCaptureReady: (capture: PracticeCapture) => void) => {
    clearReleaseTimeout();

    if (captureState === 'recording') {
      return;
    }

    if (!streamRef.current) {
      const prepared =
        lastSuccessfulModeRef.current === 'mic-only' ? await prepareMicOnly() : await prepareCameraAndMic();
      if (!prepared) {
        return;
      }
    }

    const stream = streamRef.current;
    if (!stream?.getAudioTracks().length) {
      setCaptureState('blocked');
      setMessage('Microphone access is required to record a practice attempt.');
      setLastError('Microphone access is required to record a practice attempt.');
      return;
    }

    const audioStream = new MediaStream(stream.getAudioTracks());
    const recorder = new MediaRecorder(audioStream, getRecorderOptions());
    recorderRef.current = recorder;
    chunksRef.current = [];
    framesRef.current = [];
    samplePromisesRef.current = [];
    pendingCaptureHandlerRef.current = onCaptureReady;
    dropCaptureOnStopRef.current = false;
    recordingStartedAtRef.current = Date.now();
    setDurationMs(0);
    setCaptureState('recording');
    setMessage('Recording now. Keep the target sound clean and steady.');
    setLastError(null);
    startVisualization(stream);

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onerror = () => {
      setCaptureState(stream.getVideoTracks().length ? 'ready' : 'mic-only');
      setMessage('Recording failed. Check the device state and try again.');
      setLastError('Recording failed. Check the device state and try again.');
      stopVisualization();
    };

    recorder.onstop = async () => {
      stopVisualization();
      const stoppedAt = Date.now();
      const clipDurationMs = Math.max(250, stoppedAt - recordingStartedAtRef.current);
      setDurationMs(clipDurationMs);
      const pendingSamples = [...samplePromisesRef.current];
      await Promise.all(pendingSamples);

      const nextState = stream.getVideoTracks().length ? 'ready' : 'mic-only';
      const shouldDrop = dropCaptureOnStopRef.current;
      dropCaptureOnStopRef.current = false;
      recorderRef.current = null;

      if (shouldDrop) {
        setCaptureState(nextState);
        setMessage('Recording stopped.');
        return;
      }

      const audioBlob = new Blob(chunksRef.current, {
        type: recorder.mimeType || 'audio/webm',
      });

      if (audioBlob.size < 1024) {
        setCaptureState(nextState);
        setMessage('That clip was too short. Try again with a clearer attempt.');
        setLastError('That clip was too short. Try again with a clearer attempt.');
        return;
      }

      const nextCapture: PracticeCapture = {
        audioBlob,
        frames: [...framesRef.current],
        durationMs: clipDurationMs,
        source: 'recording',
      };

      pendingCaptureHandlerRef.current?.(nextCapture);
      setCaptureState(nextState);
      setMessage(
        nextCapture.frames.length
          ? 'Capture ready. Review the sampled mouth frames and run analysis when you are ready.'
          : 'Audio capture ready. Run analysis when you are ready.'
      );
      setLastError(null);
    };

    recorder.start(200);
    captureVideoFrame();
    sampleIntervalRef.current = window.setInterval(captureVideoFrame, SAMPLE_INTERVAL_MS);
    durationIntervalRef.current = window.setInterval(() => {
      setDurationMs(Date.now() - recordingStartedAtRef.current);
    }, 200);
    autoStopTimeoutRef.current = window.setTimeout(stopRecording, MAX_CAPTURE_DURATION_MS);
  }, [captureState, clearReleaseTimeout, prepareCameraAndMic, prepareMicOnly, stopRecording]);

  const setCaptureRouteActive = useCallback((active: boolean) => {
    routeActiveRef.current = active;
    clearReleaseTimeout();

    if (active) {
      if (!streamRef.current && lastSuccessfulModeRef.current && captureState !== 'requesting') {
        void (lastSuccessfulModeRef.current === 'mic-only' ? prepareMicOnly() : prepareCameraAndMic());
      }
      return;
    }

    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      dropCaptureOnStopRef.current = true;
      clearRecordingTimers();
      recorderRef.current.stop();
    }

    releaseTimeoutRef.current = window.setTimeout(() => {
      release();
    }, RELEASE_DELAY_MS);
  }, [captureState, clearReleaseTimeout, prepareCameraAndMic, prepareMicOnly, release]);

  useEffect(() => {
    drawIdleMeter();

    return () => {
      clearReleaseTimeout();
      clearRecordingTimers();
      release();
    };
  }, [clearReleaseTimeout, release]);

  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }

    const handleVisibilityChange = () => {
      const stream = streamRef.current;
      const preview = previewRef.current;
      if (!stream) {
        return;
      }

      const hidden = document.visibilityState !== 'visible';
      stream.getVideoTracks().forEach((track) => {
        track.enabled = !hidden && routeActiveRef.current;
      });

      if (hidden) {
        stopVisualization();
        preview?.pause();
        return;
      }

      if (captureState === 'recording') {
        startVisualization(stream);
      } else {
        drawIdleMeter();
      }

      if (preview && stream.getVideoTracks().length) {
        void preview.play().catch(() => undefined);
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [captureState]);

  const value = useMemo<MediaSessionState>(
    () => ({
      deviceMode,
      captureState,
      streamReady,
      visualSupport,
      message,
      lastError,
      durationMs,
      prepareCameraAndMic,
      prepareMicOnly,
      attachPreview,
      attachMeter,
      startRecording,
      stopRecording,
      release,
      setCaptureRouteActive,
    }),
    [
      attachMeter,
      attachPreview,
      captureState,
      deviceMode,
      durationMs,
      lastError,
      message,
      prepareCameraAndMic,
      prepareMicOnly,
      release,
      setCaptureRouteActive,
      startRecording,
      stopRecording,
      streamReady,
      visualSupport,
    ]
  );

  return <MediaSessionContext.Provider value={value}>{children}</MediaSessionContext.Provider>;
}

export function useMediaSession() {
  const context = useContext(MediaSessionContext);
  if (!context) {
    throw new Error('useMediaSession must be used within MediaSessionProvider');
  }

  return context;
}
