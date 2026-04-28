export interface GemmaBrowserProgressEvent {
  status?: string;
}

export interface GemmaBrowserImageDiagnostics {
  route: string;
  index: number;
  decodePath: string;
  blobType: string;
  blobSize: number;
  constructorName: string | null;
  instanceOfRawImage: boolean;
  hasRgb: boolean;
  hasResize: boolean;
  hasToTensor: boolean;
  width: number | null;
  height: number | null;
  channels: number | null;
}

export interface GemmaBrowserRuntimeDiagnostics {
  route: string;
  stage: string;
  message?: string;
  stack?: string;
  float16Bridge?: {
    enabled: boolean;
    castCount: number;
    lastCasts: string[];
  };
  decodePath?: string;
  promptStats?: {
    length: number;
    imagePlaceholders: number;
    audioPlaceholders: number;
    requestedImages: number;
    hasAudio: boolean;
  };
  inputSummary?: Record<string, unknown>;
  imageDiagnostics: GemmaBrowserImageDiagnostics[];
}

export interface GemmaBrowserSelfTestResult {
  ok: boolean;
  route: string;
  details: string;
  diagnostics: GemmaBrowserImageDiagnostics[];
}

const DEFAULT_AUDIO_SAMPLING_RATE = 16000;
const FLOAT16_BRIDGE_FLAG = '__lisperGemmaFloat16FeedBridge';
const DECODER_FLOAT16_INPUTS = new Set(['inputs_embeds', 'per_layer_inputs']);
const lastImageDiagnostics = new Map<string, GemmaBrowserImageDiagnostics[]>();
const lastRuntimeDiagnostics = new Map<string, GemmaBrowserRuntimeDiagnostics>();
const float16BridgeState = new WeakMap<object, { castCount: number; lastCasts: string[] }>();

function setLastDiagnostics(route: string, diagnostics: GemmaBrowserImageDiagnostics[]) {
  lastImageDiagnostics.set(route, diagnostics);
}

export function getLastImageDiagnostics(route: string) {
  return lastImageDiagnostics.get(route) ?? [];
}

export function getLastGemmaBrowserDiagnostics(route: string): GemmaBrowserRuntimeDiagnostics {
  return (
    lastRuntimeDiagnostics.get(route) ?? {
      route,
      stage: 'none',
      imageDiagnostics: getLastImageDiagnostics(route),
    }
  );
}

function setRuntimeDiagnostics(route: string, diagnostics: GemmaBrowserRuntimeDiagnostics) {
  lastRuntimeDiagnostics.set(route, diagnostics);
}

function isTensorLike(value: unknown): value is { type: string; dims: number[]; data?: unknown; ort_tensor?: unknown } {
  return Boolean(
    value &&
      typeof value === 'object' &&
      typeof (value as any).type === 'string' &&
      Array.isArray((value as any).dims)
  );
}

function getExpectedInputType(session: any, name: string) {
  const metadata = session?.inputMetadata;
  if (!metadata) {
    return null;
  }

  if (Array.isArray(metadata)) {
    const match = metadata.find((entry) => entry?.name === name);
    return match?.type ?? match?.dataType ?? match?.tensorType ?? null;
  }

  const match = metadata[name];
  return match?.type ?? match?.dataType ?? match?.tensorType ?? null;
}

function expectsFloat16(session: any, name: string) {
  const expected = getExpectedInputType(session, name);
  if (typeof expected === 'string' && expected.toLowerCase().includes('float16')) {
    return true;
  }

  return session?.inputNames?.includes?.(name) && DECODER_FLOAT16_INPUTS.has(name);
}

function castFeedToFloat16(module: any, value: unknown) {
  if (!isTensorLike(value) || value.type !== 'float32' || typeof module?.Tensor !== 'function') {
    return value;
  }

  const tensor = (value as any).ort_tensor ? value : new module.Tensor(value);
  return tensor.to('float16').ort_tensor;
}

function patchSessionFloat16Feeds(module: any, session: any) {
  if (!session || session[FLOAT16_BRIDGE_FLAG] || typeof session.run !== 'function') {
    return;
  }

  const state = { castCount: 0, lastCasts: [] as string[] };
  const originalRun = session.run.bind(session);

  session.run = async (feeds: Record<string, unknown>, ...rest: unknown[]) => {
    if (!feeds || typeof feeds !== 'object') {
      return originalRun(feeds, ...rest);
    }

    let changed = false;
    const nextFeeds: Record<string, unknown> = { ...feeds };
    const castNames: string[] = [];

    for (const [name, value] of Object.entries(feeds)) {
      if (!expectsFloat16(session, name) || !isTensorLike(value) || value.type !== 'float32') {
        continue;
      }

      nextFeeds[name] = castFeedToFloat16(module, value);
      changed = nextFeeds[name] !== value;
      state.castCount += 1;
      castNames.push(name);
    }

    if (castNames.length) {
      state.lastCasts = castNames;
    }

    return originalRun(changed ? nextFeeds : feeds, ...rest);
  };

  session[FLOAT16_BRIDGE_FLAG] = true;
  float16BridgeState.set(session, state);
}

export function patchGemmaBrowserFloat16Feeds(module: any, model: any) {
  const sessions = model?.sessions;
  if (!sessions || typeof sessions !== 'object') {
    return;
  }

  for (const session of Object.values(sessions)) {
    patchSessionFloat16Feeds(module, session);
  }
}

export function getGemmaBrowserFloat16BridgeDiagnostics(model: any) {
  const sessions = model?.sessions;
  const states = sessions && typeof sessions === 'object'
    ? Object.values(sessions)
        .map((session) => (session && typeof session === 'object' ? float16BridgeState.get(session) : null))
        .filter(Boolean)
    : [];

  const castCount = states.reduce((total, state: any) => total + Number(state?.castCount ?? 0), 0);
  const lastCasts = states.flatMap((state: any) => state?.lastCasts ?? []);

  return {
    enabled: states.length > 0,
    castCount,
    lastCasts: Array.from(new Set(lastCasts)),
  };
}

function collectImageDiagnostics(
  module: any,
  route: string,
  frames: Blob[],
  images: any[],
  decodePath: string
): GemmaBrowserImageDiagnostics[] {
  return images.map((image, index) => ({
    route,
    index,
    decodePath,
    blobType: frames[index]?.type || 'unknown',
    blobSize: Number(frames[index]?.size ?? 0),
    constructorName: image?.constructor?.name ?? null,
    instanceOfRawImage: typeof module?.RawImage === 'function' ? image instanceof module.RawImage : false,
    hasRgb: typeof image?.rgb === 'function',
    hasResize: typeof image?.resize === 'function',
    hasToTensor: typeof image?.toTensor === 'function',
    width: Number.isFinite(image?.width) ? image.width : null,
    height: Number.isFinite(image?.height) ? image.height : null,
    channels: Number.isFinite(image?.channels) ? image.channels : null,
  }));
}

function buildPromptStats(prompt: string, processor: any, images: Blob[], audio?: Blob | null) {
  return {
    length: prompt.length,
    imagePlaceholders: countOccurrences(prompt, processor?.image_token),
    audioPlaceholders: countOccurrences(prompt, processor?.audio_token),
    requestedImages: images.length,
    hasAudio: Boolean(audio),
  };
}

function formatBrowserError(
  route: string,
  stage: string,
  error: unknown,
  diagnostics: GemmaBrowserImageDiagnostics[],
  extra?: Partial<GemmaBrowserRuntimeDiagnostics>
) {
  const detail = error instanceof Error ? error.message : String(error);
  const payload: GemmaBrowserRuntimeDiagnostics = {
    route,
    stage,
    message: detail,
    stack: error instanceof Error ? error.stack : undefined,
    imageDiagnostics: diagnostics,
    ...extra,
  };
  setRuntimeDiagnostics(route, payload);

  return new Error(
    `Gemma browser runtime failed during ${stage}. route=${route}. ${detail}. diagnostics=${JSON.stringify(payload)}`
  );
}

export function formatGemmaBrowserRuntimeError(args: {
  route: string;
  stage: string;
  error: unknown;
  inputSummary?: Record<string, unknown>;
  float16Bridge?: GemmaBrowserRuntimeDiagnostics['float16Bridge'];
}) {
  const previous = getLastGemmaBrowserDiagnostics(args.route);
  return formatBrowserError(args.route, args.stage, args.error, previous.imageDiagnostics, {
    ...previous,
    stage: args.stage,
    inputSummary: args.inputSummary ?? previous.inputSummary,
    float16Bridge: args.float16Bridge ?? previous.float16Bridge,
  });
}

function assertRuntimeOwnedRawImage(
  module: any,
  route: string,
  frames: Blob[],
  images: any[],
  decodePath: string,
  message?: string
) {
  const diagnostics = collectImageDiagnostics(module, route, frames, images, decodePath);
  setLastDiagnostics(route, diagnostics);

  const invalid = diagnostics.find(
    (image) => !image.instanceOfRawImage || !image.hasRgb || !image.hasResize
  );

  if (invalid) {
    throw new Error(
      message ||
        `Image canonicalization failed. decoded_images=${JSON.stringify(diagnostics)}`
    );
  }

  return diagnostics;
}

async function decodeImageFrameToRuntimeRawImage(module: any, frame: Blob) {
  if (
    typeof window !== 'undefined' &&
    typeof document !== 'undefined' &&
    typeof createImageBitmap === 'function' &&
    typeof module?.RawImage === 'function'
  ) {
    const bitmap = await createImageBitmap(frame);

    try {
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;

      const context = canvas.getContext('2d');
      if (!context) {
        throw new Error('Image decoding failed: could not create a canvas context.');
      }

      context.drawImage(bitmap, 0, 0);
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      const runtimeImage = new module.RawImage(
        new Uint8ClampedArray(imageData.data),
        canvas.width,
        canvas.height,
        4
      );
      Object.setPrototypeOf(runtimeImage, module.RawImage.prototype);
      return {
        image: runtimeImage,
        decodePath: 'canvas+RawImage',
      };
    } finally {
      bitmap.close?.();
    }
  }

  if (module?.RawImage?.read) {
    const decoded = await module.RawImage.read(frame);
    const runtimeImage = new module.RawImage(
      new Uint8ClampedArray(decoded.data),
      decoded.width,
      decoded.height,
      decoded.channels
    );
    Object.setPrototypeOf(runtimeImage, module.RawImage.prototype);
    return {
      image: runtimeImage,
      decodePath: 'RawImage.read',
    };
  }

  if (module?.load_image) {
    const decoded = await module.load_image(frame);
    const runtimeImage = new module.RawImage(
      new Uint8ClampedArray(decoded.data),
      decoded.width,
      decoded.height,
      decoded.channels
    );
    Object.setPrototypeOf(runtimeImage, module.RawImage.prototype);
    return {
      image: runtimeImage,
      decodePath: 'load_image',
    };
  }

  throw new Error('Image decoding failed: no supported browser image decoder is available.');
}

async function loadRuntimeOwnedImages(module: any, route: string, frames: Blob[]) {
  if (!frames.length) {
    return {
      decodedImages: [] as any[],
      diagnostics: [] as GemmaBrowserImageDiagnostics[],
      decodePath: 'none',
    };
  }

  const decoded = await Promise.all(
    frames.map(async (frame, index) => {
      try {
        const result = await decodeImageFrameToRuntimeRawImage(module, frame);
        return {
          index,
          ...result,
        };
      } catch (error) {
        throw new Error(
          `Image frame ${index} failed to decode. ${error instanceof Error ? error.message : String(error)}`
        );
      }
    })
  );

  const decodePath = decoded[0]?.decodePath ?? 'unknown';
  const images = decoded.map((item) => item.image);
  const diagnostics = assertRuntimeOwnedRawImage(
    module,
    route,
    frames,
    images,
    decodePath,
    'Image canonicalization failed before processor packing.'
  );

  return {
    decodedImages: images,
    diagnostics,
    decodePath,
  };
}

async function loadAudio(processor: any, module: any, audioBlob: Blob) {
  const audioUrl = URL.createObjectURL(audioBlob);
  try {
    const samplingRate = Number(
      processor?.feature_extractor?.config?.sampling_rate ?? DEFAULT_AUDIO_SAMPLING_RATE
    );
    return {
      audio: await module.read_audio(audioUrl, samplingRate),
      samplingRate,
    };
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

function countOccurrences(value: string, needle: string | null | undefined) {
  if (!needle) {
    return 0;
  }

  return value.split(needle).length - 1;
}

function summarizeTensor(value: any) {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const dims = Array.isArray(value.dims) ? value.dims : null;
  if (!dims) {
    return null;
  }

  return {
    type: value.type ?? null,
    dims,
  };
}

export function summarizeGemmaInputs(inputs: Record<string, any> | null | undefined) {
  if (!inputs) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(inputs).map(([key, value]) => [key, summarizeTensor(value) ?? typeof value])
  );
}

function assertPromptMediaAlignment(
  route: string,
  promptStats: NonNullable<GemmaBrowserRuntimeDiagnostics['promptStats']>,
  imageDiagnostics: GemmaBrowserImageDiagnostics[],
  decodePath: string
) {
  if (promptStats.requestedImages && promptStats.imagePlaceholders !== promptStats.requestedImages) {
    throw formatBrowserError(
      route,
      'processor preflight',
      new Error(
        `Prompt image placeholders (${promptStats.imagePlaceholders}) do not match selected images (${promptStats.requestedImages}).`
      ),
      imageDiagnostics,
      { promptStats, decodePath }
    );
  }

  if (promptStats.hasAudio && promptStats.audioPlaceholders < 1) {
    throw formatBrowserError(
      route,
      'processor preflight',
      new Error('Prompt has audio input but no Gemma audio placeholder.'),
      imageDiagnostics,
      { promptStats, decodePath }
    );
  }
}

export async function buildGemmaBrowserInputs(args: {
  module: any;
  processor: any;
  prompt: string;
  route: string;
  images?: Blob[];
  audio?: Blob | null;
  onProgress?: (event: GemmaBrowserProgressEvent) => void;
}) {
  const { module, processor, prompt, route, images = [], audio, onProgress } = args;
  const addSpecialTokensOptions = { add_special_tokens: false };
  const promptStats = buildPromptStats(prompt, processor, images, audio);

  let decodedImages: any[] = [];
  let imageDiagnostics: GemmaBrowserImageDiagnostics[] = [];
  let decodePath = 'none';
  let waveform: any = null;

  if (images.length) {
    onProgress?.({ status: images.length === 1 ? 'decode image' : 'decode images' });
    try {
      const decoded = await loadRuntimeOwnedImages(module, route, images);
      decodedImages = decoded.decodedImages;
      imageDiagnostics = decoded.diagnostics;
      decodePath = decoded.decodePath;
    } catch (error) {
      throw formatBrowserError(route, 'decode media', error, getLastImageDiagnostics(route), {
        promptStats,
        decodePath,
      });
    }
  } else {
    setLastDiagnostics(route, []);
  }

  if (audio) {
    onProgress?.({ status: 'decode audio' });
    try {
      ({ audio: waveform } = await loadAudio(processor, module, audio));
    } catch (error) {
      throw formatBrowserError(route, 'decode media', error, imageDiagnostics, {
        promptStats,
        decodePath,
      });
    }
  }

  assertPromptMediaAlignment(route, promptStats, imageDiagnostics, decodePath);

  onProgress?.({ status: 'processor' });
  try {
    const inputs = await processor(
      prompt,
      decodedImages.length ? decodedImages : null,
      waveform,
      addSpecialTokensOptions
    );
    const inputSummary = summarizeGemmaInputs(inputs);
    setRuntimeDiagnostics(route, {
      route,
      stage: 'processor',
      decodePath,
      promptStats,
      inputSummary,
      imageDiagnostics,
    });
    return inputs;
  } catch (error) {
    throw formatBrowserError(route, 'processor multimodal call', error, imageDiagnostics, {
      promptStats,
      decodePath,
    });
  }
}

async function buildSyntheticImage(module: any, route: string) {
  const image = new module.RawImage(new Uint8ClampedArray(4 * 32 * 32).fill(255), 32, 32, 4);
  Object.setPrototypeOf(image, module.RawImage.prototype);

  const frame = new Blob([image.data], { type: 'application/x-rawimage-selftest' });
  const diagnostics = assertRuntimeOwnedRawImage(
    module,
    route,
    [frame],
    [image],
    'synthetic',
    'Synthetic image canonicalization failed before self-test.'
  );

  return {
    image,
    diagnostics,
  };
}

export async function runGemmaImagePipelineSelfTest(args: {
  module: any;
  processor: any;
  route: string;
  images?: Blob[];
}) {
  const { module, processor, route, images = [] } = args;

  if (images.length) {
    const prompt = `${processor.image_token}\nDescribe this image in one short sentence.`;
    const inputs = await buildGemmaBrowserInputs({
      module,
      processor,
      prompt,
      route,
      images,
    });
    const inputSummary = summarizeGemmaInputs(inputs);
    return {
      ok: true,
      route,
      details: `Real image processor path passed. inputs=${Object.keys(inputSummary).join(', ')}`,
      diagnostics: getLastImageDiagnostics(route),
    } satisfies GemmaBrowserSelfTestResult;
  }

  const { image, diagnostics } = await buildSyntheticImage(module, route);

  try {
    const output = await processor.image_processor([image], { add_special_tokens: false });
    const count = Array.isArray(output?.num_soft_tokens_per_image)
      ? output.num_soft_tokens_per_image[0]
      : null;
    setRuntimeDiagnostics(route, {
      route,
      stage: 'self-test',
      decodePath: 'synthetic',
      inputSummary: summarizeGemmaInputs(output),
      imageDiagnostics: diagnostics,
    });

    return {
      ok: true,
      route,
      details: `Vision preprocessing passed. soft_tokens=${count ?? 'unknown'}`,
      diagnostics,
    } satisfies GemmaBrowserSelfTestResult;
  } catch (error) {
    throw formatBrowserError(route, 'self-test', error, diagnostics, {
      decodePath: 'synthetic',
    });
  }
}

export async function runGemmaOneTokenSmoke(args: {
  module: any;
  processor: any;
  model: any;
  route: string;
  prompt: string;
  images?: Blob[];
  audio?: Blob | null;
  onProgress?: (event: GemmaBrowserProgressEvent) => void;
}) {
  const { module, processor, model, route, prompt, images, audio, onProgress } = args;
  const inputs = await buildGemmaBrowserInputs({
    module,
    processor,
    prompt,
    route,
    images,
    audio,
    onProgress,
  });

  onProgress?.({ status: 'generate' });
  try {
    const outputs = await model.generate({
      ...inputs,
      max_new_tokens: 1,
      do_sample: false,
      temperature: 1.0,
      top_p: 0.95,
      top_k: 64,
    });
    const decoded = processor.batch_decode(outputs, { skip_special_tokens: true });
    setRuntimeDiagnostics(route, {
      ...getLastGemmaBrowserDiagnostics(route),
      route,
      stage: 'one-token-smoke',
      inputSummary: summarizeGemmaInputs(inputs),
      imageDiagnostics: getLastImageDiagnostics(route),
    });

    return {
      ok: true,
      route,
      details: `One-token generation passed. output=${JSON.stringify(decoded?.[0] ?? '')}`,
      diagnostics: getLastImageDiagnostics(route),
    } satisfies GemmaBrowserSelfTestResult;
  } catch (error) {
    throw formatGemmaBrowserRuntimeError({
      route,
      stage: 'model.generate',
      error,
      inputSummary: summarizeGemmaInputs(inputs),
      float16Bridge: getGemmaBrowserFloat16BridgeDiagnostics(model),
    });
  }
}
