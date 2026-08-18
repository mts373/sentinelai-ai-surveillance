/**
 * Captures a still frame from a video element already loaded in the browser.
 * These are real frames of the uploaded video — no AI output is synthesised.
 */
export async function captureFrame(
  video: HTMLVideoElement,
  timeSeconds: number,
  maxWidth = 960,
): Promise<string> {
  const previous = video.currentTime;
  const wasPaused = video.paused;
  if (!wasPaused) video.pause();

  await new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out while seeking the video for frame capture."));
    }, 8000);
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Video seek failed during frame capture."));
    };
    function cleanup() {
      window.clearTimeout(timer);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    }
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    try {
      video.currentTime = Math.max(0, timeSeconds);
    } catch {
      cleanup();
      reject(new Error("Video is not seekable."));
    }
  });

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) throw new Error("Video dimensions unavailable for frame capture.");

  const scale = Math.min(1, maxWidth / width);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  let dataUrl: string;
  try {
    dataUrl = canvas.toDataURL("image/jpeg", 0.82);
  } catch {
    throw new Error("Frame capture blocked by the video source (cross-origin).");
  }

  try {
    video.currentTime = previous;
  } catch {
    /* ignore */
  }
  return dataUrl;
}

/** Crops a data-URL image using normalized (0..1) rect coordinates. */
export async function cropImage(
  src: string,
  rect: { x: number; y: number; w: number; h: number },
): Promise<string> {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.crossOrigin = "anonymous";
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error("Could not load the frame for cropping."));
    el.src = src;
  });

  const sx = Math.round(rect.x * image.naturalWidth);
  const sy = Math.round(rect.y * image.naturalHeight);
  const sw = Math.max(1, Math.round(rect.w * image.naturalWidth));
  const sh = Math.max(1, Math.round(rect.h * image.naturalHeight));

  const canvas = document.createElement("canvas");
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
  try {
    return canvas.toDataURL("image/jpeg", 0.85);
  } catch {
    throw new Error("Crop blocked by the image source (cross-origin).");
  }
}
