import { forwardRef, useImperativeHandle, useRef, useState } from "react";

import { ErrorNotice } from "@/components/sentinel-ui";
import { formatSeconds } from "@/lib/sentinel";
import { cn } from "@/lib/utils";

export interface VideoPlayerHandle {
  seek: (seconds: number, play?: boolean) => void;
  element: () => HTMLVideoElement | null;
  captureFrame: () => string | null;
}

export const VideoPlayer = forwardRef<
  VideoPlayerHandle,
  {
    src: string;
    filename?: string | null | undefined;
    className?: string | undefined;
    onTime?: ((seconds: number) => void) | undefined;
    onDuration?: ((seconds: number) => void) | undefined;
  }
>(function VideoPlayer({ src, filename, className, onTime, onDuration }, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useImperativeHandle(ref, () => ({
    seek: (seconds: number, play = false) => {
      const el = videoRef.current;
      if (!el) return;
      try {
        el.currentTime = Math.max(0, seconds);
      } catch {
        return;
      }
      if (play) void el.play().catch(() => undefined);
    },
    element: () => videoRef.current,
    captureFrame: () => {
      const el = videoRef.current;
      if (!el) return null;
      try {
        const canvas = document.createElement("canvas");
        canvas.width = el.videoWidth;
        canvas.height = el.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg", 0.8);
      } catch (e) {
        console.error("Frame capture failed:", e);
        return null;
      }
    },
  }));

  return (
    <div className={cn("space-y-2", className)}>
      <div className="overflow-hidden rounded-md border border-border bg-background">
        <video
          ref={videoRef}
          src={src}
          controls
          playsInline
          preload="metadata"
          crossOrigin="anonymous"
          className="aspect-video w-full bg-black"
          onError={() =>
            setError(
              "Video playback failed — the source could not be loaded by the browser.",
            )
          }
          onLoadedMetadata={(event) => {
            const value = event.currentTarget.duration;
            if (Number.isFinite(value)) {
              setDuration(value);
              onDuration?.(value);
            }
            setError(null);
          }}
          onTimeUpdate={(event) => {
            const value = event.currentTarget.currentTime;
            setCurrent(value);
            onTime?.(value);
          }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span className="font-mono text-foreground">
          {formatSeconds(current)} / {duration !== null ? formatSeconds(duration) : "--:--"}
        </span>
        {filename ? <span className="min-w-0 truncate">{filename}</span> : null}
      </div>
      {error ? <ErrorNotice message={error} /> : null}
    </div>
  );
});
