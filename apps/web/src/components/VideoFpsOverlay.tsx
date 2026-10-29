import { useEffect, useState, type RefObject } from "react";

type VideoFpsOverlayProps = {
  videoRef: RefObject<HTMLVideoElement>;
  stream: MediaStream;
};

/** Measures frames presented by the video element, not the display refresh rate. */
export function VideoFpsOverlay({ videoRef, stream }: VideoFpsOverlayProps) {
  const [label, setLabel] = useState("—");

  useEffect(() => {
    const video = videoRef.current;
    if (!video || typeof video.requestVideoFrameCallback !== "function") {
      setLabel("不支持");
      return;
    }

    let stopped = false;
    let callbackId: number;
    let previousPresented: number | null = null;
    let frames = 0;
    let sampledFrames = 0;
    let sampledAt = performance.now();

    const reset = () => {
      previousPresented = null;
      frames = sampledFrames = 0;
      sampledAt = performance.now();
      setLabel("—");
    };
    reset();

    const onFrame: VideoFrameRequestCallback = (_now, metadata) => {
      if (stopped) return;
      if (!document.hidden) {
        if (previousPresented !== null && metadata.presentedFrames < previousPresented) reset();
        frames += previousPresented === null ? 1 : metadata.presentedFrames - previousPresented;
        previousPresented = metadata.presentedFrames;
      }
      callbackId = video.requestVideoFrameCallback(onFrame);
    };
    callbackId = video.requestVideoFrameCallback(onFrame);

    const timer = window.setInterval(() => {
      if (document.hidden) return;
      const now = performance.now();
      const elapsed = now - sampledAt;
      if (elapsed > 0) setLabel(((frames - sampledFrames) * 1000 / elapsed).toFixed(1));
      sampledFrames = frames;
      sampledAt = now;
    }, 1000);
    document.addEventListener("visibilitychange", reset);

    return () => {
      stopped = true;
      video.cancelVideoFrameCallback(callbackId);
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", reset);
    };
  }, [videoRef, stream]);

  return (
    <div
      className="pointer-events-none absolute bottom-3 left-3 z-30 rounded bg-black/65 px-2 py-1 font-mono text-xs tabular-nums text-white"
      title="浏览器实际呈现的视频帧率，约每秒更新；后台标签页暂停统计"
    >
      播放 FPS：{label}
    </div>
  );
}
