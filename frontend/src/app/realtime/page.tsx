"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { recognizeFace, FaceResult } from "@/lib/api";
import {
  faVideo,
  faPlay,
  faStop,
  faSpinner,
  faExclamationCircle,
  faCircle,
  faRoad,
  faEye,
  faBolt,
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

type Mode = "recognition" | "tracking";
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

export default function RealtimePage() {
  const [mode, setMode] = useState<Mode>("recognition");
  const [running, setRunning] = useState(false);
  const [threshold, setThreshold] = useState(0.6);
  const [error, setError] = useState<string>("");
  const [lastDetection, setLastDetection] = useState<FaceResult[]>([]);
  const [fps, setFps] = useState(0);

  // Recognition mode refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef(0);
  const fpsTimeRef = useRef(Date.now());

  // Tracking mode refs
  const trackingImgRef = useRef<HTMLImageElement>(null);
  const [trackingSource, setTrackingSource] = useState<"webcam" | "video" | "rtsp">("webcam");
  const [customSource, setCustomSource] = useState<string>("");
  const [showSourceInput, setShowSourceInput] = useState(false);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAll();
    };
  }, []);

  const stopAll = () => {
    // Stop recognition mode
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Stop tracking mode
    if (trackingImgRef.current) {
      trackingImgRef.current.src = "";
    }

    setRunning(false);
    setLastDetection([]);
    setFps(0);
    setError("");
  };

  // ========== Recognition Mode ==========
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }

      setRunning(true);
      setError("");

      // Start detection loop (2 FPS)
      intervalRef.current = setInterval(detectFrame, 500);
    } catch (err: any) {
      setError("Không thể truy cập webcam: " + err.message);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setRunning(false);
    setLastDetection([]);
    setFps(0);
  };

  const detectFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== 4) return;

    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.8),
      );

      if (!blob) return;

      const file = new File([blob], "frame.jpg", { type: "image/jpeg" });
      const response = await recognizeFace(file, threshold);

      if (response.success && response.data) {
        const faces = response.data.faces;
        setLastDetection(faces);
        drawFaces(ctx, faces, canvas.width, canvas.height);
      }

      // FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - fpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        fpsTimeRef.current = now;
      }
    } catch (err) {
      console.error("Detection error:", err);
    }
  }, [threshold]);

  const drawFaces = (
    ctx: CanvasRenderingContext2D,
    faces: FaceResult[],
    width: number,
    height: number,
  ) => {
    const video = videoRef.current;
    if (!video) return;

    ctx.drawImage(video, 0, 0, width, height);

    faces.forEach((face) => {
      const [x1, y1, x2, y2] = face.bbox;
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      ctx.strokeStyle = face.matched ? "#03dac6" : "#cf6679";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      const label = face.name === "unknown" ? "Không xác định" : `${face.name}`;
      const confidence = `${(face.confidence * 100).toFixed(0)}%`;
      const labelText = `${label} ${confidence}`;

      ctx.font = "bold 16px Arial";
      const textWidth = ctx.measureText(labelText).width;
      const labelHeight = 24;
      const labelY = y1 > labelHeight + 5 ? y1 - labelHeight : y1;

      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(x1, labelY, textWidth + 10, labelHeight);

      ctx.fillStyle = face.matched ? "#03dac6" : "#cf6679";
      ctx.fillText(labelText, x1 + 5, labelY + 17);
    });
  };

  // ========== Tracking Mode ==========
  const startTracking = () => {
    setError("");
    setRunning(true);

    // Determine source based on selection
    let sourceParam: string;
    if (trackingSource === "webcam") {
      sourceParam = "0";  // Default webcam
    } else if (trackingSource === "video") {
      sourceParam = "data/test_video.mp4";  // Test video file
    } else if (trackingSource === "rtsp") {
      sourceParam = customSource.trim() || "0";
    } else {
      sourceParam = "0";
    }

    // The image will load the MJPEG stream automatically via src
    if (trackingImgRef.current) {
      const src =
        API_BASE +
        `/api/tracking/stream?source=${encodeURIComponent(sourceParam)}&resolution=640x480&frame_skip=1&recognition_interval=2.0`;
      trackingImgRef.current.src = src;
    }
  };

  const stopTracking = () => {
    if (trackingImgRef.current) {
      trackingImgRef.current.src = "";
    }
    setRunning(false);
  };

  const handleTrackingError = (
    e: React.SyntheticEvent<HTMLImageElement, Event>,
  ) => {
    setError(
      "Không thể kết nối đến tracking stream. Kiểm tra backend có chạy không?",
    );
    setRunning(false);
  };

  // ========== Mode Switch ==========
  const switchMode = (newMode: Mode) => {
    stopAll();
    setMode(newMode);
  };

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-2 text-[#bb86fc]">
        <FontAwesomeIcon icon={faVideo} className="mr-2" />
        NhậnDiện Realtime
      </h1>
      <p className="text-gray-400 mb-6">
        Chọn chế độ: &quot;Recognition&quot; (nhận diện thường) hoặc &quot;YOLO
        Tracking&quot; (theo dõi đối tượng thông minh)
      </p>

      {/* Mode Selector */}
      <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6">
        <div className="flex gap-4 items-center flex-wrap">
          <button
            onClick={() => switchMode("recognition")}
            disabled={running}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              mode === "recognition"
                ? "bg-[#03dac6] text-[#0a0a0a]"
                : "bg-[#333] text-gray-400 hover:bg-[#444]"
            }`}
          >
            <FontAwesomeIcon icon={faEye} className="mr-2" />
            Recognition Mode
          </button>

          <button
            onClick={() => switchMode("tracking")}
            disabled={running}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              mode === "tracking"
                ? "bg-[#03dac6] text-[#0a0a0a]"
                : "bg-[#333] text-gray-400 hover:bg-[#444]"
            }`}
          >
            <FontAwesomeIcon icon={faRoad} className="mr-2" />
            YOLO Tracking Mode
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[#b71c1c] text-[#ef9a9a] border border-[#c62828] rounded-lg p-4 mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faExclamationCircle} />
          {error}
        </div>
      )}

      {/* ========== Recognition Mode UI ========== */}
      {mode === "recognition" && (
        <>
          <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6 flex gap-4 items-center flex-wrap">
            <button
              onClick={startCamera}
              disabled={running}
              className="bg-[#03dac6] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <FontAwesomeIcon icon={faPlay} className="mr-2" />
              Bắt Đầu
            </button>

            <button
              onClick={stopCamera}
              disabled={!running}
              className="bg-[#cf6679] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <FontAwesomeIcon icon={faStop} className="mr-2" />
              Dừng
            </button>

            <label className="flex items-center gap-2 ml-4">
              <span>Ngưỡng:</span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-32"
              />
              <span className="text-[#03dac6] font-mono">
                {threshold.toFixed(2)}
              </span>
            </label>

            {running && (
              <div className="ml-auto flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faCircle}
                  className="text-[#03dac6] text-xs animate-pulse"
                />
                <span className="text-sm text-gray-400">
                  Đang chạy • {fps} FPS
                </span>
              </div>
            )}
          </div>

          <div className="relative inline-block max-w-[640px] w-full mb-6">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full rounded-lg border border-[#444] bg-black"
              style={{ display: running ? "block" : "none" }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
            />

            {!running && mode === "recognition" && (
              <div className="flex items-center justify-center h-[480px] bg-[#1a1a1a] rounded-lg border border-[#444]">
                <p className="text-gray-500 text-lg">
                  Nhấn "Bắt Đầu" để kích hoạt webcam
                </p>
              </div>
            )}
          </div>

          {/* Detection Results */}
          {lastDetection.length > 0 && running && (
            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-[#03dac6]">
                Phát hiện {lastDetection.length} khuôn mặt
              </h2>
              <div className="grid gap-3">
                {lastDetection.map((face, idx) => (
                  <div
                    key={idx}
                    className={`bg-[#1a1a1a] border rounded-lg p-4 ${
                      face.matched
                        ? "border-l-4 border-l-[#03dac6]"
                        : "border-l-4 border-l-[#cf6679]"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-lg">
                        {face.name === "unknown" ? "Không xác định" : face.name}
                      </span>
                      <span className="text-gray-400">
                        {(face.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ========== Tracking Mode UI ========== */}
      {mode === "tracking" && (
        <div className="space-y-6">
          {/* Source Selection */}
          <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">Nguồn video:</h3>
            <div className="flex gap-6 items-center flex-wrap">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="trackingSource"
                  checked={trackingSource === "webcam"}
                  onChange={() => setTrackingSource("webcam")}
                  className="w-4 h-4 accent-[#bb86fc]"
                />
                <span>Webcam (0)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="trackingSource"
                  checked={trackingSource === "video"}
                  onChange={() => setTrackingSource("video")}
                  className="w-4 h-4 accent-[#bb86fc]"
                />
                <span>Video Test File</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="trackingSource"
                  checked={trackingSource === "rtsp"}
                  onChange={() => setTrackingSource("rtsp")}
                  className="w-4 h-4 accent-[#bb86fc]"
                />
                <span>RTSP / Custom</span>
              </label>
            </div>
            {trackingSource === "rtsp" && (
              <div className="mt-3 flex gap-2">
                <input
                  type="text"
                  value={customSource}
                  onChange={(e) => setCustomSource(e.target.value)}
                  placeholder="rtsp://username:password@192.168.1.100:554/stream"
                  className="flex-1 bg-[#252525] border border-[#444] rounded px-3 py-2 text-sm max-w-md"
                />
              </div>
            )}
            <p className="text-xs text-gray-500 mt-2">
              Webcam yêu cầu backend có quyền truy cập camera. Trên macOS, chạy từ terminal có thể cần cấp quyền.
            </p>
          </div>

          {/* Control Buttons */}
          <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6 flex gap-4 items-center flex-wrap">
            <button
              onClick={startTracking}
              disabled={running}
              className="bg-[#bb86fc] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <FontAwesomeIcon icon={faPlay} className="mr-2" />
              Start Tracking
            </button>

            <button
              onClick={stopTracking}
              disabled={!running}
              className="bg-[#cf6679] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <FontAwesomeIcon icon={faStop} className="mr-2" />
              Stop
            </button>

            {running && (
              <div className="ml-auto flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faBolt}
                  className="text-[#bb86fc] animate-pulse"
                />
                <span className="text-sm text-gray-400">
                  YOLO Tracking Active
                </span>
              </div>
            )}
          </div>

          <div className="relative inline-block max-w-[640px] w-full mb-6">
            <img
              ref={trackingImgRef}
              alt="Tracking Stream"
              className="w-full rounded-lg border border-[#444]"
              style={{ display: running ? "block" : "none" }}
              onError={handleTrackingError}
            />

            {!running && mode === "tracking" && (
              <div className="flex items-center justify-center h-[480px] bg-[#1a1a1a] rounded-lg border border-[#444]">
                <div className="text-center">
                  <FontAwesomeIcon
                    icon={faRoad}
                    size="3x"
                    className="text-[#bb86fc] mb-4"
                  />
                  <p className="text-gray-500 text-lg">
                    Nhấn "Start Tracking" để bắt đầu
                  </p>
                  <p className="text-gray-600 text-sm mt-2">
                    Stream từ backend với YOLO + ByteTrack
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Tracking Info */}
          {running && (
            <div className="bg-[#1a1a1a] rounded-lg p-4">
              <h3 className="text-lg font-semibold text-[#bb86fc] mb-2">
                <FontAwesomeIcon icon={faRoad} className="mr-2" />
                Tracking Info
              </h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Person tracks hiển thị trực tiếp trên video</li>
                <li>• Màu sắc khác nhau cho mỗi người được nhận diện</li>
                <li>• Track ID và tên người được hiển thị</li>
                <li>• Tự động theo dõi khi người bị che khuất</li>
                <li>• Backend xử lý video stream với YOLO + InsightFace</li>
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
