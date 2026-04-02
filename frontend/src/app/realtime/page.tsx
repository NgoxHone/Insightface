'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { recognizeFace, FaceResult } from '@/lib/api';
import { faVideo, faPlay, faStop, faSpinner, faExclamationCircle, faCircle } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

export default function RealtimePage() {
  const [running, setRunning] = useState(false);
  const [threshold, setThreshold] = useState(0.6);
  const [error, setError] = useState<string>('');
  const [lastDetection, setLastDetection] = useState<FaceResult[]>([]);
  const [fps, setFps] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef(0);
  const fpsTimeRef = useRef(Date.now());

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
      
      setRunning(true);
      setError('');
      
      // Start detection loop
      intervalRef.current = setInterval(detectFrame, 500); // 2 FPS
    } catch (err: any) {
      setError('Không thể truy cập webcam: ' + err.message);
    }
  };

  const stopCamera = () => {
    setRunning(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    // Clear canvas
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    setLastDetection([]);
    setFps(0);
  };

  const detectFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== 4) return;

    // Set canvas size to match video
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Capture frame
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      // Convert to blob
      const blob = await new Promise<Blob | null>(resolve => 
        canvas.toBlob(resolve, 'image/jpeg', 0.8)
      );
      
      if (!blob) return;

      const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
      const response = await recognizeFace(file, threshold);
      
      if (response.success && response.data) {
        const faces = response.data.faces;
        setLastDetection(faces);
        drawFaces(ctx, faces, canvas.width, canvas.height);
      }
      
      // Calculate FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - fpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        fpsTimeRef.current = now;
      }
    } catch (err) {
      console.error('Detection error:', err);
    }
  }, [threshold]);

  const drawFaces = (ctx: CanvasRenderingContext2D, faces: FaceResult[], width: number, height: number) => {
    // Draw video frame
    const video = videoRef.current;
    if (!video) return;
    
    ctx.drawImage(video, 0, 0, width, height);

    faces.forEach((face) => {
      const [x1, y1, x2, y2] = face.bbox;
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      // Draw bounding box
      ctx.strokeStyle = face.matched ? '#03dac6' : '#cf6679';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      // Draw label background
      const label = face.name === 'unknown' 
        ? 'Không xác định' 
        : `${face.name}`;
      const confidence = `${(face.confidence * 100).toFixed(0)}%`;
      const labelText = `${label} ${confidence}`;
      
      ctx.font = 'bold 16px Arial';
      const textWidth = ctx.measureText(labelText).width;
      const labelHeight = 24;
      const labelY = y1 > labelHeight + 5 ? y1 - labelHeight : y1;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x1, labelY, textWidth + 10, labelHeight);
      
      // Draw text
      ctx.fillStyle = face.matched ? '#03dac6' : '#cf6679';
      ctx.fillText(labelText, x1 + 5, labelY + 17);
    });
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-2 text-[#bb86fc]">
        <FontAwesomeIcon icon={faVideo} className="mr-2" />
        Nhận Diện Realtime
      </h1>
      <p className="text-gray-400 mb-6">Sử dụng webcam để nhận diện khuôn mặt theo thời gian thực</p>

      {/* Controls */}
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
          <span className="text-[#03dac6] font-mono">{threshold.toFixed(2)}</span>
        </label>

        {running && (
          <div className="ml-auto flex items-center gap-2">
            <FontAwesomeIcon icon={faCircle} className="text-[#03dac6] text-xs animate-pulse" />
            <span className="text-sm text-gray-400">Đang chạy • {fps} FPS</span>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[#b71c1c] text-[#ef9a9a] border border-[#c62828] rounded-lg p-4 mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faExclamationCircle} />
          {error}
        </div>
      )}

      {/* Video Container */}
      <div className="relative inline-block max-w-[640px] w-full mb-6">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full rounded-lg border border-[#444] bg-black"
          style={{ display: running ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full"
        />
        
        {!running && (
          <div className="flex items-center justify-center h-[480px] bg-[#1a1a1a] rounded-lg border border-[#444]">
            <p className="text-gray-500 text-lg">Nhấn "Bắt Đầu" để kích hoạt webcam</p>
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
                    ? 'border-l-4 border-l-[#03dac6]' 
                    : 'border-l-4 border-l-[#cf6679]'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-bold text-lg">
                    {face.name === 'unknown' ? 'Không xác định' : face.name}
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
    </div>
  );
}
