'use client';

import { useState, useRef, useCallback } from 'react';
import { recognizeFace, FaceResult } from '@/lib/api';
import { faCamera, faUpload, faSpinner, faExclamationCircle, faCheckCircle } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

export default function RecognizePage() {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [threshold, setThreshold] = useState(0.6);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<FaceResult[]>([]);
  const [error, setError] = useState<string>('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Vui lòng chọn file ảnh');
      return;
    }
    setImage(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResults([]);
    setError('');
    
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
    };
    img.src = url;
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleRecognize = async () => {
    if (!image) return;
    
    setLoading(true);
    setError('');
    setResults([]);

    try {
      const response = await recognizeFace(image, threshold);
      if (response.success && response.data) {
        setResults(response.data.faces);
        drawResults(response.data.faces);
      } else {
        setError(response.error || 'Nhận diện thất bại');
      }
    } catch (err) {
      setError('Lỗi kết nối đến server');
    } finally {
      setLoading(false);
    }
  };

  const drawResults = (faces: FaceResult[]) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const img = imgRef.current;
    if (!canvas || !ctx || !img) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    faces.forEach((face) => {
      const [x1, y1, x2, y2] = face.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      ctx.strokeStyle = face.matched ? '#03dac6' : '#cf6679';
      ctx.lineWidth = Math.max(3, width / 50);
      ctx.strokeRect(x1, y1, width, height);

      const label = face.name === 'unknown' 
        ? 'Không xác định' 
        : `${face.name} (${(face.confidence * 100).toFixed(1)}%)`;
      
      const fontSize = Math.max(14, width / 8);
      ctx.font = `bold ${fontSize}px Arial`;
      ctx.fillStyle = face.matched ? '#03dac6' : '#cf6679';
      
      const textY = y1 > fontSize + 5 ? y1 - 5 : y1 + fontSize + 5;
      ctx.fillText(label, x1, textY);

      if (face.quality_score) {
        ctx.font = `${fontSize * 0.7}px Arial`;
        ctx.fillStyle = '#888';
        ctx.fillText(`Chất lượng: ${(face.quality_score * 100).toFixed(0)}%`, x1, y2 + fontSize);
      }
    });
  };

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-2 text-[#bb86fc]">
        <FontAwesomeIcon icon={faCamera} className="mr-2" />
        Nhận Diện Khuôn Mặt
      </h1>
      <p className="text-gray-400 mb-6">Upload ảnh để phát hiện và nhận diện khuôn mặt</p>

      {/* Upload Area */}
      <div
        className="border-2 border-dashed border-[#555] rounded-lg p-8 text-center cursor-pointer hover:border-[#03dac6] hover:bg-[#1a1a1a] transition-all mb-6"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
        />
        <FontAwesomeIcon icon={faUpload} className="text-4xl text-gray-500 mb-3" />
        <p className="text-gray-400">Kéo thả ảnh vào đây hoặc click để chọn</p>
      </div>

      {/* Preview and Canvas Container */}
      {previewUrl && (
        <div className="mb-6 relative inline-block max-w-[600px]">
          <img
            src={previewUrl}
            alt="Preview"
            className="max-w-full max-h-[600px] rounded-lg border border-[#444] block"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
          />
        </div>
      )}

      {/* Controls */}
      <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6 flex gap-4 items-center flex-wrap">
        <label className="flex items-center gap-2">
          <span>Ngưỡng nhận diện:</span>
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
        
        <button
          onClick={handleRecognize}
          disabled={!image || loading}
          className="bg-[#03dac6] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
              Đang xử lý...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <FontAwesomeIcon icon={faCamera} />
              Nhận Diện
            </span>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[#b71c1c] text-[#ef9a9a] border border-[#c62828] rounded-lg p-4 mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faExclamationCircle} />
          {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-xl font-semibold text-[#03dac6] flex items-center gap-2">
            <FontAwesomeIcon icon={faCheckCircle} />
            Kết quả: Tìm thấy {results.length} khuôn mặt
          </h2>
          {results.map((face, idx) => (
            <div
              key={idx}
              className={`bg-[#1a1a1a] border rounded-lg p-4 ${
                face.matched ? 'border-l-4 border-l-[#03dac6]' : 'border-l-4 border-l-[#cf6679]'
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
              {face.quality_score && (
                <div className="text-sm text-gray-500 mt-1">
                  Chất lượng: {(face.quality_score * 100).toFixed(0)}%
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
