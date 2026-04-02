'use client';

import { useState, useRef, useCallback } from 'react';
import { recognizeFace, FaceResult } from '@/lib/api';
import { faCamera, faUpload, faSpinner, faExclamationCircle, faCheckCircle, faClock, faImage } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

export default function RecognizePage() {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [previewKey, setPreviewKey] = useState<number>(0);
  const [threshold, setThreshold] = useState(0.6);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<FaceResult[]>([]);
  const [error, setError] = useState<string>('');
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [faceImages, setFaceImages] = useState<string[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewImgRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Vui lòng chọn file ảnh');
      return;
    }
    setImage(file);
    // Revoke old URL to prevent memory leaks
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPreviewKey(prev => prev + 1); // Force re-render to prevent cache
    setResults([]);
    setFaceImages([]);
    setProcessingTime(0);
    setError('');
  }, [previewUrl]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleRecognize = async () => {
    if (!image || !previewImgRef.current) return;
    
    setLoading(true);
    setError('');
    setResults([]);
    setFaceImages([]);
    setProcessingTime(0);

    const startTime = performance.now();

    try {
      const response = await recognizeFace(image, threshold);
      const endTime = performance.now();
      setProcessingTime(endTime - startTime);

      if (response.success && response.data) {
        setResults(response.data.faces);
        // Extract face images first, then draw on canvas
        extractFaceImages(response.data.faces);
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

  const extractFaceImages = (faces: FaceResult[]) => {
    const img = previewImgRef.current;
    if (!img || !img.complete || img.naturalWidth === 0) {
      console.error('Image not ready for face extraction');
      return;
    }

    const images: string[] = [];
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    faces.forEach((face, idx) => {
      try {
        const [x1, y1, x2, y2] = face.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Add padding around face
        const padding = 0.2;
        const paddedX1 = Math.max(0, x1 - width * padding);
        const paddedY1 = Math.max(0, y1 - height * padding);
        const paddedWidth = Math.min(img.naturalWidth - paddedX1, width * (1 + 2 * padding));
        const paddedHeight = Math.min(img.naturalHeight - paddedY1, height * (1 + 2 * padding));

        tempCanvas.width = paddedWidth;
        tempCanvas.height = paddedHeight;
        tempCtx.clearRect(0, 0, paddedWidth, paddedHeight);
        tempCtx.drawImage(img, paddedX1, paddedY1, paddedWidth, paddedHeight, 0, 0, paddedWidth, paddedHeight);

        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
        images.push(dataUrl);
      } catch (err) {
        console.error(`Failed to extract face ${idx}:`, err);
        images.push('');
      }
    });

    setFaceImages(images);
  };

  const drawResults = (faces: FaceResult[]) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const img = previewImgRef.current;
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

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
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
            ref={previewImgRef}
            key={previewKey}
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
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold text-[#03dac6] flex items-center gap-2">
              <FontAwesomeIcon icon={faCheckCircle} />
              Kết quả: Tìm thấy {results.length} khuôn mặt
            </h2>
            <div className="flex items-center gap-4 text-sm text-gray-400">
              <span className="flex items-center gap-1">
                <FontAwesomeIcon icon={faClock} />
                Thời gian: {formatTime(processingTime)}
              </span>
            </div>
          </div>

          {/* Face Table */}
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg overflow-hidden">
            <div className="p-4 border-b border-[#2a2a2a]">
              <h3 className="font-semibold text-[#ededed] flex items-center gap-2">
                <FontAwesomeIcon icon={faImage} />
                Chi tiết khuôn mặt
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-[#252525]">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">#</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Ảnh khuôn mặt</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Tên</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Độ tin cậy</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Chất lượng</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Trạng thái</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#2a2a2a]">
                  {results.map((face, idx) => (
                    <tr key={idx} className="hover:bg-[#252525] transition-colors">
                      <td className="px-4 py-3 text-sm text-gray-400">{idx + 1}</td>
                      <td className="px-4 py-3">
                        {faceImages[idx] ? (
                          <img
                            src={faceImages[idx]}
                            alt={`Face ${idx + 1}`}
                            className="w-16 h-16 object-cover rounded-lg border border-[#444]"
                          />
                        ) : (
                          <div className="w-16 h-16 bg-[#252525] rounded-lg flex items-center justify-center">
                            <FontAwesomeIcon icon={faImage} className="text-gray-600" />
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`font-semibold ${face.matched ? 'text-[#03dac6]' : 'text-[#cf6679]'}`}>
                          {face.name === 'unknown' ? 'Không xác định' : face.name}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-300">
                        {(face.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-400">
                        {face.quality_score ? `${(face.quality_score * 100).toFixed(0)}%` : '-'}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                          face.matched 
                            ? 'bg-[#03dac6]/10 text-[#03dac6]' 
                            : 'bg-[#cf6679]/10 text-[#cf6679]'
                        }`}>
                          <FontAwesomeIcon icon={face.matched ? faCheckCircle : faExclamationCircle} className="text-xs" />
                          {face.matched ? 'Đã nhận diện' : 'Không khớp'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
