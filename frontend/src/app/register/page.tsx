'use client';

import { useState, useRef, useCallback } from 'react';
import { registerPerson } from '@/lib/api';
import { faUserPlus, faUpload, faSpinner, faExclamationCircle, faCheckCircle, faTrash, faImages } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

interface ImageItem {
  file: File;
  url: string;
}

export default function RegisterPage() {
  const [name, setName] = useState('');
  const [images, setImages] = useState<ImageItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((files: FileList) => {
    const newImages = Array.from(files)
      .filter(f => f.type.startsWith('image/'))
      .map(file => ({
        file,
        url: URL.createObjectURL(file)
      }));
    
    setImages(prev => [...prev, ...newImages]);
    setError('');
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files);
  }, [handleFileSelect]);

  const removeImage = (index: number) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleRegister = async () => {
    if (!name.trim() || images.length === 0) {
      setError('Vui lòng nhập tên và chọn ít nhất 1 ảnh');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await registerPerson(name.trim(), images.map(img => img.file));
      
      if (response.success) {
        setSuccess(`Đã đăng ký ${name} với ${response.data?.embeddings_extracted || 0} ảnh khuôn mặt!`);
        setName('');
        setImages([]);
      } else {
        setError(response.error || 'Đăng ký thất bại');
      }
    } catch (err) {
      setError('Lỗi kết nối đến server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-2 text-[#bb86fc]">
        <FontAwesomeIcon icon={faUserPlus} className="mr-2" />
        Đăng Ký Người Mới
      </h1>
      <p className="text-gray-400 mb-6">Thêm người mới vào hệ thống với ảnh khuôn mặt</p>

      {/* Name Input */}
      <div className="mb-6">
        <label className="block mb-2 font-medium">Họ và tên:</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Nhập tên..."
          maxLength={100}
          className="w-full max-w-md bg-[#252525] border border-[#3a3a3a] rounded-lg px-4 py-2 text-[#ededed] focus:border-[#03dac6] focus:outline-none focus:ring-2 focus:ring-[#03dac6]/20"
        />
      </div>

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
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFileSelect(e.target.files)}
        />
        <FontAwesomeIcon icon={faUpload} className="text-4xl text-gray-500 mb-3" />
        <p className="text-gray-400">
          Kéo thả nhiều ảnh vào đây hoặc click để chọn
          <br />
          <small className="text-sm">Nên có ít nhất 5 ảnh với các góc khác nhau</small>
        </p>
      </div>

      {/* Image Previews */}
      {images.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3 text-gray-400">
            <FontAwesomeIcon icon={faImages} />
            <span>{images.length} ảnh đã chọn</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {images.map((img, idx) => (
              <div key={idx} className="relative group">
                <img
                  src={img.url}
                  alt={`Ảnh ${idx + 1}`}
                  className="w-full h-32 object-cover rounded-lg border border-[#444]"
                />
                <button
                  onClick={() => removeImage(idx)}
                  className="absolute top-1 right-1 bg-[#cf6679] text-[#0a0a0a] w-6 h-6 rounded-full opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-sm font-bold"
                >
                  ✕
                </button>
                <div className="absolute bottom-1 left-1 bg-black/70 px-2 py-0.5 rounded text-xs">
                  {idx + 1}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Controls */}
      <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6 flex gap-4 items-center flex-wrap">
        <button
          onClick={handleRegister}
          disabled={!name.trim() || images.length === 0 || loading}
          className="bg-[#03dac6] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
              Đang đăng ký...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <FontAwesomeIcon icon={faCheckCircle} />
              Đăng Ký
            </span>
          )}
        </button>
        
        <button
          onClick={() => {
            setName('');
            setImages([]);
            setError('');
            setSuccess('');
          }}
          className="bg-[#cf6679] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 transition-all"
        >
          <FontAwesomeIcon icon={faTrash} className="mr-2" />
          Xóa
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[#b71c1c] text-[#ef9a9a] border border-[#c62828] rounded-lg p-4 mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faExclamationCircle} />
          {error}
        </div>
      )}

      {/* Success */}
      {success && (
        <div className="bg-[#1b5e20] text-[#a5d6a7] border border-[#2e7d32] rounded-lg p-4 mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faCheckCircle} />
          {success}
        </div>
      )}
    </div>

  );
}
