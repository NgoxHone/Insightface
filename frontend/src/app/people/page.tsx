"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  faUsers,
  faRefresh,
  faTrash,
  faCheck,
  faSpinner,
  faExclamationCircle,
  faBrain,
  faCheckCircle,
  faXmark,
  faEye,
  faSearch,
  faUserTag,
  faCameraRetro,
  faLayerGroup,
  faUserPlus,
  faUpload,
  faImages,
  faPhotoFilm,
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

interface PersonData {
  name: string;
  image_count: number;
  embeddings_count: number;
  is_trained: boolean;
  trained_at?: string;
  has_database: boolean;
  images?: Array<{ filename: string; data: string }>;
  faces?: Array<{ filename: string; data: string }>;
  loading: boolean;
  expanded: boolean;
}

export default function PeoplePage() {
  const [totalImages, setTotalImages] = useState(0);
  const [people, setPeople] = useState<PersonData[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [training, setTraining] = useState<string | null>(null);
  const [trainingAll, setTrainingAll] = useState(false);

  // Registration state
  const [showRegister, setShowRegister] = useState(false);
  const [regName, setRegName] = useState("");
  const [regImages, setRegImages] = useState<{ file: File; url: string }[]>([]);
  const [regLoading, setRegLoading] = useState(false);
  const regFileRef = useRef<HTMLInputElement>(null);

  // Filter logic
  const filteredPeople = useMemo(() => {
    return people.filter((p) =>
      p.name.toLowerCase().includes(searchTerm.toLowerCase()),
    );
  }, [people, searchTerm]);

  const loadPeople = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/people`);
      const response = await res.json();
      if (response.success) {
        setPeople(
          response.data.people.map((p: any) => ({
            ...p,
            images: [],
            faces: [],
            loading: false,
            expanded: false,
          })),
        );
        setTotalImages(response.data.total_images);
      }
    } catch (err) {
      setError("Lỗi kết nối server");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPeople();
  }, [loadPeople]);

  const loadPersonDetails = async (name: string) => {
    const person = people.find((p) => p.name === name);
    if (person?.expanded) {
      setPeople((prev) =>
        prev.map((p) => (p.name === name ? { ...p, expanded: false } : p)),
      );
      return;
    }
    setPeople((prev) =>
      prev.map((p) =>
        p.name === name ? { ...p, expanded: true, loading: true } : p,
      ),
    );
    try {
      const [imgRes, faceRes] = await Promise.all([
        fetch(`${API_BASE}/api/person/${encodeURIComponent(name)}`),
        fetch(`${API_BASE}/api/person/${encodeURIComponent(name)}/faces`),
      ]);
      const [imgData, faceData] = await Promise.all([
        imgRes.json(),
        faceRes.json(),
      ]);
      setPeople((prev) =>
        prev.map((p) =>
          p.name === name
            ? {
                ...p,
                images: imgData.success ? imgData.data.images : [],
                faces: faceData.success ? faceData.data.faces : [],
                loading: false,
              }
            : p,
        ),
      );
    } catch (err) {
      setPeople((prev) =>
        prev.map((p) => (p.name === name ? { ...p, loading: false } : p)),
      );
    }
  };

  const handleTrainAll = async () => {
    setTrainingAll(true);
    try {
      await fetch(`${API_BASE}/api/train-all`, { method: "POST" });
      setSuccess("Đã hoàn thành train toàn bộ hệ thống");
      loadPeople();
    } catch (err) {
      setError("Lỗi khi train all");
    } finally {
      setTrainingAll(false);
    }
  };

  const toggleSelect = (name: string) => {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelected(next);
  };

  const handleTrain = async (name: string) => {
    setTraining(name);
    try {
      await fetch(`${API_BASE}/api/train/${encodeURIComponent(name)}`, {
        method: "POST",
      });
      setSuccess(`Đã cập nhật model cho ${name}`);
      loadPeople();
    } catch (err) {
      setError("Lỗi khi train");
    } finally {
      setTraining(null);
    }
  };

  const handleRegFileSelect = useCallback((files: FileList) => {
    const newImages = Array.from(files)
      .filter((f) => f.type.startsWith("image/"))
      .map((file) => ({ file, url: URL.createObjectURL(file) }));
    setRegImages((prev) => [...prev, ...newImages]);
  }, []);

  const handleRegister = async () => {
    if (!regName.trim() || regImages.length === 0) {
      setError("Vui lòng nhập tên và chọn ít nhất 1 ảnh");
      return;
    }
    setRegLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("name", regName.trim());
      regImages.forEach((img) => formData.append("images", img.file));
      const res = await fetch(`${API_BASE}/api/register`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setSuccess(
          `Đã đăng ký ${regName} với ${data.data?.embeddings_extracted || 0} ảnh khuôn mặt!`,
        );
        setRegName("");
        setRegImages([]);
        setShowRegister(false);
        loadPeople();
      } else {
        setError(data.error || "Đăng ký thất bại");
      }
    } catch {
      setError("Lỗi kết nối đến server");
    } finally {
      setRegLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen pb-40 px-4 md:px-0 max-w-[1400px] mx-auto">
      {/* 1. TOP HEADER (STATIC) */}
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-5 gap-3">
        <div>
          <h1 className="text-2xl md:text-3xl font-black text-[#bb86fc] flex items-center gap-3">
            <FontAwesomeIcon
              style={{ maxWidth: 25, maxHeight: 25 }}
              icon={faUsers}
            />
            HỆ THỐNG NHÂN SỰ
          </h1>
          <p className="text-gray-500 text-sm">
            Quản lý kho dữ liệu nhận diện AI
          </p>
          {people.length > 0 && (
            <div className="flex items-center gap-3 mt-2">
              <span className="inline-flex items-center gap-1.5 bg-[#bb86fc]/10 border border-[#bb86fc]/30 text-[#bb86fc] px-3 py-1 rounded-full text-xs font-bold">
                <FontAwesomeIcon
                  icon={faUsers}
                  style={{ maxWidth: 12, maxHeight: 12 }}
                />
                {people.length} người dùng
              </span>
              <span className="inline-flex items-center gap-1.5 bg-[#03dac6]/10 border border-[#03dac6]/30 text-[#03dac6] px-3 py-1 rounded-full text-xs font-bold">
                <FontAwesomeIcon
                  icon={faPhotoFilm}
                  style={{ maxWidth: 12, maxHeight: 12 }}
                />
                {totalImages} hình ảnh
              </span>
            </div>
          )}
        </div>

        {/* Nút chức năng tổng quát */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowRegister((v) => !v)}
            className="h-10 px-4 rounded-xl bg-[#03dac6] text-black font-bold hover:opacity-90 transition-all flex items-center gap-2 shadow-lg shadow-[#03dac6]/20 text-sm"
          >
            <FontAwesomeIcon icon={faUserPlus} />
            Đăng Ký Mới
          </button>
          <button
            onClick={loadPeople}
            disabled={loading}
            className="h-10 px-4 rounded-xl bg-[#252525] text-white hover:bg-[#333] transition-all flex items-center gap-2 shadow-lg text-sm"
          >
            <FontAwesomeIcon
              icon={faRefresh}
              className={loading ? "animate-spin" : ""}
            />
            Làm mới
          </button>
          <button
            onClick={handleTrainAll}
            disabled={trainingAll}
            className="h-10 px-4 rounded-xl bg-[#bb86fc] text-black font-bold hover:opacity-90 transition-all flex items-center gap-2 shadow-lg shadow-[#bb86fc]/20 text-sm"
          >
            <FontAwesomeIcon
              icon={faBrain}
              className={trainingAll ? "animate-spin" : ""}
            />
            Train Tất Cả
          </button>
        </div>
      </div>

      {/* 2. STICKY SEARCH BAR (LUÔN DÍNH KHI CUỘN) */}
      <div className="sticky top-0 z-[100] bg-[#0a0a0a]/95 backdrop-blur py-3 border-b border-[#1a1a1a]">
        <div className="bg-[#141414] border border-[#2c2c2c] rounded-2xl p-2 flex items-center gap-3 shadow-2xl">
          <div className="relative flex-1 group">
            <FontAwesomeIcon
              style={{ maxWidth: 25, maxHeight: 25 }}
              icon={faSearch}
              className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-[#bb86fc]"
            />
            <input
              type="text"
              placeholder="Nhập tên nhân sự cần tìm..."
              className="w-full bg-transparent py-2 pl-12 pr-4 outline-none text-white"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          {searchTerm && (
            <button
              onClick={() => setSearchTerm("")}
              className="w-10 h-10 text-gray-500 hover:text-white"
            >
              <FontAwesomeIcon icon={faXmark} />
            </button>
          )}
        </div>
      </div>

      {/* Alert Messages */}
      <div className="mt-4">
        {/* Registration Panel */}
        {showRegister && (
          <div className="mb-6 bg-[#141414] border border-[#03dac6]/30 rounded-2xl p-6 animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-[#03dac6] flex items-center gap-2">
                <FontAwesomeIcon
                  icon={faUserPlus}
                  style={{ maxWidth: 20, maxHeight: 20 }}
                />
                Đăng Ký Người Mới
              </h3>
              <button
                onClick={() => {
                  setShowRegister(false);
                  setRegName("");
                  setRegImages([]);
                }}
                className="w-8 h-8 rounded-lg bg-[#252525] text-gray-400 hover:text-white transition-all flex items-center justify-center"
              >
                <FontAwesomeIcon icon={faXmark} />
              </button>
            </div>

            <div className="mb-4">
              <input
                type="text"
                value={regName}
                onChange={(e) => setRegName(e.target.value)}
                placeholder="Nhập tên người dùng..."
                maxLength={100}
                className="w-full max-w-md bg-[#252525] border border-[#3a3a3a] rounded-lg px-4 py-2 text-[#ededed] focus:border-[#03dac6] focus:outline-none focus:ring-2 focus:ring-[#03dac6]/20"
              />
            </div>

            <div
              className="border-2 border-dashed border-[#444] rounded-lg p-6 text-center cursor-pointer hover:border-[#03dac6] hover:bg-[#1a1a1a] transition-all mb-4"
              onClick={() => regFileRef.current?.click()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files.length > 0)
                  handleRegFileSelect(e.dataTransfer.files);
              }}
              onDragOver={(e) => e.preventDefault()}
            >
              <input
                ref={regFileRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={(e) =>
                  e.target.files && handleRegFileSelect(e.target.files)
                }
              />
              <FontAwesomeIcon
                icon={faUpload}
                className="text-3xl text-gray-500 mb-2"
              />
              <p className="text-gray-400 text-sm">
                Kéo thả nhiều ảnh vào đây hoặc click để chọn
                <br />
                <small>Nên có ít nhất 5 ảnh với các góc khác nhau</small>
              </p>
            </div>

            {regImages.length > 0 && (
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2 text-gray-400 text-sm">
                  <FontAwesomeIcon icon={faImages} />
                  <span>{regImages.length} ảnh đã chọn</span>
                </div>
                <div className="grid grid-cols-4 sm:grid-cols-6 xl:grid-cols-8 gap-2">
                  {regImages.map((img, idx) => (
                    <div key={idx} className="relative group aspect-square">
                      <img
                        src={img.url}
                        alt=""
                        className="w-full h-full object-cover rounded-lg border border-[#444]"
                      />
                      <button
                        onClick={() =>
                          setRegImages((prev) =>
                            prev.filter((_, i) => i !== idx),
                          )
                        }
                        className="absolute top-1 right-1 bg-[#cf6679] text-black w-5 h-5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-xs font-bold"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleRegister}
                disabled={
                  !regName.trim() || regImages.length === 0 || regLoading
                }
                className="bg-[#03dac6] text-black px-6 py-2 rounded-lg font-bold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 text-sm"
              >
                {regLoading ? (
                  <>
                    <FontAwesomeIcon
                      icon={faSpinner}
                      className="animate-spin"
                    />
                    Đang đăng ký...
                  </>
                ) : (
                  <>
                    <FontAwesomeIcon icon={faCheckCircle} />
                    Đăng Ký
                  </>
                )}
              </button>
              <button
                onClick={() => {
                  setRegName("");
                  setRegImages([]);
                }}
                className="bg-[#252525] text-gray-300 px-4 py-2 rounded-lg hover:bg-[#333] transition-all text-sm"
              >
                Xóa hết
              </button>
            </div>
          </div>
        )}

        {success && (
          <div className="mb-4 p-4 bg-green-500/10 border border-green-500/50 text-green-400 rounded-xl flex items-center gap-3 animate-slide-up">
            <FontAwesomeIcon icon={faCheckCircle} /> {success}
          </div>
        )}
        {error && (
          <div className="mb-4 p-4 bg-red-500/10 border border-red-500/50 text-red-400 rounded-xl flex items-center gap-3 animate-slide-up">
            <FontAwesomeIcon icon={faExclamationCircle} /> {error}
          </div>
        )}
      </div>

      {/* 3. MAIN LIST */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredPeople.map((person) => {
          const isSelected = selected.has(person.name);
          return (
            <div
              key={person.name}
              className={`rounded-2xl border transition-all duration-300 ${isSelected ? "border-[#03dac6] bg-[#03dac6]/5 shadow-[0_0_20px_rgba(3,218,198,0.05)]" : "border-[#2a2a2a] bg-[#111]"}`}
            >
              <div className="p-4 flex items-center gap-3">
                {/* Custom Checkbox */}
                <div
                  onClick={() => toggleSelect(person.name)}
                  className={`w-6 h-6 rounded-md border-2 flex items-center justify-center cursor-pointer transition-all ${isSelected ? "bg-[#03dac6] border-[#03dac6]" : "border-gray-700 hover:border-[#03dac6]"}`}
                >
                  {isSelected && (
                    <FontAwesomeIcon
                      icon={faCheck}
                      className="text-black text-xs"
                    />
                  )}
                </div>

                {/* Name & Summary */}
                <div
                  className="flex-1 cursor-pointer"
                  onClick={() => loadPersonDetails(person.name)}
                >
                  <h2 className="font-bold text-lg text-gray-100 leading-tight">
                    {person.name}
                  </h2>
                  <div className="flex gap-3 mt-1 flex-wrap">
                    <span className="text-xs text-gray-500 flex items-center gap-1.5 font-medium">
                      <FontAwesomeIcon
                        style={{ maxWidth: 25, maxHeight: 25 }}
                        icon={faCameraRetro}
                        className="text-gray-600"
                      />{" "}
                      {person.image_count} Ảnh gốc
                    </span>
                    <span className="text-xs text-[#bb86fc] flex items-center gap-1.5 font-bold">
                      <FontAwesomeIcon
                        style={{ maxWidth: 25, maxHeight: 25 }}
                        icon={faUserTag}
                      />{" "}
                      {person.embeddings_count} embedding
                    </span>
                  </div>
                </div>

                {/* Card Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() => handleTrain(person.name)}
                    className="w-10 h-10 rounded-lg bg-[#bb86fc]/10 text-[#bb86fc] hover:bg-[#bb86fc] hover:text-black transition-all"
                  >
                    <FontAwesomeIcon
                      style={{ maxWidth: 25, maxHeight: 25 }}
                      icon={training === person.name ? faSpinner : faBrain}
                      className={training === person.name ? "animate-spin" : ""}
                    />
                  </button>
                  <button
                    onClick={() => loadPersonDetails(person.name)}
                    className={`w-10 h-10 rounded-lg transition-all ${person.expanded ? "bg-white text-black" : "bg-[#252525] text-white"}`}
                  >
                    <FontAwesomeIcon
                      style={{ maxWidth: 25, maxHeight: 25 }}
                      icon={person.expanded ? faXmark : faEye}
                    />
                  </button>
                </div>
              </div>

              {/* 4. EXPANDED CONTENT: SEPARATED RAW vs FACES */}
              {person.expanded && (
                <div className="p-4 pt-0 space-y-6 border-t border-[#222] bg-black/30 rounded-b-2xl">
                  {/* PHẦN 1: ẢNH GỐC (RAW) */}
                  <div className="mt-4">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-1 h-4 bg-[#03dac6] rounded-full"></div>
                      <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest">
                        Dữ liệu gốc (Raw Images)
                      </h4>
                    </div>
                    {person.loading ? (
                      <div className="py-6 text-center">
                        <FontAwesomeIcon
                          style={{ maxWidth: 25, maxHeight: 25 }}
                          icon={faSpinner}
                          className="animate-spin text-[#03dac6] text-xl"
                        />
                      </div>
                    ) : (
                      <div className="grid grid-cols-4 sm:grid-cols-6 xl:grid-cols-8 gap-2">
                        {person.images?.map((img, i) => (
                          <div
                            key={i}
                            className="aspect-square rounded-lg overflow-hidden border border-white/5 opacity-50 hover:opacity-100 transition-all hover:scale-105"
                          >
                            <img
                              src={img.data}
                              className="w-full h-full object-cover"
                              alt=""
                            />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* PHẦN 2: FACES (ĐÃ CẮT) */}
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-1 h-4 bg-[#bb86fc] rounded-full animate-pulse"></div>
                      <h4 className="text-xs font-black text-[#bb86fc] uppercase tracking-widest">
                        Khuôn mặt trích xuất (Face Crops)
                      </h4>
                    </div>
                    <div className="grid grid-cols-4 sm:grid-cols-6 xl:grid-cols-8 gap-2.5">
                      {person.faces?.map((face, i) => (
                        <div
                          key={i}
                          className="relative aspect-square rounded-xl overflow-hidden border-2 border-[#bb86fc]/40 shadow-lg shadow-[#bb86fc]/5"
                        >
                          <img
                            src={face.data}
                            className="w-full h-full object-cover"
                            alt=""
                          />
                          <div className="absolute top-0 right-0 bg-[#03dac6] p-1 rounded-bl-lg shadow-md">
                            <FontAwesomeIcon
                              style={{ maxWidth: 25, maxHeight: 25 }}
                              icon={faCheck}
                              className="text-black text-[8px]"
                            />
                          </div>
                        </div>
                      ))}
                      {person.faces?.length === 0 && (
                        <div className="col-span-full py-4 text-gray-600 text-[11px] italic">
                          Chưa có khuôn mặt nào được xử lý. Nhấn Train để AI bắt
                          đầu cắt mặt.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 5. FLOATING ACTION BAR (CHỈ HIỆN KHI CHỌN) */}
      {selected.size > 0 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[110] animate-bounce-in w-max">
          <div className="bg-[#1a1a1a] border border-[#bb86fc] p-2.5 rounded-2xl shadow-[0_20px_60px_rgba(0,0,0,0.7)] flex items-center gap-4">
            <div className="px-5 py-2 bg-black/50 rounded-xl border border-white/5">
              <span className="text-[#bb86fc] font-black text-xl">
                {selected.size}
              </span>
              <span className="ml-2 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                Đã chọn
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={async () => {
                  setDeleting(true);
                  for (let name of selected)
                    await fetch(
                      `${API_BASE}/api/person/${encodeURIComponent(name)}`,
                      { method: "DELETE" },
                    );
                  setSelected(new Set());
                  loadPeople();
                  setDeleting(false);
                }}
                className="h-11 px-5 rounded-xl bg-red-600 text-white font-bold text-sm hover:bg-red-700 transition-all flex items-center gap-2"
              >
                <FontAwesomeIcon
                  style={{ maxWidth: 25, maxHeight: 25 }}
                  icon={deleting ? faSpinner : faTrash}
                  className={deleting ? "animate-spin" : ""}
                />
                Xóa {selected.size} mục
              </button>

              <button
                onClick={() => setSelected(new Set())}
                className="w-12 h-12 flex items-center justify-center text-gray-500 hover:text-white transition-all bg-[#252525] rounded-xl"
              >
                <FontAwesomeIcon
                  style={{ maxWidth: 25, maxHeight: 25 }}
                  icon={faXmark}
                  className="text-xl"
                />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {filteredPeople.length === 0 && !loading && (
        <div className="text-center py-24 bg-[#111] rounded-3xl border border-dashed border-[#333] mt-8">
          <FontAwesomeIcon
            style={{ maxWidth: 25, maxHeight: 25 }}
            icon={faLayerGroup}
            className="text-4xl text-gray-700 mb-4"
          />
          <p className="text-gray-500 font-medium italic">
            Không tìm thấy dữ liệu phù hợp với bộ lọc
          </p>
        </div>
      )}
    </div>
  );
}
