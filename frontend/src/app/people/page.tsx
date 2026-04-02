'use client';

import { useState, useEffect, useCallback } from 'react';
import { getPeople, deletePerson, trainPerson, trainAll, getPersonImages, getPersonStatus } from '@/lib/api';
import { 
  faUsers, 
  faRefresh, 
  faTrash, 
  faCheck, 
  faSpinner, 
  faExclamationCircle,
  faBrain,
  faCheckCircle,
  faClock,
  faCircleXmark,
  faImages,
  faXmark
} from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

interface PersonData {
  name: string;
  embeddings_count: number;
  trained: boolean;
  trained_at?: string;
  created_at?: string;
  images: Array<{ filename: string; data: string }>;
  loading: boolean;
}

export default function PeoplePage() {
  const [people, setPeople] = useState<PersonData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [training, setTraining] = useState<string | null>(null);
  const [trainingAll, setTrainingAll] = useState(false);
  const [viewingImages, setViewingImages] = useState<string | null>(null);
  const [loadingImages, setLoadingImages] = useState(false);
  const [totalEmbeddings, setTotalEmbeddings] = useState(0);

  const loadPeople = useCallback(async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    
    try {
      const response = await getPeople();
      if (response.success && response.data) {
        const peopleList = response.data.people || [];
        setTotalEmbeddings(response.data.total_embeddings || 0);
        
        const peopleData: PersonData[] = await Promise.all(
          peopleList.map(async (name: string) => {
            try {
              const statusRes = await getPersonStatus(name);
              const status = statusRes.success && statusRes.data ? statusRes.data : { trained: false, embedding_count: 0 };
              return {
                name,
                embeddings_count: status.embedding_count || 0,
                trained: status.trained || false,
                trained_at: status.trained_at,
                created_at: status.created_at,
                images: [],
                loading: false
              };
            } catch {
              return {
                name,
                embeddings_count: 0,
                trained: false,
                images: [],
                loading: false
              };
            }
          })
        );
        
        setPeople(peopleData);
      } else {
        setError('Không thể tải danh sách');
      }
    } catch (err) {
      setError('Lỗi kết nối đến server');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPeople();
  }, [loadPeople]);

  const loadPersonImages = async (name: string) => {
    if (viewingImages === name) {
      setViewingImages(null);
      return;
    }
    
    setLoadingImages(true);
    setViewingImages(name);
    
    try {
      const response = await getPersonImages(name);
      if (response.success && response.data) {
        setPeople(prev =>
          prev.map(p =>
            p.name === name
              ? { ...p, images: response.data.images || [] }
              : p
          )
        );
      }
    } catch (err) {
      setError('Không thể tải ảnh');
    } finally {
      setLoadingImages(false);
    }
  };

  const toggleSelect = (name: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const selectAll = () => {
    if (selected.size === people.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(people.map(p => p.name)));
    }
  };

  const handleDelete = async () => {
    if (selected.size === 0) return;
    
    if (!confirm(`Xóa ${selected.size} người đã chọn? Hành động này không thể hoàn tác.`)) {
      return;
    }

    setDeleting(true);
    
    try {
      for (const name of selected) {
        await deletePerson(name);
      }
      setSelected(new Set());
      await loadPeople();
      setSuccess(`Đã xóa ${selected.size} người dùng`);
    } catch (err) {
      setError('Lỗi khi xóa người dùng');
    } finally {
      setDeleting(false);
    }
  };

  const handleTrain = async (name: string) => {
    setTraining(name);
    setError('');
    setSuccess('');
    
    try {
      const response = await trainPerson(name);
      if (response.success) {
        setSuccess(`Đã train thành công cho ${name}`);
        await loadPeople();
      } else {
        setError(response.error || 'Train thất bại');
      }
    } catch (err) {
      setError('Lỗi khi train model');
    } finally {
      setTraining(null);
    }
  };

  const handleTrainAll = async () => {
    if (people.length === 0) return;
    
    setTrainingAll(true);
    setError('');
    setSuccess('');

    setPeople(prev => prev.map(p => ({ ...p, loading: true })));

    try {
      const response = await trainAll();
      if (response.success) {
        setSuccess(`Đã train thành công ${response.data?.success || 0}/${response.data?.total || 0} người`);
        await loadPeople();
      } else {
        setError(response.error || 'Train thất bại');
        setPeople(prev => prev.map(p => ({ ...p, loading: false })));
      }
    } catch (err) {
      setError('Lỗi khi train model');
      setPeople(prev => prev.map(p => ({ ...p, loading: false })));
    } finally {
      setTrainingAll(false);
    }
  };

  const getStatusBadge = (person: PersonData) => {
    if (person.loading) {
      return { label: 'Đang train...', color: '#bb86fc', bg: '#bb86fc/10', icon: faSpinner, spin: true };
    }
    if (person.trained) {
      return { label: 'Đã train', color: '#03dac6', bg: '#03dac6/10', icon: faCheckCircle, spin: false };
    }
    if (person.embeddings_count > 0) {
      return { label: 'Chưa train', color: '#ff9800', bg: '#ff9800/10', icon: faClock, spin: false };
    }
    return { label: 'Không có dữ liệu', color: '#cf6679', bg: '#cf6679/10', icon: faCircleXmark, spin: false };
  };

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-2 text-[#bb86fc]">
        <FontAwesomeIcon icon={faUsers} className="mr-2" />
        Quản Lý Người Dùng
      </h1>
      <p className="text-gray-400 mb-6">Quản lý, xem ảnh và train model cho người đã đăng ký</p>

      {/* Controls */}
      <div className="bg-[#1a1a1a] rounded-lg p-4 mb-6 flex gap-4 items-center flex-wrap">
        <button
          onClick={loadPeople}
          disabled={loading}
          className="bg-[#03dac6] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 transition-all"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
              Đang tải...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <FontAwesomeIcon icon={faRefresh} />
              Tải Danh Sách
            </span>
          )}
        </button>

        {people.length > 0 && (
          <>
            <button
              onClick={handleTrainAll}
              disabled={trainingAll}
              className="bg-[#bb86fc] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 transition-all"
            >
              {trainingAll ? (
                <span className="flex items-center gap-2">
                  <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                  Đang train...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <FontAwesomeIcon icon={faBrain} />
                  Train Tất Cả
                </span>
              )}
            </button>

            <button
              onClick={selectAll}
              className="bg-[#252525] text-[#ededed] px-4 py-2 rounded-lg hover:bg-[#2a2a2a] transition-all"
            >
              {selected.size === people.length ? 'Bỏ chọn tất cả' : 'Chọn tất cả'}
            </button>

            <button
              onClick={handleDelete}
              disabled={selected.size === 0 || deleting}
              className="bg-[#cf6679] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {deleting ? (
                <span className="flex items-center gap-2">
                  <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                  Đang xóa...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <FontAwesomeIcon icon={faTrash} />
                  Xóa ({selected.size})
                </span>
              )}
            </button>

            <span className="text-gray-400 text-sm ml-auto">
              Tổng: {people.length} người • {totalEmbeddings} embeddings
            </span>
          </>
        )}
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

      {/* People List */}
      {people.length === 0 && !loading ? (
        <div className="text-center py-16 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a]">
          <FontAwesomeIcon icon={faUsers} className="text-6xl text-gray-600 mb-4" />
          <p className="text-gray-500 text-lg">Chưa có ai được đăng ký</p>
          <p className="text-gray-600 text-sm mt-2">Hãy đăng ký người dùng mới ở tab "Đăng Ký"</p>
        </div>
      ) : (
        <div className="space-y-4">
          {people.map((person) => {
            const status = getStatusBadge(person);
            const isSelected = selected.has(person.name);
            const isTraining = training === person.name;
            const isViewing = viewingImages === person.name;
            
            return (
              <div
                key={person.name}
                className={`bg-[#1a1a1a] border rounded-xl overflow-hidden transition-all ${
                  isSelected ? 'border-[#03dac6] bg-[#03dac6]/5' : 'border-[#2a2a2a]'
                }`}
              >
                {/* Person Header */}
                <div className="p-5">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start gap-4 flex-1">
                      {/* Checkbox */}
                      <div 
                        className={`w-6 h-6 rounded border-2 flex items-center justify-center transition-all cursor-pointer mt-1 ${
                          isSelected 
                            ? 'bg-[#03dac6] border-[#03dac6]' 
                            : 'border-[#444] hover:border-[#03dac6]'
                        }`}
                        onClick={() => toggleSelect(person.name)}
                      >
                        {isSelected && (
                          <FontAwesomeIcon icon={faCheck} className="text-[#0a0a0a] text-xs" />
                        )}
                      </div>

                      {/* Info */}
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-[#ededed] mb-2">
                          {person.name}
                        </h3>
                        <div className="flex items-center gap-3 flex-wrap">
                          <span
                            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium"
                            style={{ color: status.color, backgroundColor: status.bg }}
                          >
                            <FontAwesomeIcon icon={status.icon} className={`text-xs ${status.spin ? 'animate-spin' : ''}`} />
                            {status.label}
                          </span>
                          <span className="text-xs text-gray-500">
                            {person.embeddings_count} embeddings
                          </span>
                          {person.trained_at && (
                            <span className="text-xs text-gray-500">
                              Train: {new Date(person.trained_at).toLocaleString('vi-VN')}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => loadPersonImages(person.name)}
                        className="bg-[#252525] hover:bg-[#2a2a2a] text-[#ededed] rounded-lg py-2 px-3 transition-all flex items-center gap-2 text-sm"
                      >
                        <FontAwesomeIcon icon={isViewing ? faXmark : faImages} />
                        {isViewing ? 'Đóng' : 'Ảnh'}
                      </button>
                      
                      <button
                        onClick={() => handleTrain(person.name)}
                        disabled={isTraining || person.loading}
                        className="bg-[#bb86fc]/10 hover:bg-[#bb86fc]/20 text-[#bb86fc] border border-[#bb86fc]/30 rounded-lg py-2 px-4 transition-all flex items-center gap-2 disabled:opacity-50 text-sm"
                      >
                        {isTraining || person.loading ? (
                          <>
                            <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                            Đang train...
                          </>
                        ) : (
                          <>
                            <FontAwesomeIcon icon={faBrain} />
                            Train
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Images Section */}
                {isViewing && (
                  <div className="border-t border-[#2a2a2a] p-5 bg-[#151515]">
                    <h4 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
                      <FontAwesomeIcon icon={faImages} />
                      Ảnh đã lưu ({person.images.length})
                    </h4>
                    
                    {loadingImages ? (
                      <div className="flex items-center justify-center py-8">
                        <FontAwesomeIcon icon={faSpinner} className="animate-spin text-2xl text-[#bb86fc]" />
                      </div>
                    ) : person.images.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">
                        <FontAwesomeIcon icon={faImages} className="text-3xl mb-2" />
                        <p>Chưa có ảnh nào được lưu</p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                        {person.images.map((img, idx) => (
                          <div key={idx} className="relative group">
                            <img
                              src={img.data}
                              alt={img.filename}
                              className="w-full h-24 object-cover rounded-lg border border-[#444]"
                            />
                            <div className="absolute bottom-1 left-1 bg-black/70 px-2 py-0.5 rounded text-xs text-white">
                              {img.filename}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
