'use client';

import { useState, useEffect } from 'react';
import { getPeople, trainPerson, trainAll } from '@/lib/api';
import { 
  faBrain, 
  faSpinner, 
  faCheckCircle, 
  faExclamationCircle,
  faRefresh,
  faUsers,
  faClock,
  faBolt
} from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

interface TrainingStatus {
  name: string;
  status: 'idle' | 'training' | 'success' | 'error';
  message?: string;
}

export default function TrainingPage() {
  const [people, setPeople] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [trainingStatuses, setTrainingStatuses] = useState<TrainingStatus[]>([]);
  const [trainingAll, setTrainingAll] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const loadPeople = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await getPeople();
      if (response.success && response.data) {
        const names = (response.data.people || []).map((p) => p.name);
        setPeople(names);
        setTrainingStatuses(
          names.map((name: string) => ({
            name,
            status: 'idle' as const
          }))
        );
      } else {
        setError('Không thể tải danh sách');
      }
    } catch (err) {
      setError('Lỗi kết nối đến server');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPeople();
  }, []);

  const handleTrainPerson = async (name: string) => {
    setTrainingStatuses(prev =>
      prev.map(s => s.name === name ? { ...s, status: 'training', message: undefined } : s)
    );
    setError('');
    setSuccess('');

    try {
      const response = await trainPerson(name);
      if (response.success) {
        setTrainingStatuses(prev =>
          prev.map(s => s.name === name ? { ...s, status: 'success', message: response.message } : s)
        );
        setSuccess(`Đã train thành công cho ${name}`);
      } else {
        setTrainingStatuses(prev =>
          prev.map(s => s.name === name ? { ...s, status: 'error', message: response.error } : s)
        );
        setError(response.error || 'Train thất bại');
      }
    } catch (err) {
      setTrainingStatuses(prev =>
        prev.map(s => s.name === name ? { ...s, status: 'error', message: 'Lỗi kết nối' } : s)
      );
      setError('Lỗi khi train model');
    }
  };

  const handleTrainAll = async () => {
    if (people.length === 0) return;
    
    setTrainingAll(true);
    setError('');
    setSuccess('');

    // Set all to training
    setTrainingStatuses(prev =>
      prev.map(s => ({ ...s, status: 'training' as const, message: undefined }))
    );

    try {
      const response = await trainAll();
      if (response.success) {
        setTrainingStatuses(prev =>
          prev.map(s => ({ ...s, status: 'success' as const, message: 'Thành công' }))
        );
        setSuccess('Đã train thành công cho tất cả người dùng');
      } else {
        setTrainingStatuses(prev =>
          prev.map(s => ({ ...s, status: 'error' as const, message: response.error }))
        );
        setError(response.error || 'Train thất bại');
      }
    } catch (err) {
      setTrainingStatuses(prev =>
        prev.map(s => ({ ...s, status: 'error' as const, message: 'Lỗi kết nối' }))
      );
      setError('Lỗi khi train model');
    } finally {
      setTrainingAll(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'training':
        return <FontAwesomeIcon icon={faSpinner} className="animate-spin text-[#bb86fc]" />;
      case 'success':
        return <FontAwesomeIcon icon={faCheckCircle} className="text-[#03dac6]" />;
      case 'error':
        return <FontAwesomeIcon icon={faExclamationCircle} className="text-[#cf6679]" />;
      default:
        return <FontAwesomeIcon icon={faClock} className="text-gray-500" />;
    }
  };

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-2 text-[#bb86fc]">
        <FontAwesomeIcon icon={faBrain} className="mr-2" />
        Training Model
      </h1>
      <p className="text-gray-400 mb-6">Huấn luyện model AI cho từng người hoặc toàn bộ hệ thống</p>

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
          <button
            onClick={handleTrainAll}
            disabled={trainingAll}
            className="bg-[#bb86fc] text-[#0a0a0a] px-6 py-2 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 transition-all"
          >
            {trainingAll ? (
              <span className="flex items-center gap-2">
                <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                Đang train tất cả...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <FontAwesomeIcon icon={faBolt} />
                Train Tất Cả ({people.length})
              </span>
            )}
          </button>
        )}

        <span className="text-gray-400 text-sm ml-auto">
          <FontAwesomeIcon icon={faUsers} className="mr-1" />
          {people.length} người dùng
        </span>
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

      {/* Training List */}
      {people.length === 0 && !loading ? (
        <div className="text-center py-16 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a]">
          <FontAwesomeIcon icon={faBrain} className="text-6xl text-gray-600 mb-4" />
          <p className="text-gray-500 text-lg">Chưa có người dùng nào</p>
          <p className="text-gray-600 text-sm mt-2">Hãy đăng ký người dùng trước khi train</p>
        </div>
      ) : (
        <div className="space-y-3">
          {trainingStatuses.map((ts) => (
            <div
              key={ts.name}
              className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4 flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                {getStatusIcon(ts.status)}
                <div>
                  <h3 className="font-semibold text-[#ededed]">{ts.name}</h3>
                  {ts.message && (
                    <p className="text-sm text-gray-400">{ts.message}</p>
                  )}
                </div>
              </div>

              <button
                onClick={() => handleTrainPerson(ts.name)}
                disabled={ts.status === 'training'}
                className="bg-[#bb86fc]/10 hover:bg-[#bb86fc]/20 text-[#bb86fc] border border-[#bb86fc]/30 rounded-lg py-2 px-4 transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {ts.status === 'training' ? (
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
          ))}
        </div>
      )}
    </div>
  );
}
