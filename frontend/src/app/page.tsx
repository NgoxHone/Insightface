'use client';

import Link from 'next/link';
import { 
  faCamera, 
  faVideo, 
  faUserPlus, 
  faUsers,
  faBrain,
  faShieldHalved,
  faBolt,
  faChartLine
} from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

const features = [
  {
    icon: faCamera,
    title: 'Nhận Diện Khuôn Mặt',
    description: 'Upload ảnh và nhận diện khuôn mặt với độ chính xác cao',
    href: '/recognize',
    color: '#03dac6'
  },
  {
    icon: faVideo,
    title: 'Realtime Detection',
    description: 'Nhận diện khuôn mặt theo thời gian thực qua webcam',
    href: '/realtime',
    color: '#bb86fc'
  },
  {
    icon: faUserPlus,
    title: 'Đăng Ký Người Dùng',
    description: 'Thêm người mới vào hệ thống với nhiều ảnh',
    href: '/register',
    color: '#ff9800'
  },
  {
    icon: faUsers,
    title: 'Quản Lý Người Dùng',
    description: 'Xem, train và xóa người dùng đã đăng ký',
    href: '/people',
    color: '#4caf50'
  },
  {
    icon: faBrain,
    title: 'Training Model',
    description: 'Train model AI cho từng người hoặc toàn bộ',
    href: '/training',
    color: '#cf6679'
  },
  {
    icon: faShieldHalved,
    title: 'Bảo Mật Cao',
    description: 'Dữ liệu được lưu trữ an toàn và mã hóa',
    href: '/people',
    color: '#2196f3'
  }
];

const stats = [
  { icon: faBolt, label: 'Tốc Độ', value: '< 1s', desc: 'Nhận diện nhanh' },
  { icon: faChartLine, label: 'Chính Xác', value: '99%', desc: 'Độ chính xác' },
  { icon: faUsers, label: 'Người Dùng', value: '∞', desc: 'Không giới hạn' },
];

export default function HomePage() {
  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <section className="mb-16">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-[#03dac6] via-[#bb86fc] to-[#cf6679] bg-clip-text text-transparent">
            Hệ Thống Nhận Diện Khuôn Mặt
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Giải pháp nhận diện khuôn mặt thông minh với công nghệ AI tiên tiến, 
            hỗ trợ nhận diện realtime và quản lý người dùng dễ dàng
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {stats.map((stat, idx) => (
            <div key={idx} className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-6 text-center">
              <FontAwesomeIcon icon={stat.icon} className="text-3xl text-[#03dac6] mb-3" />
              <div className="text-3xl font-bold text-[#ededed] mb-1">{stat.value}</div>
              <div className="text-sm text-gray-400">{stat.label}</div>
              <div className="text-xs text-gray-600 mt-1">{stat.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section>
        <h2 className="text-2xl font-bold mb-6 text-[#ededed]">Tính Năng Chính</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, idx) => (
            <Link
              key={idx}
              href={feature.href}
              className="group bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-6 hover:border-[#03dac6]/50 transition-all hover:scale-105 cursor-pointer"
            >
              <div 
                className="w-14 h-14 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform"
                style={{ backgroundColor: `${feature.color}20` }}
              >
                <FontAwesomeIcon 
                  icon={feature.icon} 
                  className="text-2xl"
                  style={{ color: feature.color }}
                />
              </div>
              <h3 className="text-lg font-bold text-[#ededed] mb-2 group-hover:text-[#03dac6] transition-colors">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-400">
                {feature.description}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Quick Actions */}
      <section className="mt-12 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-6 text-[#ededed]">Bắt Đầu Nhanh</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/register"
            className="flex items-center gap-4 bg-[#252525] hover:bg-[#2a2a2a] rounded-lg p-4 transition-all"
          >
            <div className="w-12 h-12 bg-[#03dac6]/20 rounded-lg flex items-center justify-center">
              <span className="text-2xl font-bold text-[#03dac6]">1</span>
            </div>
            <div>
              <h3 className="font-semibold text-[#ededed]">Đăng Ký Người Dùng</h3>
              <p className="text-sm text-gray-400">Thêm người mới vào hệ thống</p>
            </div>
          </Link>

          <Link
            href="/training"
            className="flex items-center gap-4 bg-[#252525] hover:bg-[#2a2a2a] rounded-lg p-4 transition-all"
          >
            <div className="w-12 h-12 bg-[#bb86fc]/20 rounded-lg flex items-center justify-center">
              <span className="text-2xl font-bold text-[#bb86fc]">2</span>
            </div>
            <div>
              <h3 className="font-semibold text-[#ededed]">Train Model</h3>
              <p className="text-sm text-gray-400">Huấn luyện AI nhận diện</p>
            </div>
          </Link>

          <Link
            href="/recognize"
            className="flex items-center gap-4 bg-[#252525] hover:bg-[#2a2a2a] rounded-lg p-4 transition-all"
          >
            <div className="w-12 h-12 bg-[#cf6679]/20 rounded-lg flex items-center justify-center">
              <span className="text-2xl font-bold text-[#cf6679]">3</span>
            </div>
            <div>
              <h3 className="font-semibold text-[#ededed]">Nhận Diện</h3>
              <p className="text-sm text-gray-400">Bắt đầu nhận diện khuôn mặt</p>
            </div>
          </Link>
        </div>
      </section>
    </div>
  );
}
