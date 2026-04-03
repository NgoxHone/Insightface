"use client";

import Link from "next/link";
import {
  faCamera,
  faVideo,
  faUsers,
  faBrain,
  faShieldHalved,
  faBolt,
  faChartLine,
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

const features = [
  {
    icon: faCamera,
    title: "Nhận Diện Khuôn Mặt",
    description: "Upload ảnh và nhận diện khuôn mặt với độ chính xác cao",
    href: "/recognize",
    color: "#03dac6",
  },
  {
    icon: faVideo,
    title: "Realtime Detection",
    description: "Nhận diện khuôn mặt theo thời gian thực qua webcam",
    href: "/realtime",
    color: "#bb86fc",
  },
  {
    icon: faUsers,
    title: "Quản Lý Người Dùng",
    description: "Đăng ký, xem, train và xóa người dùng đã đăng ký",
    href: "/people",
    color: "#4caf50",
  },
  {
    icon: faBrain,
    title: "Training Model",
    description: "Train model AI cho từng người hoặc toàn bộ",
    href: "/people",
    color: "#cf6679",
  },
  {
    icon: faShieldHalved,
    title: "Bảo Mật Cao",
    description: "Dữ liệu được lưu trữ an toàn và mã hóa",
    href: "/people",
    color: "#2196f3",
  },
];

const stats = [
  { icon: faBolt, label: "Tốc Độ", value: "< 1s", desc: "Nhận diện nhanh" },
  { icon: faChartLine, label: "Chính Xác", value: "99%", desc: "Độ chính xác" },
  { icon: faUsers, label: "Người Dùng", value: "∞", desc: "Không giới hạn" },
];

export default function HomePage() {
  return (
    <div className="animate-fade-in">
      {/* Hero Section */}

      {/* Features Grid */}
      <section>
        <h2 className="text-2xl font-bold mb-6 text-[#ededed]">
          Tính Năng Chính
        </h2>
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
              <p className="text-sm text-gray-400">{feature.description}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* Quick Actions */}
      <section className="mt-12 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-6 text-[#ededed]">
          Bắt Đầu Nhanh
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/people"
            className="flex items-center gap-4 bg-[#252525] hover:bg-[#2a2a2a] rounded-lg p-4 transition-all"
          >
            <div className="w-12 h-12 bg-[#03dac6]/20 rounded-lg flex items-center justify-center">
              <span className="text-2xl font-bold text-[#03dac6]">1</span>
            </div>
            <div>
              <h3 className="font-semibold text-[#ededed]">
                Đăng Ký & Quản Lý
              </h3>
              <p className="text-sm text-gray-400">
                Thêm người mới và quản lý dữ liệu
              </p>
            </div>
          </Link>

          <Link
            href="/people"
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
              <p className="text-sm text-gray-400">
                Bắt đầu nhận diện khuôn mặt
              </p>
            </div>
          </Link>
        </div>
      </section>
    </div>
  );
}
