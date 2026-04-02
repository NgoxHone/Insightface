"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  faHome,
  faCamera,
  faVideo,
  faUserPlus,
  faUsers,
  faBrain,
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

const navItems = [
  { href: "/", label: "Trang Chủ", icon: faHome },
  { href: "/recognize", label: "Nhận Diện", icon: faCamera },
  { href: "/realtime", label: "Realtime", icon: faVideo },
  { href: "/register", label: "Đăng Ký", icon: faUserPlus },
  { href: "/people", label: "Quản lý Train", icon: faBrain },
  // { href: '/training', label: 'Training', icon: faBrain },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-[#1a1a1a] border-r border-[#2a2a2a] flex flex-col z-50">
      {/* Logo */}
      <div className="p-6 border-b border-[#2a2a2a]">
        <Link href="/" className="flex items-center gap-3">
          <div>
            <h1 className="font-bold text-lg text-[#ededed]">FaceRec</h1>
            <p className="text-xs text-gray-500">Recognition System</p>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                isActive
                  ? "bg-[#03dac6]/10 text-[#03dac6] border-l-4 border-[#03dac6]"
                  : "text-gray-400 hover:bg-[#252525] hover:text-[#ededed]"
              }`}
            >
              <FontAwesomeIcon icon={item.icon} className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-[#2a2a2a]">
        <div className="bg-[#252525] rounded-lg p-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-[#03dac6] rounded-full animate-pulse" />
            <span className="text-xs text-gray-400">Server Status</span>
          </div>
          {/* <p className="text-xs text-gray-500">localhost:5001</p> */}
        </div>
      </div>
    </aside>
  );
}
