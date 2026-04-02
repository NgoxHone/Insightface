'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
  { href: '/', label: '🔍 Nhận Diện', icon: '🔍' },
  { href: '/realtime', label: '📹 Realtime', icon: '📹' },
  { href: '/register', label: '➕ Đăng Ký', icon: '➕' },
  { href: '/people', label: '👥 Danh Sách', icon: '👥' },
];

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="bg-[#1a1a1a] border-b border-[#2a2a2a] sticky top-0 z-50">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="text-xl font-bold text-[#03dac6]">
            🔐 Face Recognition
          </Link>
          
          <nav className="flex gap-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-[#03dac6] text-[#0a0a0a]'
                      : 'text-[#ededed] hover:bg-[#2a2a2a]'
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}
