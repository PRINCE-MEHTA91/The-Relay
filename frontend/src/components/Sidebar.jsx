import React from 'react';
import {
  MessageSquare,
  BookOpen,
  Activity,
  Settings,
  PlusCircle,
  Clock,
  Trash2,
  ExternalLink,
  ChevronRight,
  ShieldCheck,
  X,
  Sparkles,
} from 'lucide-react';

export default function Sidebar({
  activeView,
  setActiveView,
  history,
  onSelectHistory,
  onClearHistory,
  onNewChat,
  apiHealthy,
  mobileOpen,
  setMobileOpen,
}) {
  const navItems = [
    { id: 'chat', label: 'Research Chat', icon: MessageSquare },
    { id: 'papers', label: 'Papers Library', icon: BookOpen, badge: '3' },
    { id: 'status', label: 'API Status & Metrics', icon: Activity },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex w-72 flex-col border-r border-slate-800/80 bg-slate-950/95 backdrop-blur-md transition-transform duration-300 md:static md:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-800/80 px-4">
          <button
            onClick={() => {
              setActiveView('dashboard');
              setMobileOpen(false);
            }}
            className="flex items-center gap-2.5 text-left group"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-600 via-sky-500 to-blue-600 shadow-md shadow-sky-500/20 group-hover:scale-105 transition-transform">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-semibold text-white tracking-tight">The Relay</span>
                <span className="rounded-sm bg-sky-500/10 px-1.5 py-0.5 text-[10px] font-medium text-sky-400 border border-sky-500/20">
                  RAG
                </span>
              </div>
              <p className="text-[11px] text-slate-400">AI Research Assistant</p>
            </div>
          </button>

          <button
            onClick={() => setMobileOpen(false)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800/60 hover:text-white md:hidden"
            aria-label="Close sidebar"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* New Research Button */}
        <div className="p-3">
          <button
            onClick={() => {
              onNewChat();
              setActiveView('dashboard');
              setMobileOpen(false);
            }}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 px-4 py-2.5 text-sm font-medium text-white shadow-lg shadow-sky-600/20 transition-all hover:brightness-110 active:scale-[0.98]"
          >
            <PlusCircle className="h-4 w-4" />
            <span>New Research</span>
          </button>
        </div>

        {/* Navigation Links */}
        <div className="px-3 py-1 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveView(item.id);
                  setMobileOpen(false);
                }}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-slate-800/90 text-sky-400 border border-sky-500/20 shadow-xs'
                    : 'text-slate-300 hover:bg-slate-900 hover:text-white'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className={`h-4 w-4 ${isActive ? 'text-sky-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300 border border-slate-700">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Recent Searches / History Section */}
        <div className="flex-1 overflow-y-auto px-3 py-3 border-t border-slate-800/60">
          <div className="flex items-center justify-between px-2 pb-2">
            <span className="text-xs font-semibold tracking-wider text-slate-400 uppercase">
              Recent Queries
            </span>
            {history.length > 0 && (
              <button
                onClick={onClearHistory}
                className="text-[11px] text-slate-400 hover:text-red-400 flex items-center gap-1 transition-colors"
                title="Clear history"
              >
                <Trash2 className="h-3 w-3" />
                <span>Clear</span>
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="py-6 text-center">
              <Clock className="mx-auto h-5 w-5 text-slate-600 mb-1.5" />
              <p className="text-xs text-slate-400">No recent queries yet</p>
            </div>
          ) : (
            <div className="space-y-1">
              {history.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    onSelectHistory(item);
                    setMobileOpen(false);
                  }}
                  className="group flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-xs text-slate-300 hover:bg-slate-900/90 hover:text-white transition-all"
                >
                  <span className="truncate pr-2 font-normal">{item.question}</span>
                  <ChevronRight className="h-3.5 w-3.5 text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Backend API Health Status Footer */}
        <div className="border-t border-slate-800/80 p-3">
          <div className="flex items-center justify-between rounded-xl bg-slate-900/80 p-2.5 border border-slate-800">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                {apiHealthy ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500"></span>
                )}
              </span>
              <div>
                <p className="text-xs font-medium text-slate-200">
                  {apiHealthy ? 'Backend Connected' : 'Backend Offline'}
                </p>
                <p className="text-[10px] text-slate-400">127.0.0.1:8000</p>
              </div>
            </div>
            <button
              onClick={() => {
                setActiveView('status');
                setMobileOpen(false);
              }}
              className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              title="View metrics & diagnostics"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

