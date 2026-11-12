import React from 'react';
import { Menu, Sparkles, PlusCircle } from 'lucide-react';

export default function Header({
  activeView,
  onOpenMobile,
  apiHealthy,
  settings,
  onNewChat,
}) {
  const titles = {
    dashboard: 'Research Overview',
    chat: 'Research Conversation',
    papers: 'Paper Library',
    status: 'System Health & Metrics',
    settings: 'Configuration',
  };

  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-slate-800/80 bg-slate-950/80 px-4 backdrop-blur-md md:px-8">
      {/* Mobile Drawer Button & Title */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobile}
          className="rounded-lg p-2 text-slate-400 hover:bg-slate-900 hover:text-white md:hidden"
          aria-label="Open navigation"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-white tracking-tight">
            {titles[activeView] || 'The Relay'}
          </span>
          <span className="hidden sm:inline-block text-slate-600">•</span>
          <span className="hidden sm:inline-block text-xs text-slate-400">
            {settings.use_mock
              ? `Stand-in (${settings.mock_mode || 'NORMAL'})`
              : 'Groq Cloud (GPT-OSS 20B)'}
          </span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2.5">
        {/* Backend Connectivity Status Dot */}
        <div
          className={`hidden sm:flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium border ${
            apiHealthy
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${apiHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`}
          />
          <span>{apiHealthy ? 'API Online' : 'API Offline'}</span>
        </div>

        {/* New Chat shortcut if in chat view */}
        {activeView === 'chat' && (
          <button
            onClick={onNewChat}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 hover:text-white transition-colors"
          >
            <PlusCircle className="h-3.5 w-3.5" />
            <span className="hidden xs:inline">New Query</span>
          </button>
        )}
      </div>
    </header>
  );
}

