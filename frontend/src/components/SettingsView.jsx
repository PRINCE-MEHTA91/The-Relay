import React from 'react';
import {
  Settings,
  Sliders,
  Cpu,
  Zap,
  ShieldAlert,
  Trash2,
  RotateCcw,
  Sparkles,
  Info,
} from 'lucide-react';

export default function SettingsView({
  settings,
  setSettings,
  onClearHistory,
  onResetDefaults,
}) {
  const mockModes = [
    { id: 'NORMAL', label: 'NORMAL', desc: '100ms baseline latency, deterministic accurate answers' },
    { id: 'SLOW', label: 'SLOW', desc: '1.5s deliberate network delay to test client patience' },
    { id: 'OVERLOAD', label: 'OVERLOAD', desc: 'Latency increases proportionally with active concurrency' },
    { id: 'FAILURE', label: 'FAILURE', desc: 'Simulates upstream HTTP 500 internal server error' },
    { id: 'RATE_LIMITED', label: 'RATE_LIMITED', desc: 'Simulates HTTP 429 daily budget quota exhaustion' },
    { id: 'TIMEOUT', label: 'TIMEOUT', desc: 'Simulates request timeout exceeding client deadline' },
  ];

  return (
    <div className="flex-1 overflow-y-auto px-4 py-8 md:px-12 lg:px-20">
      <div className="mx-auto max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-1">
            <Settings className="h-5 w-5 text-sky-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Configuration & Preferences</h1>
          </div>
          <p className="text-xs md:text-sm text-slate-400">
            Customize retrieval depth, model provider, mock fault injection, and streaming behavior.
          </p>
        </div>

        <div className="space-y-6">
          {/* Retrieval Parameters Card */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl backdrop-blur-md">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-2">
              <Sliders className="h-4 w-4 text-sky-400" />
              Retrieval Parameters
            </h2>

            {/* Top-K Slider */}
            <div className="space-y-2 mb-6">
              <div className="flex justify-between items-center">
                <label className="text-xs font-medium text-slate-200">
                  Top-K Passages to Retrieve
                </label>
                <span className="rounded-md bg-sky-500/10 px-2 py-0.5 text-xs font-mono font-bold text-sky-400 border border-sky-500/20">
                  {settings.top_k} chunks
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="15"
                value={settings.top_k}
                onChange={(e) =>
                  setSettings((prev) => ({ ...prev, top_k: parseInt(e.target.value, 10) }))
                }
                className="w-full accent-sky-500 bg-slate-800 rounded-lg cursor-pointer"
              />
              <p className="text-[11px] text-slate-400">
                Number of most relevant passage vectors fetched from FAISS index to form the prompt context.
              </p>
            </div>

            {/* SSE Streaming Toggle */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-800/80">
              <div>
                <p className="text-xs font-medium text-slate-200">Server-Sent Events (SSE) Streaming</p>
                <p className="text-[11px] text-slate-400">
                  Stream generated response tokens progressively for fast TTFT (~50–60ms).
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.stream}
                  onChange={(e) =>
                    setSettings((prev) => ({ ...prev, stream: e.target.checked }))
                  }
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-800 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
              </label>
            </div>
          </div>

          {/* Model Provider & Mock Settings Card */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl backdrop-blur-md">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-2">
              <Cpu className="h-4 w-4 text-sky-400" />
              LLM Provider & Fault Injection
            </h2>

            {/* Provider Toggle */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
              <button
                type="button"
                onClick={() => setSettings((prev) => ({ ...prev, use_mock: false }))}
                className={`flex flex-col text-left rounded-xl p-4 border transition-all ${
                  !settings.use_mock
                    ? 'border-emerald-500 bg-emerald-950/20 text-white shadow-md'
                    : 'border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-white">Live Groq Cloud API</span>
                  <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-medium text-emerald-400">
                    Live Production LLM
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Real inference via Groq Cloud SDK using the configured <code className="text-emerald-300">GROQ_MODEL</code> and API key.
                </p>
              </button>

              <button
                type="button"
                onClick={() => setSettings((prev) => ({ ...prev, use_mock: true }))}
                className={`flex flex-col text-left rounded-xl p-4 border transition-all ${
                  settings.use_mock
                    ? 'border-amber-500 bg-amber-950/20 text-white shadow-md'
                    : 'border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-white">Mock Stand-in Service</span>
                  <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-medium text-amber-400">
                    Experiments & Benchmarks
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Deterministic offline stand-in (for load tests, synthetic fault injection, and zero quota consumption).
                </p>
              </button>
            </div>

            {/* Mock Mode Selector (if Mock is active) */}
            {settings.use_mock ? (
              <div className="space-y-3 pt-4 border-t border-slate-800/80">
                <label className="text-xs font-medium text-slate-200 flex items-center justify-between">
                  <span>Stand-in Operational Mode</span>
                  <span className="text-[11px] text-sky-400 font-mono">
                    Mode: {settings.mock_mode || 'NORMAL'}
                  </span>
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                  {mockModes.map((m) => {
                    const isSelected = (settings.mock_mode || 'NORMAL') === m.id;
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setSettings((prev) => ({ ...prev, mock_mode: m.id }))}
                        className={`rounded-xl border p-3 text-left transition-all ${
                          isSelected
                            ? 'border-sky-500/80 bg-slate-800 text-white shadow-xs'
                            : 'border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700 hover:text-slate-300'
                        }`}
                      >
                        <p className={`text-xs font-semibold ${isSelected ? 'text-sky-400' : 'text-slate-200'}`}>
                          {m.label}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-1 leading-snug">{m.desc}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 p-3.5 flex items-start gap-3 text-xs text-amber-200">
                <Info className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-amber-300">Live Quota Warning</p>
                  <p className="mt-0.5 text-amber-200/90 leading-relaxed">
                    Live requests use tokens on your Groq developer plan (30 RPM, 1,000 RPD).
                    Use sparingly to avoid exhausting daily rate limits.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Reset & Maintenance Card */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Reset & Clear Data</h2>
              <p className="text-xs text-slate-400">
                Restore default settings or purge local browser search query history.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={onResetDefaults}
                className="flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700 hover:text-white transition-colors"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>Reset Defaults</span>
              </button>

              <button
                type="button"
                onClick={onClearHistory}
                className="flex items-center gap-1.5 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-2 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition-colors"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Clear History</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

