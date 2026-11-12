import React, { useState, useEffect } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Cpu,
  Layers,
  Database,
  ShieldCheck,
  ExternalLink,
  Clock,
} from 'lucide-react';
import { fetchHealth, fetchMetrics } from '../services/api';

export default function ApiStatusView() {
  const [healthData, setHealthData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, m] = await Promise.all([
        fetchHealth().catch((e) => ({ error: e.message })),
        fetchMetrics().catch((e) => ({ error: e.message })),
      ]);
      setHealthData(h);
      setMetricsData(m);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 15000); // Auto-refresh every 15s
    return () => clearInterval(interval);
  }, []);

  const isHealthy = healthData && !healthData.error && healthData.status === 'healthy';

  return (
    <div className="flex-1 overflow-y-auto px-4 py-8 md:px-12 lg:px-20">
      <div className="mx-auto max-w-5xl">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Activity className="h-5 w-5 text-sky-400" />
              <h1 className="text-2xl font-bold text-white tracking-tight">API Status & Subsystem Health</h1>
            </div>
            <p className="text-xs md:text-sm text-slate-400">
              Live telemetry connected to FastAPI backend at <code className="text-sky-300 font-mono">127.0.0.1:8000</code>.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {lastUpdated && (
              <span className="text-xs text-slate-500">Updated: {lastUpdated}</span>
            )}
            <button
              onClick={loadStatus}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800/80 px-3.5 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700 hover:text-white transition-all disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Global Health Banner */}
        <div
          className={`rounded-2xl border p-5 mb-8 flex items-center justify-between backdrop-blur-md ${
            isHealthy
              ? 'border-emerald-500/30 bg-emerald-950/20 text-emerald-200'
              : 'border-rose-500/30 bg-rose-950/20 text-rose-200'
          }`}
        >
          <div className="flex items-center gap-3.5">
            {isHealthy ? (
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/20 text-emerald-400">
                <CheckCircle2 className="h-6 w-6" />
              </div>
            ) : (
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/20 text-rose-400">
                <AlertCircle className="h-6 w-6" />
              </div>
            )}
            <div>
              <h2 className="text-base font-bold text-white">
                {isHealthy ? 'All Systems Operational' : 'Backend Subsystems Degraded / Offline'}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                {isHealthy
                  ? 'FastAPI server, FAISS vector index, and rate limiting middlewares are healthy.'
                  : 'Ensure `python -m uvicorn app.main:app --port 8000` is running locally.'}
              </p>
            </div>
          </div>

          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-sky-400 hover:underline"
          >
            <span>OpenAPI Docs</span>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>

        {/* Telemetry Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
          {/* FAISS Vector DB */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                FAISS Vector Index
              </span>
              <div className="rounded-lg bg-sky-500/10 p-2 text-sky-400 border border-sky-500/20">
                <Database className="h-4 w-4" />
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-2xl font-bold text-white font-mono">
                  {healthData?.indexed_vectors !== undefined ? healthData.indexed_vectors : '—'}
                </p>
                <p className="text-xs text-slate-400">Total Indexed Vectors</p>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-xs text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Metadata Records:</span>
                  <span className="font-mono text-slate-200">
                    {healthData?.metadata_records || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Model:</span>
                  <span className="font-mono text-sky-400 truncate max-w-[140px]">
                    all-MiniLM-L6-v2
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Rate Limiter */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Sliding-Window Limiter
              </span>
              <div className="rounded-lg bg-emerald-500/10 p-2 text-emerald-400 border border-emerald-500/20">
                <ShieldCheck className="h-4 w-4" />
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-2xl font-bold text-white font-mono">
                  {metricsData?.rate_limiter?.max_requests || 30} RPM
                </p>
                <p className="text-xs text-slate-400">Per-IP Request Ceiling</p>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-xs text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Window Seconds:</span>
                  <span className="font-mono text-slate-200">
                    {metricsData?.rate_limiter?.window_seconds || 60}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Rejection Status:</span>
                  <span className="font-mono text-emerald-400">HTTP 429 Shedding</span>
                </div>
              </div>
            </div>
          </div>

          {/* Concurrency Queue */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Concurrency Manager
              </span>
              <div className="rounded-lg bg-purple-500/10 p-2 text-purple-400 border border-purple-500/20">
                <Cpu className="h-4 w-4" />
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-2xl font-bold text-white font-mono">
                  {metricsData?.concurrency?.max_concurrency || 10} Slots
                </p>
                <p className="text-xs text-slate-400">Max Semaphore Capacity</p>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-xs text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Active Running:</span>
                  <span className="font-mono text-slate-200">
                    {metricsData?.concurrency?.active_running || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Queued Waiters:</span>
                  <span className="font-mono text-slate-200">
                    {metricsData?.concurrency?.queued_waiters || 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* API Endpoint Directory */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 shadow-xl">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-2">
            <Layers className="h-4 w-4 text-sky-400" />
            Backend REST API Endpoints
          </h2>

          <div className="space-y-2 text-xs font-mono">
            {[
              { method: 'GET', path: '/', desc: 'API Root & welcome status' },
              { method: 'GET', path: '/health', desc: 'Subsystem health & FAISS vector count' },
              { method: 'GET', path: '/metrics', desc: 'Concurrency queue & rate limiter stats' },
              { method: 'POST', path: '/ask', desc: 'Batch research question answering (JSON)' },
              { method: 'POST', path: '/ask/stream', desc: 'Progressive token streaming via SSE' },
              { method: 'GET', path: '/docs', desc: 'OpenAPI interactive Swagger documentation' },
            ].map((ep, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between rounded-xl bg-slate-950/70 p-3 border border-slate-800/80"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`rounded-md px-2 py-0.5 text-[11px] font-bold ${
                      ep.method === 'GET'
                        ? 'bg-sky-500/10 text-sky-400 border border-sky-500/20'
                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    }`}
                  >
                    {ep.method}
                  </span>
                  <span className="font-semibold text-slate-200">{ep.path}</span>
                </div>
                <span className="text-slate-400 font-sans text-xs">{ep.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

