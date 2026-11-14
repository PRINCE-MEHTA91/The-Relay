import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ChatView from './components/ChatView';
import AutonomousResearchView from './components/AutonomousResearchView';
import PapersLibrary from './components/PapersLibrary';
import ApiStatusView from './components/ApiStatusView';
import SettingsView from './components/SettingsView';
import { askQuestion, streamQuestion, fetchHealth, fetchResearchAnswer } from './services/api';

const DEFAULT_SETTINGS = {
  top_k: 5,
  stream: true,
  use_mock: false,
  mock_mode: 'NORMAL',
  research_mode: 'autonomous', // 'autonomous' | 'corpus'
};

const STORAGE_KEY_SETTINGS = 'the_relay_settings_v2';
const STORAGE_KEY_HISTORY = 'the_relay_history_v2';

export default function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [apiHealthy, setApiHealthy] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [groqInfo, setGroqInfo] = useState({ configured: false, model: 'openai/gpt-oss-20b' });
  const activeControllerRef = useRef(null);

  // Settings state persisted in localStorage
  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_SETTINGS);
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  // Recent queries history persisted in localStorage
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_HISTORY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Save settings on update
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(settings));
    } catch {
      // ignore
    }
  }, [settings]);

  // Save history on update
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(history));
    } catch {
      // ignore
    }
  }, [history]);

  // Periodic health check
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetchHealth();
        setApiHealthy(res && res.status === 'healthy');
        if (res) {
          setGroqInfo({
            configured: !!res.groq_configured,
            model: res.groq_model || 'openai/gpt-oss-20b',
          });
        }
      } catch {
        setApiHealthy(false);
      }
    };
    checkStatus();
    const timer = setInterval(checkStatus, 15000);
    return () => clearInterval(timer);
  }, []);

  // Save query to history
  const addQueryToHistory = (q) => {
    setHistory((prev) => {
      const filtered = prev.filter((item) => item.question.toLowerCase() !== q.toLowerCase());
      return [
        {
          id: Date.now().toString(),
          question: q,
          timestamp: new Date().toLocaleTimeString(),
        },
        ...filtered,
      ].slice(0, 20); // keep last 20
    });
  };

  // Main submission handler
  const handleSendMessage = async (questionText) => {
    if (!questionText.trim() || isStreaming) return;

    addQueryToHistory(questionText);
    setActiveView('chat');

    const userMessage = {
      role: 'user',
      content: questionText,
      timestamp: new Date().toLocaleTimeString(),
    };

    const initialAssistantMessage = {
      role: 'assistant',
      content: '',
      isPartial: true,
      citations: [],
      status: 'pending',
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    const assistantIndex = messages.length + 1;
    const startTime = performance.now();

    // 1. Autonomous Academic Research Mode (Searches arXiv, S2, Crossref live)
    if (settings.research_mode === 'autonomous') {
      setIsStreaming(true);
      try {
        const res = await fetchResearchAnswer({
          question: questionText,
          max_papers: 3,
          sources: ['arxiv', 'semantic_scholar', 'crossref'],
          top_k_evidence: settings.top_k || 6,
          use_mock: settings.use_mock,
        });

        const elapsed = Math.round(performance.now() - startTime);
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[assistantIndex]) {
            updated[assistantIndex] = {
              role: 'assistant',
              content: res.answer,
              citations: res.citations || [],
              papers_used: res.papers_used || [],
              evidence_snippets: res.evidence_snippets || [],
              limitations: res.limitations || [],
              status: res.status === 'success' ? 'verified' : (res.status === 'INSUFFICIENT_EVIDENCE' ? 'REFUSAL_VERIFIED' : res.status),
              validation_status: res.status === 'success' ? 'VERIFIED' : 'UNVERIFIED',
              input_tokens: res.input_tokens || 0,
              output_tokens: res.output_tokens || 0,
              latency_ms: res.execution_time_ms || elapsed,
              provider: res.provider || (settings.use_mock ? 'mock' : 'groq'),
              model: res.model,
              search_queries: res.search_queries || [],
              sources_queried: res.sources_queried || [],
              total_papers_found: res.total_papers_found,
              isPartial: false,
              isAutonomous: true,
            };
          }
          return updated;
        });
      } catch (err) {
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[assistantIndex]) {
            updated[assistantIndex] = {
              role: 'assistant',
              content: '',
              isPartial: false,
              isError: true,
              errorMessage: err.message || 'Error communicating with autonomous research backend',
            };
          }
          return updated;
        });
      } finally {
        setIsStreaming(false);
      }
    }
    // 2. Fixed Local Corpus Q&A (3 Papers) - Streaming Mode (SSE)
    else if (settings.stream) {
      setIsStreaming(true);

      const controller = streamQuestion(
        {
          question: questionText,
          top_k: settings.top_k,
          use_mock: settings.use_mock,
          mock_mode: settings.mock_mode,
        },
        {
          onInit: (initData) => {
            setMessages((prev) => {
              const updated = [...prev];
              if (updated[assistantIndex]) {
                updated[assistantIndex].retrieved_chunks = initData.retrieved_chunks;
              }
              return updated;
            });
          },
          onToken: (token, accumulated) => {
            setMessages((prev) => {
              const updated = [...prev];
              if (updated[assistantIndex]) {
                updated[assistantIndex].content = accumulated;
              }
              return updated;
            });
          },
          onDone: (finalData) => {
            const elapsed = Math.round(performance.now() - startTime);
            setIsStreaming(false);
            activeControllerRef.current = null;
            setMessages((prev) => {
              const updated = [...prev];
              if (updated[assistantIndex]) {
                updated[assistantIndex] = {
                  ...updated[assistantIndex],
                  content: finalData.answer,
                  citations: finalData.citations || [],
                  status: finalData.status || 'verified',
                  validation_status: finalData.validation_status || 'VERIFIED',
                  input_tokens: finalData.input_tokens || 0,
                  output_tokens: finalData.output_tokens || 0,
                  latency_ms: finalData.latency_ms || elapsed,
                  provider: finalData.provider,
                  model: finalData.model,
                  isPartial: false,
                };
              }
              return updated;
            });
          },
          onError: (err) => {
            setIsStreaming(false);
            activeControllerRef.current = null;
            setMessages((prev) => {
              const updated = [...prev];
              if (updated[assistantIndex]) {
                updated[assistantIndex] = {
                  ...updated[assistantIndex],
                  isPartial: false,
                  isError: true,
                  errorMessage: err.message || 'Stream connection error',
                };
              }
              return updated;
            });
          },
        }
      );
      activeControllerRef.current = controller;
    }
    // 2. Batch Mode (Non-streaming JSON)
    else {
      setIsStreaming(true);
      try {
        const res = await askQuestion({
          question: questionText,
          top_k: settings.top_k,
          use_mock: settings.use_mock,
          mock_mode: settings.mock_mode,
        });

        setMessages((prev) => {
          const updated = [...prev];
          if (updated[assistantIndex]) {
            updated[assistantIndex] = {
              role: 'assistant',
              content: res.answer,
              citations: res.citations || [],
              status: res.status || (res.is_valid ? 'verified' : 'unverified'),
              validation_status: res.validation?.validation_status || (res.is_valid ? 'VERIFIED' : 'UNVERIFIED'),
              input_tokens: res.input_tokens || 0,
              output_tokens: res.output_tokens || 0,
              latency_ms: res.latency_ms || Math.round(performance.now() - startTime),
              provider: res.provider || (settings.use_mock ? 'mock' : 'groq'),
              model: res.model,
              isPartial: false,
            };
          }
          return updated;
        });
      } catch (err) {
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[assistantIndex]) {
            updated[assistantIndex] = {
              role: 'assistant',
              content: '',
              isPartial: false,
              isError: true,
              errorMessage: err.message || 'Error communicating with backend',
            };
          }
          return updated;
        });
      } finally {
        setIsStreaming(false);
      }
    }
  };

  const handleStopStream = () => {
    if (activeControllerRef.current) {
      activeControllerRef.current.abort();
      activeControllerRef.current = null;
    }
    setIsStreaming(false);
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last && last.role === 'assistant') {
        last.isPartial = false;
      }
      return updated;
    });
  };

  const handleNewChat = () => {
    handleStopStream();
    setMessages([]);
    setActiveView('dashboard');
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY_HISTORY);
    } catch {
      // ignore
    }
  };

  const handleResetDefaults = () => {
    setSettings(DEFAULT_SETTINGS);
  };

  const handleRetryLast = () => {
    if (messages.length >= 2) {
      const lastUser = [...messages].reverse().find((m) => m.role === 'user');
      if (lastUser) {
        // remove failed assistant response
        setMessages((prev) => prev.slice(0, prev.length - 1));
        handleSendMessage(lastUser.content);
      }
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans text-slate-100">
      {/* Sidebar */}
      <Sidebar
        activeView={activeView}
        setActiveView={setActiveView}
        history={history}
        onSelectHistory={(item) => handleSendMessage(item.question)}
        onClearHistory={handleClearHistory}
        onNewChat={handleNewChat}
        apiHealthy={apiHealthy}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header
          activeView={activeView}
          onOpenMobile={() => setMobileOpen(true)}
          apiHealthy={apiHealthy}
          settings={settings}
          onNewChat={handleNewChat}
        />

        {/* Prominent Provider Warning / Notification Banners */}
        {settings.use_mock ? (
          <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-xs text-amber-300 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold uppercase tracking-wider text-[10px] bg-amber-500/20 px-2 py-0.5 rounded border border-amber-500/30">
                Mock Stand-in Active
              </span>
              <span>
                Deterministic simulation mode ({settings.mock_mode || 'NORMAL'}) active for offline experiments. Uncheck Mock mode in Settings to use live Groq Cloud API.
              </span>
            </div>
            <button
              onClick={() => setSettings((prev) => ({ ...prev, use_mock: false }))}
              className="underline hover:text-white font-medium text-[11px] shrink-0"
            >
              Switch to Live Groq (GPT-OSS 20B) →
            </button>
          </div>
        ) : !groqInfo.configured && apiHealthy ? (
          <div className="bg-rose-500/10 border-b border-rose-500/20 px-4 py-2 text-xs text-rose-300 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold uppercase tracking-wider text-[10px] bg-rose-500/20 px-2 py-0.5 rounded border border-rose-500/30">
                GROQ_API_KEY Not Configured
              </span>
              <span>
                Add your Groq API key to .env, or switch to Mock Stand-in mode in Settings.
              </span>
            </div>
            <button
              onClick={() => setSettings((prev) => ({ ...prev, use_mock: true }))}
              className="underline hover:text-white font-medium text-[11px] shrink-0"
            >
              Enable Mock Stand-in Mode →
            </button>
          </div>
        ) : null}

        <main className="flex flex-1 overflow-hidden">
          {activeView === 'dashboard' && (
            <Dashboard
              onSearch={handleSendMessage}
              settings={settings}
              setSettings={setSettings}
              onOpenSettings={() => setActiveView('settings')}
              onOpenPapers={() => setActiveView('papers')}
              onOpenAutonomous={() => setActiveView('autonomous')}
            />
          )}

          {activeView === 'autonomous' && (
            <AutonomousResearchView settings={settings} />
          )}

          {activeView === 'chat' && (
            <ChatView
              messages={messages}
              isStreaming={isStreaming}
              onSendMessage={handleSendMessage}
              onStopStream={handleStopStream}
              onRetryLast={handleRetryLast}
              settings={settings}
              setSettings={setSettings}
            />
          )}

          {activeView === 'papers' && (
            <PapersLibrary
              onQueryPaper={(query) => {
                handleSendMessage(query);
              }}
            />
          )}

          {activeView === 'status' && <ApiStatusView />}

          {activeView === 'settings' && (
            <SettingsView
              settings={settings}
              setSettings={setSettings}
              onClearHistory={handleClearHistory}
              onResetDefaults={handleResetDefaults}
            />
          )}
        </main>
      </div>
    </div>
  );
}
