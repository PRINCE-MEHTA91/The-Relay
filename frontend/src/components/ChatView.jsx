import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Send,
  Sparkles,
  User,
  ShieldCheck,
  ShieldAlert,
  Copy,
  Check,
  Clock,
  Cpu,
  Layers,
  ChevronDown,
  ChevronUp,
  Square,
  AlertTriangle,
  FileText,
  RotateCw,
  Globe,
  BookOpen,
  ExternalLink,
  Info,
  AlertCircle,
} from 'lucide-react';

export default function ChatView({
  messages,
  isStreaming,
  onSendMessage,
  onStopStream,
  onRetryLast,
  settings,
  setSettings,
}) {
  const [input, setInput] = useState('');
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [expandedCitations, setExpandedCitations] = useState({});
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isStreaming) {
      onSendMessage(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputResize = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const toggleCitation = (msgIdx, citIdx) => {
    const key = `${msgIdx}-${citIdx}`;
    setExpandedCitations((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-slate-950">
      {/* Scrollable Message List */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-12 lg:px-24 space-y-6">
        {messages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={idx}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-4xl mx-auto w-full`}
            >
              {/* Message Header */}
              <div className="flex items-center gap-2 mb-1.5 px-1">
                {isUser ? (
                  <>
                    <span className="text-xs font-medium text-slate-400">You</span>
                    <div className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-slate-300">
                      <User className="h-3 w-3" />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex h-5 w-5 items-center justify-center rounded-full bg-sky-500/20 text-sky-400 border border-sky-500/30">
                      <Sparkles className="h-3 w-3" />
                    </div>
                    <span className="text-xs font-medium text-sky-400">The Relay Research</span>
                    {msg.isPartial && (
                      <span className="flex items-center gap-1 text-[10px] text-amber-400 animate-pulse">
                        <span className="h-1.5 w-1.5 rounded-full bg-amber-400"></span>
                        Synthesizing...
                      </span>
                    )}
                  </>
                )}
              </div>

              {/* Message Card */}
              <div
                className={`w-full rounded-2xl px-5 py-4 shadow-sm transition-all ${
                  isUser
                    ? 'bg-slate-900 border border-slate-800 text-slate-100 max-w-2xl ml-auto'
                    : 'bg-slate-900/90 border border-slate-850 text-slate-200'
                }`}
              >
                {/* User Text */}
                {isUser ? (
                  <p className="whitespace-pre-wrap text-sm md:text-[15px] leading-relaxed">{msg.content}</p>
                ) : (
                  /* Assistant Markdown Text */
                  <div className="markdown-content text-sm md:text-[15px] leading-relaxed">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        table: ({ node, ...props }) => (
                          <div className="overflow-x-auto my-3 rounded-lg border border-slate-800 bg-slate-950/40">
                            <table className="w-full text-left border-collapse text-xs md:text-sm" {...props} />
                          </div>
                        ),
                        th: ({ node, ...props }) => (
                          <th className="border-b border-slate-800 bg-slate-900/90 px-3 py-2 font-semibold text-slate-200" {...props} />
                        ),
                        td: ({ node, ...props }) => (
                          <td className="border-b border-slate-800/50 px-3 py-2 text-slate-300" {...props} />
                        ),
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {msg.isPartial && (
                      <span className="inline-block w-2 h-4 bg-sky-400 animate-pulse ml-1 translate-y-0.5" />
                    )}
                  </div>
                )}

                {/* Assistant Metadata & Performance Footprint */}
                {!isUser && !msg.isPartial && (
                  <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
                    <div className="flex flex-wrap items-center gap-2">
                      {/* Verification Badge */}
                      {msg.status === 'verified' && (
                        <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-400 border border-emerald-500/20">
                          <ShieldCheck className="h-3 w-3" />
                          <span>Grounded & Verified</span>
                        </span>
                      )}
                      {msg.status === 'REFUSAL_VERIFIED' && (
                        <span className="inline-flex items-center gap-1 rounded-md bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium text-cyan-400 border border-cyan-500/20">
                          <ShieldCheck className="h-3 w-3" />
                          <span>Honest Refusal (Out of Domain)</span>
                        </span>
                      )}
                      {msg.status === 'unverified' && (
                        <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-400 border border-amber-500/20">
                          <ShieldAlert className="h-3 w-3" />
                          <span>Citation Warning</span>
                        </span>
                      )}

                      {/* Latency */}
                      {msg.latency_ms && (
                        <span className="inline-flex items-center gap-1 text-[11px] text-slate-400">
                          <Clock className="h-3 w-3 text-slate-500" />
                          <span>{msg.latency_ms} ms</span>
                        </span>
                      )}

                      {/* Tokens */}
                      {msg.input_tokens !== undefined && (
                        <span className="inline-flex items-center gap-1 text-[11px] text-slate-400">
                          <Cpu className="h-3 w-3 text-slate-500" />
                          <span>{msg.input_tokens} in / {msg.output_tokens || 0} out</span>
                        </span>
                      )}

                      {/* Provider Badge */}
                      {msg.provider && (
                        <span className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[10px] font-mono border ${
                          msg.provider === 'mock'
                            ? 'bg-amber-500/10 text-amber-300 border-amber-500/20'
                            : 'bg-sky-500/10 text-sky-300 border-sky-500/20'
                        }`}>
                          <span>{msg.provider === 'mock' ? 'Mock LLM' : 'Groq Cloud'}</span>
                        </span>
                      )}
                    </div>

                    {/* Copy Button */}
                    <button
                      onClick={() => copyToClipboard(msg.content, idx)}
                      className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
                      title="Copy response markdown"
                    >
                      {copiedIndex === idx ? (
                        <>
                          <Check className="h-3 w-3 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3 w-3" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                )}

                {/* Papers Consulted in Autonomous Research */}
                {!isUser && msg.papers_used && msg.papers_used.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-800/80">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <BookOpen className="h-3.5 w-3.5 text-sky-400" />
                        <span>Academic Literature Ingested ({msg.papers_used.length})</span>
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {msg.papers_used.map((paper, pIdx) => {
                        const isFull = paper.content_type === 'full_text';
                        return (
                          <div
                            key={pIdx}
                            className="rounded-xl border border-slate-800 bg-slate-950/70 p-3 flex flex-col justify-between space-y-2 hover:border-slate-700 transition-colors"
                          >
                            <div>
                              <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                                {isFull ? (
                                  <span className="rounded-sm bg-emerald-500/10 border border-emerald-500/30 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-400">
                                    Full Text ({paper.page_count}p)
                                  </span>
                                ) : (
                                  <span className="rounded-sm bg-amber-500/10 border border-amber-500/30 px-1.5 py-0.5 text-[10px] font-semibold text-amber-400">
                                    Abstract Only
                                  </span>
                                )}
                                {paper.year && (
                                  <span className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400 font-mono">
                                    {paper.year}
                                  </span>
                                )}
                                {paper.sources && paper.sources.map((s) => (
                                  <span key={s} className="rounded-sm bg-sky-500/10 text-sky-400 px-1.5 py-0.5 text-[10px] font-mono capitalize">
                                    {s}
                                  </span>
                                ))}
                              </div>
                              <p className="text-xs font-semibold text-white line-clamp-2 leading-snug">
                                {paper.title}
                              </p>
                              {paper.authors && paper.authors.length > 0 && (
                                <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">
                                  {paper.authors.join(', ')}
                                </p>
                              )}
                            </div>

                            <div className="flex items-center justify-end gap-2 pt-1 border-t border-slate-800/60">
                              {paper.url && (
                                <a
                                  href={paper.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="inline-flex items-center gap-1 text-[11px] text-sky-400 hover:underline"
                                >
                                  <span>View Paper</span>
                                  <ExternalLink className="h-2.5 w-2.5" />
                                </a>
                              )}
                              {paper.pdf_url && (
                                <a
                                  href={paper.pdf_url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-200 hover:bg-slate-700"
                                >
                                  PDF ↗
                                </a>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Source Citations Section */}
                {!isUser && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-800/80">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <FileText className="h-3.5 w-3.5 text-sky-400" />
                        <span>Source Citations ({msg.citations.length})</span>
                      </span>
                    </div>

                    <div className="grid grid-cols-1 gap-2">
                      {msg.citations.map((cit, cIdx) => {
                        const isExpanded = expandedCitations[`${idx}-${cIdx}`];
                        const isAbstract = cit.evidence_type === 'abstract_only' || cit.page_number === 'Abstract' || cit.is_abstract_only;
                        const paperName = cit.paper_title || cit.paper_name;
                        const passageText = cit.evidence_excerpt || cit.source_passage;

                        return (
                          <div
                            key={cIdx}
                            className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 transition-colors hover:border-slate-700"
                          >
                            <div
                              onClick={() => toggleCitation(idx, cIdx)}
                              className="flex items-center justify-between cursor-pointer select-none"
                            >
                              <div className="flex items-center gap-2">
                                <span className="flex h-5 w-5 items-center justify-center rounded-md bg-sky-500/10 text-[10px] font-bold text-sky-400 border border-sky-500/20">
                                  {cIdx + 1}
                                </span>
                                <span className="text-xs font-medium text-slate-200">
                                  {paperName}
                                </span>
                                {isAbstract ? (
                                  <span className="rounded-sm bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 text-[10px] font-mono">
                                    Abstract
                                  </span>
                                ) : (
                                  <span className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                                    p. {cit.page_number}
                                  </span>
                                )}
                                {cit.chunk_id !== undefined && (
                                  <span className="rounded-sm bg-slate-800/60 px-1.5 py-0.5 text-[10px] text-slate-400 font-mono">
                                    chunk #{cit.chunk_id}
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                {cit.pdf_url && (
                                  <a
                                    href={cit.pdf_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    className="text-slate-400 hover:text-sky-400"
                                    title="Open PDF"
                                  >
                                    <ExternalLink className="h-3 w-3" />
                                  </a>
                                )}
                                <button className="text-slate-500 hover:text-slate-300">
                                  {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                                </button>
                              </div>
                            </div>

                            {/* Expanded Passage Excerpt */}
                            {isExpanded && passageText && (
                              <div className="mt-2.5 rounded-lg bg-slate-900/90 p-2.5 border border-slate-800 text-xs text-slate-300 font-mono leading-relaxed">
                                <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-1 font-sans">
                                  Retrieved Passage Excerpt:
                                </p>
                                <p className="italic">{passageText}</p>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Research Limitations Callout */}
                {!isUser && msg.limitations && msg.limitations.length > 0 && (
                  <div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200 space-y-1">
                    <div className="flex items-center gap-1.5 font-semibold text-amber-300">
                      <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
                      <span>Research Limitations & Evidence Caveats</span>
                    </div>
                    <ul className="list-disc pl-5 space-y-0.5 text-amber-200/90">
                      {msg.limitations.map((lim, lIdx) => (
                        <li key={lIdx}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Error Banner if Request Failed */}
                {msg.isError && (
                  <div className="mt-3 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
                    <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="font-semibold text-rose-200">Execution Error</p>
                      <p className="mt-0.5">{msg.errorMessage || 'Failed to complete research query.'}</p>
                    </div>
                    {onRetryLast && (
                      <button
                        onClick={onRetryLast}
                        className="flex items-center gap-1 rounded-md bg-rose-500/20 px-2 py-1 text-[11px] text-rose-200 hover:bg-rose-500/30 transition-colors"
                      >
                        <RotateCw className="h-3 w-3" />
                        <span>Retry</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Fixed Bottom Input Form */}
      <div className="border-t border-slate-800/90 bg-slate-950/90 p-4 backdrop-blur-md">
        <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
          <div className="relative flex flex-col rounded-2xl border border-slate-800 bg-slate-900/90 p-2 focus-within:border-sky-500/70 focus-within:ring-1 focus-within:ring-sky-500/20 shadow-lg">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputResize}
              onKeyDown={handleKeyDown}
              placeholder={isStreaming ? 'Generating response...' : 'Ask a follow-up research question...'}
              disabled={isStreaming}
              rows={1}
              className="w-full resize-none bg-transparent px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-hidden leading-relaxed disabled:opacity-50"
            />

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/60 pt-2 px-2 mt-1">
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                {/* Engine Mode Toggle */}
                {setSettings && (
                  <button
                    type="button"
                    onClick={() =>
                      setSettings((prev) => ({
                        ...prev,
                        research_mode: prev.research_mode === 'autonomous' ? 'corpus' : 'autonomous',
                      }))
                    }
                    className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-semibold border transition-all ${
                      settings.research_mode === 'autonomous'
                        ? 'bg-sky-500/15 text-sky-300 border-sky-500/30 hover:bg-sky-500/25'
                        : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-white'
                    }`}
                    title="Toggle between Autonomous Live Academic Search and Local 3-Paper RAG"
                  >
                    {settings.research_mode === 'autonomous' ? (
                      <>
                        <Globe className="h-3 w-3 text-sky-400" />
                        <span>Autonomous Agent (arXiv+S2+Crossref)</span>
                      </>
                    ) : (
                      <>
                        <FileText className="h-3 w-3 text-slate-400" />
                        <span>Corpus Q&A (3 Papers)</span>
                      </>
                    )}
                  </button>
                )}

                <span>Top-K: {settings.top_k}</span>
                <span>•</span>
                <span>{settings.use_mock ? `Mock (${settings.mock_mode || 'NORMAL'})` : 'Live Groq'}</span>
              </div>

              <div className="flex items-center gap-2">
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={onStopStream}
                    className="flex items-center gap-1.5 rounded-xl bg-rose-600/80 hover:bg-rose-600 px-3 py-1.5 text-xs font-medium text-white shadow-xs transition-colors"
                  >
                    <Square className="h-3 w-3 fill-white" />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className="flex items-center gap-1.5 rounded-xl bg-sky-500 px-3.5 py-1.5 text-xs font-medium text-white shadow-sm shadow-sky-500/30 hover:bg-sky-400 active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <span>Send</span>
                    <Send className="h-3 w-3" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

