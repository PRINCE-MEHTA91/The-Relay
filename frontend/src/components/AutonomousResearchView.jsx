import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Search,
  Sparkles,
  BookOpen,
  FileText,
  CheckCircle2,
  Clock,
  Cpu,
  ExternalLink,
  ShieldCheck,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Layers,
  Globe,
  DownloadCloud,
  FileSearch,
  RotateCcw,
  Sliders,
  Filter,
  Check,
  Copy,
  Info,
} from 'lucide-react';
import { fetchResearchAnswer, fetchResearchSources } from '../services/api';

const SAMPLE_RESEARCH_PROMPTS = [
  {
    title: 'Direct Preference Optimization',
    prompt: 'How does Direct Preference Optimization (DPO) eliminate the need for a separate reward model in RLHF?',
    tag: 'LLM Alignment',
  },
  {
    title: 'Kolmogorov-Arnold Networks',
    prompt: 'What are Kolmogorov-Arnold Networks (KANs) and how do their learnable spline activation functions compare to MLPs?',
    tag: 'Deep Learning',
  },
  {
    title: 'FlashAttention IO-Awareness',
    prompt: 'How does FlashAttention utilize tiling and GPU SRAM memory hierarchy to compute exact attention with reduced HBM memory access?',
    tag: 'Systems & Efficiency',
  },
  {
    title: 'Vision-Language Scaling Laws',
    prompt: 'What empirical scaling laws govern performance and data efficiency in contrastive multimodal vision-language pretraining?',
    tag: 'Multimodal AI',
  },
];

const RESEARCH_STAGES = [
  { id: 1, label: 'Formulating search queries & search plan', desc: 'Expanding natural language into targeted boolean academic queries' },
  { id: 2, label: 'Searching academic sources (arXiv, Crossref, S2)', desc: 'Parallel discovery and deduplication of literature' },
  { id: 3, label: 'Retrieving open-access full-text PDFs', desc: 'Fetching full-text PDFs via HTTPS, fallback to abstracts' },
  { id: 4, label: 'Extracting pages & building scoped FAISS index', desc: 'In-memory PyMuPDF extraction and isolated dense embedding index' },
  { id: 5, label: 'Synthesizing evidence-grounded answer', desc: 'Comparative multi-paper synthesis with page & abstract citations' },
];

export default function AutonomousResearchView({ settings }) {
  const [question, setQuestion] = useState('');
  const [maxPapers, setMaxPapers] = useState(3);
  const [topKEvidence, setTopKEvidence] = useState(6);
  const [selectedSources, setSelectedSources] = useState(['arxiv', 'semantic_scholar', 'crossref']);
  const [availableSources, setAvailableSources] = useState([]);
  
  // Execution states
  const [loading, setLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(1);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // UI accordion toggles
  const [showAllEvidence, setShowAllEvidence] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState({});
  const [copied, setCopied] = useState(false);

  const stageTimerRef = useRef(null);
  const clockTimerRef = useRef(null);

  // Fetch supported sources on mount
  useEffect(() => {
    fetchResearchSources()
      .then((data) => {
        if (data && data.sources) {
          setAvailableSources(data.sources);
        }
      })
      .catch((err) => console.warn('Could not load sources list:', err));
  }, []);

  // Handle stage simulation while awaiting response
  useEffect(() => {
    if (loading) {
      setCurrentStage(1);
      setElapsedMs(0);
      const start = performance.now();

      clockTimerRef.current = setInterval(() => {
        setElapsedMs(Math.round(performance.now() - start));
      }, 100);

      // Advance stages realistically to give user clear visibility
      const stageTimeouts = [
        setTimeout(() => setCurrentStage(2), 900),
        setTimeout(() => setCurrentStage(3), 2200),
        setTimeout(() => setCurrentStage(4), 4000),
        setTimeout(() => setCurrentStage(5), 5800),
      ];

      return () => {
        clearInterval(clockTimerRef.current);
        stageTimeouts.forEach(clearTimeout);
      };
    }
  }, [loading]);

  const toggleSource = (sourceId) => {
    if (selectedSources.includes(sourceId)) {
      if (selectedSources.length === 1) return; // Keep at least one source
      setSelectedSources(selectedSources.filter((s) => s !== sourceId));
    } else {
      setSelectedSources([...selectedSources, sourceId]);
    }
  };

  const handleExecuteResearch = async (qText = null) => {
    const targetQ = (qText || question).trim();
    if (!targetQ || loading) return;

    if (qText) setQuestion(qText);
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const data = await fetchResearchAnswer({
        question: targetQ,
        max_papers: maxPapers,
        sources: selectedSources,
        top_k_evidence: topKEvidence,
        use_mock: settings.use_mock,
      });

      setResult(data);
      setCurrentStage(5);
    } catch (err) {
      console.error('Autonomous research query failed:', err);
      setError(err.message || 'An error occurred during autonomous research synthesis.');
    } finally {
      setLoading(false);
      clearInterval(clockTimerRef.current);
    }
  };

  const handleCopyAnswer = () => {
    if (!result?.answer) return;
    navigator.clipboard.writeText(result.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleCitation = (idx) => {
    setExpandedCitations((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-950 px-4 py-8 md:px-10 lg:px-16 text-slate-100">
      <div className="mx-auto max-w-5xl space-y-8">
        {/* Header Hero */}
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-3.5 py-1 text-xs font-semibold text-sky-400 backdrop-blur-md">
            <Globe className="h-3.5 w-3.5" />
            <span>Autonomous Academic Search & Grounded Synthesis</span>
          </div>

          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white bg-gradient-to-r from-white via-slate-100 to-sky-300 bg-clip-text text-transparent">
            Autonomous Hybrid Research Agent
          </h1>
          <p className="text-sm md:text-base text-slate-400 max-w-3xl leading-relaxed">
            Enter any open-ended research question. The agent autonomously formulates queries, searches{' '}
            <span className="text-slate-200 font-medium">arXiv</span>,{' '}
            <span className="text-slate-200 font-medium">Semantic Scholar</span>, and{' '}
            <span className="text-slate-200 font-medium">Crossref</span>, retrieves open-access full-text PDFs, builds a query-scoped FAISS index, and produces an evidence-grounded comparative answer with exact page citations.
          </p>
        </div>

        {/* Input & Control Station */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl backdrop-blur-md space-y-4">
          <div className="space-y-2">
            <div className="relative">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleExecuteResearch();
                  }
                }}
                disabled={loading}
                rows={3}
                placeholder="Ask any research or scientific question (e.g. 'What are the architectural trade-offs between dense Transformers and Mixture-of-Experts?')"
                className="w-full resize-none rounded-xl border border-slate-700/80 bg-slate-950/80 px-4 py-3 text-sm md:text-base text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-hidden focus:ring-1 focus:ring-sky-500/30 transition-all disabled:opacity-60"
              />
            </div>
          </div>

          {/* Quick-Prompt Suggestions */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sparkles className="h-3 w-3 text-sky-400" />
              <span>Suggested Academic Inquiries</span>
            </span>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_RESEARCH_PROMPTS.map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  disabled={loading}
                  onClick={() => handleExecuteResearch(item.prompt)}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-1.5 text-xs text-slate-300 hover:border-sky-500/40 hover:bg-sky-500/10 hover:text-white transition-all disabled:opacity-40 text-left"
                >
                  <span className="font-medium text-sky-400">{item.title}:</span>
                  <span className="truncate max-w-xs">{item.prompt}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Configuration Controls Bar */}
          <div className="border-t border-slate-800/80 pt-4 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-300">
            {/* Source Toggles */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-slate-400 font-medium flex items-center gap-1">
                <Filter className="h-3.5 w-3.5 text-slate-500" />
                <span>Sources:</span>
              </span>
              {[
                { id: 'arxiv', label: 'arXiv' },
                { id: 'semantic_scholar', label: 'Semantic Scholar' },
                { id: 'crossref', label: 'Crossref' },
              ].map((src) => {
                const active = selectedSources.includes(src.id);
                return (
                  <button
                    key={src.id}
                    type="button"
                    disabled={loading}
                    onClick={() => toggleSource(src.id)}
                    className={`rounded-md px-2.5 py-1 font-medium transition-all border ${
                      active
                        ? 'bg-sky-500/20 text-sky-300 border-sky-500/40'
                        : 'bg-slate-950/40 text-slate-500 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {src.label}
                  </button>
                );
              })}
            </div>

            {/* Retrieval Parameters */}
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Papers:</span>
                <select
                  value={maxPapers}
                  disabled={loading}
                  onChange={(e) => setMaxPapers(Number(e.target.value))}
                  className="rounded-md border border-slate-800 bg-slate-950 px-2 py-1 text-slate-200 focus:outline-hidden"
                >
                  <option value={2}>2 Papers</option>
                  <option value={3}>3 Papers (Recommended)</option>
                  <option value={5}>5 Papers</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-slate-400">Top Evidence:</span>
                <select
                  value={topKEvidence}
                  disabled={loading}
                  onChange={(e) => setTopKEvidence(Number(e.target.value))}
                  className="rounded-md border border-slate-800 bg-slate-950 px-2 py-1 text-slate-200 focus:outline-hidden"
                >
                  <option value={4}>4 Chunks</option>
                  <option value={6}>6 Chunks</option>
                  <option value={8}>8 Chunks</option>
                </select>
              </div>

              {/* Submit Button */}
              <button
                type="button"
                disabled={loading || !question.trim()}
                onClick={() => handleExecuteResearch()}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 px-5 py-2 font-medium text-white shadow-lg shadow-sky-500/25 hover:brightness-110 active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Search className="h-4 w-4" />
                <span>{loading ? 'Researching...' : 'Conduct Research'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Live Multi-Stage Progress Display */}
        {loading && (
          <div className="rounded-2xl border border-sky-500/30 bg-slate-900/90 p-6 shadow-xl backdrop-blur-md space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500/20 text-sky-400 border border-sky-500/30 animate-pulse">
                  <Sparkles className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="font-semibold text-white text-sm md:text-base">
                    Autonomous Research Pipeline Active
                  </h3>
                  <p className="text-xs text-slate-400">
                    Executing dynamic paper retrieval and citation-grounded synthesis
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1.5 rounded-md bg-slate-950 px-2.5 py-1 text-xs text-slate-300 font-mono border border-slate-800">
                <Clock className="h-3.5 w-3.5 text-sky-400" />
                <span>{(elapsedMs / 1000).toFixed(1)}s elapsed</span>
              </div>
            </div>

            {/* Stages Timeline */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              {RESEARCH_STAGES.map((stg) => {
                const isPassed = currentStage > stg.id;
                const isCurrent = currentStage === stg.id;
                return (
                  <div
                    key={stg.id}
                    className={`rounded-xl border p-3.5 transition-all flex flex-col justify-between ${
                      isPassed
                        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                        : isCurrent
                        ? 'border-sky-500/50 bg-sky-500/10 text-sky-200 ring-1 ring-sky-500/30'
                        : 'border-slate-800/80 bg-slate-950/40 text-slate-500'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] font-mono font-bold">Step {stg.id}</span>
                      {isPassed ? (
                        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                      ) : isCurrent ? (
                        <span className="relative flex h-2.5 w-2.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-sky-500"></span>
                        </span>
                      ) : (
                        <span className="h-2 w-2 rounded-full bg-slate-700"></span>
                      )}
                    </div>
                    <div>
                      <p className="text-xs font-semibold leading-tight mb-1">{stg.label}</p>
                      <p className="text-[10px] text-slate-400 leading-normal line-clamp-2">{stg.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Error Notification */}
        {error && (
          <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5 text-xs md:text-sm text-rose-300 flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold text-rose-200">Autonomous Research Failure</p>
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Final Synthesized Research Result */}
        {result && (
          <div className="space-y-6">
            {/* Answer Card */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-6 md:p-8 shadow-2xl backdrop-blur-md space-y-6">
              {/* Answer Header Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-1 text-xs font-semibold text-emerald-400">
                    <ShieldCheck className="h-4 w-4" />
                    <span>Evidence-Grounded Synthesis</span>
                  </span>
                  <span className="inline-flex items-center gap-1 rounded-md bg-slate-950 border border-slate-800 px-2.5 py-1 text-xs text-slate-300 font-mono">
                    <Clock className="h-3 w-3 text-slate-400" />
                    <span>{result.execution_time_ms} ms</span>
                  </span>
                  <span className="inline-flex items-center gap-1 rounded-md bg-sky-500/10 border border-sky-500/20 px-2.5 py-1 text-xs text-sky-300 font-mono">
                    <Cpu className="h-3 w-3 text-sky-400" />
                    <span>{result.provider === 'mock' ? 'Mock Stand-in' : result.model || 'Groq Cloud'}</span>
                  </span>
                  {result.input_tokens !== undefined && result.input_tokens !== null && (
                    <span className="inline-flex items-center gap-1 rounded-md bg-slate-950 border border-slate-800 px-2.5 py-1 text-xs text-slate-300 font-mono">
                      <Cpu className="h-3 w-3 text-slate-400" />
                      <span>{result.input_tokens} in / {result.output_tokens || 0} out</span>
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleCopyAnswer}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-slate-300 hover:text-white hover:border-slate-700 transition-colors"
                >
                  {copied ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" />
                      <span>Copy Markdown</span>
                    </>
                  )}
                </button>
              </div>

              {/* Research Answer Body */}
              <div className="markdown-content text-sm md:text-base leading-relaxed text-slate-200">
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
                  {result.answer}
                </ReactMarkdown>
              </div>

              {/* Research Limitations & Nuance Callout */}
              {result.limitations && result.limitations.length > 0 && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-200 space-y-2">
                  <div className="flex items-center gap-2 font-semibold text-amber-300">
                    <AlertCircle className="h-4 w-4 text-amber-400" />
                    <span>Research Limitations & Evidence Caveats</span>
                  </div>
                  <ul className="list-disc pl-5 space-y-1 text-amber-200/90 leading-relaxed">
                    {result.limitations.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Papers Consulted Metadata Cards */}
            {result.papers_used && result.papers_used.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <BookOpen className="h-4 w-4 text-sky-400" />
                    <span>Academic Literature Ingested ({result.papers_used.length})</span>
                  </h3>
                  <span className="text-xs text-slate-500">
                    Extracted via PyMuPDF in-memory pipeline
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {result.papers_used.map((p, idx) => {
                    const isFullText = p.content_type === 'full_text';
                    return (
                      <div
                        key={p.paper_id || idx}
                        className="rounded-xl border border-slate-800 bg-slate-900/90 p-4 flex flex-col justify-between space-y-3 hover:border-slate-700 transition-colors"
                      >
                        <div className="space-y-2">
                          {/* Badges Bar */}
                          <div className="flex flex-wrap items-center gap-1.5">
                            {isFullText ? (
                              <span className="inline-flex items-center gap-1 rounded-sm bg-emerald-500/10 border border-emerald-500/30 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-400">
                                <FileText className="h-2.5 w-2.5" />
                                <span>Full Text ({p.page_count}p)</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 rounded-sm bg-amber-500/10 border border-amber-500/30 px-1.5 py-0.5 text-[10px] font-semibold text-amber-400">
                                <Info className="h-2.5 w-2.5" />
                                <span>Abstract Only</span>
                              </span>
                            )}

                            {p.year && (
                              <span className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                                {p.year}
                              </span>
                            )}

                            {p.sources && p.sources.map((s) => (
                              <span
                                key={s}
                                className="rounded-sm bg-sky-500/10 text-sky-400 border border-sky-500/20 px-1.5 py-0.5 text-[10px] font-mono capitalize"
                              >
                                {s}
                              </span>
                            ))}
                          </div>

                          {/* Paper Title */}
                          <h4 className="text-xs md:text-sm font-semibold text-white leading-snug line-clamp-2">
                            {p.title}
                          </h4>

                          {/* Authors */}
                          {p.authors && p.authors.length > 0 && (
                            <p className="text-[11px] text-slate-400 line-clamp-1">
                              {p.authors.join(', ')}
                            </p>
                          )}
                        </div>

                        {/* Links & Ingest Message */}
                        <div className="border-t border-slate-800/60 pt-2.5 flex items-center justify-between text-xs">
                          <span className="text-[10px] text-slate-500 truncate max-w-[150px]" title={p.ingest_message}>
                            {p.ingest_message}
                          </span>

                          <div className="flex items-center gap-2">
                            {p.url && (
                              <a
                                href={p.url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-0.5 text-[11px] text-sky-400 hover:text-sky-300 hover:underline"
                              >
                                <span>Source</span>
                                <ExternalLink className="h-2.5 w-2.5" />
                              </a>
                            )}
                            {p.pdf_url && (
                              <a
                                href={p.pdf_url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-0.5 rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-200 hover:bg-slate-700"
                              >
                                <DownloadCloud className="h-2.5 w-2.5 text-sky-400" />
                                <span>PDF</span>
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Specific Page & Abstract Citation Cards */}
            {result.citations && result.citations.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <FileSearch className="h-4 w-4 text-emerald-400" />
                    <span>Evidence Citations ({result.citations.length})</span>
                  </h3>
                  <span className="text-xs text-slate-500">
                    Validated against query-scoped FAISS embeddings
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {result.citations.map((cit, cIdx) => {
                    const isExpanded = expandedCitations[cIdx];
                    return (
                      <div
                        key={cIdx}
                        className="rounded-xl border border-slate-800 bg-slate-900/70 p-3 hover:border-slate-700 transition-colors"
                      >
                        <div
                          onClick={() => toggleCitation(cIdx)}
                          className="flex items-center justify-between cursor-pointer select-none"
                        >
                          <div className="flex items-center gap-2">
                            <span className="flex h-5 w-5 items-center justify-center rounded-md bg-emerald-500/10 text-[10px] font-bold text-emerald-400 border border-emerald-500/20">
                              {cIdx + 1}
                            </span>
                            <span className="text-xs font-medium text-slate-200 line-clamp-1">
                              {cit.paper_title}
                            </span>
                            {cit.is_abstract_only ? (
                              <span className="rounded-sm bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 text-[10px] font-mono">
                                Abstract
                              </span>
                            ) : (
                              <span className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                                p. {cit.page_number}
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
                            <button type="button" className="text-slate-500 hover:text-slate-300">
                              {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                            </button>
                          </div>
                        </div>

                        {/* Expandable Evidence Excerpt */}
                        {isExpanded && cit.evidence_excerpt && (
                          <div className="mt-2.5 rounded-lg bg-slate-950 p-2.5 border border-slate-800/80 text-xs text-slate-300 font-mono leading-relaxed">
                            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1 font-sans">
                              Indexed Evidence Excerpt:
                            </p>
                            <p className="italic">{cit.evidence_excerpt}</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Expandable Raw Evidence Snippets Explorer */}
            {result.evidence_snippets && result.evidence_snippets.length > 0 && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <button
                  type="button"
                  onClick={() => setShowAllEvidence(!showAllEvidence)}
                  className="flex items-center justify-between w-full text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <Layers className="h-3.5 w-3.5 text-sky-400" />
                    <span>Dense Vector Retrieval Evidence Chunks ({result.evidence_snippets.length})</span>
                  </span>
                  {showAllEvidence ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>

                {showAllEvidence && (
                  <div className="mt-3 space-y-2 border-t border-slate-800/80 pt-3">
                    {result.evidence_snippets.map((snip, sIdx) => (
                      <div key={sIdx} className="rounded-lg bg-slate-950 p-3 border border-slate-800/60 text-xs font-mono space-y-1">
                        <div className="flex items-center justify-between text-[11px] text-slate-400 font-sans">
                          <span className="font-semibold text-slate-300">{snip.paper_title}</span>
                          <div className="flex items-center gap-2">
                            <span className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-[10px]">
                              p. {snip.page_number}
                            </span>
                            <span className="rounded-sm bg-sky-500/10 text-sky-400 px-1.5 py-0.5 text-[10px]">
                              score: {snip.similarity_score}
                            </span>
                          </div>
                        </div>
                        <p className="text-slate-400 italic font-mono text-[11px] leading-relaxed">
                          "{snip.snippet}"
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
