import React, { useState } from 'react';
import {
  Search,
  Sparkles,
  ArrowRight,
  BookOpen,
  Cpu,
  Layers,
  ShieldCheck,
  Zap,
  Sliders,
  Globe,
} from 'lucide-react';

export default function Dashboard({ onSearch, settings, onOpenSettings, onOpenPapers, onOpenAutonomous }) {
  const [question, setQuestion] = useState('');

  const suggestions = [
    {
      title: 'Scaled Dot-Product Attention',
      query: 'What is the mathematical formulation of Scaled Dot-Product Attention in the Transformer?',
      paper: 'Attention Is All You Need',
      icon: Cpu,
      color: 'from-amber-500/20 to-orange-500/20 text-amber-400 border-amber-500/30',
    },
    {
      title: 'Moshi Spoken Dialogue',
      query: 'How does Moshi support real-time full-duplex spoken dialogue using separate audio codec streams?',
      paper: 'Moshi: Speech-Text Model',
      icon: Zap,
      color: 'from-purple-500/20 to-violet-500/20 text-purple-400 border-purple-500/30',
    },
    {
      title: 'Dynamic Chunking Mechanism',
      query: 'How does dynamic chunking determine variable-length boundaries in hierarchical sequence modeling?',
      paper: 'Dynamic Chunking',
      icon: Layers,
      color: 'from-emerald-500/20 to-teal-500/20 text-emerald-400 border-emerald-500/30',
    },
    {
      title: 'Self-Attention Complexity',
      query: 'Compare the per-layer computational complexity of self-attention layers with recurrent and convolutional layers.',
      paper: 'Attention Is All You Need',
      icon: BookOpen,
      color: 'from-sky-500/20 to-blue-500/20 text-sky-400 border-sky-500/30',
    },
    {
      title: 'Helium 7B Parameter Architecture',
      query: 'What is the base text language model in Moshi and what is its parameter count?',
      paper: 'Moshi: Speech-Text Model',
      icon: Cpu,
      color: 'from-pink-500/20 to-rose-500/20 text-pink-400 border-pink-500/30',
    },
    {
      title: 'Negative Refusal Control',
      query: 'What are the recipe ingredients and temperature steps for baking French chocolate croissants?',
      paper: 'Out-of-Domain Query',
      icon: ShieldCheck,
      color: 'from-slate-500/20 to-zinc-500/20 text-slate-300 border-slate-500/30',
    },
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onSearch(question.trim());
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto px-4 py-8 md:px-12 lg:px-20">
      <div className="mx-auto max-w-4xl">
        {/* Hero Branding */}
        <div className="text-center mb-10 pt-4">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-3.5 py-1 text-xs font-medium text-sky-400 backdrop-blur-md mb-4 shadow-sm shadow-sky-500/20">
            <Sparkles className="h-3.5 w-3.5" />
            <span>arXiv Research Corpus • Attention • Moshi • Dynamic Chunking</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-3 bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
            Where research meets grounded answers.
          </h1>
          <p className="text-base text-slate-400 max-w-2xl mx-auto">
            Ask technical questions across arXiv papers. Answers are synthesized with dense vector retrieval,
            concurrency guardrails, and verified page-level citations.
          </p>
        </div>

        {/* Autonomous Research Mode Highlight Banner */}
        <div className="mb-6 rounded-2xl border border-sky-500/30 bg-gradient-to-r from-sky-500/10 via-slate-900/80 to-blue-500/10 p-4 shadow-lg flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500/20 text-sky-400 border border-sky-500/30 shrink-0">
              <Globe className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white text-sm">Need Autonomous Academic Search?</span>
                <span className="rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/40 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider">
                  Phase 3 Live
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Explore any scientific question beyond the 3 local papers: automated arXiv, Semantic Scholar & Crossref search with open-access PDF retrieval.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onOpenAutonomous}
            className="flex items-center gap-1.5 rounded-xl bg-sky-500 px-3.5 py-2 text-xs font-semibold text-white shadow-md shadow-sky-500/20 hover:bg-sky-400 transition-all shrink-0"
          >
            <span>Launch Autonomous Agent</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Central Search Form */}
        <div className="relative mb-12">
          <form
            onSubmit={handleSubmit}
            className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-3 shadow-2xl shadow-sky-950/40 backdrop-blur-xl transition-all focus-within:border-sky-500/80 focus-within:ring-2 focus-within:ring-sky-500/20"
          >
            <div className="relative">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a technical research question (e.g., 'What is the formula for scaled dot-product attention?')..."
                rows={3}
                className="w-full resize-none bg-transparent px-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-hidden text-sm md:text-base leading-relaxed"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/80 pt-2.5 px-2">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="flex items-center gap-1.5 rounded-md bg-slate-800/80 px-2.5 py-1 text-slate-300 border border-slate-700">
                  <span className="font-semibold text-sky-400">Top-K:</span> {settings.top_k}
                </span>

                <span className="flex items-center gap-1.5 rounded-md bg-slate-800/80 px-2.5 py-1 text-slate-300 border border-slate-700">
                  <span className="font-semibold text-sky-400">Provider:</span>{' '}
                  {settings.use_mock ? `Mock (${settings.mock_mode || 'NORMAL'})` : 'Live Groq (20B)'}
                </span>

                <button
                  type="button"
                  onClick={onOpenSettings}
                  className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
                  title="Configure retrieval parameters"
                >
                  <Sliders className="h-4 w-4" />
                </button>
              </div>

              <div className="flex items-center gap-2">
                <span className="hidden sm:inline text-[11px] text-slate-500">
                  Press <kbd className="rounded-sm bg-slate-800 px-1.5 py-0.5 text-slate-400 border border-slate-700">Enter ↵</kbd> to submit
                </span>
                <button
                  type="submit"
                  disabled={!question.trim()}
                  className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 px-4 py-2 text-xs md:text-sm font-medium text-white shadow-md shadow-sky-600/30 transition-all hover:brightness-110 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <span>Query</span>
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Suggested Queries Grid */}
        <div className="mb-12">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-sky-400" />
              Suggested Benchmark Queries
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {suggestions.map((item, idx) => {
              const Icon = item.icon;
              return (
                <button
                  key={idx}
                  onClick={() => onSearch(item.query)}
                  className="group flex flex-col justify-between rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 text-left transition-all hover:border-slate-700 hover:bg-slate-900/90 hover:shadow-lg active:scale-[0.99]"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-white group-hover:text-sky-400 transition-colors">
                        {item.title}
                      </span>
                      <div className={`rounded-lg border p-1.5 ${item.color}`}>
                        <Icon className="h-3.5 w-3.5" />
                      </div>
                    </div>
                    <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                      {item.query}
                    </p>
                  </div>
                  <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-800/60 pt-2">
                    <span className="truncate">{item.paper}</span>
                    <span className="flex items-center gap-1 text-sky-400 opacity-0 group-hover:opacity-100 transition-opacity">
                      Ask now <ArrowRight className="h-3 w-3" />
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Ingested Paper Library Quick Link */}
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/40 p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4 text-left">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-sky-500/10 border border-sky-500/20 text-sky-400">
              <BookOpen className="h-6 w-6" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Three Seminal arXiv Research Papers</h3>
              <p className="text-xs text-slate-400">
                Indexed in FAISS with 215 chunks (500t & 250t) using Sentence Transformers.
              </p>
            </div>
          </div>
          <button
            onClick={onOpenPapers}
            className="shrink-0 flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800 hover:text-white transition-colors"
          >
            <span>Explore Library</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

