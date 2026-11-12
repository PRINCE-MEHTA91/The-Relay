import React, { useState } from 'react';
import {
  BookOpen,
  ExternalLink,
  Layers,
  FileText,
  Search,
  Sparkles,
  ArrowRight,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

export default function PapersLibrary({ onQueryPaper }) {
  const [filter, setFilter] = useState('');

  const papers = [
    {
      id: 'attention',
      title: 'Attention Is All You Need',
      arxivId: 'arXiv:1706.03762',
      version: 'v7',
      pages: 15,
      authors: 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin',
      affiliation: 'Google Brain & Google Research',
      pdfUrl: 'https://arxiv.org/pdf/1706.03762v7',
      arxivUrl: 'https://arxiv.org/abs/1706.03762v7',
      abstract:
        'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.',
      keyConcepts: [
        'Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V',
        'Multi-Head Attention across h representation subspaces',
        'Computational Complexity: O(n^2 * d) per layer vs O(n * d^2) for RNNs',
        'Sinusoidal Positional Encoding and residual connections',
      ],
      sampleQueries: [
        'What is the mathematical formulation of Scaled Dot-Product Attention?',
        'Why does the dot product require scaling by the square root of d_k?',
        'Compare the computational complexity per layer of self-attention with recurrent layers.',
      ],
      chunkCount500: 45,
      chunkCount250: 82,
    },
    {
      id: 'moshi',
      title: 'Moshi: a speech-text foundation model for real-time dialogue',
      arxivId: 'arXiv:2410.00037',
      version: 'v1',
      pages: 66,
      authors: 'Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, Neil Zeghidour',
      affiliation: 'Kyutai',
      pdfUrl: 'https://arxiv.org/pdf/2410.00037v1',
      arxivUrl: 'https://arxiv.org/abs/2410.00037v1',
      abstract:
        'Current spoken dialogue systems rely on pipelined cascades of speech-to-text, text-to-text language modeling, and text-to-speech synthesis, introducing artificial turn-taking latencies. Moshi is a speech-text foundation model enabling full-duplex spoken interaction with a theoretical 160ms latency, modeling user and system audio simultaneously as interleaved codec streams.',
      keyConcepts: [
        'Helium 7B text language model foundation',
        'Mimi 12.5 Hz neural audio codec with split Residual Vector Quantization (RVQ)',
        'Full-duplex real-time spoken dialogue with parallel user and system audio streams',
        'Joint text-audio token modeling without separate ASR and TTS cascading',
      ],
      sampleQueries: [
        'What is Moshi and how does it support full-duplex spoken dialogue?',
        'What is the base text language model in Moshi and what is its parameter count?',
        'How does Mimi use split residual vector quantization (RVQ)?',
      ],
      chunkCount500: 105,
      chunkCount250: 198,
    },
    {
      id: 'dynamic_chunking',
      title: 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling',
      arxivId: 'arXiv:2507.07955',
      version: 'v2',
      pages: 39,
      authors: 'Sukjun Hwang, Brandon Wang, Albert Gu',
      affiliation: 'Carnegie Mellon University & Cartesia AI',
      pdfUrl: 'https://arxiv.org/pdf/2507.07955v2',
      arxivUrl: 'https://arxiv.org/abs/2507.07955v2',
      abstract:
        'Hierarchical sequence models offer high efficiency on long sequences by processing them at multiple granularities, but rely on static chunking rules. Dynamic Chunking learns variable-length boundary segmentations end-to-end using Routing and Smoothing Modules, allowing adaptive token grouping with monotonic channel width scaling.',
      keyConcepts: [
        'Routing Module determines boundary decision scores dynamically',
        'Smoothing Module makes boundary segmentation differentiable for backprop',
        'Monotonic width scaling: D_0 <= D_1 <= ... <= D_S across hierarchy stages',
        'End-to-end adaptive sequence modeling without heuristic token chunking',
      ],
      sampleQueries: [
        'How does dynamic chunking differ from fixed-size chunking in hierarchical sequence modeling?',
        'What is the role of the Routing Module in dynamic chunking?',
        'Why does model width scale monotonically across hierarchical stages?',
      ],
      chunkCount500: 65,
      chunkCount250: 124,
    },
  ];

  const filtered = papers.filter(
    (p) =>
      p.title.toLowerCase().includes(filter.toLowerCase()) ||
      p.authors.toLowerCase().includes(filter.toLowerCase()) ||
      p.abstract.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="flex-1 overflow-y-auto px-4 py-8 md:px-12 lg:px-20">
      <div className="mx-auto max-w-5xl">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <BookOpen className="h-5 w-5 text-sky-400" />
              <h1 className="text-2xl font-bold text-white tracking-tight">Research Paper Library</h1>
            </div>
            <p className="text-xs md:text-sm text-slate-400">
              The 3 authoritative arXiv papers ingested and indexed into The Relay vector database.
            </p>
          </div>

          {/* Filter */}
          <div className="relative w-full md:w-72">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter papers or concepts..."
              className="w-full rounded-xl border border-slate-800 bg-slate-900 px-9 py-2 text-xs text-slate-200 placeholder-slate-500 focus:border-sky-500/80 focus:outline-hidden"
            />
          </div>
        </div>

        {/* Papers Grid */}
        <div className="space-y-6">
          {filtered.map((paper) => (
            <div
              key={paper.id}
              className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-6 shadow-xl backdrop-blur-md transition-all hover:border-slate-700 hover:bg-slate-900/80"
            >
              {/* Paper Title & Badges */}
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <div className="flex-1">
                  <div className="flex flex-wrap items-center gap-2 mb-1.5">
                    <span className="rounded-md bg-sky-500/10 px-2 py-0.5 text-xs font-semibold text-sky-400 border border-sky-500/20 font-mono">
                      {paper.arxivId} {paper.version}
                    </span>
                    <span className="rounded-md bg-slate-800 px-2 py-0.5 text-xs text-slate-300 border border-slate-700">
                      {paper.pages} pages
                    </span>
                    <span className="rounded-md bg-slate-800/80 px-2 py-0.5 text-xs text-slate-400">
                      {paper.affiliation}
                    </span>
                  </div>
                  <h2 className="text-lg font-bold text-white tracking-tight">{paper.title}</h2>
                  <p className="text-xs text-slate-400 mt-1">{paper.authors}</p>
                </div>

                <div className="flex items-center gap-2">
                  <a
                    href={paper.pdfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 hover:text-white transition-colors"
                  >
                    <span>arXiv PDF</span>
                    <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              </div>

              {/* Abstract */}
              <p className="text-xs md:text-sm text-slate-300 leading-relaxed mb-4">
                {paper.abstract}
              </p>

              {/* Key Concepts */}
              <div className="mb-4 rounded-xl bg-slate-950/70 p-4 border border-slate-800/80">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-sky-400" />
                  Key Mathematical & Architectural Concepts
                </h3>
                <ul className="space-y-1.5 text-xs text-slate-300">
                  {paper.keyConcepts.map((concept, cIdx) => (
                    <li key={cIdx} className="flex items-start gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-sky-400 mt-1.5 shrink-0" />
                      <span>{concept}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Sample Queries */}
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Query this Paper:
                </h4>
                <div className="flex flex-wrap gap-2">
                  {paper.sampleQueries.map((q, qIdx) => (
                    <button
                      key={qIdx}
                      onClick={() => onQueryPaper(q)}
                      className="group flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-1.5 text-xs text-slate-300 hover:border-sky-500/50 hover:bg-slate-900 hover:text-sky-300 transition-all text-left"
                    >
                      <span className="truncate max-w-xs">{q}</span>
                      <ArrowRight className="h-3 w-3 text-slate-500 group-hover:text-sky-400 shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

