/**
 * API Service for The Relay Backend
 * Communicates with FastAPI endpoints (/ask, /ask/stream, /health, /metrics).
 */

const API_BASE_URL = ''; // Relative path leverages Vite dev server proxy to http://127.0.0.1:8000

/**
 * Fetch health status of the backend and FAISS index
 */
export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Health check failed with HTTP ${res.status}`);
  }
  return await res.json();
}

/**
 * Fetch concurrency and rate limiting runtime metrics
 */
export async function fetchMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Metrics fetch failed with HTTP ${res.status}`);
  }
  return await res.json();
}

/**
 * Send a non-streaming research question to /ask
 * @param {Object} params
 * @param {string} params.question
 * @param {number} params.top_k
 * @param {boolean} params.use_mock
 * @param {string|null} params.mock_mode
 */
export async function askQuestion({ question, top_k = 5, use_mock = false, mock_mode = null }) {
  const payload = {
    question: question.trim(),
    top_k,
    use_mock,
    stream: false,
  };
  if (use_mock && mock_mode) {
    payload.mock_mode = mock_mode;
  }

  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errDetail = `Server returned HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch {
      // ignore
    }
    const err = new Error(errDetail);
    err.status = res.status;
    throw err;
  }

  return await res.json();
}

/**
 * Send a streaming research question to /ask with stream=true using Server-Sent Events (SSE)
 * @param {Object} params
 * @param {string} params.question
 * @param {number} params.top_k
 * @param {boolean} params.use_mock
 * @param {string|null} params.mock_mode
 * @param {Function} onInit - Callback when retrieved chunks metadata arrives
 * @param {Function} onToken - Callback when a token arrives: onToken(tokenString, accumulatedText)
 * @param {Function} onDone - Callback when stream finishes with metadata: onDone(finalMetadata)
 * @param {Function} onError - Callback when error occurs
 * @returns {AbortController} - Controller to cancel stream if needed
 */
export function streamQuestion(
  { question, top_k = 5, use_mock = false, mock_mode = null },
  { onInit, onToken, onDone, onError }
) {
  const controller = new AbortController();

  const payload = {
    question: question.trim(),
    top_k,
    use_mock,
    stream: true,
  };
  if (use_mock && mock_mode) {
    payload.mock_mode = mock_mode;
  }

  (async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        let errMessage = `Server error HTTP ${response.status}`;
        try {
          const errData = await response.json();
          errMessage = errData.detail || errMessage;
        } catch {
          // ignore
        }
        const error = new Error(errMessage);
        error.status = response.status;
        onError(error);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let accumulatedContent = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        // Keep the last partial line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const jsonStr = trimmed.slice(6).trim();
            if (!jsonStr) continue;

            try {
              const eventData = JSON.parse(jsonStr);

              // 1. Initial event with retrieved chunks count
              if (eventData.type === 'init') {
                if (onInit) onInit(eventData);
              }
              // 2. Token emission
              else if (eventData.token !== undefined && !eventData.done) {
                accumulatedContent += eventData.token;
                if (onToken) onToken(eventData.token, accumulatedContent);
              }
              // 3. Final event with citations and validation
              else if (eventData.done) {
                if (onDone) {
                  onDone({
                    answer: eventData.answer || accumulatedContent,
                    citations: eventData.citations || [],
                    status: eventData.status || 'verified',
                    validation_status: eventData.validation_status || 'VERIFIED',
                    input_tokens: eventData.input_tokens || 0,
                    output_tokens: eventData.output_tokens || 0,
                    retries_attempted: eventData.retries_attempted || 0,
                    latency_ms: eventData.latency_ms,
                    provider: eventData.provider || (payload.use_mock ? 'mock' : 'groq'),
                    model: eventData.model,
                  });
                }
              }
            } catch (parseErr) {
              console.warn('Failed to parse SSE JSON chunk:', jsonStr, parseErr);
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Stream request aborted by user');
      } else {
        onError(err);
      }
    }
  })();

  return controller;
}

/**
 * Fetch supported academic repositories (/research/sources)
 */
export async function fetchResearchSources() {
  const res = await fetch(`${API_BASE_URL}/research/sources`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch academic sources: HTTP ${res.status}`);
  }
  return await res.json();
}

/**
 * Autonomously search academic literature without synthesis (/research/search)
 * @param {Object} params
 * @param {string} params.question
 * @param {string[]} [params.sources]
 * @param {number} [params.max_results_per_source]
 * @param {boolean} [params.use_mock]
 */
export async function searchAcademicLiterature({
  question,
  sources = ['arxiv', 'semantic_scholar', 'crossref'],
  max_results_per_source = 5,
  use_mock = false,
}) {
  const payload = {
    question: question.trim(),
    sources,
    max_results_per_source,
    use_mock,
  };

  const res = await fetch(`${API_BASE_URL}/research/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errDetail = `Server returned HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch {
      // ignore
    }
    const err = new Error(errDetail);
    err.status = res.status;
    throw err;
  }

  return await res.json();
}

/**
 * Conduct end-to-end autonomous research: query expansion -> literature search -> PDF download -> in-memory indexing -> synthesis (/research/answer)
 * @param {Object} params
 * @param {string} params.question
 * @param {number} [params.max_papers]
 * @param {string[]} [params.sources]
 * @param {number} [params.top_k_evidence]
 * @param {boolean} [params.use_mock]
 */
export async function fetchResearchAnswer({
  question,
  max_papers = 3,
  sources = ['arxiv', 'semantic_scholar', 'crossref'],
  top_k_evidence = 6,
  use_mock = false,
}) {
  const payload = {
    question: question.trim(),
    max_papers,
    sources,
    top_k_evidence,
    use_mock,
  };

  const res = await fetch(`${API_BASE_URL}/research/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errDetail = `Server returned HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch {
      // ignore
    }
    const err = new Error(errDetail);
    err.status = res.status;
    throw err;
  }

  return await res.json();
}


