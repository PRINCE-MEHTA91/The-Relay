"""Query planner decomposing natural-language research questions into targeted academic search queries."""

import re
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def generate_fallback_queries(question: str) -> List[str]:
    """Deterministic academic query expansion fallback without LLM."""
    clean = re.sub(r"[^\w\s\-]", "", question).strip()
    words = clean.split()
    stop_words = {
        "what", "is", "the", "and", "how", "does", "it", "to", "in", "of", "for",
        "a", "an", "explain", "compare", "why", "are", "with", "between", "on", "can"
    }
    keywords = [w for w in words if w.lower() not in stop_words]

    queries = []
    # 1. Full clean keywords query
    if keywords:
        queries.append(" ".join(keywords[:6]))

    # 2. Key phrases
    if "compare" in question.lower() or "difference" in question.lower() or "versus" in question.lower():
        queries.append(f"{' '.join(keywords[:4])} comparison benchmark")
    else:
        queries.append(f"{' '.join(keywords[:4])} architecture sequence modeling")

    # 3. Specific theoretical query
    if len(keywords) >= 2:
        queries.append(f"{keywords[0]} {keywords[1]} survey")

    # Deduplicate while preserving order
    seen = set()
    result = []
    for q in queries:
        if q not in seen and len(q.strip()) > 3:
            seen.add(q)
            result.append(q)

    return result[:3] if result else [clean]


class QueryPlanner:
    """Generates focused search queries from natural language research questions using Groq or fallback."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name

    def plan_queries(self, question: str, use_mock: bool = False, max_queries: int = 3) -> List[str]:
        """Generate academic search queries from a user research question."""
        clean_question = question.strip()
        if not clean_question:
            return []

        if use_mock:
            return generate_fallback_queries(clean_question)[:max_queries]

        try:
            from core.groq_service import send_chat_completion

            system_prompt = (
                "You are an academic literature search specialist.\n"
                "Your task is to convert a user's research question into 2 to 4 distinct, highly targeted "
                "search queries suitable for academic search engines (arXiv, Semantic Scholar, Crossref).\n"
                "Focus on core theoretical concepts, algorithms, architectural names, and domain terminology.\n\n"
                "OUTPUT FORMAT: Return ONLY a valid JSON array of strings. Do not include markdown or explanations.\n"
                "Example: [\"query one\", \"query two\", \"query three\"]"
            )
            user_prompt = f"Research Question: {clean_question}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = send_chat_completion(messages=messages, model=self.model_name)
            if response.get("status") == "success" and response.get("content"):
                raw_content = response["content"].strip()
                # Strip markdown code fences if present
                if "```" in raw_content:
                    raw_content = re.sub(r"```(?:json)?", "", raw_content).replace("```", "").strip()

                parsed = json.loads(raw_content)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    cleaned_queries = [q.strip() for q in parsed if q.strip()]
                    if cleaned_queries:
                        return cleaned_queries[:max_queries]

        except Exception as e:
            logger.warning(f"Groq query planning failed, falling back to deterministic planner: {e}")

        return generate_fallback_queries(clean_question)[:max_queries]

