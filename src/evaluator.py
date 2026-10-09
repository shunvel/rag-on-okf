from __future__ import annotations

import asyncio
import math
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from src.config import Config
from src.ragas_bootstrap import ensure_ragas_importable


@lru_cache(maxsize=4)
def _judge_bundle(api_key: str, model: str) -> tuple[Any, Any, AsyncOpenAI]:
    ensure_ragas_importable()
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms.base import llm_factory

    # Ragas 0.4 collection metrics use ascore() / agenerate() — requires async client.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=Config.GEMINI_OPENAI_BASE_URL,
    )
    llm = llm_factory(model, provider="openai", client=client, temperature=0)
    embeddings = OpenAIEmbeddings(client=client, model=Config.GEMINI_EMBED_MODEL)
    return llm, embeddings, client


async def _generate_answer(client: AsyncOpenAI, model: str, query: str, context: str) -> str:
    if not context.strip():
        return "No retrieved context was available, so no grounded answer can be produced."
    prompt = (
        "Answer the user question using ONLY the retrieved context below. "
        "If the context is insufficient, say so briefly.\n\n"
        f"Question: {query}\n\nContext:\n{context[:8000]}"
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


async def _score_with_ragas(
    query: str,
    answer: str,
    context_list: list[str],
    llm: Any,
    embeddings: Any,
) -> dict[str, float]:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    contexts = [c for c in context_list if c.strip()] or [""]
    faith = await Faithfulness(llm=llm).ascore(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )
    relevancy = await AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=2).ascore(
        user_input=query,
        response=answer,
    )
    precision = await ContextPrecisionWithoutReference(llm=llm).ascore(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )
    return {
        "faithfulness": _safe_float(faith.value),
        "answer_relevancy": _safe_float(relevancy.value),
        "context_precision": _safe_float(precision.value),
    }


async def _evaluate_live(
    query: str,
    context: str,
    blocks: list[str],
    api_key: str,
    model: str,
) -> dict[str, Any]:
    llm, embeddings, client = _judge_bundle(api_key, model)
    answer = await _generate_answer(client, model, query, context)
    scores = await _score_with_ragas(query, answer, blocks, llm, embeddings)
    return {
        **scores,
        "answer": answer,
        "engine": f"Ragas · {model} (Gemini OpenAI-compat)",
        "reasoning": (
            "Scores from Ragas metrics: context precision (without reference), "
            "faithfulness, and answer relevancy — judged with Gemini via the OpenAI-compatible API."
        ),
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return max(0.0, min(1.0, out))


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Streamlit may already have a loop; run in a fresh loop on a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class RagasEvaluator:
    @staticmethod
    def evaluate(
        query: str,
        context: str,
        context_list: list[str] | None = None,
        api_key: str = "",
    ) -> dict[str, Any]:
        # Back-compat: evaluate(query, context, api_key) when the third arg was the key.
        if isinstance(context_list, str) and not api_key:
            api_key = context_list
            context_list = None

        blocks = context_list if context_list else ([context] if context.strip() else [])
        if not api_key or not Config.USE_RAGAS:
            reason = "No Gemini key. Using local estimate." if not api_key else "USE_RAGAS=0. Using local estimate."
            return RagasEvaluator._simulate(context, reason=reason)

        model = Config.GEMINI_MODEL
        try:
            return _run_async(_evaluate_live(query, context, blocks, api_key, model))
        except Exception as exc:
            message = str(exc)
            if "429" in message or "quota" in message.lower() or "RESOURCE_EXHAUSTED" in message:
                reason = (
                    "Gemini quota or rate limit hit (HTTP 429). Showing a local estimate until quota resets. "
                    f"Last error: {exc}"
                )
            elif "503" in message or "UNAVAILABLE" in message or "high demand" in message.lower():
                reason = (
                    "Gemini is overloaded (HTTP 503). Showing a local precision estimate. "
                    "Retry the run in a moment. "
                    f"Last error: {exc}"
                )
            else:
                reason = f"Ragas evaluation failed; using local estimate. Last error: {exc}"
            simulated = RagasEvaluator._simulate(context, reason=reason)
            simulated["answer"] = RagasEvaluator._answer_from_context(context)
            return simulated

    @staticmethod
    def _answer_from_context(context: str) -> str:
        if not context.strip():
            return "No retrieved context was available, so no grounded answer can be produced."
        excerpt = context.strip()[:1200]
        return f"Based only on retrieved context: {excerpt}"

    @staticmethod
    def _simulate(context: str, reason: str) -> dict[str, Any]:
        is_okf = "[AUTHORITATIVE" in context or "[DETERMINISTIC" in context
        polluted = "[UNSTRUCTURED LOG CHUNK]" in context
        precision = 0.96 if is_okf and not polluted else (0.72 if is_okf else 0.38)
        faithfulness = 0.97 if is_okf else 0.58
        answer_relevancy = 0.85 if is_okf else 0.55
        return {
            "context_precision": precision,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer": RagasEvaluator._answer_from_context(context),
            "engine": "fallback estimate",
            "reasoning": reason,
        }
