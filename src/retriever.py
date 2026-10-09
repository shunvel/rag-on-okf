from __future__ import annotations

import os
import re
from typing import Any

from src.database import KnowledgeStore
from src.parsers import OKFParser


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def phrase_match(query: str, terms: list[str]) -> str | None:
    lowered = _normalize(query)
    tokens = {token.strip(",.?!():;\"'") for token in lowered.split() if token}
    ranked = sorted({term.lower().strip() for term in terms if term}, key=len, reverse=True)
    for term in ranked:
        if " " in term:
            if term in lowered:
                return term
        elif term in tokens:
            return term
    return None


class OKFRetriever:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def retrieve_semantic_broker(self, query: str) -> dict[str, Any]:
        """RAG search first, OKF intercept second (probabilistic discovery)."""
        matched_docs = self.store.hybrid_search(query)
        logs = [
            "Semantic Broker: hybrid search — Qdrant vectors + BM25, merged with Reciprocal Rank Fusion."
        ]
        context_blocks: list[str] = []
        crawled: list[str] = []
        route = "unstructured_only"

        for doc in matched_docs:
            if doc["type"] == "okf_concept":
                route = "okf_intercept"
                logs.append(f"OKF intercept: `{os.path.basename(doc['path'])}` ranked in the hybrid stack.")
                block, extra_logs, extra_nodes = self._hydrate_okf(doc, label="AUTHORITATIVE OKF TRUTH")
                if block:
                    context_blocks.append(block)
                    crawled.extend(extra_nodes)
                    logs.extend(extra_logs)
            elif doc["type"] == "warehouse_fact":
                logs.append(f"Warehouse snapshot hit: {doc.get('title')}")
                context_blocks.append(f"[WAREHOUSE SNAPSHOT]\n{doc['text']}")
            else:
                logs.append(f"Unstructured log hit (ID {doc['id']})")
                context_blocks.append(f"[UNSTRUCTURED LOG CHUNK]\n{doc['text']}")

        return {
            "pattern": "semantic_broker",
            "route": route,
            "context": "\n\n===\n\n".join(context_blocks),
            "logs": logs,
            "score_rows": matched_docs,
            "crawled_nodes": list(dict.fromkeys(crawled)),
        }

    def retrieve_deterministic_gateway(self, query: str) -> dict[str, Any]:
        """OKF first, RAG search second (deterministic routing)."""
        logs = ["Deterministic Gateway: matching query against OKF titles, tags, and aliases."]
        matched_doc = self._alias_lookup(query)

        if matched_doc:
            logs.append(
                f"Dictionary hit on `{matched_doc['title']}`. Bypassing Qdrant vector search."
            )
            block, extra_logs, crawled = self._hydrate_okf(matched_doc, label="DETERMINISTIC GATEWAY TRUTH")
            logs.extend(extra_logs)
            return {
                "pattern": "deterministic_gateway",
                "route": "okf_hit",
                "context": block,
                "logs": logs,
                "score_rows": [],
                "crawled_nodes": list(dict.fromkeys(crawled)),
            }

        logs.append("No OKF title/tag/alias match. Falling back to hybrid RAG.")
        fallback_docs = self.store.hybrid_search(query)
        context_blocks: list[str] = []
        crawled: list[str] = []
        route = "rag_fallback"
        for doc in fallback_docs:
            if doc["type"] == "okf_concept":
                okf_data = OKFParser.parse_file(doc["path"])
                if okf_data:
                    context_blocks.append(f"[FALLBACK RECOVERY]\n{okf_data.get('body')}")
                    crawled.append(okf_data["metadata"].get("title", ""))
            else:
                prefix = "WAREHOUSE SNAPSHOT" if doc["type"] == "warehouse_fact" else "UNSTRUCTURED LOG CHUNK"
                context_blocks.append(f"[{prefix}]\n{doc['text']}")
        return {
            "pattern": "deterministic_gateway",
            "route": route,
            "context": "\n\n===\n\n".join(context_blocks),
            "logs": logs,
            "score_rows": fallback_docs,
            "crawled_nodes": list(dict.fromkeys(crawled)),
        }

    def _alias_lookup(self, query: str) -> dict[str, Any] | None:
        for doc in self.store.raw_documents:
            if doc["type"] != "okf_concept":
                continue
            terms = [doc.get("title", ""), *doc.get("aliases", [])]
            if phrase_match(query, terms):
                return doc
        return None

    def _hydrate_okf(self, doc: dict[str, Any], label: str) -> tuple[str, list[str], list[str]]:
        logs: list[str] = []
        crawled: list[str] = []
        okf_data = OKFParser.parse_file(doc["path"])
        if not okf_data:
            return "", logs, crawled
        meta = okf_data["metadata"]
        title = meta.get("title", "Untitled")
        crawled.append(title)
        node_type = str(meta.get("type", "concept")).upper()
        block = (
            f"[{label}] {title} ({node_type})\n"
            f"• Registered Source: {meta.get('resource')}\n"
            f"• Active Spec:\n{okf_data.get('body')}"
        )
        for relation in meta.get("links") or []:
            base_dir = os.path.dirname(doc["path"])
            full_path = os.path.normpath(os.path.join(base_dir, relation))
            linked = OKFParser.parse_file(full_path)
            if not linked:
                continue
            linked_title = linked["metadata"].get("title", os.path.basename(full_path))
            logs.append(f"Graph edge crawled: `{linked_title}`")
            crawled.append(linked_title)
            block += (
                f"\n\n[ADJACENT GRAPH NODE] {linked_title}\n"
                f"Location: {linked['metadata'].get('resource')}\n"
                f"{linked.get('body')}"
            )
        return block, logs, crawled
