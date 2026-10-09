from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from src.config import Config
from src.database import KnowledgeStore
from src.evaluator import RagasEvaluator
from src.retriever import OKFRetriever
from src.visualize import (
    neighbor_frame,
    neighborhood,
    neighborhood_inference,
    pipeline_flow_html,
    retrieval_explain_html,
)

st.set_page_config(page_title="RAG vs OKF — Complement Lab", layout="wide")

PRESETS = [
    "Show me the formula for NRC",
    "How do we write a query for our NRC metrics?",
    "What does money lost when accounts leave mean?",
    "Show me how to compute CAC attribution",
    "What is the official ARPU definition?",
    "Show me common subscription billing errors",
]

PRESET_WHY = {
    "Show me the formula for NRC": "Named KPI — OKF should bypass search; RAG still pays vector cost.",
    "How do we write a query for our NRC metrics?": "Ticker language (NRC) — OKF dictionary hit.",
    "What does money lost when accounts leave mean?": "Paraphrase — RAG dense search should find NRC; OKF aliases may miss.",
    "Show me how to compute CAC attribution": "Named KPI — OKF hit on CAC, then graph walk to MRR.",
    "What is the official ARPU definition?": "Named KPI — OKF hit on ARPU.",
    "Show me common subscription billing errors": "No canonical metric — both sides may retrieve tickets/noise.",
}

ROUTE_LABELS = {
    "okf_hit": "OKF hit — search bypassed",
    "okf_intercept": "OKF intercept after hybrid search",
    "rag_fallback": "RAG fallback (no alias hit)",
    "unstructured_only": "Unstructured / warehouse only",
}


@st.cache_resource(show_spinner="Connecting to Qdrant and loading embeddings...")
def load_store() -> KnowledgeStore:
    store = KnowledgeStore()
    if not store.healthcheck():
        raise ConnectionError(
            "Cannot reach Qdrant on the configured host/port. Start it with `docker compose up -d` "
            "(Docker CLI: /Applications/Docker.app/Contents/Resources/bin/docker). "
            "This project maps host 6343 → container 6333."
        )
    store.ensure_ready(force_reseed=False)
    return store


def render_pattern_tag(kind: str, family: str, color: str) -> None:
    st.markdown(
        f"<div style='margin:0 0 0.6rem 0'>"
        f"<span style='background:{color};color:#fff;font-weight:700;letter-spacing:0.04em;"
        f"padding:0.22rem 0.65rem;border-radius:6px;font-size:0.85rem'>{kind}</span>"
        f"<span style='margin-left:0.55rem;font-size:0.92rem;color:#24292f'>{family}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_stepper(steps: list[tuple[str, bool]]) -> None:
    chips = []
    for label, active in steps:
        tone = "#1f6feb" if active else "#8c959f"
        chips.append(
            f"<span style='border:1px solid {tone};color:{tone};padding:0.2rem 0.55rem;"
            f"border-radius:999px;margin-right:0.35rem;font-size:0.8rem'>{label}</span>"
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def render_column(
    title: str,
    subtitle: str,
    result: dict,
    metrics: dict,
    latency_ms: float,
    extra_delta: str | None,
    kind: str,
    family: str,
    color: str,
) -> None:
    render_pattern_tag(kind, family, color)
    st.subheader(title)
    st.caption(subtitle)
    st.metric("Latency", f"{latency_ms:.1f} ms", delta=extra_delta)
    st.caption(ROUTE_LABELS.get(result["route"], result["route"]))
    with st.expander("Architecture trace", expanded=True):
        for log in result["logs"]:
            st.markdown(f"- {log}")
    st.text_area("Prompt context", value=result["context"], height=240, key=f"ctx-{title}")
    st.markdown(f"**{Config.GEMINI_MODEL} — LLM-as-judge**")
    st.slider("Context precision", 0.0, 1.0, float(metrics["context_precision"]), disabled=True, key=f"p-{title}")
    st.slider("Faithfulness / groundedness", 0.0, 1.0, float(metrics["faithfulness"]), disabled=True, key=f"f-{title}")
    st.info(metrics["reasoning"])


st.markdown(
    """
<div style="border:1px solid #d0d7de;border-radius:12px;padding:1.1rem 1.25rem 1.15rem;margin-bottom:1rem;background:#f6f8fa">
  <div style="font-size:0.78rem;letter-spacing:0.08em;font-weight:700;color:#656d76;text-transform:uppercase;margin-bottom:0.45rem">
    Enterprise knowledge retrieval lab
  </div>
  <div style="font-size:1.85rem;font-weight:750;line-height:1.2;color:#1f2328;margin-bottom:0.45rem">
    RAG vs OKF — same query, two composition orders
  </div>
  <div style="margin-bottom:0.7rem">
    <span style="background:#1f6feb;color:#fff;font-weight:700;padding:0.18rem 0.55rem;border-radius:6px;font-size:0.8rem">RAG first, then OKF</span>
    <span style="color:#8c959f;margin:0 0.55rem">vs</span>
    <span style="background:#1a7f37;color:#fff;font-weight:700;padding:0.18rem 0.55rem;border-radius:6px;font-size:0.8rem">OKF first, then RAG</span>
  </div>
  <div style="font-size:1rem;color:#424a53;line-height:1.45">
    Same question, two orders. Left searches first, then attaches the official definition.
    Right looks up the official name first, and searches only if it does not know the name.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Runtime")
    try:
        store = load_store()
        retriever = OKFRetriever(store)
        st.success(f"Qdrant connected · {store.collection_point_count()} points")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    sidebar_key = st.text_input("Google AI Studio key", type="password", help="Session only. Not written to disk.")
    active_key = sidebar_key.strip() or Config.GEMINI_API_KEY
    if active_key:
        st.caption("Gemini judge is live.")
    else:
        st.warning("No Gemini key. Judge will simulate scores.")

legend_l, legend_r = st.columns(2)
with legend_l:
    render_pattern_tag("RAG first, then OKF", "Search → then official definition", "#1f6feb")
    st.markdown(
        "Best for messy wording (“money lost when accounts leave”). "
        "Always searches. Then, if an OKF metric shows up, it loads the real formula."
    )
with legend_r:
    render_pattern_tag("OKF first, then RAG", "Official name → search only if unknown", "#1a7f37")
    st.markdown(
        "Best when the user says the metric name (“NRC”, “CAC”). "
        "Looks up the file immediately. Searches only if no name matches."
    )

st.markdown("### Query")
preset = st.selectbox("Preset", PRESETS)
st.caption(PRESET_WHY.get(preset, ""))
custom = st.text_input("Or type a custom query", value="")
query = custom.strip() or preset
run = st.button("Run comparison", type="primary")

if run:
    start_a = time.perf_counter()
    broker = retriever.retrieve_semantic_broker(query)
    latency_a = (time.perf_counter() - start_a) * 1000

    start_b = time.perf_counter()
    gateway = retriever.retrieve_deterministic_gateway(query)
    latency_b = (time.perf_counter() - start_b) * 1000

    metrics_a = RagasEvaluator.evaluate(query, broker["context"], active_key)
    metrics_b = RagasEvaluator.evaluate(query, gateway["context"], active_key)

    st.session_state["last_run"] = {
        "query": query,
        "broker": broker,
        "gateway": gateway,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "latency_a": latency_a,
        "latency_b": latency_b,
    }

if "last_run" not in st.session_state:
    st.info("Choose a preset and run the comparison to execute both pipelines on the same query.")
    st.stop()

result = st.session_state["last_run"]
broker = result["broker"]
gateway = result["gateway"]
st.caption(f"Last query: {result['query']}")

broker_steps = [
    ("1. Hybrid search", True),
    ("2. Intercept OKF", broker["route"] == "okf_intercept"),
    ("3. Crawl graph", bool(broker["crawled_nodes"])),
]
gateway_hit = gateway["route"] == "okf_hit"
gateway_steps = [
    ("1. Alias / tag route", True),
    ("2. Bypass search" if gateway_hit else "2. Miss → RAG fallback", True),
    ("3. Hydrate graph", bool(gateway["crawled_nodes"])),
]

faster = gateway_hit and result["latency_b"] < result["latency_a"]
delta_b = f"-{(result['latency_a'] - result['latency_b']):.1f} ms vs Broker" if faster else "Fallback or similar cost"

col_a, col_b = st.columns(2)
with col_a:
    render_stepper(broker_steps)
    render_column(
        "RAG first, then OKF",
        "Search the whole catalog. If an official metric ranks, attach that OKF file.",
        broker,
        result["metrics_a"],
        result["latency_a"],
        None,
        kind="RAG first, then OKF",
        family="Search → official definition",
        color="#1f6feb",
    )
with col_b:
    render_stepper(gateway_steps)
    render_column(
        "OKF first, then RAG",
        "Match the metric name. Skip search on a hit. Search only when the name is unknown.",
        gateway,
        result["metrics_b"],
        result["latency_b"],
        delta_b,
        kind="OKF first, then RAG",
        family="Official name → search if needed",
        color="#1a7f37",
    )

prec_a = float(result["metrics_a"]["context_precision"])
prec_b = float(result["metrics_b"]["context_precision"])
if gateway_hit and prec_b >= prec_a:
    verdict = "Gateway won precision and skipped Qdrant — known KPI language should take this path in production."
elif broker["route"] == "okf_intercept" and not gateway_hit:
    verdict = "Broker recovered a paraphrase the alias map missed. Vector search acted as the fuzzy translator into OKF."
elif prec_a > prec_b:
    verdict = "Broker context scored higher this run. Inspect traces for pollution vs intercept."
else:
    verdict = "Compare traces: Gateway fallback and Broker search can both surface warehouse or Slack noise."

st.success(verdict)

st.markdown("### What this query actually did")
st.caption("Bright boxes ran. Grey boxes were skipped. ★ in the charts = RAG kept that document.")
components.html(pipeline_flow_html(result["query"], broker, gateway), height=340, scrolling=False)

retrieved_ids = {int(row["id"]) for row in broker.get("score_rows") or []}
rows = neighborhood(store, result["query"], retrieved_ids)

st.markdown("##### Why RAG picked these documents")
st.caption("Two simple lists (meaning vs exact words), then one combined bar. ★ = kept in the prompt.")
components.html(retrieval_explain_html(result["query"], rows), height=320, scrolling=False)
st.dataframe(neighbor_frame(rows), hide_index=True, width="stretch")
st.info(neighborhood_inference(rows, gateway_hit))
