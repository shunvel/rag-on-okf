from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from src.database import KnowledgeStore

TYPE_COLORS = {
    "okf_concept": "#1f6feb",
    "warehouse_fact": "#9a6700",
    "rag_unstructured": "#cf222e",
}

TYPE_LABELS = {
    "okf_concept": "OKF",
    "warehouse_fact": "Warehouse",
    "rag_unstructured": "Slack / ticket",
}

__all__ = [
    "neighbor_frame",
    "neighborhood",
    "neighborhood_inference",
    "pipeline_flow_html",
    "retrieval_explain_html",
]


def pretty_label(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    kind = row.get("type")
    if kind == "okf_concept":
        return title
    if kind == "warehouse_fact":
        return f"Numbers {title}"
    if "Slack" in text:
        return "Slack: " + _clip(_after_colon(text), 34)
    if re.search(r"[Tt]icket", text):
        head = text.split(":")[0].strip()
        return _clip(head.replace("Support ", "").replace("Customer ", ""), 40)
    if "Disclosure" in text:
        return "Q&A: billing errors"
    if "All-hands" in text:
        return "All-hands: ARPU mix-up"
    if "PagerDuty" in text or "Engineering" in text:
        return _clip(text.split(":")[0], 40)
    return _clip(title or text, 40)


def _after_colon(text: str) -> str:
    return text.split(":", 1)[-1].strip()


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def neighborhood(store: KnowledgeStore, query: str, retrieved_ids: set[int]) -> list[dict[str, Any]]:
    rows = []
    for scored in store.hybrid_scores(query):
        item = {
            **scored,
            "type_label": TYPE_LABELS.get(scored.get("type", ""), scored.get("type", "")),
            "color": TYPE_COLORS.get(scored.get("type", ""), "#8c959f"),
            "cosine": float(scored.get("dense_score", 0.0)),
            "bm25": float(scored.get("bm25_score", 0.0)),
            "rrf": float(scored.get("rrf_score", 0.0)),
            "retrieved": int(scored["id"]) in retrieved_ids,
        }
        item["label"] = pretty_label(item)
        rows.append(item)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def neighbor_frame(rows: list[dict[str, Any]], limit: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "#": row["rank"],
                "Kept by RAG": "yes" if row["retrieved"] else "—",
                "Kind": row["type_label"],
                "Document": row["label"],
                "Meaning #": row["dense_rank"],
                "Words #": row["bm25_rank"],
                "RRF": round(row["rrf"], 4),
            }
            for row in rows[:limit]
        ]
    )


def neighborhood_inference(rows: list[dict[str, Any]], gateway_bypassed: bool) -> str:
    if not rows:
        return "No Qdrant documents to compare."
    nearest = rows[0]
    retrieved = [row for row in rows if row["retrieved"]]
    noise = [row for row in retrieved if row["type"] != "okf_concept"]
    okf = [row for row in retrieved if row["type"] == "okf_concept"]
    bits = [
        f"**{nearest['label']}** is closest overall "
        f"(meaning #{nearest['dense_rank']}, words #{nearest['bm25_rank']})."
    ]
    if okf:
        bits.append("RAG found official OKF: " + ", ".join(f"**{row['label']}**" for row in okf) + ".")
    if noise:
        bits.append("It also kept chatter: " + ", ".join(row["label"] for row in noise) + ".")
    if gateway_bypassed:
        bits.append("The OKF column skipped this list — it already knew the metric name.")
    else:
        bits.append("The OKF column did not recognize a name, so it searched this same list.")
    return " ".join(bits)


def retrieval_explain_html(query: str, rows: list[dict[str, Any]], limit: int = 5) -> str:
    docs = [
        {
            "name": row["label"],
            "kind": row["type_label"],
            "color": row["color"],
            "v": int(row["dense_rank"]),
            "b": int(row["bm25_rank"]),
            "rrf": round(float(row["rrf"]), 4),
            "used": bool(row["retrieved"]),
        }
        for row in rows[:limit]
    ]
    payload = json.dumps({"query": query, "docs": docs})
    template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; color: #1f2328; }
    .wrap { display: grid; grid-template-columns: 1fr 1.15fr; gap: 16px; }
    .card { border: 1px solid #d0d7de; border-radius: 12px; padding: 14px; background: #fff; }
    h3 { margin: 0 0 6px; font-size: 15px; }
    p.sub { margin: 0 0 10px; font-size: 12px; color: #656d76; line-height: 1.4; }
    .lists { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    ol { margin: 0; padding-left: 22px; }
    li { margin: 0 0 8px; font-size: 13px; line-height: 1.3; }
    .kept { font-weight: 700; }
    .legend { font-size: 11px; color: #656d76; margin-top: 8px; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 4px 0 10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h3>Two search lists</h3>
      <p class="sub">Same documents, two orderings. Left = “what it means”. Right = “exact words”. Bold = RAG kept it.</p>
      <div class="lists">
        <div>
          <strong>By meaning</strong>
          <ol id="listV"></ol>
        </div>
        <div>
          <strong>By words</strong>
          <ol id="listB"></ol>
        </div>
      </div>
    </div>
    <div class="card">
      <h3>Combined winner (RRF)</h3>
      <p class="sub">We add the two place-in-line scores. Longer bar = better joint rank. Bold outline = kept in the prompt.</p>
      <svg id="bars" width="100%" viewBox="0 0 520 220"></svg>
    </div>
  </div>
  <div class="legend">
    <span class="dot" style="background:#1f6feb;margin-left:0"></span>Official OKF
    <span class="dot" style="background:#cf222e"></span>Slack / ticket
    <span class="dot" style="background:#9a6700"></span>Warehouse numbers
  </div>
  <script>
    const docs = __DATA__.docs;
    const li = (d) => `<li class="${d.used ? "kept" : ""}" style="color:${d.color}">${d.used ? "★ " : ""}${d.name}</li>`;
    document.getElementById("listV").innerHTML = [...docs].sort((a,b) => a.v - b.v).map(li).join("");
    document.getElementById("listB").innerHTML = [...docs].sort((a,b) => a.b - b.b).map(li).join("");

    const bars = d3.select("#bars");
    const maxR = d3.max(docs, d => d.rrf) || 0.01;
    docs.forEach((d, i) => {
      const y = 18 + i * 40;
      const w = 12 + (d.rrf / maxR) * 250;
      bars.append("text").attr("x", 0).attr("y", y + 4).attr("font-size", 12)
        .attr("font-weight", d.used ? 700 : 500).text((d.used ? "★ " : "") + d.name);
      bars.append("rect").attr("x", 200).attr("y", y - 9).attr("rx", 7)
        .attr("width", w).attr("height", 18)
        .attr("fill", d.color).attr("opacity", d.used ? 1 : 0.35)
        .attr("stroke", d.used ? "#1f2328" : "none");
      bars.append("text").attr("x", 208 + w).attr("y", y + 4).attr("font-size", 11)
        .attr("fill", "#656d76").text("meaning #" + d.v + " · words #" + d.b);
    });
  </script>
</body>
</html>"""
    return template.replace("__DATA__", payload)


def pipeline_flow_html(query: str, broker: dict[str, Any], gateway: dict[str, Any]) -> str:
    payload = {
        "query": query,
        "ragHits": [pretty_label(row) for row in broker.get("score_rows") or []],
        "ragRoute": broker.get("route"),
        "ragCrawled": broker.get("crawled_nodes") or [],
        "okfRoute": gateway.get("route"),
        "okfCrawled": gateway.get("crawled_nodes") or [],
        "okfBypass": gateway.get("route") == "okf_hit",
        "okfHits": [pretty_label(row) for row in gateway.get("score_rows") or []],
    }
    data = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; color: #1f2328; }}
    .hint {{ font-size: 12px; color: #656d76; margin: 0 0 8px 4px; }}
  </style>
</head>
<body>
  <p class="hint">What ran for this query. Bright boxes happened. Grey boxes were skipped.</p>
  <svg id="flow" width="100%" viewBox="0 0 1180 320"></svg>
  <script>
    const data = {data};
    const svg = d3.select("#flow");
    function boxes(y, items) {{
      const w = 150, h = 52, gap = 16; let x = 20;
      return items.map(item => {{ const b = Object.assign({{x, y, w, h}}, item); x += w + gap; return b; }});
    }}
    function draw(label, color, y, items) {{
      svg.append("text").attr("x", 20).attr("y", y - 22).attr("font-size", 12).attr("font-weight", 700)
        .attr("fill", color).text(label);
      const laid = boxes(y, items);
      laid.forEach((n, i) => {{
        if (i < laid.length - 1) {{
          svg.append("line").attr("x1", n.x + n.w).attr("y1", y + 26).attr("x2", laid[i+1].x - 6).attr("y2", y + 26)
            .attr("stroke", n.on && laid[i+1].on ? color : "#d0d7de").attr("stroke-width", 2);
        }}
        svg.append("rect").attr("x", n.x).attr("y", n.y).attr("rx", 10).attr("width", n.w).attr("height", n.h)
          .attr("fill", n.on ? (n.warn ? "#fff8c5" : "#fff") : "#f6f8fa")
          .attr("stroke", n.on ? (n.warn ? "#9a6700" : color) : "#d0d7de").attr("stroke-width", n.on ? 2.2 : 1);
        svg.append("text").attr("x", n.x + 10).attr("y", n.y + 20).attr("font-size", 12).attr("font-weight", 700)
          .attr("fill", n.on ? "#1f2328" : "#8c959f").text(n.label);
        svg.append("text").attr("x", n.x + 10).attr("y", n.y + 38).attr("font-size", 10).attr("fill", "#656d76")
          .text(n.sub);
      }});
    }}
    const hits = (data.ragHits || []).join(", ") || "none";
    draw("RAG · search, then intercept OKF", "#1f6feb", 58, [
      {{ label: "Query", sub: data.query.slice(0, 40), on: true }},
      {{ label: "Vector + BM25", sub: "merged with RRF", on: true }},
      {{ label: "Top hits", sub: hits.slice(0, 40), on: true }},
      {{ label: "OKF intercept", sub: data.ragRoute === "okf_intercept" ? "yes" : "no", on: data.ragRoute === "okf_intercept" }},
      {{ label: "Linked metric", sub: (data.ragCrawled || []).join(" → ").slice(0, 40) || "none", on: (data.ragCrawled || []).length > 0 }},
      {{ label: "Prompt", sub: "context window", on: true }},
    ]);
    draw("OKF · name lookup, search only if needed", "#1a7f37", 210, [
      {{ label: "Query", sub: data.query.slice(0, 40), on: true }},
      {{ label: "Alias check", sub: "title / NRC / CAC", on: true }},
      {{ label: data.okfBypass ? "HIT — skip search" : "MISS", sub: data.okfBypass ? "no Qdrant call" : "unknown wording", on: true, warn: !data.okfBypass }},
      {{ label: data.okfBypass ? "Load OKF file" : "Same RRF search", sub: data.okfBypass ? (data.okfCrawled[0] || "node") : ((data.okfHits || []).join(", ").slice(0, 36) || "hybrid"), on: true }},
      {{ label: "Linked metric", sub: (data.okfCrawled || []).join(" → ").slice(0, 40) || "none", on: (data.okfCrawled || []).length > 0 }},
      {{ label: "Prompt", sub: data.okfBypass ? "clean definition" : "may include noise", on: true }},
    ]);
  </script>
</body>
</html>"""
