"""app_ui.py — Streamlit web interface for the Medical KG SPARQL-RAG chatbot.
Usage: streamlit run src/rag/app_ui.py
"""

import json
import os
import sys
import time

import streamlit as st

# Add project root to path so we can import from lab_rag_sparql_gen
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_RAG_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.rag.lab_rag_sparql_gen import (
    MODEL,
    TTL_FILE,
    _extract_sparql_block,
    answer_no_rag,
    build_schema_summary,
    generate_answer,
    generate_sparql,
    load_graph,
    run_sparql,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedKG Chatbot",
    layout="wide",
)

# ── Sidebar — graph statistics ────────────────────────────────────────────────

STATS_PATH = os.path.join(_PROJECT_ROOT, "kg_artifacts", "stats.json")

with st.sidebar:
    st.title("MedKG Chatbot")
    st.caption("SPARQL-RAG · Ollama · RDFLib")

    st.divider()
    st.subheader("Knowledge Graph Stats")

    if os.path.isfile(STATS_PATH):
        with open(STATS_PATH, encoding="utf-8") as f:
            stats = json.load(f)
        st.metric("Triplets", f"{stats.get('total_triples', 0):,}")
        st.metric("Entities", f"{stats.get('total_entities', 0):,}")
        st.metric("Relations", f"{stats.get('total_relations', 0):,}")
    else:
        st.warning("stats.json not found in kg_artifacts/")

    st.divider()
    st.subheader("Settings")
    model_name = st.text_input("Ollama model", value=MODEL)
    enable_repair = st.toggle("SPARQL self-repair", value=True)

    st.divider()
    st.caption(f"Graph: `{os.path.basename(TTL_FILE)}`")

# ── Load graph (cached so it only loads once) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge graph...")
def get_graph_and_schema(path: str):
    g = load_graph(path)
    schema = build_schema_summary(g)
    return g, schema

try:
    graph, schema = get_graph_and_schema(TTL_FILE)
except SystemExit:
    st.error(
        f"Could not load the knowledge graph from `{TTL_FILE}`. "
        "Run the MedKG pipeline first to generate the graph files."
    )
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {question, baseline, rag, sparql, repaired}

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Medical Knowledge Graph — SPARQL-RAG Chatbot")
st.caption(
    "Each answer is shown twice: **Baseline** (LLM parametric memory only) "
    "vs **RAG** (Knowledge Graph via SPARQL). "
    "This side-by-side view validates the RAG evaluation described in the report."
)

# Display past conversation
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.write(msg["question"])

    with st.chat_message("assistant"):
        col_baseline, col_rag = st.columns(2)

        with col_baseline:
            st.markdown("#### Baseline — LLM only")
            st.info(msg["baseline"])

        with col_rag:
            st.markdown("#### RAG — Knowledge Graph")
            st.success(msg["rag"])

        with st.expander("Technical details — SPARQL query"):
            if msg["repaired"]:
                st.warning("Self-repair was activated: the first query failed and was corrected by the LLM.")
            else:
                st.success("Self-repair was not needed.")
            st.code(msg["sparql"], language="sparql")

# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a medical question (symptoms, treatments, medications...)")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        col_baseline, col_rag = st.columns(2)

        # --- Baseline (LLM only) ---
        with col_baseline:
            st.markdown("#### Baseline — LLM only")
            with st.spinner("Querying LLM (no graph)..."):
                t0 = time.time()
                baseline_answer = answer_no_rag(question, model=model_name)
                baseline_time = time.time() - t0
            if baseline_answer.startswith("ERROR:"):
                st.error(baseline_answer)
            else:
                st.info(baseline_answer)
            st.caption(f"{baseline_time:.1f}s")

        # --- RAG (Knowledge Graph) ---
        with col_rag:
            st.markdown("#### RAG — Knowledge Graph")
            with st.spinner("Generating SPARQL and querying graph..."):
                t1 = time.time()
                sparql_raw = generate_sparql(question, schema, model=model_name)
                sparql_first = _extract_sparql_block(sparql_raw)
                rows, final_sparql = run_sparql(
                    graph, sparql_first, question, schema,
                    model=model_name, enable_repair=enable_repair,
                )
                rag_answer = generate_answer(question, rows, model=model_name)
                rag_time = time.time() - t1

            # Self-repair detection: the final query differs from the first attempt
            was_repaired = enable_repair and (final_sparql.strip() != sparql_first.strip())

            if rag_answer.startswith("No relevant"):
                st.warning(rag_answer)
            else:
                st.success(rag_answer)
            st.caption(f"{rag_time:.1f}s · {len(rows)} graph result(s)")

        with st.expander("Technical details — SPARQL query"):
            if was_repaired:
                st.warning("Self-repair was activated: the first query failed and was corrected by the LLM.")
            else:
                st.success("Self-repair was not needed.")
            st.code(final_sparql, language="sparql")

    st.session_state.messages.append({
        "question": question,
        "baseline": baseline_answer,
        "rag": rag_answer,
        "sparql": final_sparql,
        "repaired": was_repaired,
    })
