"""Phase 6 — Streamlit UI: a one-page ask-the-corpus demo + the evaluation table.

  streamlit run app.py

Tab 1 (Ask) retrieves and answers with grounded citations (the Phase 5 path).
Tab 2 (Evaluation) shows the Phase 4 deliverable table from eval_runs, split by
question type. The retriever (embedding model + reranker) is loaded once and cached.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rag.db import connect
from rag.generation import generate
from rag.retrieval import Retriever

st.set_page_config(page_title="rag-eval-harness", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


st.title("rag-eval-harness")
ask, evaltab = st.tabs(["Ask", "Evaluation table"])

with ask:
    st.caption("Retrieve over the corpus and answer with citations grounded in the retrieved chunks.")
    q = st.text_input("Question", placeholder="e.g. What discount does a CCP apply to collateral?")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("retrieval mode", ["vector_rerank", "vector", "hybrid", "hybrid_rerank", "lexical"])
    size = c2.selectbox("chunk size (tokens)", [500, 250, 1000])
    if st.button("Answer", type="primary") and q.strip():
        r = get_retriever()
        with connect() as conn:
            r.configure_session(conn)
            rid = conn.execute("SELECT id FROM ingestion_runs WHERE chunk_size=%s LIMIT 1", (size,)).fetchone()[0]
            hits = getattr(r, mode)(conn, q, rid)[:5]
            if not hits:
                st.warning("No chunks retrieved.")
            else:
                rows = {cid: (fn, pg, txt) for cid, fn, pg, txt in conn.execute(
                    "SELECT c.id, d.filename, c.page_number, c.text FROM chunks c "
                    "JOIN documents d ON d.id=c.document_id WHERE c.id = ANY(%s)",
                    ([h.chunk_id for h in hits],)).fetchall()}
                with st.spinner("Generating…"):
                    gen = generate(q, [rows[h.chunk_id][2] for h in hits])
                st.markdown(gen.text)
                st.caption(f"model={gen.model} · tokens in={gen.input_tokens} out={gen.output_tokens} "
                           f"· {mode} @ {size}-token chunks")
                st.subheader("Sources")
                for i, h in enumerate(hits, 1):
                    fn, pg, txt = rows[h.chunk_id]
                    with st.expander(f"[{i}] {fn} · p{pg}"):
                        st.write(txt)

with evaltab:
    st.caption("Phase 4 results from eval_runs — split by question type, never pooled. recall shown as rate.")
    with connect() as conn:
        rows = conn.execute(
            "SELECT question_type, chunk_size, retrieval_mode, embedding_model, num_questions, "
            "recall_at_5, recall_at_10, mrr, mean_tokens_at_5, latency_p50_ms "
            "FROM eval_runs ORDER BY question_type, chunk_size, retrieval_mode").fetchall()
    if not rows:
        st.info("No eval_runs yet — run `python scripts/score.py`.")
    else:
        df = pd.DataFrame(rows, columns=["type", "chunk", "mode", "model", "N",
                                         "recall@5", "recall@10", "MRR", "tok@5", "p50_ms"])
        for t in ["cited", "descriptive"]:
            sub = df[df["type"] == t]
            if len(sub):
                st.subheader(t)
                st.dataframe(
                    sub.drop(columns=["type"]).style.format(
                        {"recall@5": "{:.2f}", "recall@10": "{:.2f}", "MRR": "{:.3f}",
                         "tok@5": "{:.0f}", "p50_ms": "{:.1f}"}),
                    use_container_width=True, hide_index=True)
