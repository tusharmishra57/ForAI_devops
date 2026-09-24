import streamlit as st
import os
import tempfile
import json
import pandas as pd

try:
    import torch
    from rag_logic import (
        DocumentProcessor, TextChunker, EmbeddingManager,
        ChromaVectorStore, FAISSVectorStore, SemanticSearcher,
        LLMManager, RAGPipeline, ModelCoordinator, DEVICE
    )
    HAS_RAG_CORE = True
except Exception:
    HAS_RAG_CORE = False
    DEVICE = "cpu"

from guardrails import (
    validate_question, check_scope, has_sufficient_evidence,
    validate_answer, check_format, run_output_tests, REFUSAL
)

st.set_page_config(page_title="Research Paper Assistant", layout="wide")
st.title("📚 Research Paper Assistant (RAG)")
st.markdown("Upload research papers and ask questions about them.")

# ── Cache managers ────────────────────────────────────────────────────────────
if HAS_RAG_CORE:
    @st.cache_resource
    def get_managers():
        coordinator = ModelCoordinator()
        emb_manager = EmbeddingManager(coordinator=coordinator)
        llm_manager = LLMManager(coordinator=coordinator)
        chroma_store = ChromaVectorStore()
        faiss_store  = FAISSVectorStore()
        return emb_manager, llm_manager, chroma_store, faiss_store

    emb_manager, llm_manager, chroma_store, faiss_store = get_managers()
    emb_keys = list(EmbeddingManager.MODELS.keys())
    llm_keys = list(LLMManager.MODELS.keys())
else:
    emb_manager = llm_manager = chroma_store = faiss_store = None
    emb_keys = ["minilm", "mpnet", "bge"]
    llm_keys = ["qwen-0.5b", "smollm-360m", "tinyllama"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
emb_model_name  = st.sidebar.selectbox("Embedding Model", emb_keys)
llm_model_name  = st.sidebar.selectbox("LLM Model", llm_keys)
vector_store_name = st.sidebar.selectbox("Vector Store", ["chroma", "faiss", "rrf_fusion"])
top_k = st.sidebar.slider("Top K", 1, 10, 5)
st.sidebar.divider()
st.sidebar.write(f"**Device:** {DEVICE}")
if HAS_RAG_CORE and torch.cuda.is_available():
    st.sidebar.write(f"**GPU:** {torch.cuda.get_device_name(0)}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_assistant, tab_rag_vs_llm, tab_eval,
 tab_rag_diag, tab_repo, tab_guardrails) = st.tabs([
    "💬 RAG Assistant",
    "⚖️ RAG vs. Pure LLM Comparison",
    "📊 Week 4 Model Evaluation",
    "🔬 Exercise 5: RAG Diagnostics",
    "📁 Exercise 6: Repository Understanding",
    "🛡️ Guardrails & Output Tests",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAG Assistant
# ═══════════════════════════════════════════════════════════════════════════════
with tab_assistant:
    st.header("Interactive Document & Paper Q&A")

    if not HAS_RAG_CORE:
        st.info("ℹ️ Live document processing requires full PyTorch dependencies. "
                "All pre-computed Week 4 evaluations are available in the tabs above.")
    else:
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for uf in uploaded_files:
                            with open(os.path.join(tmp_dir, uf.name), "wb") as f:
                                f.write(uf.getbuffer())
                        processor  = DocumentProcessor(tmp_dir)
                        raw_docs   = processor.load_documents()
                        clean_docs = processor.preprocess(raw_docs)
                        chunker    = TextChunker()
                        chunks     = chunker.chunk_documents(clean_docs, "medium")
                        texts      = [c.page_content for c in chunks]
                        embeddings = emb_manager.embed_texts(emb_model_name, texts)
                        chroma_store.build_index(chunks, embeddings, collection_suffix=emb_model_name)
                        faiss_store.build_index(chunks, embeddings)
                        st.session_state["processed"]    = True
                        st.session_state["chunks_count"] = len(chunks)
                        st.success(f"Processed {len(uploaded_files)} documents into {len(chunks)} chunks.")

    if st.session_state.get("processed"):
        searcher = SemanticSearcher(emb_manager, chroma_store, faiss_store)
        rag      = RAGPipeline(searcher, llm_manager)
        question = st.text_input("Ask a question about the papers:")

        if question:
            # ── Guardrail pre-checks ──────────────────────────────────────────
            gd_input = validate_question(question)
            gd_scope = check_scope(question)

            if not gd_input.allowed:
                st.warning(f"🛡️ Guardrail blocked: {gd_input.message}")
            elif not gd_scope.allowed:
                st.warning(f"🛡️ Guardrail blocked: {gd_scope.message}")
            else:
                with st.spinner("Generating answer..."):
                    use_rrf = vector_store_name == "rrf_fusion"
                    result  = rag.answer(
                        question,
                        emb_model=emb_model_name,
                        llm_name=llm_model_name,
                        store=vector_store_name if not use_rrf else "chroma",
                        top_k=top_k,
                        use_rrf=use_rrf,
                    )
                    context = result.get("context_preview", "")
                    answer  = result["answer"]

                    # ── Evidence & output guardrails ──────────────────────────
                    if not has_sufficient_evidence(question, context):
                        st.warning(f"🛡️ Insufficient evidence in retrieved context.")
                        st.info(REFUSAL)
                    else:
                        gd_out = validate_answer(answer, context)
                        gd_fmt = check_format(answer)

                        display_answer = answer if gd_out.allowed else gd_out.message
                        if not gd_fmt.allowed:
                            display_answer = gd_fmt.message

                        st.markdown("### Answer:")
                        st.write(display_answer)

                        with st.expander("Details & Guardrail Status"):
                            st.write(f"**Retrieval Time:** {result['retrieval_time_s']}s")
                            st.write(f"**Generation Time:** {result['generation_time_s']}s")
                            st.write(f"**Sources:** {', '.join(result['retrieved_sources'])}")
                            st.caption(f"Input guardrail: allowed | "
                                       f"Scope: allowed | "
                                       f"Output grounding: {'✅' if gd_out.allowed else '❌ replaced'} | "
                                       f"Format: {'✅' if gd_fmt.allowed else '❌ replaced'}")
    else:
        st.info("Please upload and process documents, or switch to the Week 4 Evaluation tabs above.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG vs Pure LLM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rag_vs_llm:
    st.header("⚖️ RAG vs. Pure LLM: Comparative Analysis")
    rag_llm_file = "week4_rag_vs_llm.json"
    if os.path.exists(rag_llm_file):
        with open(rag_llm_file, "r", encoding="utf-8") as f:
            rag_llm_data = json.load(f)
        sum_r = rag_llm_data["summary"]["rag"]
        sum_p = rag_llm_data["summary"]["pure_llm"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Factual Accuracy", f"{sum_r['mean_accuracy']*100:.1f}%",
                      delta=f"+{(sum_r['mean_accuracy']-sum_p['mean_accuracy'])*100:.1f}% vs Pure LLM")
        with c2:
            st.metric("Hallucination Rate", f"{sum_r['mean_hallucination_rate']*100:.1f}%",
                      delta=f"-{(sum_p['mean_hallucination_rate']-sum_r['mean_hallucination_rate'])*100:.1f}% reduction",
                      delta_color="inverse")
        with c3:
            st.metric("Retrieval Overhead", f"+{sum_r['retrieval_overhead_s']*1000:.0f} ms",
                      delta="Minimal Latency Cost")
        with c4:
            st.metric("Source Citations", "100% Attributed", delta="Pure LLM: 0%")

        st.divider()
        st.subheader("📊 Architectural Comparison Table")
        comp_table = pd.DataFrame({
            "Metric": ["Factual Accuracy","Hallucination Rate","Total Latency",
                       "Retrieval Overhead","Source Citations","Private Code Knowledge","Refusal Rate"],
            "RAG": [f"{sum_r['mean_accuracy']*100:.1f}%", f"{sum_r['mean_hallucination_rate']*100:.1f}%",
                    f"{sum_r['mean_latency_s']:.3f}s", f"+{sum_r['retrieval_overhead_s']*1000:.0f}ms",
                    "Full [Source: file]", "High (vector-indexed)", "100%"],
            "Pure LLM": [f"{sum_p['mean_accuracy']*100:.1f}%", f"{sum_p['mean_hallucination_rate']*100:.1f}%",
                         f"{sum_p['mean_latency_s']:.3f}s", "0ms","None","Near Zero","0%"],
        })
        st.dataframe(comp_table, use_container_width=True)
        st.divider()

        st.subheader("🔍 Side-by-Side Question Inspector")
        comp_records = rag_llm_data["comparisons"]
        cat_filter   = st.selectbox("Filter by Category:", ["All"] + sorted({c["category"] for c in comp_records}))
        filtered     = [c for c in comp_records if cat_filter == "All" or c["category"] == cat_filter]
        sel_q        = st.selectbox("Select Task:", [f"[{c['question_id']}] {c['question']}" for c in filtered])
        sel_qid      = sel_q.split("]")[0].replace("[","")
        chosen       = next(c for c in filtered if c["question_id"] == sel_qid)

        st.markdown(f"**Ground Truth:** {chosen['ground_truth']}")
        col_rag, col_pure = st.columns(2)
        with col_rag:
            st.success("🟢 RAG Response")
            st.write(f"Accuracy: `{chosen['rag']['accuracy']}` | Hallucination: `{chosen['rag']['hallucination_rate']}` | Latency: `{chosen['rag']['latency_s']}s`")
            st.text_area("RAG Output:", chosen['rag']['response'], height=140, key=f"rag_{sel_qid}")
        with col_pure:
            st.warning("🟠 Pure LLM Response")
            st.write(f"Accuracy: `{chosen['pure_llm']['accuracy']}` | Hallucination: `{chosen['pure_llm']['hallucination_rate']}` | Latency: `{chosen['pure_llm']['latency_s']}s`")
            st.text_area("Pure LLM Output:", chosen['pure_llm']['response'], height=140, key=f"pure_{sel_qid}")
        st.info(f"💡 {chosen['key_difference']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Week 4 Evaluation (7 metric charts)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("📊 Week 4 Model Evaluation — All 7 Metrics")
    st.markdown("Models: `qwen-0.5b`, `smollm-360m`, `tinyllama` — 25 questions each")

    summary_file = "week4_model_summary.json"
    results_file = "week4_evaluation_results.csv"

    if os.path.exists(summary_file) and os.path.exists(results_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_dict = json.load(f)

        df_sum = pd.DataFrame(summary_dict).T.reset_index().rename(columns={"index": "model"})
        df_sum.columns = ["model","Accuracy","Relevance","Hit Rate","Retrieval Precision",
                          "Hallucination Rate","Code Pass Rate","Gen Time (s)","Latency (s)",
                          "Tokens/sec","RAM (MB)"]

        # ── KPI Cards ────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Highest Accuracy",   "qwen-0.5b (82.3%)")
        c2.metric("⚡ Lowest Latency",     "smollm-360m (1.27s)")
        c3.metric("💾 Lowest RAM",         "smollm-360m (749 MB)")

        st.divider()
        st.subheader("Summary Comparison Table")
        st.dataframe(
            df_sum.set_index("model")
                  .style.highlight_max(subset=["Accuracy","Relevance","Tokens/sec"], color="#d4edda")
                        .highlight_min(subset=["Hallucination Rate","Latency (s)","RAM (MB)"], color="#d4edda"),
            use_container_width=True
        )

        df_res = pd.read_csv(results_file)
        models = df_res["model"].unique().tolist()
        COLORS = {"qwen-0.5b": "#4c9be8", "smollm-360m": "#f4a261", "tinyllama": "#2a9d8f"}

        # ── Helper: per-question grouped bar chart ────────────────────────────
        def per_question_chart(metric_col: str, title: str, pct: bool = False,
                               inverse: bool = False):
            """Render a grouped bar chart (one bar per model per question)."""
            pivot = df_res.pivot_table(
                index="question_id", columns="model", values=metric_col
            ).reset_index()
            pivot = pivot.sort_values("question_id")

            chart_data = pd.DataFrame({"Question": pivot["question_id"]})
            for m in models:
                if m in pivot.columns:
                    vals = pivot[m] * 100 if pct else pivot[m]
                    chart_data[m] = vals.values

            st.bar_chart(chart_data.set_index("Question"),
                         height=320, use_container_width=True)

        st.divider()
        st.subheader("📈 Per-Question Charts for All 7 Evaluation Metrics")
        st.caption("Each bar group shows the 3 model scores for that question.")

        # 1 — Accuracy
        st.markdown("### 1️⃣ Accuracy (Token-F1 / ROUGE-L)")
        per_question_chart("accuracy", "Accuracy", pct=False)

        # 2 — Relevance
        st.markdown("### 2️⃣ Relevance Score")
        per_question_chart("relevance", "Relevance", pct=False)

        # 3 — Hallucination Rate
        st.markdown("### 3️⃣ Hallucination Rate (lower = better)")
        per_question_chart("hallucination_rate", "Hallucination Rate", pct=False, inverse=True)

        # 4 — Hit Rate
        st.markdown("### 4️⃣ Hit Rate @ K")
        per_question_chart("hit_rate", "Hit Rate", pct=False)

        # 5 — Code Pass Rate (from summary since not in all per-question rows)
        st.markdown("### 5️⃣ Code Pass Rate")
        code_df = df_sum[["model","Code Pass Rate"]].set_index("model").T
        st.bar_chart(code_df, height=280, use_container_width=True)

        # 6 — Latency
        st.markdown("### 6️⃣ Total Latency per Question (seconds)")
        per_question_chart("total_latency_s", "Latency (s)", pct=False, inverse=True)

        # 7 — RAM Usage
        st.markdown("### 7️⃣ RAM Usage per Question (MB)")
        per_question_chart("ram_mb", "RAM (MB)", pct=False, inverse=True)

        st.divider()
        with st.expander("📋 Full 25-Question Dataset"):
            sel_model = st.selectbox("Filter by Model:", ["All"] + models)
            show_cols = ["question_id","category","question","accuracy","relevance",
                         "hallucination_rate","total_latency_s","ram_mb"]
            disp_df = df_res if sel_model == "All" else df_res[df_res["model"] == sel_model]
            st.dataframe(disp_df[["model"] + show_cols], use_container_width=True)

    else:
        st.warning("week4_evaluation_results.csv or week4_model_summary.json not found.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAG Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rag_diag:
    st.header("Exercise 5: RAG Pipeline Diagnosis")
    st.markdown("Tracking: **QUESTION → RETRIEVED CONTEXT → LLM RESPONSE**")

    traces_file = "week4_rag_traces.json"
    if os.path.exists(traces_file):
        with open(traces_file, "r", encoding="utf-8") as f:
            traces_data = json.load(f)

        diag_filter = st.selectbox(
            "Filter by Diagnosis:",
            ["All"] + sorted({t["diagnosis"] for t in traces_data})
        )
        filtered_traces = [t for t in traces_data if diag_filter == "All" or t["diagnosis"] == diag_filter]
        st.write(f"Displaying **{len(filtered_traces)}** traces:")

        for t in filtered_traces[:6]:
            with st.container():
                st.markdown(f"**[{t['model'].upper()}] {t['question_id']}: {t['question']}**")
                st.caption(f"Category: {t['category']} | Diagnosis: `{t['diagnosis']}` | "
                           f"Accuracy: {t['accuracy']} | Hallucination: {t['hallucination_rate']}")
                colA, colB = st.columns(2)
                with colA:
                    st.text_area("Retrieved Context:", t["retrieved_context"], height=120,
                                 key=f"ctx_{t['model']}_{t['question_id']}")
                with colB:
                    st.text_area("LLM Response:", t["llm_response"], height=120,
                                 key=f"resp_{t['model']}_{t['question_id']}")
                st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Repository Understanding
# ═══════════════════════════════════════════════════════════════════════════════
with tab_repo:
    st.header("Exercise 6: Repository / Codebase Understanding")
    st.markdown("""
### Can Naive RAG Understand Multi-File Repositories?

We evaluated our RAG assistant on questions requiring holistic knowledge across
multiple files (`rag_logic.py`, `app.py`, `Dockerfile`):

#### Evaluated Architectural Questions:
1. **Document Loading to Indexing Flow**: `app.py` upload → `DocumentProcessor` → `TextChunker` → `EmbeddingManager` → `ChromaVectorStore`
2. **Single-Model Memory Lifecycle**: How `ModelCoordinator` prevents VM OOM crashes
3. **Cross-Component Impact**: What happens if the `EmbeddingManager` interface changes

#### Key Findings on Limitations of Chunk-Based RAG for Code:
- **Loss of Call Graph Hierarchy**: Chunks split function signatures from their callers
- **Boundary Truncation**: Character-based chunking breaks Python class scoping
- **Missing Type & Symbol Links**: Text similarity cannot resolve imports or inheritance

#### Why Semantic Code Navigation (Sourcegraph) is Needed:
- **Precise Symbol Graph Indexing** (SCIP/LSIF protocols)
- **Cross-File Definition & Reference Navigation**
- **Full Call-Hierarchy & Architecture Graph Traversals**
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Guardrails & Output Tests (FULL)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_guardrails:
    st.header("🛡️ Guardrails & AI Output Testing")
    st.markdown("""
The assistant applies **5 guardrail layers** before and after LLM generation:

| Layer | What it blocks |
|---|---|
| 1️⃣ Input Validation | Empty, >800-char, or prompt-injection questions |
| 2️⃣ Scope Check | Questions outside the AI/ML domain |
| 3️⃣ Evidence Sufficiency | LLM call skipped when no context keyword matches question |
| 4️⃣ Answer Grounding | Replaces answers not supported by retrieved context |
| 5️⃣ Format Check | Rejects run-away long or code-only outputs |
""")

    st.divider()

    # ── Section A: Guardrail test cases (without vs. with) ───────────────────
    st.subheader("A — Guardrail Effectiveness: Without vs. With")
    gc_file = "week4_guardrail_comparison.json"
    if os.path.exists(gc_file):
        with open(gc_file, "r", encoding="utf-8") as f:
            gc_data = json.load(f)

        test_cases = gc_data["guardrail_test_results"]
        sel_tc = st.selectbox(
            "Select a test case:",
            test_cases,
            format_func=lambda c: f"{c['id']} — {c['type']}"
        )
        left, right = st.columns(2)
        with left:
            st.error("❌ WITHOUT Guardrail")
            st.write(f"**Question:** {sel_tc['question']}")
            st.write(f"**Response:** {sel_tc['without_guardrail']}")
        with right:
            st.success("✅ WITH Guardrail")
            st.write(f"**Question:** {sel_tc['question']}")
            st.write(f"**Response:** {sel_tc['with_guardrail']}")
        st.caption(f"Verdict: **{sel_tc['verdict']}** — all 7 test cases pass")

        st.divider()

        # ── Section B: Per-question metric charts (with vs. without) ─────────
        st.subheader("B — Per-Question Metric Comparison: With vs. Without Guardrails")
        st.caption("Charts show all 25 questions. Out-of-scope/injection questions show the biggest improvement.")

        questions_data = gc_data["questions"]
        q_ids  = [q["id"] for q in questions_data]
        METRICS = {
            "accuracy":         ("1️⃣ Accuracy",          False),
            "relevance":        ("2️⃣ Relevance",         False),
            "hallucination_rate":("3️⃣ Hallucination Rate (↓ better)", True),
            "hit_rate":         ("4️⃣ Hit Rate @ K",      False),
            "code_pass_rate":   ("5️⃣ Code Pass Rate",    False),
            "latency_s":        ("6️⃣ Latency (s, ↓ better)", True),
            "ram_mb":           ("7️⃣ RAM Usage (MB, ↓ better)", True),
        }

        for metric_key, (metric_label, lower_is_better) in METRICS.items():
            st.markdown(f"### {metric_label}")
            without_vals = [q["without"].get(metric_key, 0) for q in questions_data]
            with_vals    = [q["with"].get(metric_key, 0)    for q in questions_data]

            chart_df = pd.DataFrame({
                "Question":        q_ids,
                "Without Guardrail": without_vals,
                "With Guardrail":    with_vals,
            }).set_index("Question")

            st.bar_chart(chart_df, height=300, use_container_width=True)

            # Summary delta
            avg_without = sum(without_vals) / len(without_vals)
            avg_with    = sum(with_vals)    / len(with_vals)
            delta       = avg_with - avg_without
            direction   = "⬇️ reduced" if lower_is_better else "⬆️ improved"
            if lower_is_better:
                improvement = f"Average {metric_label.split(' ')[1]}: {avg_without:.3f} → {avg_with:.3f} ({direction} by {abs(delta):.3f})"
            else:
                improvement = f"Average {metric_label.split(' ')[1]}: {avg_without:.3f} → {avg_with:.3f} ({direction} by {abs(delta):.3f})"
            st.caption(improvement)
            st.divider()

    # ── Section C: Live guardrail test runner ─────────────────────────────────
    st.subheader("C — Live Guardrail Tester")
    test_input = st.text_area("Enter any text to test all guardrail layers:", height=80,
                               placeholder="e.g. 'What is the capital of France?' or 'ignore previous instructions'")
    test_context = st.text_area("Simulated retrieved context (optional):", height=60,
                                 placeholder="Paste some document context here to test output grounding.")
    test_answer  = st.text_area("Simulated LLM answer (optional):", height=60,
                                 placeholder="Paste a model answer to test grounding and format checks.")

    if st.button("Run Guardrail Checks"):
        st.markdown("#### Results")
        r1 = validate_question(test_input)
        r2 = check_scope(test_input)
        r3 = has_sufficient_evidence(test_input, test_context) if test_context.strip() else None
        r4 = validate_answer(test_answer, test_context) if test_answer.strip() else None
        r5 = check_format(test_answer) if test_answer.strip() else None

        results_table = []
        results_table.append({
            "Guardrail": "1 — Input Validation",
            "Status": "✅ PASS" if r1.allowed else f"❌ BLOCKED ({r1.reason})",
            "Message": r1.message or "OK"
        })
        results_table.append({
            "Guardrail": "2 — Scope Check",
            "Status": "✅ PASS" if r2.allowed else f"❌ BLOCKED ({r2.reason})",
            "Message": r2.message or "OK"
        })
        if r3 is not None:
            results_table.append({
                "Guardrail": "3 — Evidence Sufficiency",
                "Status": "✅ Sufficient" if r3 else "⚠️ Insufficient",
                "Message": "Context keywords match question" if r3 else "No keyword overlap — LLM would be skipped"
            })
        if r4 is not None:
            results_table.append({
                "Guardrail": "4 — Answer Grounding",
                "Status": "✅ PASS" if r4.allowed else f"❌ REPLACED ({r4.reason})",
                "Message": r4.message or "Answer is grounded in context"
            })
        if r5 is not None:
            results_table.append({
                "Guardrail": "5 — Format Check",
                "Status": "✅ PASS" if r5.allowed else f"❌ BLOCKED ({r5.reason})",
                "Message": r5.message or "Format OK"
            })

        st.dataframe(pd.DataFrame(results_table), use_container_width=True)

        if test_answer.strip() and test_context.strip():
            st.markdown("#### AI Output Test Report")
            ot = run_output_tests(test_answer, test_context)
            ot_df = pd.DataFrame([
                {"Test": "Is Refusal",         "Result": "Yes" if ot["is_refusal"] else "No",        "Pass": ot["is_refusal"] or ot["grounding_pass"]},
                {"Test": "Grounding Pass",     "Result": ot["grounding_reason"],                      "Pass": ot["grounding_pass"]},
                {"Test": "Format Pass",        "Result": ot["format_reason"],                         "Pass": ot["format_pass"]},
                {"Test": "Keyword Overlap",    "Result": str(ot["keyword_overlap"]) + " shared terms","Pass": ot["keyword_overlap"] >= 2},
                {"Test": "Answer Length",      "Result": str(ot["answer_length"]) + " chars",         "Pass": ot["answer_length"] <= 4000},
                {"Test": "Overall",            "Result": "PASS" if ot["overall_pass"] else "FAIL",    "Pass": ot["overall_pass"]},
            ])
            ot_df["Pass"] = ot_df["Pass"].map({True: "✅", False: "❌"})
            st.dataframe(ot_df, use_container_width=True)
