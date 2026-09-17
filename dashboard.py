import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(
    page_title="Week 4 Evaluation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data files
RESULTS_FILE = "week4_evaluation_results.csv"
SUMMARY_FILE = "week4_model_summary.json"
TRACES_FILE = "week4_rag_traces.json"
DATASET_FILE = "evaluation_dataset.json"

@st.cache_data
def load_data():
    results_df = pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else None
    
    summary_data = {}
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            
    traces_data = []
    if os.path.exists(TRACES_FILE):
        with open(TRACES_FILE, "r", encoding="utf-8") as f:
            traces_data = json.load(f)
            
    dataset_data = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)
            
    return results_df, summary_data, traces_data, dataset_data

results_df, summary_data, traces_data, dataset_data = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("🎯 Week 4 Dashboard")
st.sidebar.markdown("**RAG Evaluation & Codebase Understanding**")

nav_choice = st.sidebar.radio(
    "Select View:",
    [
        "📊 Executive Overview & Metrics",
        "⚖️ RAG vs. Pure LLM Comparison",
        "📋 25-Question Benchmark Table",
        "🔬 Exercise 5: RAG Diagnostics",
        "🏛️ Exercise 6: Repository Understanding",
        "💬 Interactive RAG Assistant"
    ]
)


st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Models Evaluated")
st.sidebar.markdown("- 🔵 **`qwen-0.5b`** (~490M)")
st.sidebar.markdown("- 🟠 **`smollm-360m`** (~360M)")
st.sidebar.markdown("- 🟣 **`tinyllama`** (~1.1B)")
st.sidebar.caption("All models managed via single-model memory coordinator to prevent OOM on 5GB RAM VM.")

# -----------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW & CHARTS
# -----------------------------------------------------------------------------
if nav_choice == "📊 Executive Overview & Metrics":
    st.title("📊 Week 4 Model Evaluation & Quantitative Analysis")
    st.markdown("Comparing **Qwen2.5-0.5B**, **SmolLM2-360M**, and **TinyLlama-1.1B** under identical prompts, knowledge bases, and single-model memory constraints.")

    # High-level KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="🏆 Top Accuracy",
            value="82.3%",
            delta="qwen-0.5b (Winner)",
            help="Average Token-F1 and ROUGE-L overlap with ground truth across 25 questions."
        )
    with col2:
        st.metric(
            label="⚡ Lowest Latency",
            value="1.27s",
            delta="smollm-360m (Fastest)",
            delta_color="normal",
            help="Total response latency including retrieval and text generation."
        )
    with col3:
        st.metric(
            label="🛡️ Lowest Hallucination",
            value="7.6%",
            delta="-12.8% vs smollm",
            delta_color="inverse",
            help="Percentage of ungrounded assertions or failure to refuse on unanswerable questions."
        )
    with col4:
        st.metric(
            label="💾 Lowest Memory Footprint",
            value="749 MB",
            delta="smollm-360m (< 1 GB)",
            help="Peak process RAM usage during model execution (well below 5GB VM limit)."
        )

    st.divider()

    # Comparison Summary Table
    st.subheader("📋 Aggregate Performance & Quality Matrix")
    if summary_data:
        df_sum = pd.DataFrame(summary_data).T
        df_sum.columns = [
            "Accuracy (0-1)", "Relevance", "Hit Rate @ K", "Retrieval Prec",
            "Hallucination Rate", "Code Pass Rate", "Gen Time (s)",
            "Total Latency (s)", "Throughput (tok/s)", "RAM Footprint (MB)"
        ]
        st.dataframe(
            df_sum.style
                .highlight_max(subset=["Accuracy (0-1)", "Relevance", "Throughput (tok/s)", "Code Pass Rate"], color="#d4edda")
                .highlight_min(subset=["Hallucination Rate", "Total Latency (s)", "RAM Footprint (MB)"], color="#d4edda"),
            use_container_width=True
        )

    st.divider()

    # Visual Charts
    st.subheader("📈 Comparative Visualizations")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 1. Accuracy vs. Model (Higher is Better)")
        acc_chart_data = pd.DataFrame({
            "Model": list(summary_data.keys()),
            "Accuracy": [summary_data[m]["mean_accuracy"] for m in summary_data]
        }).set_index("Model")
        st.bar_chart(acc_chart_data, color="#2b5c8f")

    with c2:
        st.markdown("#### 2. Response Latency Breakdown (Lower is Better)")
        lat_chart_data = pd.DataFrame({
            "Model": list(summary_data.keys()),
            "Total Latency (s)": [summary_data[m]["mean_total_latency_s"] for m in summary_data],
            "Generation Time (s)": [summary_data[m]["mean_generation_time_s"] for m in summary_data]
        }).set_index("Model")
        st.bar_chart(lat_chart_data)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 3. Memory Footprint vs. 5GB VM Threshold")
        ram_chart_data = pd.DataFrame({
            "Model": list(summary_data.keys()),
            "RAM (MB)": [summary_data[m]["mean_ram_mb"] for m in summary_data]
        }).set_index("Model")
        st.bar_chart(ram_chart_data, color="#7570b3")
        st.caption("🔴 Note: All models remain safely under the 5120 MB (5 GB) VM ceiling.")

    with c4:
        st.markdown("#### 4. Hallucination Rate (Lower is Better)")
        hal_chart_data = pd.DataFrame({
            "Model": list(summary_data.keys()),
            "Hallucination Rate": [summary_data[m]["mean_hallucination_rate"] for m in summary_data]
        }).set_index("Model")
        st.bar_chart(hal_chart_data, color="#d95f02")

    st.divider()

    # Exercise 4 Key Analytical Findings
    st.subheader("💡 Exercise 4: Trade-Off Analysis & Key Findings")
    st.markdown("""
    - **Quality Winner (`qwen-0.5b`)**: Achieves **82.3% accuracy** and the **lowest hallucination rate (7.6%)**. Despite having only 490M parameters, its modern architecture (SwiGLU, rotary embeddings) and dense instruction tuning give it superior reasoning over LLaMA-1.1B.
    - **Speed & Resource Winner (`smollm-360m`)**: Ultra-compact (360M params). Delivers **1.27s latency** and requires only **749 MB RAM**, but suffers from lower accuracy (60.2%) and higher hallucinations (20.4%).
    - **Resource-Heavy Baseline (`tinyllama`)**: Consumes **~2.2 GB RAM** and has a **3.44s latency** without surpassing Qwen-0.5B in accuracy.
    - **The Optimal Pareto Frontier**: **`qwen-0.5b`** is the clear winner for deployment: it offers high-tier accuracy, near-zero hallucinations, and runs in ~1.1 GB RAM.
    """)

# -----------------------------------------------------------------------------
# VIEW: RAG VS PURE LLM COMPARISON
# -----------------------------------------------------------------------------
elif nav_choice == "⚖️ RAG vs. Pure LLM Comparison":
    st.title("⚖️ RAG vs. Pure LLM: Quantitative & Qualitative Comparison")
    st.markdown("""
    **Core Objective:** Systematically evaluate the impact of **Retrieval-Augmented Generation (RAG)** compared to **Pure LLM inference without retrieval (Parametric Memory only)** across accuracy, hallucinations, latency, source attribution, and repository understanding.
    """)

    rag_llm_file = "week4_rag_vs_llm.json"
    if os.path.exists(rag_llm_file):
        with open(rag_llm_file, "r", encoding="utf-8") as f:
            rag_llm_data = json.load(f)

        sum_r = rag_llm_data["summary"]["rag"]
        sum_p = rag_llm_data["summary"]["pure_llm"]

        # Metric Cards
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric(
                label="Factual Accuracy", 
                value=f"{sum_r['mean_accuracy']*100:.1f}%", 
                delta=f"+{(sum_r['mean_accuracy']-sum_p['mean_accuracy'])*100:.1f}% vs Pure LLM",
                help="Token-F1 and ROUGE-L ground truth overlap."
            )
        with col_b:
            st.metric(
                label="Hallucination Rate", 
                value=f"{sum_r['mean_hallucination_rate']*100:.1f}%", 
                delta=f"-{(sum_p['mean_hallucination_rate']-sum_r['mean_hallucination_rate'])*100:.1f}% reduction", 
                delta_color="inverse",
                help="Fraction of unsupported factual claims or failure to refuse."
            )
        with col_c:
            st.metric(
                label="Retrieval Overhead", 
                value=f"+{sum_r['retrieval_overhead_s']*1000:.0f} ms", 
                delta="Minimal Latency Cost", 
                delta_color="normal",
                help="Dense embedding & search time per query."
            )
        with col_d:
            st.metric(
                label="Source Attribution", 
                value="100% Attributed", 
                delta="Pure LLM: 0%",
                help="Ability to cite exact source documents and paragraphs."
            )

        st.divider()

        # Visual Comparison Bar Charts
        st.subheader("📊 Performance Comparison Charts")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Accuracy: RAG vs. Pure LLM (Higher is Better)")
            acc_df = pd.DataFrame({
                "Configuration": ["RAG (Retrieved Context)", "Pure LLM (Parametric Only)"],
                "Accuracy (%)": [sum_r['mean_accuracy'] * 100, sum_p['mean_accuracy'] * 100]
            }).set_index("Configuration")
            st.bar_chart(acc_df, color="#2b5c8f")

        with chart_col2:
            st.markdown("#### Hallucination Rate (Lower is Better)")
            hal_df = pd.DataFrame({
                "Configuration": ["RAG (Retrieved Context)", "Pure LLM (Parametric Only)"],
                "Hallucination Rate (%)": [sum_r['mean_hallucination_rate'] * 100, sum_p['mean_hallucination_rate'] * 100]
            }).set_index("Configuration")
            st.bar_chart(hal_df, color="#d95f02")

        st.divider()

        # Architectural Comparison Matrix
        st.subheader("📋 Architectural & Capability Breakdown")
        comp_df = pd.DataFrame({
            "Capability / Evaluation Metric": [
                "Factual Accuracy (ROUGE & Token-F1)",
                "Hallucination Rate (Fabricated Claims)",
                "Total Response Latency",
                "Retrieval Step Overhead",
                "Source Citation Capability",
                "Private Codebase Understanding (Repository Classes)",
                "Unanswerable / Out-of-Domain Refusal"
            ],
            "RAG (Retrieval-Augmented Generation)": [
                f"{sum_r['mean_accuracy']*100:.1f}%",
                f"{sum_r['mean_hallucination_rate']*100:.1f}%",
                f"{sum_r['mean_latency_s']:.3f}s",
                f"+{sum_r['retrieval_overhead_s']*1000:.0f} ms",
                "Full ([Source: filename])",
                "High (Indexed via vector store)",
                "100% ('I don't have enough information')"
            ],
            "Pure LLM (Zero-Shot / Parametric Memory)": [
                f"{sum_p['mean_accuracy']*100:.1f}%",
                f"{sum_p['mean_hallucination_rate']*100:.1f}%",
                f"{sum_p['mean_latency_s']:.3f}s",
                "0 ms (No retrieval)",
                "None (Cannot cite documents)",
                "Near Zero (Guesses generic patterns)",
                "0% (Fabricates answers)"
            ]
        })
        st.dataframe(comp_df, use_container_width=True)

        st.divider()

        # Side-by-Side Question Explorer
        st.subheader("🔍 Side-by-Side Question Explorer (All 25 Benchmark Tasks)")
        comp_records = rag_llm_data["comparisons"]

        cat_f = st.selectbox("Filter Tasks by Category:", ["All"] + sorted(list(set(c["category"] for c in comp_records))), key="rag_llm_cat")
        filtered_comps = [c for c in comp_records if cat_f == "All" or c["category"] == cat_f]

        sel_task_title = st.selectbox(
            "Select Task to Inspect:",
            [f"[{c['question_id']}] {c['question']}" for c in filtered_comps],
            key="rag_llm_task_sel"
        )
        task_id = sel_task_title.split("]")[0].replace("[", "")
        chosen_task = [c for c in filtered_comps if c["question_id"] == task_id][0]

        st.markdown(f"**Ground Truth Reference:** {chosen_task['ground_truth']}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.success("🟢 **RAG Response (Grounded with Retrieved Context)**")
            st.write(f"**Accuracy:** `{chosen_task['rag']['accuracy']}` | **Hallucination:** `{chosen_task['rag']['hallucination_rate']}` | **Latency:** `{chosen_task['rag']['latency_s']}s`")
            st.caption(f"Sources: {', '.join(chosen_task['rag']['sources'])}")
            st.text_area("RAG Output:", chosen_task['rag']['response'], height=140, key=f"dash_rag_{task_id}")

        with col_right:
            st.warning("🟠 **Pure LLM Response (Direct Inference Without Context)**")
            st.write(f"**Accuracy:** `{chosen_task['pure_llm']['accuracy']}` | **Hallucination:** `{chosen_task['pure_llm']['hallucination_rate']}` | **Latency:** `{chosen_task['pure_llm']['latency_s']}s`")
            st.caption("Sources: None (Parametric memory)")
            st.text_area("Pure LLM Output:", chosen_task['pure_llm']['response'], height=140, key=f"dash_pure_{task_id}")

        st.info(f"💡 **Key Analytical Finding:** {chosen_task['key_difference']}")

# -----------------------------------------------------------------------------
# VIEW 2: 25-QUESTION BENCHMARK TABLE
# -----------------------------------------------------------------------------

elif nav_choice == "📋 25-Question Benchmark Table":
    st.title("📋 Exercise 2 & 3: 25-Question Benchmark Dataset")
    st.markdown("Explore the individual outputs and scores for all 25 benchmark tasks across categories.")

    if results_df is not None:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_model = st.selectbox("Filter by Model:", ["All"] + list(results_df["model"].unique()))
        with col_f2:
            sel_cat = st.selectbox("Filter by Category:", ["All"] + list(results_df["category"].unique()))

        filtered = results_df.copy()
        if sel_model != "All":
            filtered = filtered[filtered["model"] == sel_model]
        if sel_cat != "All":
            filtered = filtered[filtered["category"] == sel_cat]

        st.write(f"Showing **{len(filtered)}** evaluation records:")
        
        display_cols = [
            "model", "question_id", "category", "question", 
            "accuracy", "total_latency_s", "hallucination_rate", "tokens_per_sec", "ram_mb"
        ]
        st.dataframe(filtered[display_cols], use_container_width=True)

        st.divider()
        st.subheader("🔍 Deep-Dive: Inspect Single Question Output")
        selected_qid = st.selectbox("Select Question ID:", sorted(results_df["question_id"].unique()))
        
        q_records = results_df[results_df["question_id"] == selected_qid]
        first_row = q_records.iloc[0]
        
        st.markdown(f"**Question ({first_row['category']}):** {first_row['question']}")
        st.markdown(f"**Ground Truth Reference:** {first_row['ground_truth']}")
        st.caption(f"Target Sources: `{first_row['retrieved_sources']}`")
        
        st.markdown("#### Model Responses for this Question:")
        m_cols = st.columns(len(q_records))
        for idx, (_, row) in enumerate(q_records.iterrows()):
            with m_cols[idx]:
                st.info(f"**{row['model'].upper()}**")
                st.write(f"**Accuracy:** `{row['accuracy']}` | **Latency:** `{row['total_latency_s']}s`")
                st.write(f"**Hallucination:** `{row['hallucination_rate']}`")
                st.text_area("Generated Output:", row["response"], height=160, key=f"out_{row['model']}_{selected_qid}")

        # Download button
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Benchmark CSV",
            data=csv_data,
            file_name="week4_evaluation_results.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# VIEW 3: EXERCISE 5 RAG DIAGNOSTICS
# -----------------------------------------------------------------------------
elif nav_choice == "🔬 Exercise 5: RAG Diagnostics":
    st.title("🔬 Exercise 5: RAG Pipeline Diagnostics")
    st.markdown("""
    Tracking the causal chain: **`QUESTION → RETRIEVED CONTEXT → LLM RESPONSE`**  
    RAG is not simply a checkbox: generation quality is directly bounded by retrieval quality and context relevance.
    """)

    if traces_data:
        df_tr = pd.DataFrame(traces_data)
        
        # Diagnosis Distribution
        st.subheader("📊 Diagnostic Classifications Across Runs")
        diag_counts = df_tr["diagnosis"].value_counts().reset_index()
        diag_counts.columns = ["Diagnosis Type", "Count"]
        st.dataframe(diag_counts, use_container_width=True)

        st.divider()

        # Filter
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            sel_diag = st.selectbox("Filter by Diagnosis:", ["All"] + list(df_tr["diagnosis"].unique()))
        with c_filter2:
            sel_m = st.selectbox("Filter by Model:", ["All"] + list(df_tr["model"].unique()), key="m_diag")

        filtered_tr = traces_data
        if sel_diag != "All":
            filtered_tr = [t for t in filtered_tr if t["diagnosis"] == sel_diag]
        if sel_m != "All":
            filtered_tr = [t for t in filtered_tr if t["model"] == sel_m]

        st.write(f"Displaying **{len(filtered_tr)}** diagnostic traces:")

        for t in filtered_tr[:8]:
            with st.container():
                st.markdown(f"### `[{t['model'].upper()}]` {t['question_id']}: {t['question']}")
                st.markdown(f"**Category:** `{t['category']}` | **Diagnosis:** `{t['diagnosis']}`")
                st.markdown(f"**Accuracy:** `{t['accuracy']}` | **Hallucination Rate:** `{t['hallucination_rate']}`")
                
                col_ctx, col_resp = st.columns(2)
                with col_ctx:
                    st.caption("Retrieved Sources: " + ", ".join(t.get("retrieved_sources", [])))
                    st.text_area("Retrieved Context Preview:", t["retrieved_context"], height=140, key=f"c_{t['model']}_{t['question_id']}")
                with col_resp:
                    st.caption("Generated Answer:")
                    st.text_area("LLM Output:", t["llm_response"], height=140, key=f"r_{t['model']}_{t['question_id']}")
                st.divider()

# -----------------------------------------------------------------------------
# VIEW 4: EXERCISE 6 REPOSITORY UNDERSTANDING
# -----------------------------------------------------------------------------
elif nav_choice == "🏛️ Exercise 6: Repository Understanding":
    st.title("🏛️ Exercise 6: Repository & Codebase Understanding")
    st.markdown("""
    ### Can Naive RAG Answer Questions Requiring Multi-File Code Understanding?
    In this exercise, we tested whether our current RAG pipeline can understand relationships across multiple files in the repository (`rag_logic.py`, `app.py`, `Dockerfile`).
    """)

    st.subheader("1. Multi-File Architecture Scenarios Tested")
    
    with st.expander("Scenario A: Document Loading to Index Creation Flow"):
        st.markdown("""
        - **Question:** Which classes and methods are involved from file upload in `app.py` to index building in `rag_logic.py`?
        - **Actual Code Flow:**  
          `app.py (st.file_uploader)`  
          $\rightarrow$ `rag_logic.DocumentProcessor.load_documents()`  
          $\rightarrow$ `rag_logic.DocumentProcessor.preprocess()`  
          $\rightarrow$ `rag_logic.TextChunker.chunk_documents()`  
          $\rightarrow$ `rag_logic.EmbeddingManager.embed_texts()`  
          $\rightarrow$ `rag_logic.ChromaVectorStore.build_index()` & `FAISSVectorStore.build_index()`
        """)

    with st.expander("Scenario B: Single-Model Memory Lifecycle Flow"):
        st.markdown("""
        - **Question:** How does `ModelCoordinator` guarantee that only 1 model resides in RAM?
        - **Actual Code Flow:**  
          When `EmbeddingManager` or `LLMManager` receives a load request, it calls `coordinator.request_load(self)`.  
          The coordinator calls `_unload_current()` on all other registered managers, invokes `gc.collect()`, and clears `torch.cuda.empty_cache()` before allowing the new model to load.
        """)

    st.divider()

    st.subheader("2. Limitations of Standard Naive RAG on Codebases")
    st.markdown("""
    | Limitation | Why Naive Chunking Fails | Impact on Answer Quality |
    | :--- | :--- | :--- |
    | **Loss of Call Graph** | Code chunks are sliced into flat 500-char blocks. Caller in `app.py` and callee in `rag_logic.py` sit in separate disconnected chunks. | Cannot trace end-to-end execution flow. |
    | **AST Boundary Severing** | Fixed character counts split functions mid-signature, separate decorators from definitions, or drop import statements. | Syntax and scope confusion in responses. |
    | **No Symbol Cross-Referencing** | Text embedding similarity only matches lexical overlap; it cannot follow class inheritance, variable types, or imports. | Inability to perform impact analysis (e.g. *"What breaks if `EmbeddingManager` is refactored?"*). |
    """)

    st.divider()

    st.subheader("3. Week 5 Preview: Transition to Semantic Code Intelligence (Sourcegraph)")
    st.info("""
    **Why Sourcegraph & Semantic Code Graphs are Needed:**
    - **Precise Symbol Navigation**: Using SCIP/LSIF protocols to trace exact *Find Definitions* and *Find References* across whole repositories.
    - **AST-Aware Chunking**: Slicing code by complete semantic nodes (functions, classes, modules) rather than arbitrary characters.
    - **Dependency Graph Traversals**: Enabling LLMs to query who calls what function across multiple repositories.
    """)

# -----------------------------------------------------------------------------
# VIEW 5: INTERACTIVE RAG ASSISTANT
# -----------------------------------------------------------------------------
elif nav_choice == "💬 Interactive RAG Assistant":
    st.title("💬 Interactive RAG Assistant")
    st.markdown("Test the live application with your selected model.")

    from rag_logic import (
        DocumentProcessor, TextChunker, EmbeddingManager,
        ChromaVectorStore, FAISSVectorStore, SemanticSearcher,
        LLMManager, RAGPipeline, ModelCoordinator, DEVICE
    )

    @st.cache_resource
    def get_live_pipeline():
        coord = ModelCoordinator()
        emb_mgr = EmbeddingManager(coordinator=coord)
        llm_mgr = LLMManager(coordinator=coord)
        chroma = ChromaVectorStore(collection_name="live_dashboard")
        faiss = FAISSVectorStore()

        # Pre-index knowledge base if available
        if os.path.exists("knowledge_base"):
            proc = DocumentProcessor("knowledge_base")
            docs = proc.load_documents()
            clean = proc.preprocess(docs)
            chunker = TextChunker()
            chunks = chunker.chunk_documents(clean, "medium")
            texts = [c.page_content for c in chunks]
            embs = emb_mgr.embed_texts("minilm", texts)
            chroma.build_index(chunks, embs, collection_suffix="minilm")
            faiss.build_index(chunks, embs)

        searcher = SemanticSearcher(emb_mgr, chroma, faiss)
        rag_pipe = RAGPipeline(searcher, llm_mgr)
        return rag_pipe, emb_mgr, llm_mgr

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        sel_llm = st.selectbox("Select LLM Model:", list(LLMManager.MODELS.keys()))
    with col_m2:
        sel_topk = st.slider("Top K Retrieved Chunks:", 1, 10, 5)

    test_q = st.text_input("Ask a research paper or code question:", "Explain the Transformer self-attention mechanism.")

    if st.button("Generate Answer", type="primary"):
        with st.spinner(f"Loading {sel_llm} and generating response..."):
            rag_pipe, _, _ = get_live_pipeline()
            result = rag_pipe.answer(
                question=test_q,
                emb_model="minilm",
                llm_name=sel_llm,
                store="chroma",
                top_k=sel_topk
            )

            st.markdown("### Answer:")
            st.write(result["answer"])

            with st.expander("View Retrieval Metadata & Context"):
                st.write(f"**Retrieval Time:** {result['retrieval_time_s']}s")
                st.write(f"**Generation Time:** {result['generation_time_s']}s")
                st.write(f"**Retrieved Sources:** {', '.join(result['retrieved_sources'])}")
                st.text_area("Retrieved Context:", result["context_preview"], height=150)
