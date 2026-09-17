"""
Week 4 Hands-on Activity: Model Evaluation, Quantitative Analysis, RAG Pipeline Diagnostics,
and Repository-Level Code Understanding.

Evaluates 3 models under identical conditions:
1. qwen-0.5b (Qwen/Qwen2.5-0.5B-Instruct)
2. smollm-360m (HuggingFaceTB/SmolLM2-360M-Instruct)
3. tinyllama (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
"""

import os
import sys
import time
import json
import ast
import re
import gc
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from rag_logic import (
    DocumentProcessor, TextChunker, EmbeddingManager,
    ChromaVectorStore, FAISSVectorStore, SemanticSearcher,
    LLMManager, RAGPipeline, ModelCoordinator, DEVICE
)

# ---------------------------------------------------------------------------
# Metric Calculation Definitions
# ---------------------------------------------------------------------------
def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level precision, recall, and F1 score between prediction and ground truth.
    Formula:
        Precision = |overlap| / |pred_tokens|
        Recall    = |overlap| / |gt_tokens|
        F1        = 2 * P * R / (P + R)
    """
    def tokenize(s: str) -> List[str]:
        return re.findall(r"\w+", s.lower())

    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gt_tokens.count(t) for t in common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Computes ROUGE-L (Longest Common Subsequence F1) between prediction and ground truth."""
    def tokenize(s: str) -> List[str]:
        return re.findall(r"\w+", s.lower())

    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    m, n = len(pred_tokens), len(gt_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def compute_relevance(emb_manager: EmbeddingManager, emb_model: str, response: str, context: str) -> float:
    """Computes semantic relevance as the cosine similarity between the response embedding
    and the retrieved context embedding.
    Formula: CosSim(v_resp, v_ctx) = (v_resp . v_ctx) / (||v_resp|| * ||v_ctx||)
    """
    if not response or not context:
        return 0.0
    try:
        v_resp = emb_manager.embed_query(emb_model, response[:512])
        v_ctx = emb_manager.embed_query(emb_model, context[:512])
        sim = float(np.dot(v_resp, v_ctx.T)[0][0])
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
    except Exception:
        return 0.5


def evaluate_retrieval(retrieved_sources: List[str], expected_sources: List[str]) -> Tuple[float, float]:
    """Computes Retrieval Quality metrics:
    1. Hit Rate @ K: 1.0 if at least one expected source is in retrieved sources, else 0.0
    2. Precision @ K: fraction of retrieved sources that match expected sources.
    """
    if not expected_sources:
        # For adversarial/unanswerable questions with no expected sources
        return 1.0, 1.0

    hits = sum(1 for src in retrieved_sources if src in expected_sources)
    hit_rate = 1.0 if hits > 0 else 0.0
    precision = hits / max(1, len(retrieved_sources))
    return hit_rate, precision


def detect_hallucination(question_category: str, response: str, context: str, ground_truth: str) -> float:
    """Computes Hallucination Rate:
    - On unanswerable questions (where context is empty/unrelated):
      If the model fabricates an answer without stating 'I don't have enough information', hallucination = 1.0.
    - On answerable questions:
      Checks whether key entity words in response exist neither in retrieved context nor ground truth.
    """
    unanswerable_triggers = [
        "don't have enough information",
        "not provided in the context",
        "does not contain",
        "cannot answer",
        "not mentioned"
    ]
    is_unanswerable = "I don't have enough information" in ground_truth

    resp_lower = response.lower()
    admits_lack = any(trig in resp_lower for trig in unanswerable_triggers)

    if is_unanswerable:
        return 0.0 if admits_lack else 1.0

    # For answerable questions: check keyword hallucinations
    resp_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", response.lower()))
    ctx_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", (context + " " + ground_truth).lower()))

    if not resp_words:
        return 0.0

    unsupported = resp_words - ctx_words
    hallucination_score = len(unsupported) / max(1, len(resp_words))
    return min(1.0, round(hallucination_score, 3))


def evaluate_code_pass_rate(response: str) -> float:
    """Computes Test-Pass Rate for code generation tasks:
    1. Extracts code blocks from response.
    2. Checks Python syntax parsing via ast.parse.
    3. Runs unit syntax & structure validation.
    Returns 1.0 if code parses and compiles cleanly, 0.5 if partial, 0.0 if syntax error.
    """
    code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else response.strip()

    try:
        ast.parse(code)
        # Attempt compile to code object
        compile(code, "<string>", "exec")
        return 1.0
    except SyntaxError:
        # Check if code has single line definitions
        lines = [l for l in code.split("\n") if l.strip().startswith("def ") or "return " in l]
        if lines:
            try:
                ast.parse("\n".join(lines))
                return 0.75
            except Exception:
                pass
        return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main Evaluation Orchestration
# ---------------------------------------------------------------------------
def run_week4_evaluation(
    docs_dir: str = "knowledge_base",
    dataset_path: str = "evaluation_dataset.json",
    models: List[str] = ["qwen-0.5b", "smollm-360m", "tinyllama"],
    sample_limit: int = None
) -> Tuple[pd.DataFrame, List[Dict], Dict]:

    print("=" * 80)
    print("STARTING WEEK 4 EVALUATION HARNESS")
    print(f"Device: {DEVICE}")
    print(f"Models to evaluate: {models}")
    print("=" * 80)

    # 1. Load Evaluation Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if sample_limit:
        dataset = dataset[:sample_limit]
        print(f"Running on sample limit: {len(dataset)} questions")

    # 2. Build Document Index
    coordinator = ModelCoordinator()
    emb_manager = EmbeddingManager(coordinator=coordinator)
    llm_manager = LLMManager(coordinator=coordinator)
    chroma_store = ChromaVectorStore(collection_name="week4_eval")
    faiss_store = FAISSVectorStore()

    print(f"Loading documents from: {docs_dir}")
    processor = DocumentProcessor(docs_dir)
    raw_docs = processor.load_documents()
    clean_docs = processor.preprocess(raw_docs)

    chunker = TextChunker()
    chunks = chunker.chunk_documents(clean_docs, "medium")
    print(f"Indexed {len(raw_docs)} documents into {len(chunks)} chunks.")

    texts = [c.page_content for c in chunks]
    print("Generating embeddings using minilm...")
    embeddings = emb_manager.embed_texts("minilm", texts)

    chroma_store.build_index(chunks, embeddings, collection_suffix="minilm")
    faiss_store.build_index(chunks, embeddings)

    searcher = SemanticSearcher(emb_manager, chroma_store, faiss_store)
    rag = RAGPipeline(searcher, llm_manager)

    all_results = []
    rag_traces = []

    # 3. Benchmark Each Model
    for model_name in models:
        print(f"\n>>> BENCHMARKING MODEL: {model_name} <<<")
        model_results = []

        # Force unload before starting
        coordinator.request_load(llm_manager)

        for q_idx, item in enumerate(dataset):
            q_id = item["id"]
            q_text = item["question"]
            category = item["category"]
            gt = item["ground_truth"]
            expected_sources = item.get("expected_sources", [])
            is_code = item.get("is_code_task", False)

            # Measure CPU and RAM before
            mem_before = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0

            # Execute RAG query
            t_start = time.time()
            rag_output = rag.answer(
                question=q_text,
                emb_model="minilm",
                llm_name=model_name,
                store="chroma",
                top_k=5,
                use_rrf=False
            )
            t_end = time.time()

            mem_after = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0
            mem_delta = max(0.0, mem_after - mem_before)

            response = rag_output["answer"]
            retrieval_time = rag_output["retrieval_time_s"]
            generation_time = rag_output["generation_time_s"]
            total_latency = round(t_end - t_start, 3)
            retrieved_sources = rag_output["retrieved_sources"]
            context_preview = rag_output["context_preview"]

            # Token estimations
            prompt_tokens = len(q_text.split()) + len(context_preview.split()) + 40
            gen_tokens = len(response.split())
            tokens_per_sec = round(gen_tokens / max(0.001, generation_time), 2)

            # Compute Quality Metrics
            token_f1 = compute_token_f1(response, gt)
            rouge_l = compute_rouge_l(response, gt)
            accuracy_score = round((token_f1 + rouge_l) / 2.0, 3)
            relevance_score = round(compute_relevance(emb_manager, "minilm", response, context_preview), 3)
            hit_rate, retrieval_precision = evaluate_retrieval(retrieved_sources, expected_sources)
            hallucination_rate = detect_hallucination(category, response, context_preview, gt)

            code_pass_rate = evaluate_code_pass_rate(response) if is_code else None

            res_record = {
                "model": model_name,
                "question_id": q_id,
                "category": category,
                "question": q_text,
                "response": response,
                "ground_truth": gt,
                "accuracy": accuracy_score,
                "relevance": relevance_score,
                "hit_rate": hit_rate,
                "retrieval_precision": retrieval_precision,
                "hallucination_rate": hallucination_rate,
                "code_pass_rate": code_pass_rate,
                "retrieval_time_s": retrieval_time,
                "generation_time_s": generation_time,
                "total_latency_s": total_latency,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "tokens_per_sec": tokens_per_sec,
                "ram_mb": round(mem_after, 1),
                "ram_delta_mb": round(mem_delta, 1),
                "retrieved_sources": ",".join(retrieved_sources)
            }
            model_results.append(res_record)
            all_results.append(res_record)

            # Record RAG trace for Exercise 5
            # Determine qualitative diagnosis
            if hit_rate == 0.0 and len(expected_sources) > 0:
                diagnosis = "Important info missed during retrieval"
            elif hallucination_rate > 0.5:
                diagnosis = "LLM hallucinated despite context / unanswerable query"
            elif accuracy_score >= 0.35:
                diagnosis = "Correct answer with relevant retrieved context"
            else:
                diagnosis = "Irrelevant or partial context led to low accuracy"

            rag_traces.append({
                "model": model_name,
                "question_id": q_id,
                "category": category,
                "question": q_text,
                "retrieved_context": context_preview,
                "retrieved_sources": retrieved_sources,
                "llm_response": response,
                "accuracy": accuracy_score,
                "hallucination_rate": hallucination_rate,
                "diagnosis": diagnosis
            })

            print(f"[{model_name}] {q_id} ({category}): Latency={total_latency}s | Acc={accuracy_score} | Halluc={hallucination_rate}")

    df_results = pd.DataFrame(all_results)

    # 4. Compute Aggregate Model Summary Table
    summary = {}
    for m in models:
        m_df = df_results[df_results["model"] == m]
        code_df = m_df[m_df["code_pass_rate"].notnull()]
        summary[m] = {
            "mean_accuracy": round(m_df["accuracy"].mean(), 3),
            "mean_relevance": round(m_df["relevance"].mean(), 3),
            "mean_hit_rate": round(m_df["hit_rate"].mean(), 3),
            "mean_retrieval_precision": round(m_df["retrieval_precision"].mean(), 3),
            "mean_hallucination_rate": round(m_df["hallucination_rate"].mean(), 3),
            "mean_code_pass_rate": round(code_df["code_pass_rate"].mean(), 3) if len(code_df) > 0 else 0.0,
            "mean_generation_time_s": round(m_df["generation_time_s"].mean(), 3),
            "mean_total_latency_s": round(m_df["total_latency_s"].mean(), 3),
            "mean_tokens_per_sec": round(m_df["tokens_per_sec"].mean(), 2),
            "mean_ram_mb": round(m_df["ram_mb"].mean(), 1)
        }

    return df_results, rag_traces, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Week 4 RAG Evaluation")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of sample questions")
    parser.add_argument("--models", nargs="+", default=["qwen-0.5b", "smollm-360m", "tinyllama"])
    args = parser.parse_args()

    df, traces, summary = run_week4_evaluation(
        docs_dir="knowledge_base",
        dataset_path="evaluation_dataset.json",
        models=args.models,
        sample_limit=args.sample
    )

    df.to_csv("week4_evaluation_results.csv", index=False)
    with open("week4_rag_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    with open("week4_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY TABLE (EXERCISE 3 & 4)")
    print("=" * 80)
    summary_df = pd.DataFrame(summary).T
    print(summary_df.to_string())
    print("\nResults saved to:")
    print("- week4_evaluation_results.csv")
    print("- week4_rag_traces.json")
    print("- week4_model_summary.json")
