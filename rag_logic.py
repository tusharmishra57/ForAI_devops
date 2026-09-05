import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

import gc

# Device setting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelCoordinator:
    """Global coordinator ensuring only ONE model is in memory at any time.

    Every model manager (EmbeddingManager, LLMManager) registers itself here.
    Before loading a model, a manager calls `coordinator.request_load(self)` which
    unloads every *other* registered manager first.
    """

    def __init__(self):
        self._managers: List = []  # list of registered managers

    def register(self, manager):
        """Register a manager so the coordinator can unload it when needed."""
        if manager not in self._managers:
            self._managers.append(manager)

    def request_load(self, requester):
        """Called by a manager right before it loads a model.

        Unloads every other manager so that only the requester's model will
        be resident in memory.
        """
        for mgr in self._managers:
            if mgr is not requester:
                mgr._unload_current()
        # Run GC + clear GPU cache once after all evictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class DocumentProcessor:
    """Handles loading and preprocessing of research documents."""
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.raw_documents: List[Document] = []

    def load_documents(self) -> List[Document]:
        documents = []
        if not os.path.exists(self.docs_dir):
            return []
        for fname in os.listdir(self.docs_dir):
            fpath = os.path.join(self.docs_dir, fname)
            if fname.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(fpath)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = fname
                        doc.metadata["type"] = "pdf"
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
            elif fname.lower().endswith(".txt"):
                try:
                    loader = TextLoader(fpath, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = fname
                        doc.metadata["type"] = "text"
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        self.raw_documents = documents
        return documents

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.strip()
        return text

    def preprocess(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
        return documents

class TextChunker:
    """Chunks documents using different chunk sizes."""
    def __init__(self):
        self.chunk_configs = {
            "small":  {"chunk_size": 200,  "chunk_overlap": 40},
            "medium": {"chunk_size": 500,  "chunk_overlap": 100},
            "large":  {"chunk_size": 1000, "chunk_overlap": 200},
        }

    def chunk_documents(self, documents: List[Document], config_name: str = "medium") -> List[Document]:
        cfg = self.chunk_configs[config_name]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_config"] = config_name
        return chunks

class EmbeddingManager:
    """Manages embedding models — only ONE model across the entire app at a time."""
    MODELS = {
        "minilm":   "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet":    "sentence-transformers/all-mpnet-base-v2",
        "bge":      "BAAI/bge-small-en-v1.5",
    }
    def __init__(self, coordinator: Optional[ModelCoordinator] = None):
        self._current_name: Optional[str] = None
        self._current_model: Optional[SentenceTransformer] = None
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

    def _unload_current(self):
        """Free the currently loaded embedding model from memory."""
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
            self._current_name = None

    def load_model(self, name: str) -> SentenceTransformer:
        if self._current_name == name and self._current_model is not None:
            return self._current_model
        # Ask coordinator to unload ALL other managers first
        if self._coordinator is not None:
            self._coordinator.request_load(self)
        # Also unload our own previous model
        self._unload_current()
        model_id = self.MODELS[name]
        model = SentenceTransformer(model_id, device=DEVICE)
        self._current_name = name
        self._current_model = model
        return model

    def embed_texts(self, name: str, texts: List[str]) -> np.ndarray:
        model = self.load_model(name)
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")

    def embed_query(self, name: str, query: str) -> np.ndarray:
        model = self.load_model(name)
        embedding = model.encode([query], normalize_embeddings=True)
        return np.array(embedding, dtype="float32")

class ChromaVectorStore:
    """ChromaDB-based vector store."""
    def __init__(self, collection_name: str = "rag_research"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = None

    def build_index(self, chunks: List[Document], embeddings: np.ndarray, collection_suffix: str = ""):
        coll_name = f"{self.collection_name}_{collection_suffix}"
        try:
            self.client.delete_collection(coll_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self.collection: return []
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return output

class FAISSVectorStore:
    """FAISS-based vector store."""
    def __init__(self):
        self.index = None
        self.chunks: List[Document] = []

    def build_index(self, chunks: List[Document], embeddings: np.ndarray):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None: return []
        scores, indices = self.index.search(query_embedding, top_k)
        output = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            output.append({
                "id": f"doc_{idx}",
                "text": self.chunks[idx].page_content,
                "metadata": self.chunks[idx].metadata,
                "score": float(scores[0][i]),
            })
        return output

class SemanticSearcher:
    def __init__(self, emb_manager: EmbeddingManager, chroma_store: ChromaVectorStore, faiss_store: FAISSVectorStore):
        self.emb_manager = emb_manager
        self.chroma_store = chroma_store
        self.faiss_store = faiss_store

    def search(self, query: str, emb_model: str = "minilm", store: str = "chroma", top_k: int = 5) -> List[Dict]:
        q_emb = self.emb_manager.embed_query(emb_model, query)
        if store == "chroma":
            results = self.chroma_store.search(q_emb, top_k)
        else:
            results = self.faiss_store.search(q_emb, top_k)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def reciprocal_rank_fusion(self, query: str, emb_model: str = "minilm", top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
        chroma_results = self.search(query, emb_model, "chroma", top_k * 2)
        faiss_results   = self.search(query, emb_model, "faiss",  top_k * 2)
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}
        for rank, r in enumerate(chroma_results):
            doc_id = r["text"][:60]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[doc_id] = r
        for rank, r in enumerate(faiss_results):
            doc_id = r["text"][:60]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[doc_id] = r
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_map[doc_id] for doc_id, _ in ranked]

class LLMManager:
    """Manages LLM pipelines — only ONE model across the entire app at a time."""
    MODELS = {
        "qwen-0.5b":   "Qwen/Qwen2.5-0.5B-Instruct",
        "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "tinyllama":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen-1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    }
    def __init__(self, coordinator: Optional[ModelCoordinator] = None):
        self._current_name: Optional[str] = None
        self._current_pipe = None
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

    def _unload_current(self):
        """Free the currently loaded LLM pipeline from memory."""
        if self._current_pipe is not None:
            # Explicitly delete model & tokenizer held by the pipeline
            if hasattr(self._current_pipe, 'model'):
                del self._current_pipe.model
            if hasattr(self._current_pipe, 'tokenizer'):
                del self._current_pipe.tokenizer
            del self._current_pipe
            self._current_pipe = None
            self._current_name = None

    def load_model(self, name: str):
        if name not in self.MODELS:
            name = "qwen-0.5b"
        if self._current_name == name and self._current_pipe is not None:
            return
        # Ask coordinator to unload ALL other managers first
        if self._coordinator is not None:
            self._coordinator.request_load(self)
        # Also unload our own previous model
        self._unload_current()
        model_id = self.MODELS[name]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
        )
        self._current_name = name
        self._current_pipe = pipe

    def generate(self, name: str, prompt: str) -> str:
        if name not in self.MODELS:
            name = "qwen-0.5b"
        if self._current_name != name or self._current_pipe is None:
            self.load_model(name)
        pipe = self._current_pipe
        output = pipe(prompt, return_full_text=False)
        return output[0]["generated_text"].strip()

class RAGPipeline:
    PROMPT_TEMPLATE = """You are a helpful research paper assistant. Answer the question based ONLY on the provided context from research papers. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."

### Context (Retrieved from Research Papers):
{context}

### Question:
{question}

### Answer:"""

    def __init__(self, searcher: SemanticSearcher, llm_manager: LLMManager):
        self.searcher = searcher
        self.llm_manager = llm_manager

    def build_context(self, retrieved_docs: List[Dict], max_chars: int = 2000) -> str:
        context_parts = []
        total_chars = 0
        for doc in retrieved_docs:
            text = doc["text"]
            if total_chars + len(text) > max_chars:
                text = text[:max_chars - total_chars]
            source = doc.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{text}")
            total_chars += len(text)
            if total_chars >= max_chars: break
        return "\n\n".join(context_parts)

    def answer(self, question: str, emb_model: str = "minilm", llm_name: str = "qwen-0.5b", store: str = "chroma", top_k: int = 5, use_rrf: bool = False) -> Dict:
        t0 = time.time()
        if use_rrf:
            retrieved = self.searcher.reciprocal_rank_fusion(question, emb_model, top_k)
        else:
            retrieved = self.searcher.search(question, emb_model, store, top_k)
        retrieval_time = time.time() - t0
        context = self.build_context(retrieved)
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
        t1 = time.time()
        answer_text = self.llm_manager.generate(llm_name, prompt)
        generation_time = time.time() - t1
        return {
            "question": question,
            "answer": answer_text,
            "llm": llm_name,
            "emb_model": emb_model,
            "vector_store": store if not use_rrf else "rrf_fusion",
            "top_k": top_k,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
            "retrieved_sources": list(set([r.get("metadata", {}).get("source", "?") for r in retrieved])),
            "context_preview": context[:500],
        }
