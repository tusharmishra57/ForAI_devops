import streamlit as st
import os
import tempfile
import torch
from rag_logic import (
    DocumentProcessor, TextChunker, EmbeddingManager, 
    ChromaVectorStore, FAISSVectorStore, SemanticSearcher, 
    LLMManager, RAGPipeline, DEVICE
)

st.set_page_config(page_title="Research Paper Assistant", layout="wide")

st.title("📚 Research Paper Assistant (RAG)")
st.markdown("Upload research papers and ask questions about them.")

# Cache managers
@st.cache_resource
def get_managers():
    emb_manager = EmbeddingManager()
    llm_manager = LLMManager()
    chroma_store = ChromaVectorStore()
    faiss_store = FAISSVectorStore()
    return emb_manager, llm_manager, chroma_store, faiss_store

emb_manager, llm_manager, chroma_store, faiss_store = get_managers()

# Sidebar for configuration
st.sidebar.header("Configuration")
emb_model_name = st.sidebar.selectbox("Embedding Model", list(EmbeddingManager.MODELS.keys()))
llm_model_name = st.sidebar.selectbox("LLM Model", list(LLMManager.MODELS.keys()))
vector_store_name = st.sidebar.selectbox("Vector Store", ["chroma", "faiss", "rrf_fusion"])
top_k = st.sidebar.slider("Top K", 1, 10, 5)

# File upload
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uploaded_file in uploaded_files:
                    with open(os.path.join(tmp_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                processor = DocumentProcessor(tmp_dir)
                raw_docs = processor.load_documents()
                clean_docs = processor.preprocess(raw_docs)
                
                chunker = TextChunker()
                chunks = chunker.chunk_documents(clean_docs, "medium")
                
                texts = [c.page_content for c in chunks]
                embeddings = emb_manager.embed_texts(emb_model_name, texts)
                
                chroma_store.build_index(chunks, embeddings, collection_suffix=emb_model_name)
                faiss_store.build_index(chunks, embeddings)
                
                st.session_state["processed"] = True
                st.session_state["chunks_count"] = len(chunks)
                st.success(f"Processed {len(uploaded_files)} documents into {len(chunks)} chunks.")

if st.session_state.get("processed"):
    searcher = SemanticSearcher(emb_manager, chroma_store, faiss_store)
    rag = RAGPipeline(searcher, llm_manager)
    
    question = st.text_input("Ask a question about the papers:")
    
    if question:
        with st.spinner("Generating answer..."):
            use_rrf = vector_store_name == "rrf_fusion"
            result = rag.answer(
                question, 
                emb_model=emb_model_name, 
                llm_name=llm_model_name, 
                store=vector_store_name if not use_rrf else "chroma",
                top_k=top_k,
                use_rrf=use_rrf
            )
            
            st.markdown(f"### Answer:")
            st.write(result["answer"])
            
            with st.expander("Details"):
                st.write(f"**Retrieval Time:** {result['retrieval_time_s']}s")
                st.write(f"**Generation Time:** {result['generation_time_s']}s")
                st.write(f"**Sources:** {', '.join(result['retrieved_sources'])}")
                st.markdown("**Context Preview:**")
                st.text(result["context_preview"])
else:
    st.info("Please upload and process documents to start.")

st.sidebar.divider()
st.sidebar.write(f"**Device:** {DEVICE}")
if torch.cuda.is_available():
    st.sidebar.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
