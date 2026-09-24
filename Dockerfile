FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

 
COPY app.py rag_logic.py dashboard.py evaluate_models.py guardrails.py test_guardrails.py ./
COPY evaluation_dataset.json week4_evaluation_results.csv week4_model_summary.json week4_rag_traces.json week4_rag_vs_llm.json week4_guardrail_comparison.json ./
COPY knowledge_base ./knowledge_base


RUN mkdir -p /home/app/.cache/huggingface /home/app/.streamlit \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8501

VOLUME ["/home/app/.cache/huggingface"]

CMD ["streamlit", "run", "app.py"]