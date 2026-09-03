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
    && pip install -r requirements.txt

COPY app.py rag_logic.py ./

RUN mkdir -p /home/app/.cache/huggingface /home/app/.streamlit \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8501

VOLUME ["/home/app/.cache/huggingface"]

CMD ["streamlit", "run", "app.py"]