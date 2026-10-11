FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_PROJECT_ROOT=/app \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && pip install --no-cache-dir ".[all]"

EXPOSE 8501
CMD ["streamlit", "run", "src/rag_system/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
