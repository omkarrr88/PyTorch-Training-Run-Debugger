FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (largest layer, cached separately)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch excluded from requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete 2>/dev/null; \
    rm -rf /usr/local/lib/python3.12/site-packages/gradio/templates 2>/dev/null; \
    true

# Copy application code
COPY ml_training_debugger/ ml_training_debugger/
COPY server/ server/
COPY openenv.yaml .
COPY baseline_heuristic.py .
COPY README.md .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
