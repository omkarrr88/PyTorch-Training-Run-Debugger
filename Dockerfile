FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU-only first (largest layer, cached)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
