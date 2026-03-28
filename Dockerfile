FROM python:3.12-slim

WORKDIR /app

# Install system deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install ALL Python deps + safe cleanup in ONE layer.
# Docker layers are immutable — cleanup in a separate RUN saves nothing.
# PyTorch CPU-only (~280MB wheel, ~460MB installed) is the minimum for real
# torch.nn.Module, torch.autograd, and state_dict() support.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    # Remove non-essential torch components (safe — verified these don't break imports)
    rm -rf /usr/local/lib/python3.12/site-packages/torch/test \
           /usr/local/lib/python3.12/site-packages/torch/include \
           /usr/local/lib/python3.12/site-packages/torch/share \
           /usr/local/lib/python3.12/site-packages/torch/utils/benchmark \
           /usr/local/lib/python3.12/site-packages/torch/utils/bottleneck \
           /usr/local/lib/python3.12/site-packages/torch/utils/tensorboard \
           /usr/local/lib/python3.12/site-packages/torch/lib/*.a \
           /usr/local/lib/python3.12/site-packages/torch/lib/libtorchbind_test.so \
           /usr/local/lib/python3.12/site-packages/torch/lib/libjitbackend_test.so \
           /usr/local/lib/python3.12/site-packages/torch/lib/libbackend_with_compiler.so \
           /usr/local/lib/python3.12/site-packages/caffe2 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages/torch -name "*.pyi" -delete 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; \
    true

# Copy application code
COPY ml_training_debugger/ ml_training_debugger/
COPY server/ server/
COPY openenv.yaml .
COPY baseline_heuristic.py .
COPY baseline_inference.py .
COPY README.md .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
