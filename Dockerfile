# ---- Stage 1: Builder — install + strip aggressively ----
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends binutils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --no-compile \
        torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --no-compile -r requirements.txt && \
    #
    # === STRIP DEBUG SYMBOLS FROM ALL .so FILES (saves ~100-200MB) ===
    find /usr/local/lib/python3.12/site-packages -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "*.so.*" -exec strip --strip-unneeded {} + 2>/dev/null; \
    #
    # === TORCH CLEANUP ===
    rm -rf /usr/local/lib/python3.12/site-packages/torch/test \
           /usr/local/lib/python3.12/site-packages/torch/include \
           /usr/local/lib/python3.12/site-packages/torch/share \
           /usr/local/lib/python3.12/site-packages/torch/bin/FileStore* \
           /usr/local/lib/python3.12/site-packages/torch/bin/HashStore* \
           /usr/local/lib/python3.12/site-packages/torch/bin/TCPStore* \
           /usr/local/lib/python3.12/site-packages/torch/bin/protoc* \
           /usr/local/lib/python3.12/site-packages/torch/bin/test_* \
           /usr/local/lib/python3.12/site-packages/torch/utils/benchmark \
           /usr/local/lib/python3.12/site-packages/torch/utils/bottleneck \
           /usr/local/lib/python3.12/site-packages/torch/utils/tensorboard \
           /usr/local/lib/python3.12/site-packages/torch/lib/*.a \
           /usr/local/lib/python3.12/site-packages/torch/lib/libtorchbind_test.so \
           /usr/local/lib/python3.12/site-packages/torch/lib/libjitbackend_test.so \
           /usr/local/lib/python3.12/site-packages/torch/lib/libbackend_with_compiler.so \
           /usr/local/lib/python3.12/site-packages/torch/lib/libaoti_custom_ops.so \
           /usr/local/lib/python3.12/site-packages/torch/lib/libshm_windows \
           /usr/local/lib/python3.12/site-packages/caffe2 \
    #
    # === BLOATED TRANSITIVE DEPS ===
           /usr/local/lib/python3.12/site-packages/gradio \
           /usr/local/lib/python3.12/site-packages/gradio_client \
           /usr/local/lib/python3.12/site-packages/hf_gradio \
           /usr/local/lib/python3.12/site-packages/pandas \
           /usr/local/lib/python3.12/site-packages/PIL \
           /usr/local/lib/python3.12/site-packages/Pillow* \
           /usr/local/lib/python3.12/site-packages/pillow* \
           /usr/local/lib/python3.12/site-packages/networkx \
           /usr/local/lib/python3.12/site-packages/scipy \
           /usr/local/lib/python3.12/site-packages/matplotlib \
           /usr/local/lib/python3.12/site-packages/hf_xet \
           /usr/local/lib/python3.12/site-packages/ffmpy \
           /usr/local/lib/python3.12/site-packages/pydub \
           /usr/local/lib/python3.12/site-packages/groovy \
           /usr/local/lib/python3.12/site-packages/tomlkit \
           /usr/local/lib/python3.12/site-packages/semantic_version* \
           /usr/local/lib/python3.12/site-packages/safehttpx* \
           /usr/local/lib/python3.12/site-packages/brotli* \
           /usr/local/lib/python3.12/site-packages/Brotli* \
           /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/setuptools \
           /usr/local/lib/python3.12/site-packages/docutils \
           /usr/local/lib/python3.12/site-packages/cryptography \
           /usr/local/lib/python3.12/site-packages/cryptography* \
           /usr/local/lib/python3.12/site-packages/pytz 2>/dev/null; \
    #
    # === FILE-LEVEL CLEANUP ===
    find /usr/local/lib/python3.12/site-packages -name "*.pyi" -delete 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "tests" -type d -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.12/site-packages -name "test" -type d -exec rm -rf {} + 2>/dev/null; \
    # Remove stale dist-info for packages we already deleted
    rm -rf /usr/local/lib/python3.12/site-packages/gradio*.dist-info \
           /usr/local/lib/python3.12/site-packages/pandas*.dist-info \
           /usr/local/lib/python3.12/site-packages/Pillow*.dist-info \
           /usr/local/lib/python3.12/site-packages/hf_xet*.dist-info \
           /usr/local/lib/python3.12/site-packages/Brotli*.dist-info \
           /usr/local/lib/python3.12/site-packages/networkx*.dist-info \
           /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/pip*.dist-info 2>/dev/null; \
    true

# ---- Stage 2: Runtime — minimal clean image ----
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only what's needed from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

COPY ml_training_debugger/ ml_training_debugger/
COPY server/ server/
COPY openenv.yaml .
COPY baseline_heuristic.py .
COPY baseline_inference.py .
COPY inference.py .
COPY demo.py .
COPY README.md .
COPY validation/reports/ validation/reports/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
