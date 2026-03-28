#!/bin/bash
set -euo pipefail

echo "=== PyTorch Training Run Debugger — Pre-Submission Smoke Test ==="
echo ""

# 1. Run tests
echo "=== 1. Running test suite ==="
source .venv/bin/activate
pytest tests/ -v --cov=ml_training_debugger --cov-report=term-missing
echo ""

# 2. Code formatting check
echo "=== 2. Code formatting ==="
black --check ml_training_debugger/ server/ tests/ || { echo "Run: black ml_training_debugger/ server/ tests/"; exit 1; }
ruff check ml_training_debugger/ server/ tests/ || { echo "Run: ruff check --fix"; exit 1; }
isort --check ml_training_debugger/ server/ tests/ --profile black || { echo "Run: isort --profile black"; exit 1; }
echo "PASS: formatting OK"
echo ""

# 3. Baseline reproducibility
echo "=== 3. Baseline reproducibility ==="
python baseline_heuristic.py > /tmp/run1.json 2>/dev/null
python baseline_heuristic.py > /tmp/run2.json 2>/dev/null
diff /tmp/run1.json /tmp/run2.json && echo "PASS: bit-exact reproducible" || { echo "FAIL: non-reproducible"; exit 1; }
echo ""

# 4. Docker build
echo "=== 4. Docker build ==="
docker build -t pytorch-debugger .
IMAGE_SIZE=$(docker images pytorch-debugger --format "{{.Size}}")
echo "Image size: $IMAGE_SIZE"
echo ""

# 5. Docker run + health check
echo "=== 5. Docker run + endpoint checks ==="
docker run -d -p 7860:7860 --name smoke-test pytorch-debugger
sleep 10

curl -f http://localhost:7860/health || { echo "FAIL: health"; docker stop smoke-test; docker rm smoke-test; exit 1; }
echo ""
curl -f http://localhost:7860/tasks || { echo "FAIL: tasks"; docker stop smoke-test; docker rm smoke-test; exit 1; }
echo ""
curl -f -X POST http://localhost:7860/grader || { echo "FAIL: grader"; docker stop smoke-test; docker rm smoke-test; exit 1; }
echo ""

# 6. Cleanup
docker stop smoke-test && docker rm smoke-test
rm -f /tmp/run1.json /tmp/run2.json

echo ""
echo "=== ALL CHECKS PASSED ==="
