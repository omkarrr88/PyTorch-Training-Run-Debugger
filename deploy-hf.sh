#!/bin/bash
# Deploy to Hugging Face Spaces
# Usage: ./deploy-hf.sh <your-hf-username>/<space-name>
# Example: ./deploy-hf.sh omkarrr88/pytorch-training-debugger

set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
    echo "Usage: ./deploy-hf.sh <username>/<space-name>"
    exit 1
fi

echo "=== Deploying to HF Space: $SPACE ==="

# Ensure huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    pip install huggingface_hub
fi

# Clone or create the space
if [ ! -d ".hf-space" ]; then
    echo "Cloning space..."
    git clone "https://huggingface.co/spaces/$SPACE" .hf-space || {
        echo "Creating new space..."
        huggingface-cli repo create "$SPACE" --type space --space-sdk docker
        git clone "https://huggingface.co/spaces/$SPACE" .hf-space
    }
fi

# Copy files to space
echo "Copying files..."
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    --exclude='.hf-space' --exclude='tests' --exclude='validation' \
    --exclude='.claude' --exclude='*.pyc' --exclude='run*.json' \
    --exclude='.env' --exclude='.coverage' --exclude='uv.lock' \
    . .hf-space/

# Copy validation report (pre-computed)
mkdir -p .hf-space/validation/reports
cp -r validation/reports/fidelity_report.json .hf-space/validation/reports/ 2>/dev/null || true

cd .hf-space

# Add openenv tag to README if not present
if ! grep -q "tags:" README.md 2>/dev/null; then
    cat > README.md.header <<'EOF'
---
title: PyTorch Training Run Debugger
emoji: 🔧
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
---

EOF
    cat README.md >> README.md.header
    mv README.md.header README.md
fi

# Commit and push
git add -A
git commit -m "Deploy: PyTorch Training Run Debugger" || echo "No changes to commit"
git push

echo "=== Deployed! ==="
echo "Space URL: https://huggingface.co/spaces/$SPACE"
echo "Health: https://${SPACE/\//-}.hf.space/health"
