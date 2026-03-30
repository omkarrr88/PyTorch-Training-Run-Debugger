---
name: Docker torch stripping — what breaks and final optimized approach
description: Lessons learned from Docker optimization. Final image 885MB using torch 2.5.1 + multi-stage + strip. Which dirs break, which are safe.
type: feedback
---

## Final Optimized Dockerfile Approach (885MB)

1. **Use torch 2.5.1+cpu** (not latest 2.11.0) — smaller wheel, libtorch_cpu.so strips to 329MB
2. **Multi-stage build**: builder installs + strips, runtime copies only site-packages
3. **`strip --strip-unneeded`** on ALL .so files in one RUN layer
4. **`--no-compile`** flag on pip install (skip .pyc generation)
5. **Remove bloated transitive deps** in same layer: gradio (155MB), pandas (42MB), PIL, pip, setuptools

## Do NOT Remove (breaks `import torch` or runtime)

- `torch/testing` → required by `torch.autograd.gradcheck`
- `torch/distributed` → required by `torch._jit_internal`
- `torch/cuda` → required at `_initExtension`
- `torch/_inductor`, `torch/_dynamo` → required by `torch.optim` (optimizer init)
- `torch/_functorch` → required by core init
- `torch/fx` → required by `_functorch`
- `torch/sparse`, `torch/nested`, `torch/masked` → required by `torch.nn`
- `torch/onnx`, `torch/ao`, `torch/_export`, `torch/jit` → required at import time
- `torchgen` → required by `torch.utils._python_dispatch`
- `sympy` + `mpmath` → required by `torch._dynamo.utils`
- `numpy` + `numpy.libs` → required by `torch.storage`
- `beartype` → required by `fastmcp` → `openenv-core`
- `pygments` → required by `rich` → `fastmcp`
- `torch/bin/torch_shm_manager` → required at `_initExtension`

## Safe to Remove (verified working after removal)

- `torch/test`, `torch/include`, `torch/share` — dev/test files
- `torch/bin/*` EXCEPT `torch_shm_manager` — test binaries (47MB)
- `torch/utils/benchmark`, `torch/utils/bottleneck`, `torch/utils/tensorboard`
- `torch/lib/*.a`, `torch/lib/libtorchbind_test.so`, `torch/lib/libjitbackend_test.so`, etc.
- `caffe2/` — not used
- `gradio`, `gradio_client`, `hf_gradio` — pulled by openenv-core, not needed at runtime
- `pandas`, `PIL/Pillow`, `networkx`, `scipy`, `matplotlib`
- `pip`, `setuptools`, `docutils`, `cryptography`, `pytz`
- `ffmpy`, `pydub`, `groovy`, `tomlkit`, `semantic_version`, `safehttpx`, `brotli`
- All `.pyi` files, `__pycache__`, `.pyc`, stale `.dist-info`

## Older Torch NOT Smaller
torch 2.2.0+cpu was 179MB wheel but installed to 932MB (numpy version mismatch, no strip benefit). torch 2.5.1+cpu at 885MB is the sweet spot.
