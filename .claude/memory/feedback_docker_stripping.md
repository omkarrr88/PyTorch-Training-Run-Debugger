---
name: Docker torch stripping — what breaks
description: Lessons learned from aggressive PyTorch stripping in Docker. Which dirs are safe to remove and which break imports.
type: feedback
---

Do NOT remove these torch directories in Docker — they break `import torch`:

- `torch/cuda` → `ModuleNotFoundError: No module named 'torch.cuda'` (imported at `_initExtension`)
- `torch/distributed` → `ModuleNotFoundError` (imported via `torch._jit_internal`)
- `torch/testing` → `ModuleNotFoundError` (imported via `torch.autograd.gradcheck`)
- `torch/jit` → Required by core torch init
- `torch/fx` → Required by `torch._functorch`
- `torch/_functorch` → Required by core init
- `torch/sparse`, `torch/nested`, `torch/masked` → Required by `torch.nn`

**Why:** PyTorch's `__init__.py` eagerly imports these modules during initialization. Even CPU-only builds reference them.

**Safe to remove** (verified working): `torch/test`, `torch/include`, `torch/share`, `torch/utils/benchmark`, `torch/utils/bottleneck`, `torch/utils/tensorboard`, `torch/lib/*.a`, `torch/lib/libtorchbind_test.so`, `torch/lib/libjitbackend_test.so`, `torch/lib/libbackend_with_compiler.so`, `caffe2/`, `torch/_inductor`, `torch/_dynamo`, `torch/onnx`, `torch/_export`, `torch/compiler`, `torch/package`, `torch/profiler`, `torch/export`, `.pyi` files

**How to apply:** Always combine pip install + cleanup in ONE Docker RUN layer. Separate layers don't reduce size.

**`strip --strip-debug` on .so files**: Did NOT reduce `libtorch_cpu.so` size (426MB → 426MB). The pre-built CPU wheel has no debug symbols.
