#!/usr/bin/env python3
"""Prove the GPU is real and usable — no silent CPU fallback.

Prints, in order:
  - torch version and the CUDA version it was built against
  - torch.cuda.is_available()
  - device name and compute capability
  - total and free device memory
  - a real matmul on the device, timed, result verified against the CPU answer
  - BAAI/bge-base-en-v1.5 loaded and 64 short strings embedded on the device, timed,
    reported as embeddings/second (for comparison against CPU numbers later)

If CUDA is not available, this STOPS with a nonzero exit and says why. It never
falls back to CPU and calls that a pass. Failures are not swallowed.

Usage:  python scripts/gpu_check.py
"""
from __future__ import annotations

import sys
import time

import torch


def main() -> int:
    print(f"torch version      : {torch.__version__}")
    print(f"torch CUDA build   : {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"cuda.is_available(): {available}")
    if not available:
        print(
            "\nSTOP: torch.cuda.is_available() is False. The GPU is not usable from "
            "this interpreter.\nLikely causes: a CPU-only torch wheel, a driver/toolkit "
            "mismatch, or no visible device.\nThis is a hard failure — not falling back "
            "to CPU.",
            file=sys.stderr,
        )
        # Surface whatever torch knows about why initialization failed.
        try:
            torch.cuda.init()
        except Exception as e:  # noqa: BLE001 - we want the raw reason printed
            print(f"cuda.init() raised: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    print(f"device name        : {torch.cuda.get_device_name(0)}")
    print(f"compute capability : {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
    print(f"torch arch_list    : {torch.cuda.get_arch_list()}")

    free_b, total_b = torch.cuda.mem_get_info(0)
    gib = 1024**3
    print(f"device memory total: {total_b / gib:.2f} GiB")
    print(f"device memory free : {free_b / gib:.2f} GiB")

    # --- Real matmul on device, timed, verified against CPU ---
    n = 4096
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    # Warm up so the timing excludes cuBLAS/kernel JIT on first launch.
    (a @ b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    matmul_ms = (time.perf_counter() - t0) * 1000.0

    c_cpu = a.cpu() @ b.cpu()
    max_abs_diff = (c.cpu() - c_cpu).abs().max().item()
    matches = torch.allclose(c.cpu(), c_cpu, rtol=1e-3, atol=1e-3)
    gflops = (2.0 * n**3) / (matmul_ms / 1000.0) / 1e9
    print(f"matmul {n}x{n} fp32 : {matmul_ms:.2f} ms on device  (~{gflops:.0f} GFLOP/s)")
    print(f"matmul vs CPU      : matches={matches}  max_abs_diff={max_abs_diff:.2e}")
    if not matches:
        print(
            "\nSTOP: device matmul does not match the CPU result. The kernels ran but "
            "produced wrong numbers — do not trust this GPU build.",
            file=sys.stderr,
        )
        return 2

    # --- Embed 64 short strings with bge-base-en-v1.5 on the device, timed ---
    from sentence_transformers import SentenceTransformer

    texts = [f"Short evaluation sentence number {i} about retrieval." for i in range(64)]
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    # Confirm the model's parameters actually live on CUDA.
    param_device = next(model.parameters()).device
    print(f"bge model device   : {param_device}")
    # Warm up (first batch pays model compilation / allocator costs).
    model.encode(texts, batch_size=64, normalize_embeddings=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True)
    torch.cuda.synchronize()
    embed_s = time.perf_counter() - t0
    print(f"bge embed 64 strs  : {embed_s * 1000:.2f} ms  -> {len(texts) / embed_s:.1f} embeddings/sec")
    print(f"embedding shape    : {tuple(emb.shape)}")

    print("\nGPU check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
