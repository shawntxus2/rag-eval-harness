#!/usr/bin/env python3
"""Prove the GPU is real and usable — no silent CPU fallback.

Prints, in order:
  - torch version and the CUDA version it was built against
  - torch.cuda.is_available()
  - device name and compute capability
  - total and free device memory
  - a real matmul on the device, timed, result verified against the CPU answer
  - BAAI/bge-base-en-v1.5 loaded and a REPRESENTATIVE sample embedded on the device
    (256 chunks sized to CHUNK_SIZE tokens — what Phase 1 will actually feed it),
    timed, reported as embeddings/second AND tokens/second, plus the forward dtype.
    tokens/sec is the figure that stays comparable when CHUNK_SIZE changes.

If CUDA is not available, this STOPS with a nonzero exit and says why. It never
falls back to CPU and calls that a pass. Failures are not swallowed.

Usage:  python scripts/gpu_check.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

# Domain-flavoured filler so the sample tokenizes like real corpus text, not
# repeated single tokens (which would embed unrealistically fast).
_FILLER = (
    "The covered clearing agency applies a haircut to collateral and computes "
    "net settlement obligations through multilateral netting, managing "
    "counterparty credit exposure with initial and variation margin. Upon a "
    "participant default the waterfall draws on margin, the guaranty fund, and "
    "assessment powers; novation interposes the central counterparty as buyer "
    "to every seller and seller to every buyer across the settlement cycle. "
)


def _sample_chunks(model, n: int, target_tokens: int) -> list[str]:
    """Return n distinct strings each ~target_tokens long under the model tokenizer."""
    tok = model.tokenizer
    ids = tok(_FILLER * 40, add_special_tokens=False)["input_ids"]
    # Special tokens ([CLS]/[SEP]) eat 2 slots; leave room so we land near target.
    body = max(1, target_tokens - 2)
    out = []
    for i in range(n):
        # Rotate the token window per-chunk so strings differ (defeats any caching).
        start = (i * 7) % max(1, len(ids) - body)
        text = tok.decode(ids[start:start + body])
        out.append(f"[chunk {i}] {text}")
    return out


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

    # --- Embed a representative sample with bge-base-en-v1.5 on the device ---
    # 256 chunks sized to the chunk length Phase 1 will actually use, so the
    # throughput number reflects the real workload rather than toy short strings.
    from sentence_transformers import SentenceTransformer

    from rag.config import settings

    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    param = next(model.parameters())
    print(f"bge model device   : {param.device}")
    print(f"bge forward dtype  : {param.dtype}")

    target_tokens = min(settings.chunk_size, model.max_seq_length)
    n_chunks = 256
    texts = _sample_chunks(model, n_chunks, target_tokens)
    # Exact tokens the model will process (truncated at max_seq_length).
    enc = model.tokenizer(texts, truncation=True, max_length=model.max_seq_length)
    total_tokens = sum(len(ids) for ids in enc["input_ids"])
    print(f"sample             : {n_chunks} chunks @ ~{total_tokens // n_chunks} tokens "
          f"(CHUNK_SIZE={settings.chunk_size}, model max={model.max_seq_length})")

    bs = settings.embed_batch_size
    model.encode(texts, batch_size=bs, normalize_embeddings=True)  # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=bs, normalize_embeddings=True)
    torch.cuda.synchronize()
    embed_s = time.perf_counter() - t0
    print(f"bge embed sample   : {embed_s * 1000:.1f} ms @ batch {bs}  ->  "
          f"{n_chunks / embed_s:.1f} embeddings/sec  |  {total_tokens / embed_s:,.0f} tokens/sec")
    print(f"embedding shape    : {tuple(emb.shape)}")

    print("\nGPU check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
