#!/usr/bin/env python3
"""Embedding precision sweep: fp32 vs bf16 vs fp16 on this GPU.

Task 4 measured batch size but never asked what dtype the forward pass runs in.
The default is fp32. On Blackwell, lower precision is likely worth more than batch
size. This measures, for BAAI/bge-base-en-v1.5 at the real chunk length:

  - throughput (embeddings/sec and tokens/sec) per dtype
  - whether lower-precision embeddings still AGREE with fp32 closely enough that
    retrieval RANKING would not change (the only tolerance that matters here):
      * cosine(fp32, lowp) per vector  (min / mean)
      * top-1 unchanged rate, mean top-5 / top-10 overlap, max rank displacement,
        over a corpus where every chunk is also used as a query

Reports and recommends. Changes no defaults.

Usage:  python scripts/dtype_bench.py
"""
from __future__ import annotations

import glob
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from sentence_transformers import SentenceTransformer

from rag.config import settings

# A varied, domain-flavoured pool so chunks differ in content (a meaningful
# ranking test needs distinct neighbours, not 256 near-duplicates).
_POOL = """
The covered clearing agency applies a haircut to collateral and computes net
settlement obligations through multilateral netting. Counterparty credit
exposure is managed with initial margin and variation margin collected intraday.
Upon a participant default the loss waterfall draws first on the defaulter's
margin, then the mutualized guaranty fund, then assessment powers against
surviving members. Novation interposes the central counterparty as buyer to
every seller and seller to every buyer. A derivatives clearing organization must
maintain financial resources sufficient to withstand the default of the one or
two participants creating the largest combined exposure in extreme but plausible
market conditions. Payment system risk arises from settlement lag, liquidity
shortfalls, and operational disruption; recovery planning and orderly wind-down
preserve continuity of critical services. Information security governance,
business continuity management, and third-party outsourcing controls bound
operational resilience. Custody of securities, funds transfer, and the delivery
versus payment model reduce principal risk at settlement. Stress testing,
backtesting of margin models, and reverse stress testing size the resources.
Liquidity risk is distinct from credit risk and is covered by qualifying liquid
resources and committed facilities across relevant currencies and settlement
banks. Cyber resilience requires the ability to resume critical operations within
two hours of a disruption and to complete settlement by the end of the day.
"""


def _real_chunks(n: int, corpus="corpus") -> list[str]:
    """Diverse ~chunk-length passages pulled from the actual corpus PDFs.

    A meaningful ranking test needs genuinely distinct neighbours; real
    documents give that where repeated synthetic filler does not. Returns [] if
    no corpus is present (caller falls back to synthetic text).
    """
    from pypdf import PdfReader

    out: list[str] = []
    for f in sorted(glob.glob(f"{corpus}/*.pdf")):
        try:
            r = PdfReader(f)
        except Exception:  # noqa: BLE001
            continue
        npg = len(r.pages)
        for p in range(5, npg - 3, max(3, npg // 12)):
            try:
                t = re.sub(r"\s+", " ", (r.pages[p].extract_text() or "")).strip()
            except Exception:  # noqa: BLE001
                t = ""
            if len(t) > 1200:
                out.append(t[:3000])
            if len(out) >= n:
                return out
    return out


def _chunks(model: SentenceTransformer, n: int, target_tokens: int) -> list[str]:
    real = _real_chunks(n)
    if len(real) >= 32:
        return real
    tok = model.tokenizer
    ids = tok(_POOL * 30, add_special_tokens=False)["input_ids"]
    body = max(1, target_tokens - 2)
    step = max(1, (len(ids) - body) // n)
    return [
        f"[chunk {i}] " + tok.decode(ids[(i * step) % (len(ids) - body):][:body])
        for i in range(n)
    ]


def _embed(model: SentenceTransformer, texts: list[str], bs: int) -> torch.Tensor:
    return model.encode(
        texts, batch_size=bs, normalize_embeddings=True, convert_to_tensor=True
    ).float()


def _timed(model: SentenceTransformer, texts: list[str], bs: int) -> float:
    model.encode(texts, batch_size=bs, normalize_embeddings=True)  # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.encode(texts, batch_size=bs, normalize_embeddings=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _ranking_agreement(ref: torch.Tensor, other: torch.Tensor, k: int = 10) -> dict:
    """Every row is a query against all other rows; compare top-k neighbours."""
    n = ref.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=ref.device)
    sr = ref @ ref.T
    so = other @ other.T
    sr[eye] = -2.0  # exclude self
    so[eye] = -2.0
    top_r = sr.topk(k, dim=1).indices
    top_o = so.topk(k, dim=1).indices
    top1 = (top_r[:, 0] == top_o[:, 0]).float().mean().item()
    ov5 = _overlap(top_r[:, :5], top_o[:, :5])
    ov10 = _overlap(top_r, top_o)
    # Max displacement: for each ref top-1, where does it rank in `other`?
    rank_in_other = (so.argsort(dim=1, descending=True) == top_r[:, :1]).float().argmax(1)
    max_disp = int(rank_in_other.max().item())
    return {"top1_same": top1, "top5_overlap": ov5, "top10_overlap": ov10,
            "max_disp_of_ref_top1": max_disp}


def _overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    out = [len(set(a[i].tolist()) & set(b[i].tolist())) / a.shape[1] for i in range(a.shape[0])]
    return sum(out) / len(out)


def main() -> int:
    if not torch.cuda.is_available():
        print("STOP: CUDA not available.", file=sys.stderr)
        return 1

    target_tokens = min(settings.chunk_size, 512)
    bs = settings.embed_batch_size

    base = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    texts = _chunks(base, 384, target_tokens)
    n = len(texts)
    src = "real corpus/*.pdf" if len(_real_chunks(1)) else "synthetic"
    enc = base.tokenizer(texts, truncation=True, max_length=base.max_seq_length)
    total_tokens = sum(len(x) for x in enc["input_ids"])
    print(f"model: BAAI/bge-base-en-v1.5  | {n} {src} chunks @ ~{total_tokens // n} tokens "
          f"(CHUNK_SIZE={settings.chunk_size})  | batch={bs}  | device={next(base.parameters()).device}")
    print(f"default forward dtype: {next(base.parameters()).dtype}\n")

    ref_emb = _embed(base, texts, bs)

    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    ref_s = None
    print(f"{'dtype':>6} {'emb/s':>9} {'tok/s':>10} {'speedup':>8} "
          f"{'cos_min':>9} {'cos_mean':>9} {'top1=':>7} {'top5ov':>7} {'top10ov':>8} {'maxdisp':>8}")
    for name, dt in dtypes.items():
        m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
        if dt != torch.float32:
            m = m.to(dt)
        dt_s = _timed(m, texts, bs)
        emb = _embed(m, texts, bs)
        if name == "fp32":
            ref_s = dt_s
            ref_emb = emb  # use this dtype's own fp32 as the reference
        cos = (ref_emb * emb).sum(dim=1)  # both normalized -> cosine
        rk = _ranking_agreement(ref_emb, emb)
        speedup = ref_s / dt_s if ref_s else 1.0
        print(f"{name:>6} {n/dt_s:>9.1f} {total_tokens/dt_s:>10,.0f} {speedup:>7.2f}x "
              f"{cos.min().item():>9.5f} {cos.mean().item():>9.5f} "
              f"{rk['top1_same']*100:>6.1f}% {rk['top5_overlap']*100:>6.1f}% "
              f"{rk['top10_overlap']*100:>7.1f}% {rk['max_disp_of_ref_top1']:>8d}")
        del m
        torch.cuda.empty_cache()
    print("\nAgreement is measured against fp32. top1=/top5ov/top10ov are ranking")
    print("stability vs fp32; maxdisp is the worst rank the fp32 top-1 neighbour fell to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
