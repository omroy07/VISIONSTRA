import numpy as np

from . import llm_metrics as m
from .timing import Timer, latency_stats


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def run(cases, generate, halluc_thr=0.6, price_in=None, price_out=None):
    """cases: [{"question": str, "context": str (opt), "reference": str (opt)}]
    generate(question, context) -> {"text": str, "input_tokens": int|None, "output_tokens": int|None}

    Returns (metrics, not_applicable, per_case_rows).
    """
    rows, lat = [], []
    tok_in = tok_out = 0
    tokens_known = True

    for c in cases:
        q, ctx, ref = c["question"], c.get("context", ""), c.get("reference", "")
        with Timer() as t:
            r = generate(q, ctx)
        lat.append(t.ms)
        text = r["text"]

        if r.get("input_tokens") is None or r.get("output_tokens") is None:
            tokens_known = False
        else:
            tok_in += r["input_tokens"]
            tok_out += r["output_tokens"]

        rows.append({
            "question": q,
            "answer": text,
            "latency_ms": round(t.ms, 1),
            "exact_match": m.exact_match(text, ref) if ref else None,
            "token_f1": m.token_f1(text, ref) if ref else None,
            "contains_ref": m.contains_reference(text, ref) if ref else None,
            "relevance": m.relevance(text, q),
            "groundedness": m.groundedness(text, ctx) if ctx else None,
        })

    na = {}
    metrics = {"n_cases": len(cases), "latency": latency_stats(lat)}

    if any(r["exact_match"] is not None for r in rows):
        metrics["accuracy_exact"] = _mean(r["exact_match"] for r in rows)
        metrics["accuracy_contains_ref"] = _mean(r["contains_ref"] for r in rows)
        metrics["accuracy_token_f1"] = _mean(r["token_f1"] for r in rows)
    else:
        na["accuracy"] = "no reference answers in the test cases"

    metrics["relevance"] = _mean(r["relevance"] for r in rows)

    grounded = [r["groundedness"] for r in rows if r["groundedness"] is not None]
    if grounded:
        metrics["groundedness"] = _mean(grounded)
        metrics["hallucination_rate"] = round(sum(g < halluc_thr for g in grounded) / len(grounded), 4)
        metrics["hallucination_threshold"] = halluc_thr
    else:
        na["groundedness"] = "no context in the test cases"
        na["hallucination_rate"] = "no context in the test cases"

    if tokens_known:
        metrics["tokens"] = m.token_cost(tok_in, tok_out, price_in, price_out)
        if metrics["tokens"]["cost_usd"] is None:
            na["cost_usd"] = "no prices given (--price-in / --price-out)"
    else:
        na["tokens"] = "model wrapper did not report token counts"

    return metrics, na, rows
