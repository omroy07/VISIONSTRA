"""Text metrics for LLM answers.

These are lexical heuristics, not semantic judgments. They are cheap, deterministic and
need no extra model, but they can miss paraphrases. See docs/methodology.md for the limits.
"""
import re
from collections import Counter

STOP = set("""a an the is are was were be been of in on at to for from by with and or but
as it its this that these those what which who whom how many much do does did has have had
i you he she we they them his her their our your not no so than then there here into about""".split())


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_words(text):
    return [w for w in words(text) if w not in STOP]


def exact_match(pred, ref):
    return float(words(pred) == words(ref))


def token_f1(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def contains_reference(pred, ref):
    """True if every word of the reference shows up in the answer (good for short factual refs)."""
    r = words(ref)
    return float(bool(r) and all(w in set(words(pred)) for w in r))


def relevance(pred, question):
    """Share of the question's content words that the answer picks up on."""
    q = set(content_words(question))
    if not q:
        return None
    return len(q & set(content_words(pred))) / len(q)


def groundedness(pred, context):
    """Share of the answer's content words that are found in the context."""
    a = content_words(pred)
    if not a:
        return None
    ctx = set(content_words(context))
    return sum(w in ctx for w in a) / len(a)


def token_cost(in_tokens, out_tokens, price_in=None, price_out=None):
    """Prices are USD per 1M tokens and have to be passed in; no defaults on purpose."""
    out = {"input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
    if price_in is None or price_out is None:
        out["cost_usd"] = None
    else:
        out["cost_usd"] = round(in_tokens / 1e6 * price_in + out_tokens / 1e6 * price_out, 6)
    return out
