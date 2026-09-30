import time
import numpy as np


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000


def latency_stats(samples_ms):
    """mean / p50 / p95 / max in ms, plus fps computed from the mean."""
    if len(samples_ms) == 0:
        return None
    s = np.asarray(samples_ms, dtype=float)
    mean = float(s.mean())
    return {
        "n": len(s),
        "mean_ms": round(mean, 2),
        "p50_ms": round(float(np.percentile(s, 50)), 2),
        "p95_ms": round(float(np.percentile(s, 95)), 2),
        "max_ms": round(float(s.max()), 2),
        "fps": round(1000.0 / mean, 2) if mean > 0 else None,
    }
