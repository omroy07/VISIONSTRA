import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def env_info():
    info = {"python": platform.python_version(), "os": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3
        ).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def save(kind, name, dataset, metrics, config, not_applicable=None, out_dir="results"):
    """One json file per (kind, name). Re-running with the same name overwrites it."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rec = {
        "kind": kind,
        "name": name,
        "dataset": dataset,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "env": env_info(),
        "metrics": metrics,
        "not_applicable": not_applicable or {},
    }
    path = Path(out_dir) / f"{kind}_{name}.json"
    path.write_text(json.dumps(rec, indent=2))
    return path


def load(path):
    return json.loads(Path(path).read_text())


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def compare(paths, skip=("per_class_ap50",)):
    """Return {metric: [value per file]} for metrics found in any of the runs."""
    runs = [load(p) for p in paths]
    flat = [flatten({k: v for k, v in r["metrics"].items() if k not in skip}) for r in runs]
    keys = []
    for f in flat:
        for k in f:
            if k not in keys:
                keys.append(k)
    return [r["name"] for r in runs], {k: [f.get(k, "n/a") for f in flat] for k in keys}
