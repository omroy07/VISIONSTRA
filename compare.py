"""Side-by-side table of saved runs.

    python compare.py results/cv_yolov8n-nms0.7.json results/cv_yolov8n-nms0.5.json
"""
import sys

from evalkit.store import compare

if len(sys.argv) < 3:
    sys.exit("usage: python compare.py run1.json run2.json [...]")

names, table = compare(sys.argv[1:])
width = max(len(k) for k in table) + 2
print("metric".ljust(width) + "".join(n.ljust(24) for n in names))
print("-" * (width + 24 * len(names)))
for metric, vals in table.items():
    print(metric.ljust(width) + "".join(str(v).ljust(24) for v in vals))
