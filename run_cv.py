"""Benchmark a detector on a folder of images + YOLO-format label files.

    python run_cv.py --name yolov8n-nms0.7
    python run_cv.py --name yolov8n-nms0.5 --nms-iou 0.5

Metrics come from the predictions vs. the labels; nothing is hardcoded.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from evalkit.cv_metrics import evaluate
from evalkit.store import save
from evalkit.timing import Timer, latency_stats
from models.yolo_onnx import YoloOnnx

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_labels(path, w, h):
    """YOLO txt (cls cx cy w h, normalised) -> xyxy pixels."""
    boxes, cls = [], []
    if path.exists():
        for line in path.read_text().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            cls.append(c)
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "cls": np.array(cls, int)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label for this run / model version")
    ap.add_argument("--model", default="models/weights/yolov8n.onnx")
    ap.add_argument("--images", default="data/coco128/images/train2017")
    ap.add_argument("--labels", default="data/coco128/labels/train2017")
    ap.add_argument("--conf", type=float, default=0.25, help="score cutoff for precision/recall/F1")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a match for P/R/F1")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N images")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if not Path(a.model).exists() or not Path(a.images).exists():
        sys.exit("model or dataset missing, run: python data/download.py")

    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    if a.limit:
        files = files[: a.limit]

    # decode all images up front so disk/JPEG time doesn't end up in the latency numbers
    imgs, gts = {}, {}
    for f in files:
        im = cv2.imread(str(f))
        imgs[f.stem] = im
        gts[f.stem] = load_labels(Path(a.labels) / f"{f.stem}.txt", im.shape[1], im.shape[0])

    # low conf so the PR curve (mAP) sees everything; the P/R/F1 cutoff is applied in evaluate()
    model = YoloOnnx(a.model, conf=0.001, nms_iou=a.nms_iou, threads=a.threads)

    for im in list(imgs.values())[: a.warmup]:
        model(im)

    preds, times = {}, []
    for k, im in imgs.items():
        with Timer() as t:
            preds[k] = model(im)
        times.append(t.ms)

    res = evaluate(gts, preds, conf_thr=a.conf, iou_thr=a.iou)
    metrics = {
        "precision": round(res["precision"], 4),
        "recall": round(res["recall"], 4),
        "f1": round(res["f1"], 4),
        "map50": round(res["map50"], 4),
        "map50_95": round(res["map50_95"], 4),
        "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
        "n_images": len(files),
        "n_gt_boxes": int(sum(len(g["cls"]) for g in gts.values())),
        "latency": latency_stats(times),
        "per_class_ap50": {str(k): round(v, 4) for k, v in res["per_class_ap50"].items()},
    }
    config = {k: v for k, v in vars(a).items() if k != "out"}
    path = save("cv", a.name, str(a.images), metrics, config, out_dir=a.out)

    show = {k: v for k, v in metrics.items() if k != "per_class_ap50"}
    print(json.dumps(show, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
