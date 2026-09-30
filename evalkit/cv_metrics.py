"""Detection metrics. Boxes are xyxy in pixels, numpy arrays.

gts:   {image_id: {"boxes": (n,4), "cls": (n,)}}
preds: {image_id: {"boxes": (m,4), "scores": (m,), "cls": (m,)}}
"""
import numpy as np

IOU_RANGE = np.arange(0.5, 0.96, 0.05)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_class(gt_boxes, pred_boxes, pred_scores, thr):
    """Greedy match for one image + one class.
    Returns (tp flags, scores), both in descending-score order."""
    order = np.argsort(-np.asarray(pred_scores), kind="stable")
    pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)[order]
    scores = np.asarray(pred_scores)[order]
    tp = np.zeros(len(order), dtype=bool)
    if len(gt_boxes) and len(order):
        ious = iou_matrix(pred_boxes, gt_boxes)
        used = set()
        for i in range(len(order)):
            # best gt that hasn't been claimed yet
            cand = [(ious[i, j], j) for j in range(ious.shape[1]) if j not in used]
            if not cand:
                break
            best, j = max(cand)
            if best >= thr:
                tp[i] = True
                used.add(j)
    return tp, scores


def average_precision(tp, n_gt):
    """tp must be sorted by score desc. All-point interpolated AP."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _collect(gts, preds, cls_id, thr):
    """tp flags (sorted by score) for one class over the whole dataset, plus gt count."""
    all_tp, all_sc, n_gt = [], [], 0
    for img, g in gts.items():
        gmask = g["cls"] == cls_id
        n_gt += int(gmask.sum())
        p = preds.get(img)
        if p is None:
            continue
        pmask = p["cls"] == cls_id
        if not pmask.any():
            continue
        tp, sc = match_class(g["boxes"][gmask], p["boxes"][pmask], p["scores"][pmask], thr)
        all_tp.append(tp)
        all_sc.append(sc)
    if not all_tp:
        return np.zeros(0, dtype=bool), n_gt
    tp = np.concatenate(all_tp)
    sc = np.concatenate(all_sc)
    return tp[np.argsort(-sc, kind="stable")], n_gt


def map_score(gts, preds, thrs):
    """AP averaged over classes present in the ground truth and over the given IoU thresholds."""
    classes = sorted({int(c) for g in gts.values() for c in g["cls"]})
    if not classes:
        return None, {}
    per_class = {}
    for c in classes:
        aps = []
        for t in thrs:
            tp, n_gt = _collect(gts, preds, c, t)
            aps.append(average_precision(tp, n_gt))
        per_class[c] = float(np.mean(aps))
    return float(np.mean(list(per_class.values()))), per_class


def precision_recall_f1(gts, preds, conf_thr=0.25, iou_thr=0.5):
    """Micro-averaged over classes, using only predictions with score >= conf_thr."""
    tp = fp = fn = 0
    classes = {int(c) for g in gts.values() for c in g["cls"]}
    classes |= {int(c) for p in preds.values() for c in p["cls"]}
    for img, g in gts.items():
        p = preds.get(img)
        for c in classes:
            gmask = g["cls"] == c
            n_g = int(gmask.sum())
            pmask = None if p is None else (p["cls"] == c) & (p["scores"] >= conf_thr)
            if pmask is None or not pmask.any():
                fn += n_g
                continue
            flags, _ = match_class(g["boxes"][gmask], p["boxes"][pmask], p["scores"][pmask], iou_thr)
            tp += int(flags.sum())
            fp += int((~flags).sum())
            fn += n_g - int(flags.sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(gts, preds, conf_thr=0.25, iou_thr=0.5):
    out = precision_recall_f1(gts, preds, conf_thr, iou_thr)
    out["map50"], per_class = map_score(gts, preds, [0.5])
    out["map50_95"], _ = map_score(gts, preds, IOU_RANGE)
    out["per_class_ap50"] = per_class
    return out
