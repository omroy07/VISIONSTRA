# Methodology

## What was evaluated

- **CV:** YOLOv8n (COCO-pretrained, ONNX export, 640x640 input) run with onnxruntime on CPU, on COCO128.
- **LLM:** the runner and metrics are implemented and unit tested; a real model run needs an API key (see README).

## Test setup (CV)

1. All images are decoded once before timing starts.
2. 5 warm-up inferences are run and not timed.
3. Each image is timed on its own: letterbox resize, model run, decoding of the output, NMS.
4. The detector keeps everything with score >= 0.001, so the precision/recall curve is complete for mAP.
5. Predictions and labels are compared with `evalkit/cv_metrics.py`.

## CV metrics

- **Matching:** for each image and class, predictions are taken in descending score order. Each is matched to the unclaimed ground-truth box with the highest IoU, if that IoU passes the threshold. Otherwise it is a false positive. Unmatched ground-truth boxes are false negatives.
- **Precision / recall / F1:** computed over all classes together, using only predictions with score >= `--conf` (default 0.25) at IoU >= `--iou` (default 0.5).
- **AP:** area under the precision-recall curve, all-point interpolation. Built from all predictions of a class across the dataset.
- **mAP50:** mean AP over classes present in the labels, IoU 0.5. **mAP50-95:** the same, averaged over IoU 0.50 to 0.95 in steps of 0.05.
- **Latency:** per-image wall-clock time in ms; mean, median, p95 and max are reported. **FPS** = 1000 / mean latency (single image at a time, no batching).

Sanity check: the published COCO val mAP50-95 for YOLOv8n is 37.3. We get 44.3 on COCO128, which is higher, as expected since COCO128 comes from the training split. The metric functions are also checked against hand-computed cases in `tests/test_cv_metrics.py`.

## LLM metrics

All are simple word-overlap heuristics, chosen because they are deterministic, free and need no second model.

| metric | how it is computed |
|---|---|
| accuracy | exact match, "answer contains every word of the reference", and token F1, each against `reference` |
| relevance | share of the question's content words (stop words removed) that appear in the answer |
| groundedness | share of the answer's content words that appear in the context |
| hallucination rate | share of answers with groundedness below `--halluc-thr` (default 0.6) |
| latency | wall-clock time per call, same stats as CV |
| tokens / cost | token counts from the API response; cost = tokens x price per 1M tokens supplied by the user |

Limits: a correct answer that is a paraphrase can score low on groundedness and relevance, and a wrong answer that reuses context words can score high. Treat these as a cheap first signal. For a stronger check, add a judge-model scorer that returns the same per-case fields and compare it with these on a few dozen hand-checked answers. The raw answers are saved next to each LLM result for exactly that kind of review.

## Metrics that don't apply

If the inputs for a metric are missing, the metric is left out of `metrics` and listed in `not_applicable` with a reason: no references, no context, no token counts from the model wrapper, no prices. Nothing is silently set to 0.

## Reproducing a run

Each result json stores the config (all CLI arguments), OS, Python version, CPU count and git commit. To repeat a run: check out that commit, run `python data/download.py`, and pass the same arguments. Accuracy metrics come out identical on repeated runs (checked). Latency does not, and it depends on hardware, so compare latency only between runs on the same machine.

The ONNX file comes from a third-party GitHub repo (URL in `data/download.py`). To use the official weights instead: `pip install ultralytics`, `yolo export model=yolov8n.pt format=onnx`, and put the file at `models/weights/yolov8n.onnx`.

## Comparing versions

Give each run a distinct `--name` and run `python compare.py <run1.json> <run2.json>`. Example in the README: NMS IoU 0.7 vs 0.5 (lower NMS threshold removes more duplicate boxes, so precision goes up, 0.713 to 0.757, with little change in recall).
