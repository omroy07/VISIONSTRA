# Model evaluation framework

Small toolkit to benchmark AI models with numbers instead of impressions. It covers two cases:

- **Object detection (CV):** precision, recall, F1, mAP@0.5, mAP@0.5:0.95, latency, FPS
- **LLM answers:** accuracy, relevance, groundedness, hallucination rate, latency, token usage and cost

Every number is computed from a dataset at run time. Each run is saved as a json file, so model versions or settings can be compared later.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                # 21 tests, should all pass
```

## Run the CV benchmark

```
python data/download.py         # YOLOv8n (ONNX) + COCO128, about 20 MB
python run_cv.py --name yolov8n-nms0.7
python run_cv.py --name yolov8n-nms0.5 --nms-iou 0.5
python compare.py results/cv_yolov8n-nms0.7.json results/cv_yolov8n-nms0.5.json
```

Use your own data: point `--images` and `--labels` at a folder of images and a folder of YOLO-format `.txt` labels (`class cx cy w h`, normalised). Use your own model: replace `models/yolo_onnx.py` with any callable that takes a BGR image and returns `{"boxes": xyxy, "scores": ..., "cls": ...}`.

## Run the LLM benchmark

```
pip install anthropic
set ANTHROPIC_API_KEY=...       # Linux/Mac: export
python run_llm.py --name sonnet-run1 --model claude-sonnet-5 --price-in 3 --price-out 15
```

Test cases live in `data/llm/cases.json` (`question`, optional `context`, optional `reference`). If a metric can't be computed, for example groundedness without a context, it is listed under `not_applicable` in the result instead of being reported as 0. Prices are per 1M tokens and must be passed in; without them cost is shown as not applicable.

To test another LLM, write a function `generate(question, context) -> {"text", "input_tokens", "output_tokens"}` like the one in `models/claude_api.py`.

## Sample results

YOLOv8n on COCO128 (128 images, 929 boxes), conf 0.25, IoU 0.5 for P/R/F1:

| run | precision | recall | F1 | mAP50 | mAP50-95 | latency (mean) | FPS |
|---|---|---|---|---|---|---|---|
| yolov8n-nms0.7 | 0.713 | 0.498 | 0.587 | 0.594 | 0.443 | 76.8 ms | 13.0 |
| yolov8n-nms0.5 | 0.757 | 0.494 | 0.598 | 0.604 | 0.441 | 79.1 ms | 12.7 |

Raw files are in `results/`. Please read these before quoting the numbers:

- COCO128 is taken from COCO train2017, which YOLOv8 was trained on, so accuracy here is optimistic. It shows the pipeline works, not how the model generalises. Use a held-out set for real decisions.
- Latency/FPS were measured on an Intel Core i3-1115G4 (2 cores), 8 GB RAM, Windows, CPU only (onnxruntime). They cover preprocessing, inference and NMS, but not image decoding. Latency varies slightly from run to run and between machines; the accuracy metrics do not.
- The LLM side is covered by unit tests with fake models only. No real LLM run is included yet because it needs an API key.

More detail on the formulas and choices is in `docs/methodology.md`.

## Layout

```
evalkit/     metrics, timing, llm loop, result storage
models/      YOLO ONNX wrapper, Anthropic wrapper
data/        download script, LLM test cases
run_cv.py  run_llm.py  compare.py
tests/  results/  docs/
```
