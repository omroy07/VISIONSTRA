"""Fetch the YOLOv8n ONNX model and the COCO128 dataset (128 images with YOLO-format labels).

    python data/download.py

Nothing is stored in git; files go to models/weights/ and data/coco128/.
"""
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_URL = "https://raw.githubusercontent.com/Hyuto/yolov8-onnxruntime-web/master/public/model/yolov8n.onnx"
DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    model = ROOT / "models" / "weights" / "yolov8n.onnx"
    if not model.exists():
        fetch(MODEL_URL, model)

    data_dir = ROOT / "data" / "coco128"
    if not data_dir.exists():
        zpath = ROOT / "data" / "coco128.zip"
        fetch(DATA_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ROOT / "data")
        zpath.unlink()
    print("done")
