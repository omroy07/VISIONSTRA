"""YOLOv8 detector running on onnxruntime (CPU by default).

Expects an ONNX export of a YOLOv8 detection model: input 1x3x640x640,
output 1x(4+nc)x8400. Get one with `yolo export model=yolov8n.pt format=onnx`
or see data/download.py.
"""
import cv2
import numpy as np
import onnxruntime as ort


class YoloOnnx:
    def __init__(self, path, conf=0.001, nms_iou=0.7, threads=None, size=640):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.nms_iou, self.size = conf, nms_iou, size

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def __call__(self, img_bgr):
        """img_bgr: HxWx3 uint8 (cv2.imread). Returns dict with boxes(xyxy px), scores, cls."""
        h, w = img_bgr.shape[:2]
        canvas, r, left, top = self._letterbox(img_bgr)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0][0].T  # (8400, 4+nc)

        scores_all = out[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(out)), cls]
        keep = scores >= self.conf
        out, cls, scores = out[keep], cls[keep], scores[keep]
        if len(out) == 0:
            return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, int)}

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        # undo the letterbox
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        xywh = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), self.conf, self.nms_iou)
        idx = np.array(idx).reshape(-1).astype(int)
        return {"boxes": xyxy[idx], "scores": scores[idx], "cls": cls[idx]}
