"""
POST /detect in backend/app.py, with a fake YOLO model.

Needs Flask, OpenCV and NumPy (backend/requirements.txt); skipped otherwise.
No model weights, camera or network are used.
"""

import io
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import flask  # noqa: F401
    import numpy as np
except ImportError:          # pragma: no cover - depends on the environment
    cv2 = None


class _Box:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [xyxy]
        self.cls = [cls]
        self.conf = [conf]


class _FakeYOLO:
    """Returns whatever boxes the test put in ``next_boxes``."""
    names = {0: "person", 1: "bicycle", 2: "car"}
    next_boxes = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, frame, **kwargs):
        return [types.SimpleNamespace(boxes=list(_FakeYOLO.next_boxes))]


@unittest.skipIf(cv2 is None, "Flask, OpenCV and NumPy are required for the API test")
class TestDetectEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._saved_ultralytics = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=_FakeYOLO)
        sys.modules.pop("app", None)
        import app as backend_app
        cls.app = backend_app
        cls.client = backend_app.app.test_client()
        ok, encoded = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))
        cls.jpeg = encoded.tobytes()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app", None)
        if cls._saved_ultralytics is None:
            sys.modules.pop("ultralytics", None)
        else:
            sys.modules["ultralytics"] = cls._saved_ultralytics

    def post(self, boxes, stream_id="test"):
        _FakeYOLO.next_boxes = boxes
        return self.client.post("/detect", data={
            "frame": (io.BytesIO(self.jpeg), "frame.jpg"), "stream_id": stream_id,
        }, content_type="multipart/form-data")

    def test_task_scene_is_ranked_and_the_primary_is_announced(self):
        # Boxes sized so the estimator gives ~8 m, ~5 m and ~2 m.
        boxes = [
            _Box([100, 300, 144, 449], 0, 0.9),    # person, height 149 px -> 8 m
            _Box([500, 300, 752, 510], 2, 0.9),    # car,    height 210 px -> 5 m
            _Box([900, 200, 1110, 568], 1, 0.9),   # bicycle,height 368 px -> 2 m
        ]
        response = self.post(boxes, stream_id="scene")
        self.assertEqual(response.status_code, 200)
        ranked = response.get_json()
        self.assertEqual([d["object"] for d in ranked], ["bicycle", "car", "person"])
        self.assertEqual([d["priority"] for d in ranked], ["HIGH", "MEDIUM", "LOW"])
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertIn("breakdown", ranked[0])
        self.assertEqual(ranked[0]["confidence"], 0.9)
        # Exactly one sentence, on the primary hazard.
        self.assertIn("Warning: bicycle", ranked[0]["announcement"])
        self.assertEqual(sum("announcement" in d for d in ranked), 1)

    def test_repeat_frame_is_not_announced_again(self):
        boxes = [_Box([900, 200, 1110, 568], 1, 0.9)]
        first = self.post(boxes, stream_id="repeat").get_json()
        second = self.post(boxes, stream_id="repeat").get_json()
        self.assertIn("announcement", first[0])
        self.assertNotIn("announcement", second[0])
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

    def test_streams_are_independent(self):
        boxes = [_Box([900, 200, 1110, 568], 1, 0.9)]
        self.post(boxes, stream_id="tab-a")
        other = self.post(boxes, stream_id="tab-b").get_json()
        self.assertIn("announcement", other[0])

    def test_empty_frame(self):
        self.assertEqual(self.post([], stream_id="empty").get_json(), [])

    def test_bad_requests(self):
        self.assertEqual(self.client.post("/detect").status_code, 400)
        bad_image = self.client.post("/detect", data={
            "frame": (io.BytesIO(b"not a jpeg"), "x.jpg")}, content_type="multipart/form-data")
        self.assertEqual(bad_image.status_code, 400)
        self.assertEqual(self.post([], stream_id="../etc").status_code, 400)


if __name__ == "__main__":
    unittest.main()
