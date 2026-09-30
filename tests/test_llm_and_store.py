import numpy as np
import pytest

from evalkit import llm_metrics as m
from evalkit.llm_eval import run
from evalkit.store import compare, save
from evalkit.timing import latency_stats


def test_exact_and_f1():
    assert m.exact_match("The Pacific Ocean.", "pacific ocean") == 0.0  # extra word "the"
    assert m.exact_match("Pacific Ocean", "pacific ocean") == 1.0
    assert m.token_f1("a b c", "a b c") == 1.0
    assert m.token_f1("x y", "a b") == 0.0


def test_contains_reference():
    assert m.contains_reference("It is 212 degrees F", "212") == 1.0
    assert m.contains_reference("no idea", "212") == 0.0


def test_relevance_and_groundedness():
    assert m.relevance("Canberra is the capital of Australia", "Which city is the capital of Australia?") > 0.5
    assert m.relevance("bananas", "capital of Australia") == 0.0
    ctx = "Mars is the fourth planet from the Sun"
    assert m.groundedness("Mars is the fourth planet", ctx) == 1.0
    assert m.groundedness("dragons breathe fire", ctx) == 0.0
    assert m.groundedness("", ctx) is None


def test_cost():
    assert m.token_cost(1_000_000, 1_000_000, 3, 15)["cost_usd"] == 18
    assert m.token_cost(10, 10)["cost_usd"] is None


CASES = [
    {"question": "Capital of Australia?", "context": "The capital of Australia is Canberra.", "reference": "Canberra"},
    {"question": "Largest ocean?", "context": "The Pacific is the largest ocean.", "reference": "Pacific"},
]


def echo_reference(q, ctx):
    ref = "Canberra" if "Australia" in q else "Pacific"
    return {"text": ref, "input_tokens": 10, "output_tokens": 2}


def test_run_good_model():
    met, na, rows = run(CASES, echo_reference, price_in=1, price_out=2)
    assert met["accuracy_exact"] == 1.0
    assert met["hallucination_rate"] == 0.0
    assert met["tokens"]["total_tokens"] == 24
    assert met["tokens"]["cost_usd"] == pytest.approx(20 * 1e-6 + 4 * 2e-6)
    assert na == {}
    assert len(rows) == 2


def test_run_hallucinating_model():
    bad = lambda q, c: {"text": "purple dragons everywhere", "input_tokens": 5, "output_tokens": 3}
    met, na, _ = run(CASES, bad)
    assert met["accuracy_exact"] == 0.0
    assert met["hallucination_rate"] == 1.0
    assert "cost_usd" in na  # no prices were given


def test_run_marks_missing_inputs_not_applicable():
    cases = [{"question": "Say hi to the world"}]
    met, na, _ = run(cases, lambda q, c: {"text": "hi world"})
    assert "accuracy" in na and "groundedness" in na and "hallucination_rate" in na and "tokens" in na
    assert "accuracy_exact" not in met


def test_latency_stats():
    s = latency_stats([10, 20, 30])
    assert s["mean_ms"] == 20 and s["fps"] == 50
    assert latency_stats([]) is None


def test_save_and_compare(tmp_path):
    p1 = save("cv", "a", "ds", {"precision": 0.8, "latency": {"mean_ms": 10}}, {"x": 1}, out_dir=tmp_path)
    p2 = save("cv", "b", "ds", {"precision": 0.9, "latency": {"mean_ms": 12}, "extra": 1}, {"x": 2}, out_dir=tmp_path)
    names, table = compare([p1, p2])
    assert names == ["a", "b"]
    assert table["precision"] == [0.8, 0.9]
    assert table["latency.mean_ms"] == [10, 12]
    assert table["extra"] == ["n/a", 1]


def test_yolo_postprocess_maps_back_to_original_pixels():
    from models.yolo_onnx import YoloOnnx

    # fake session: one anchor, class 2, box centred at (320,320) size 128x128 in letterbox space
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[0, 0:4, 0] = [320, 320, 128, 128]
    out[0, 4 + 2, 0] = 0.9

    class FakeSess:
        def run(self, *_):
            return [out]

    y = YoloOnnx.__new__(YoloOnnx)
    y.sess, y.inp, y.conf, y.nms_iou, y.size = FakeSess(), "images", 0.001, 0.7, 640

    img = np.zeros((320, 640, 3), dtype=np.uint8)  # h=320, w=640 -> r=1, pad top/bottom 160
    res = y(img)
    assert res["cls"].tolist() == [2]
    # letterbox y range 256..384 -> minus 160 pad = 96..224 ; x stays 256..384
    assert res["boxes"][0] == pytest.approx([256, 96, 384, 224], abs=0.5)
