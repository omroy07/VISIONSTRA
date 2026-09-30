import numpy as np
import pytest

from evalkit.cv_metrics import average_precision, evaluate, iou_matrix


def gt(boxes, cls):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "cls": np.array(cls, int)}


def pr(boxes, scores, cls):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "cls": np.array(cls, int)}


def test_iou_basic():
    a = [[0, 0, 10, 10]]
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0.0
    # half overlap: inter 50, union 150
    assert iou_matrix(a, [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)


def test_iou_empty():
    assert iou_matrix(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


def test_perfect_detection():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    p = {"a": pr([[0, 0, 10, 10]], [0.9], [0])}
    r = evaluate(g, p)
    assert (r["precision"], r["recall"], r["f1"]) == (1, 1, 1)
    assert r["map50"] == pytest.approx(1)
    assert r["map50_95"] == pytest.approx(1)


def test_missed_object():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    r = evaluate(g, {"a": pr([], [], [])})
    assert r["recall"] == 0 and r["fn"] == 1 and r["map50"] == 0


def test_missing_image_in_predictions():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    r = evaluate(g, {})
    assert r["fn"] == 1 and r["recall"] == 0


def test_false_positive_lowers_precision():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    p = {"a": pr([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8], [0, 0])}
    r = evaluate(g, p)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == 1


def test_duplicate_detection_is_a_false_positive():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    p = {"a": pr([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])}
    r = evaluate(g, p)
    assert (r["tp"], r["fp"]) == (1, 1)


def test_wrong_class_does_not_match():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    p = {"a": pr([[0, 0, 10, 10]], [0.9], [1])}
    r = evaluate(g, p)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_conf_threshold_only_affects_prf_not_map():
    g = {"a": gt([[0, 0, 10, 10]], [0])}
    p = {"a": pr([[0, 0, 10, 10]], [0.1], [0])}
    r = evaluate(g, p, conf_thr=0.25)
    assert r["recall"] == 0
    assert r["map50"] == pytest.approx(1)


def test_average_precision_hand_computed():
    # 2 gt, ranked predictions: tp, fp, tp
    # precision 1, .5, .667 / recall .5, .5, 1 -> AP = .5*1 + .5*.667
    ap = average_precision(np.array([True, False, True]), n_gt=2)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_map_is_lower_at_strict_iou():
    g = {"a": gt([[0, 0, 100, 100]], [0])}
    p = {"a": pr([[0, 0, 100, 80]], [0.9], [0])}  # iou 0.8
    r = evaluate(g, p)
    assert r["map50"] == pytest.approx(1)
    assert r["map50_95"] < r["map50"]
