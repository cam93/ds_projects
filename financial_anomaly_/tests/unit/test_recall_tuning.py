import numpy as np
import pytest

from scripts.tune_recall import recall_threshold


def test_recall_threshold_preserves_ties_and_meets_target():
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.2, 0.8, 0.1])
    threshold = recall_threshold(labels, scores, 0.70)
    assert threshold == 0.8
    assert ((scores >= threshold) & (labels == 1)).sum() / labels.sum() >= 0.70
    assert recall_threshold(labels, scores, 1) == 0.2


@pytest.mark.parametrize("target", [0, 1.1])
def test_invalid_recall_target_rejected(target):
    with pytest.raises(ValueError):
        recall_threshold([1, 0], [0.9, 0.1], target)


def test_single_class_calibration_rejected():
    with pytest.raises(ValueError):
        recall_threshold([1, 1], [0.9, 0.1], 0.85)


@pytest.mark.parametrize(
    "labels,scores", [([1, 0], [0.9]), ([1, 0], [float("nan"), 0.1]), ([0.5, 0], [0.9, 0.1])]
)
def test_invalid_calibration_data_rejected(labels, scores):
    with pytest.raises(ValueError):
        recall_threshold(labels, scores, 0.86)
