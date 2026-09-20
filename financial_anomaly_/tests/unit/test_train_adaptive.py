import pytest

from scripts.train_adaptive import high_recall_threshold


def test_selects_precision_over_highest_threshold_when_recall_is_satisfied():
    # Both cutoffs exceed 85% recall; including the last fraud improves precision.
    labels = [1] * 6 + [0, 1]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3]
    assert high_recall_threshold(labels, scores) == 0.3


def test_recall_requirement_is_strict_and_preserves_ties():
    labels = [1] * 20 + [0]
    scores = [0.9] * 17 + [0.2] * 3 + [0.2]
    assert high_recall_threshold(labels, scores) == 0.2


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_rejects_single_class_validation(labels):
    with pytest.raises(ValueError):
        high_recall_threshold(labels, [0.9, 0.1])
