import numpy as np

from src.evaluate import bootstrap


def test_bootstrap_keeps_paired_predictions_together() -> None:
    actual = np.array([1, 1, 2, 2, 3, 3])
    baseline = np.array([1, 2, 2, 1, 3, 1])
    selected = np.array([1, 1, 2, 2, 3, 3])

    result = bootstrap(actual, baseline, selected, [1, 2, 3])

    assert result["selected_accuracy_95_percent_ci"] == [1.0, 1.0]
    assert result["macro_f1_difference_95_percent_ci"][0] >= 0
    assert result["baseline_only_correct"] == 0
    assert result["selected_only_correct"] == 3
