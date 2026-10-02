"""Fit the selected classifiers and evaluate once on the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.linear import build_model


SEED = 42
BOOTSTRAP_SAMPLES = 2_000


def scores(actual: np.ndarray, predicted: np.ndarray, labels: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
    }


def bootstrap(actual: np.ndarray, baseline: np.ndarray, selected: np.ndarray, labels: list[int]) -> dict:
    generator = np.random.default_rng(SEED)
    sample_count = len(actual)
    accuracy_samples = []
    f1_samples = []
    difference_samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indexes = generator.integers(0, sample_count, sample_count)
        truth = actual[indexes]
        current = selected[indexes]
        previous = baseline[indexes]
        accuracy_samples.append(accuracy_score(truth, current))
        current_f1 = f1_score(truth, current, labels=labels, average="macro", zero_division=0)
        previous_f1 = f1_score(truth, previous, labels=labels, average="macro", zero_division=0)
        f1_samples.append(current_f1)
        difference_samples.append(current_f1 - previous_f1)
    interval = lambda values: [float(value) for value in np.quantile(values, [0.025, 0.975])]
    baseline_only = int(np.sum((baseline == actual) & (selected != actual)))
    selected_only = int(np.sum((selected == actual) & (baseline != actual)))
    discordant = baseline_only + selected_only
    return {
        "resamples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "selected_accuracy_95_percent_ci": interval(accuracy_samples),
        "selected_macro_f1_95_percent_ci": interval(f1_samples),
        "macro_f1_difference_95_percent_ci": interval(difference_samples),
        "mcnemar_exact_two_sided_p": float(binomtest(min(baseline_only, selected_only), discordant, p=0.5).pvalue) if discordant else 1.0,
        "baseline_only_correct": baseline_only,
        "selected_only_correct": selected_only,
    }


def evaluate(data_dir: Path = Path("data/processed"), results_dir: Path = Path("results")) -> dict:
    training = pd.concat(
        [pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "validation.csv")],
        ignore_index=True,
    )
    test = pd.read_csv(data_dir / "test.csv")
    label_table = pd.read_csv(data_dir / "labels.csv")
    labels = label_table["label"].astype(int).tolist()
    label_names = label_table["en_label"].tolist()
    predictions = {}
    for name in ("word_bigram", "word_character"):
        model = build_model(name)
        model.fit(training["text"], training["label"])
        predictions[name] = model.predict(test["text"])
        if name == "word_character":
            artifact = Path("artifacts/final.joblib")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, artifact, compress=3)

    actual = test["label"].to_numpy()
    final = predictions["word_character"]
    baseline = predictions["word_bigram"]
    result = {
        "training_rows": len(training),
        "test_rows": len(test),
        "selection_metric": "validation macro F1",
        "baseline": scores(actual, baseline, labels),
        "selected": scores(actual, final, labels),
        "uncertainty": bootstrap(actual, baseline, final, labels),
        "classes": classification_report(actual, final, labels=labels, target_names=label_names, output_dict=True, zero_division=0),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    output = pd.DataFrame({"row_id": np.arange(len(test)), "label": actual})
    output["word_bigram_prediction"] = baseline
    output["final_prediction"] = final
    output.to_csv(results_dir / "test_predictions.csv", index=False)
    matrix = confusion_matrix(actual, final, labels=labels)
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(results_dir / "confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(10, 8))
    picture = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), label_names, rotation=65, ha="right")
    axis.set_yticks(range(len(labels)), label_names)
    axis.set_xlabel("Predicted topic")
    axis.set_ylabel("Actual topic")
    figure.colorbar(picture, ax=axis)
    figure.tight_layout()
    (results_dir / "figures").mkdir(exist_ok=True)
    figure.savefig(results_dir / "figures/confusion_matrix.png", dpi=180)
    plt.close(figure)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
