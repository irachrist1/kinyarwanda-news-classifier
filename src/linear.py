"""Train and evaluate linear text classifiers on the prepared split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


CONFIGS = {
    "majority": {"kind": "majority"},
    "word_unigram": {"kind": "word", "ngram_range": (1, 1)},
    "word_bigram": {"kind": "word", "ngram_range": (1, 2)},
    "character": {"kind": "character", "ngram_range": (3, 5)},
    "word_character": {"kind": "combined"},
}


def build_model(name: str):
    config = CONFIGS[name]
    if config["kind"] == "majority":
        return DummyClassifier(strategy="most_frequent")
    if config["kind"] == "combined":
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True, dtype=np.float32)),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=150_000, sublinear_tf=True, dtype=np.float32)),
        ])
    elif config["kind"] == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=config["ngram_range"], min_df=2,
            max_features=100_000, sublinear_tf=True, dtype=np.float32
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=config["ngram_range"], min_df=2,
            max_features=150_000, sublinear_tf=True, dtype=np.float32
        )
    return make_pipeline(vectorizer, LinearSVC(C=1.0, random_state=42))


def train(name: str, data_dir: Path, model_dir: Path, results_dir: Path) -> dict:
    train_data = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    model = build_model(name)
    model.fit(train_data["text"], train_data["label"])
    predictions = model.predict(validation["text"])
    metrics = {
        "experiment": name,
        "validation_accuracy": accuracy_score(validation["label"], predictions),
        "validation_macro_f1": f1_score(validation["label"], predictions, average="macro"),
        "validation_weighted_f1": f1_score(validation["label"], predictions, average="weighted"),
        "train_rows": len(train_data),
        "validation_rows": len(validation),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{name}.joblib")
    (results_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame({"label": validation["label"], "prediction": predictions}).to_csv(
        results_dir / f"{name}_validation_predictions.csv", index=False
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=CONFIGS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    args = parser.parse_args()
    print(json.dumps(train(args.name, args.data_dir, args.model_dir, args.results_dir), indent=2))
