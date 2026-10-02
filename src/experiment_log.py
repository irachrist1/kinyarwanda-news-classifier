"""Rebuild the experiment log from measured validation metrics."""

import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("majority", "", "Most frequent class", "Reference score for class imbalance"),
    ("word_unigram", "", "Word unigram TF-IDF with linear SVM", "Strong, interpretable text baseline"),
    ("word_bigram", "word_unigram", "Add word bigrams", "Two-word phrases may identify topics more precisely"),
    ("character", "word_unigram", "Replace word features with character 3–5-grams", "Subword patterns may help with Kinyarwanda morphology and spelling variation"),
    ("word_character", "word_bigram", "Add character 3–5-gram features", "Combining word and subword cues may improve minority-class recall"),
    ("gru", "", "Bidirectional GRU with learned embeddings", "A sequence model may capture word order missed by TF-IDF"),
    ("lstm", "gru", "Replace GRU cells with LSTM cells", "LSTM memory gates may retain longer article context"),
    ("lstm_8epochs", "lstm", "Raise maximum epochs from 4 to 8", "More updates may help before validation loss stops falling"),
    ("lstm_8epochs_320tokens", "lstm_8epochs", "Raise input length from 160 to 320 tokens", "More article context may improve topic recognition"),
]


def build_log(results_dir: Path = Path("results")) -> None:
    rows = []
    for name, reference, change, hypothesis in EXPERIMENTS:
        metrics = json.loads((results_dir / "experiments" / f"{name}.json").read_text())
        rows.append({
            "experiment": name,
            "reference": reference,
            "change": change,
            "hypothesis": hypothesis,
            "validation_accuracy": metrics["validation_accuracy"],
            "validation_macro_f1": metrics["validation_macro_f1"],
            "validation_weighted_f1": metrics["validation_weighted_f1"],
        })
    with (results_dir / "experiment_log.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    build_log()
