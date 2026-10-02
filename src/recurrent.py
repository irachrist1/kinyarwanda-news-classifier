"""Train compact recurrent classifiers with a shared tokenization setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


SEED = 42
VOCABULARY_SIZE = 20_000
SEQUENCE_LENGTH = 160
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 4


def build_model(kind: str, training_text: np.ndarray, class_count: int, sequence_length: int) -> tf.keras.Model:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCABULARY_SIZE, output_mode="int", output_sequence_length=sequence_length
    )
    vectorizer.adapt(training_text)
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    tokens = vectorizer(inputs)
    vectors = tf.keras.layers.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, mask_zero=True)(tokens)
    recurrent = tf.keras.layers.GRU if kind == "gru" else tf.keras.layers.LSTM
    features = tf.keras.layers.Bidirectional(recurrent(HIDDEN_SIZE))(vectors)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(
    kind: str, data_dir: Path, model_dir: Path, results_dir: Path,
    epochs: int = EPOCHS, sequence_length: int = SEQUENCE_LENGTH
) -> dict:
    tf.keras.utils.set_random_seed(SEED)
    training = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    labels = sorted(training["label"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    x_train = training["text"].astype(str).to_numpy()
    x_validation = validation["text"].astype(str).to_numpy()
    y_train = training["label"].map(label_to_index).to_numpy(dtype=np.int32)
    y_validation = validation["label"].map(label_to_index).to_numpy(dtype=np.int32)
    model = build_model(kind, x_train, len(labels), sequence_length)
    experiment = kind if epochs == EPOCHS else f"{kind}_{epochs}epochs"
    if sequence_length != SEQUENCE_LENGTH:
        experiment += f"_{sequence_length}tokens"
    history = model.fit(
        x_train, y_train, validation_data=(x_validation, y_validation),
        batch_size=BATCH_SIZE, epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
        verbose=2,
    )
    probabilities = model.predict(x_validation, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.array(labels)[probabilities.argmax(axis=1)]
    metrics = {
        "experiment": experiment,
        "validation_accuracy": accuracy_score(validation["label"], predictions),
        "validation_macro_f1": f1_score(validation["label"], predictions, average="macro"),
        "validation_weighted_f1": f1_score(validation["label"], predictions, average="weighted"),
        "epochs_completed": len(history.history["loss"]),
        "epochs": epochs,
        "vocabulary_size": VOCABULARY_SIZE,
        "sequence_length": sequence_length,
        "embedding_size": EMBEDDING_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "history": history.history,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{experiment}.keras")
    (results_dir / f"{experiment}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame({"label": validation["label"], "prediction": predictions}).to_csv(
        results_dir / f"{experiment}_validation_predictions.csv", index=False
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["gru", "lstm"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    args = parser.parse_args()
    print(json.dumps(train(args.kind, args.data_dir, args.model_dir, args.results_dir, args.epochs, args.sequence_length), indent=2))
