"""Prepare the KINNEWS corpus with article-level separation between splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_URL = "https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus"
DOWNLOAD_URL = "https://drive.google.com/drive/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox"
SEED = 42


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(raw_dir: Path, output_dir: Path, seed: int = SEED) -> dict:
    paths = [raw_dir / "raw" / "train.csv", raw_dir / "raw" / "test.csv"]
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    original_count = len(data)
    for column in ["title", "content", "url", "en_label", "kin_label"]:
        data[column] = data[column].map(normalize)
    data = data[data["title"].ne("") & data["content"].ne("")].copy()
    after_missing = len(data)
    data["title_key"] = data["title"].str.casefold()
    data["content_key"] = data["content"].str.casefold()

    conflicting_titles = data.groupby("title_key")["label"].nunique()
    conflicting_contents = data.groupby("content_key")["label"].nunique()
    bad_titles = set(conflicting_titles[conflicting_titles > 1].index)
    bad_contents = set(conflicting_contents[conflicting_contents > 1].index)
    data = data[~data["title_key"].isin(bad_titles) & ~data["content_key"].isin(bad_contents)].copy()
    after_conflicts = len(data)
    data = data.drop_duplicates("title_key").drop_duplicates("content_key").copy()
    data = data.sort_values(["label", "title_key"]).reset_index(drop=True)
    data["text"] = data["title"] + "\n\n" + data["content"]
    labels = (
        data.groupby("label")[["kin_label", "en_label"]]
        .agg(lambda values: values.mode().iloc[0])
        .reset_index()
        .sort_values("label")
    )

    train, remaining = train_test_split(data, test_size=0.30, stratify=data["label"], random_state=seed)
    validation, test = train_test_split(
        remaining, test_size=0.50, stratify=remaining["label"], random_state=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["label", "title", "content", "text", "url"]
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame[columns].sort_values(["label", "title"]).to_csv(output_dir / f"{name}.csv", index=False)
    labels.to_csv(output_dir / "labels.csv", index=False)
    manifest = {
        "source": SOURCE_URL,
        "download": DOWNLOAD_URL,
        "raw_sha256": {path.name + "_" + path.parent.name: sha256(path) for path in paths},
        "seed": seed,
        "original_rows": original_count,
        "rows_after_missing_text": after_missing,
        "rows_after_conflicting_labels": after_conflicts,
        "unique_articles": len(data),
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "label_counts": {str(int(label)): int(count) for label, count in data["label"].value_counts().sort_index().items()},
        "cleaning": "NFC Unicode and whitespace normalization; remove blank text, conflicting labels for identical titles or bodies, then deduplicate by title and body; stratified 70/15/15 split.",
        "published_split_note": "The publisher's split has duplicate articles within and across train/test; this project makes a new article-level split.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.raw_dir, arguments.output_dir), indent=2, ensure_ascii=False))
