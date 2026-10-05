"""Load the final model and return its topic names."""

from functools import lru_cache
import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
ENGLISH_DISPLAY = {"politic": "Politics", "sport": "Sports", "relationship": "Relationships"}


@lru_cache(maxsize=1)
def load_resources(root: Path = ROOT):
    model = joblib.load(root / "artifacts" / "final.joblib")
    labels = json.loads((root / "artifacts" / "labels.json").read_text())
    return model, labels


def classify(text: str) -> dict[str, str]:
    if not text.strip():
        raise ValueError("Enter some news text first.")
    model, labels = load_resources()
    prediction = str(int(model.predict([text])[0]))
    name = labels[prediction]
    return {
        "label": prediction,
        "kinyarwanda": name["kinyarwanda"].capitalize(),
        "english": ENGLISH_DISPLAY.get(name["english"], name["english"].capitalize()),
    }
