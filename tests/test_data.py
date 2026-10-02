from pathlib import Path

import pandas as pd

from src.data import prepare


def test_prepared_splits_have_no_shared_articles(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "raw"
    raw.mkdir(parents=True)
    rows = []
    for label in (1, 2):
        for number in range(20):
            rows.append({"label": label, "kin_label": f"class {label}", "en_label": f"class {label}", "url": f"https://example.org/{label}/{number}", "title": f"Title {label} {number}", "content": f"Body {label} {number}"})
    rows.extend([rows[0].copy(), {**rows[1], "label": 2}])
    pd.DataFrame(rows[:22]).to_csv(raw / "train.csv", index=False)
    pd.DataFrame(rows[22:]).to_csv(raw / "test.csv", index=False)

    output = tmp_path / "processed"
    manifest = prepare(raw.parent, output)
    splits = [pd.read_csv(output / f"{name}.csv") for name in ("train", "validation", "test")]
    assert manifest["unique_articles"] == 39
    assert sum(len(split) for split in splits) == 39
    assert len(set.union(*(set(split["title"]) for split in splits))) == 39
