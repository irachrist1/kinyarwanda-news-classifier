"""Copy the exact evaluated model and shared inference code for deployment."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
for folder, names in {
    "artifacts": ["final.joblib", "labels.json"],
    "src": ["__init__.py", "inference.py"],
}.items():
    (WEB / folder).mkdir(exist_ok=True)
    for name in names:
        shutil.copy2(ROOT / folder / name, WEB / folder / name)
print("Deployment files copied from the final repository artifacts.")
