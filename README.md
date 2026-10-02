# Kinyarwanda news topic classification

This project studies topic classification for Kinyarwanda news articles using the [KINNEWS corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus). It includes a reproducible data pipeline, baseline and neural models, evaluation, and an interactive web application.

## Prepare the data

Create a Python environment and install `requirements.txt`. Download the authors' [KINNEWS folder](https://drive.google.com/drive/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox) into `data/raw`, then prepare the split:

```bash
python -m pip install -r requirements.txt
gdown --folder 'https://drive.google.com/drive/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox' -O data/raw
python -m src.data
```

The data is not stored in this repository. [DATASET.md](DATASET.md) documents the source, quality audit, and split. Project details and verified results will be added as each stage is completed.
