# Kinyarwanda news topic classification

This project studies topic classification for Kinyarwanda news articles using the [KINNEWS corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus). It includes a reproducible data pipeline, baseline and neural models, evaluation, and an interactive web application.

## Prepare the data

Create a Python environment and install `requirements.txt`. Download the authors' [KINNEWS folder](https://drive.google.com/drive/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox) into `data/raw`, then prepare the split:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
gdown --folder 'https://drive.google.com/drive/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox' -O data/raw
python -m src.data
```

The data is not stored in this repository. [DATASET.md](DATASET.md) documents the source, quality audit, and split.

## Experiments

The [experiment log](results/experiment_log.csv) records each change, hypothesis, and measured validation result. To rerun the main comparison and held-out evaluation:

```bash
python -m src.linear word_unigram
python -m src.linear word_bigram
python -m src.linear character
python -m src.linear word_character
python -m src.evaluate
```

Install `requirements-training.txt` to rerun the GRU and LSTM experiments. Each command in `src/recurrent.py` takes one to several minutes on an Apple M2 Pro CPU.

The combined word and character TF-IDF classifier was selected using validation macro F1. On the 1,625-article test set, it reached **81.05% accuracy** and **0.740 macro F1**. The word-bigram baseline reached 80.49% and 0.733. The paired bootstrap 95% interval for their macro F1 difference includes zero, so the observed improvement is uncertain. The test set was used only after selecting the approach.

The trained model is stored in `artifacts/final.joblib`. It is a serialized scikit-learn object; load only this trusted repository artifact.

The [research report](report/report.pdf) includes the full method, results, error analysis, and references. Its [editable version](report/report.docx) and [rubric audit](report/rubric_audit.md) are also included. To rebuild both formats from the saved results, install `requirements-report.txt` and run `python report/build_report.py`, then run `npm install --prefix report` and `node report/build_docx.mjs`.

## Run the app

```bash
streamlit run app.py
```

The app accepts Kinyarwanda news text and returns one of the 14 topics. Its model is the tested classifier in this repository, so it does not need a separate model service.

The HTTP interface in `web/` uses the same `src/inference.py` and model artifact. To deploy it with the Vercel CLI, run `python web/prepare.py`, then `vercel --cwd web --prod`. The preparation step copies only the evaluated model, label map, and shared inference module into the deployment folder.

[Open the deployed app](https://kinyarwanda-news-topics.vercel.app/). The public workflow was checked with sports, health, and overlapping fashion/entertainment inputs, plus an empty input.

## Colab

The [end-to-end Colab notebook](https://colab.research.google.com/github/irachrist1/kinyarwanda-news-classifier/blob/main/notebooks/kinyarwanda_news_end_to_end.ipynb) downloads the source data and runs preparation, training, evaluation, and inference in order.
