# Rubric audit

This maps the 60-point rubric to files and actions that can be checked before submission.

| Criterion | Points | Evidence | Remaining check |
| --- | ---: | --- | --- |
| Problem, language, dataset | 8 | Report sections 1, 2, 4; `DATASET.md`; `results/data_manifest.json`; real KINNEWS source | Explain why Kinyarwanda news readers and archives benefit |
| Model and methodology | 10 | Report section 5; `src/linear.py`; `src/recurrent.py`; demo script screens 4–5 | Explain input, TF-IDF weights, decision scores, embeddings, bidirectional cells, hyperparameters |
| Baseline and experiments | 8 | `results/experiment_log.csv` with change and hypothesis; report Tables 1–3; demo script screens 4–6 | Explain why the majority and word baselines matter, then each controlled change |
| Evaluation and errors | 7 | `results/test_metrics.json`; `results/error_examples.csv`; confusion matrix; report sections 6–7 | Show one success and two failures, explain macro F1 and uncertainty |
| Research report | 10 | `report/report.pdf`, editable `report/report.docx`, 11 required sections, references, real figures and tables | Add the direct video link and rebuild both files |
| Web deployment | 7 | `app.py`; packaged `artifacts/final.joblib`; [live app](https://kinyarwanda-news-topics.streamlit.app/) | Show three inputs at the public link in the video |
| Code and reproducibility | 5 | `src/`, one Colab notebook, `requirements.txt`, README, versioned results and model | Add the direct video URL to README |
| Technical defense | 5 | Timed demo script; readable source and results | Record the demo in your own words and rehearse likely viva questions |

## Submission gates

- [x] Individual classification project with real Kinyarwanda articles.
- [x] Disjoint seeded train, validation, and test groups; source hashes and class counts recorded.
- [x] Linear and recurrent training runs logged; final numbers come from actual runs.
- [x] Report PDF and editable Word document generated from saved results.
- [x] Public web app loads the same saved classifier and returns a topic.
- [x] Public app accessible, with direct link in README and report.
- [ ] Ten-minute demo recorded, with direct link in README and report.
- [ ] PDF uploaded before 18 October at 11:59pm, course local time.
- [ ] All Intranet tasks complete; attendance remains at or above 85%.

The brief gives both 7–10 and 10–15 minutes for the video, so exactly 10 minutes satisfies both windows. The separate report submission accepts PDF.
