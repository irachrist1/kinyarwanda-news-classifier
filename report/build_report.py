"""Build the report source and PDF from versioned experiment results."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
RESULTS = ROOT / "results"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def load() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest = json.loads((RESULTS / "data_manifest.json").read_text())
    metrics = json.loads((RESULTS / "test_metrics.json").read_text())
    log = pd.read_csv(RESULTS / "experiment_log.csv")
    errors = pd.read_csv(RESULTS / "error_examples.csv")
    links = json.loads((REPORT / "links.json").read_text())
    label_artifact = json.loads((ROOT / "artifacts/labels.json").read_text())
    labels = pd.DataFrame([
        {"label": int(number), "en_label": names["english"]}
        for number, names in label_artifact.items()
    ])
    return manifest, metrics, log, errors, links, labels


def chart(manifest: dict, labels: pd.DataFrame) -> Path:
    counts = manifest["label_counts"]
    names = [str(labels.loc[labels.label == int(key), "en_label"].iloc[0]) for key in counts]
    values = [counts[key] for key in counts]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    figure.patch.set_facecolor("white")
    axis.bar(names, values, color="#156b78")
    axis.set_ylabel("Unique articles")
    axis.tick_params(axis="x", rotation=60)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output = REPORT / "figures" / "class_counts.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def build_blocks() -> list[dict]:
    manifest, metrics, log, errors, links, labels = load()
    selected = metrics["selected"]
    baseline = metrics["baseline"]
    uncertainty = metrics["uncertainty"]
    count_chart = chart(manifest, labels)
    blocks = []

    def add(kind: str, **values):
        blocks.append({"type": kind, **values})

    add("title", text="Leakage-aware Kinyarwanda news topic classification")
    add("subtitle", text="Individual NLP summative project | Christian Tonny | October 2026")
    add("link", label="GitHub repository", url=links["repository"])
    add("link", label="Demo video", url=links["demo_video"], missing="Not yet recorded; add the direct video link before submission.")
    add("link", label="Live system", url=links["live_system"], missing="Public deployment pending; run locally with `streamlit run app.py` in the meantime.")
    add("heading", text="1. Abstract")
    add("paragraph", text=f"This study tests whether a compact text classifier can assign Kinyarwanda news articles to 14 topics. The KINNEWS source contains {manifest['original_rows']:,} published rows; after removing blank text, conflicting labels, and duplicates, {manifest['unique_articles']:,} unique articles remained. A seeded, stratified split reserved {manifest['split_sizes']['test']:,} articles for held-out evaluation. Word and character TF-IDF features with a linear support vector classifier reached {selected['accuracy']*100:.2f}% test accuracy and {selected['macro_f1']:.3f} macro F1, compared with {baseline['accuracy']*100:.2f}% and {baseline['macro_f1']:.3f} for the word-bigram baseline. The paired 95% bootstrap interval for the macro F1 difference was [{uncertainty['macro_f1_difference_95_percent_ci'][0]:.3f}, {uncertainty['macro_f1_difference_95_percent_ci'][1]:.3f}], so this small improvement is uncertain. Recurrent models with learned embeddings performed worse under the tested training budget. Errors were concentrated in overlapping categories and smaller classes. The final classifier is packaged in a web interface and can be reproduced from the public repository.")

    add("heading", text="2. Introduction")
    add("paragraph", text="Kinyarwanda news publishers produce articles across politics, sport, health, culture, and other topics. Automatic topic assignment can help readers and archives route or search articles, especially when editorial labels differ across sites. KINNEWS provides a public benchmark for this language (Niyongabo et al., 2020). The intended users here are news readers and archive staff who need a first-pass topic suggestion, not an authoritative editorial decision.")
    add("paragraph", text="The project asks three questions: (1) do character features add useful information to a word-based classifier; (2) can small GRU or LSTM models with learned embeddings match that baseline; and (3) which categories and examples remain difficult? The contribution is a reproducible, leakage-aware assessment and deployable classifier, not a claim that Kinyarwanda topic classification is new.")

    add("heading", text="3. Related Work")
    add("paragraph", text="Niyongabo et al. (2020) introduced KINNEWS and KIRNEWS with monolingual and cross-lingual news classification baselines, including TF-IDF and neural approaches. Their corpus supplies the articles and original topic labels used here. MasakhaNEWS extended African-language news topic benchmarking to 16 languages and evaluated classical and pretrained model approaches (Adelani et al., 2023). These studies show the value of local-language datasets while making source quality and evaluation design central concerns.")
    add("paragraph", text="Linear support vector machines are a long-standing approach to sparse, high-dimensional text features (Joachims, 1998). We test word and character TF-IDF views because word phrases capture topic terms while character sequences may capture spelling and morphological variation. The latter is a hypothesis, not a guaranteed advantage. For a sequence-aware comparison, we use gated recurrent units introduced by Cho et al. (2014) and long short-term memory cells introduced by Hochreiter and Schmidhuber (1997). Published KINNEWS scores are not directly comparable here because this project removes repeated articles and uses a newly generated split.")

    add("heading", text="4. Dataset and Data Preparation")
    add("paragraph", text=f"The authors' raw KINNEWS CSVs provide numeric labels, Kinyarwanda and English label names, source URLs, titles, and article bodies (Niyongabo et al., 2020). The two files have {manifest['original_rows']:,} rows in total. Five bodies are blank. The published raw files repeat 10,179 URLs, and 160 URLs have conflicting topic labels. In the authors' cleaned split, 678 exact title-and-body pairs appear in both train and test. That overlap could let a classifier recognize articles rather than generalize.")
    add("paragraph", text=f"The preparation script normalizes Unicode to NFC and collapses whitespace, removes blank text, discards title or body groups with conflicting labels, then keeps one record per normalized title and body. This leaves {manifest['unique_articles']:,} unique articles. A fixed seed of {manifest['seed']} creates a stratified 70/15/15 split with {manifest['split_sizes']['train']:,} training, {manifest['split_sizes']['validation']:,} validation, and {manifest['split_sizes']['test']:,} test articles. No title, body, or URL is shared across the resulting splits. Source file SHA-256 hashes and exact class counts are in the repository's data manifest.")
    add("figure", path=str(count_chart.relative_to(ROOT)), caption="Figure 1. Topic counts after conflict removal and deduplication. Source: prepared KINNEWS data.")
    add("paragraph", text="The class distribution is uneven: the largest class has 2,070 articles and the smallest has 109. Median article length is 291 words. Several publishers dominate the corpus, and publication dates are absent. A random split therefore measures generalization to unseen articles from broadly similar sources, not to future time periods, new publishers, or conversational text.")

    add("heading", text="5. Methodology")
    add("paragraph", text="The baseline predicts the most frequent class. The main statistical baseline uses word-unigram TF-IDF and a linear support vector classifier. Controlled variations add word bigrams, replace words with character 3- to 5-grams, and add character 3- to 5-grams to word unigrams and bigrams. The vectorizers use a minimum document frequency of two and sublinear term frequency; word and character vocabularies are capped at 100,000 and 150,000 features. LinearSVC uses C = 1.0. TF-IDF and the classifier are fitted only on the training partition during model selection.")
    add("paragraph", text="For the sequence comparison, a Keras text vectorizer keeps up to 20,000 tokens and pads or truncates to 160 tokens. A 64-dimensional embedding layer, bidirectional GRU or LSTM with 48 hidden units, dropout of 0.2, and a 14-way softmax form the network. Training uses Adam at 0.001, batch size 64, sparse categorical cross-entropy, seed 42, and validation-loss early stopping. The longer-run variant raises only the maximum epoch count from four to eight; another variant raises only sequence length from 160 to 320 relative to that run. The embeddings are learned from this project's training set rather than imported from a pretrained model.")
    add("paragraph", text="Macro F1 averages class F1 values equally, so it is the selection metric for the imbalanced 14-class problem. Accuracy, macro precision, macro recall, and weighted F1 provide complementary views. The test split was left untouched until the combined word and character approach was selected on validation macro F1.")

    add("heading", text="6. Experiments and Results")
    linear = log.iloc[:5]
    recurrent = log.iloc[5:]
    add("table", caption="Table 1. Linear-model validation experiments; each non-anchor row names one change from its reference.", rows=[["Experiment", "Single change", "Accuracy", "Macro F1"]] + [[str(r.experiment), str(r.change), f"{r.validation_accuracy*100:.1f}%", f"{r.validation_macro_f1:.3f}"] for r in linear.itertuples()])
    add("table", caption="Table 2. Recurrent-model validation experiments.", rows=[["Experiment", "Single change", "Accuracy", "Macro F1"]] + [[str(r.experiment), str(r.change), f"{r.validation_accuracy*100:.1f}%", f"{r.validation_macro_f1:.3f}"] for r in recurrent.itertuples()])
    add("paragraph", text="Adding word bigrams raised validation accuracy but barely changed macro F1. Character-only features had slightly stronger macro F1 than word features. Adding character features to the word-bigram model gave the highest validation macro F1 (0.730), with 81.5% accuracy. The GRU and LSTM did not match the linear models. Extending LSTM training helped, but doubling input length hurt validation performance; that run stopped early when validation loss rose. These outcomes support the simpler approach for this dataset and compute budget, while not ruling out better tuned or pretrained sequence models.")
    add("table", caption="Table 3. Held-out test results after refitting the two linear approaches on training plus validation data.", rows=[["Model", "Accuracy", "Macro P", "Macro R", "Macro F1"] , ["Word bigram", f"{baseline['accuracy']*100:.2f}%", f"{baseline['macro_precision']:.3f}", f"{baseline['macro_recall']:.3f}", f"{baseline['macro_f1']:.3f}"], ["Word + character", f"{selected['accuracy']*100:.2f}%", f"{selected['macro_precision']:.3f}", f"{selected['macro_recall']:.3f}", f"{selected['macro_f1']:.3f}"]])
    add("paragraph", text=f"Across {uncertainty['resamples']:,} paired bootstrap resamples, the selected model's 95% accuracy interval was [{uncertainty['selected_accuracy_95_percent_ci'][0]*100:.1f}%, {uncertainty['selected_accuracy_95_percent_ci'][1]*100:.1f}%] and its macro F1 interval was [{uncertainty['selected_macro_f1_95_percent_ci'][0]:.3f}, {uncertainty['selected_macro_f1_95_percent_ci'][1]:.3f}]. The interval for the macro F1 difference included zero. The exact paired McNemar test on correctness gave p = {uncertainty['mcnemar_exact_two_sided_p']:.3f}, with {uncertainty['selected_only_correct']} cases correct only for the combined model and {uncertainty['baseline_only_correct']} only for the baseline. The evidence supports similar test performance rather than a reliable win for the extra feature set.")

    add("heading", text="7. Error Analysis and Discussion")
    classes = metrics["classes"]
    selected_classes = ["sport", "religion", "politic", "health", "fashion", "education", "history"]
    add("table", caption="Table 4. Selected class-level test results; support is the number of true examples.", rows=[["Class", "Support", "Precision", "Recall", "F1"]] + [[name, str(int(classes[name]["support"])), f"{classes[name]['precision']:.3f}", f"{classes[name]['recall']:.3f}", f"{classes[name]['f1-score']:.3f}"] for name in selected_classes])
    add("figure", path="results/figures/confusion_matrix.png", caption="Figure 2. Held-out confusion matrix for the selected model. Darker cells contain more articles.")
    add("paragraph", text="Sport is comparatively clear (F1 = 0.957), while history (0.517) and education (0.524) are harder. The largest confusion is health predicted as politics (34 articles). These class differences reflect both support and overlap in vocabulary, so a single aggregate score would hide useful information.")
    for row in errors.itertuples():
        label = "Correct" if row.status == "success" else "Error"
        add("example", text=f"{label}: '{row.title}' | source label: {row.actual}; prediction: {row.predicted}. {row.interpretation_hypothesis}")
    add("paragraph", text="The interpretations above are hypotheses based on article titles, source categories, and predicted labels; they were not adjudicated by a Kinyarwanda-speaking annotator. Some examples may expose broad or noisy publisher categories rather than a purely model-induced error. A manual review of full articles would be needed to separate those causes.")

    add("heading", text="8. Deployment")
    add("paragraph", text="The public Streamlit interface accepts a Kinyarwanda headline or article and calls the packaged scikit-learn pipeline directly. It returns the Kinyarwanda topic name and an English gloss. No external model API is required, and decision margins are not presented as calibrated probabilities. The public deployment was tested with three distinct text inputs and an empty-input warning. The repository includes the exact model artifact and dependency list; the live URL appears in the project links above. The app can also be run locally with `streamlit run app.py`.")

    add("heading", text="9. Limitations and Future Work")
    add("paragraph", text="Deduplication removed nearly half of the published rows and all title/body groups with contradictory labels. This improves evaluation separation but changes the benchmark population. The random split still shares publishers and possible stylistic patterns across partitions. The corpus lacks dates, independent label adjudication, and broad representation of informal Kinyarwanda. A classifier trained on article bodies may be unreliable on a short headline or unrelated text, and it has no calibrated abstention mechanism.")
    add("paragraph", text="Future work should include a Kinyarwanda-speaking review of disputed labels, a held-out publisher or time-based collection, and calibration or out-of-domain detection. A pretrained language model could be tested with enough compute and a single-change experimental design. Larger or longer recurrent models should be evaluated only after controlling token coverage and regularization. Any public deployment should be checked again close to assessment because free hosting may sleep or change.")

    add("heading", text="10. Conclusion")
    add("paragraph", text=f"A reproducible, article-level split of KINNEWS yielded a practical 14-topic Kinyarwanda news classifier. The combined word and character model achieved {selected['accuracy']*100:.2f}% held-out accuracy and {selected['macro_f1']:.3f} macro F1, but its advantage over the word-bigram baseline was not statistically clear. Recurrent models with learned embeddings underperformed in the tested configurations. The most useful lesson is that data duplication, label quality, and class-wise errors matter as much as model complexity for this task.")

    add("heading", text="11. References")
    references = [
        "Adelani, D. I., et al. (2023). MasakhaNEWS: News topic classification for African languages. Proceedings of IJCNLP-AACL, 144-159. https://doi.org/10.18653/v1/2023.ijcnlp-main.10",
        "Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. Proceedings of EMNLP, 1724-1734. https://doi.org/10.3115/v1/D14-1179",
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735",
        "Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. Proceedings of ECML. https://www.cs.cornell.edu/~tj/publications/joachims_98a.pdf",
        "Niyongabo, R. A., Hong, Q., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. Proceedings of COLING, 5507-5521. https://doi.org/10.18653/v1/2020.coling-main.480",
        "KINNEWS and KIRNEWS Corpus. (2020). Dataset, labels, and documentation. https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus",
        "scikit-learn developers. (2026). TfidfVectorizer and LinearSVC documentation. https://scikit-learn.org/stable/modules/feature_extraction.html and https://scikit-learn.org/stable/modules/svm.html",
        "TensorFlow developers. (2026). Keras recurrent layers and TextVectorization documentation. https://www.tensorflow.org/api_docs/python/tf/keras/layers",
        "Streamlit developers. (2026). Streamlit documentation. https://docs.streamlit.io/",
    ]
    for reference in references:
        add("reference", text=reference)
    return blocks


def write_markdown(blocks: list[dict]) -> None:
    lines = []
    for block in blocks:
        kind = block["type"]
        if kind == "title":
            lines += [f"# {block['text']}", ""]
        elif kind == "subtitle":
            lines += [block["text"], ""]
        elif kind == "heading":
            lines += [f"## {block['text']}", ""]
        elif kind == "link":
            value = block.get("url")
            lines += [f"**{block['label']}:** {value if value else block['missing']}", ""]
        elif kind == "table":
            rows = block["rows"]
            lines += [f"**{block['caption']}**", "", "| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |" ]
            lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
            lines += [""]
        elif kind == "figure":
            lines += [f"![{block['caption']}]({block['path']})", "", f"*{block['caption']}*", ""]
        else:
            lines += [block["text"], ""]
    (REPORT / "report.md").write_text("\n".join(lines))


class NumberedCanvas:
    def __init__(self, canvas, doc):
        self.canvas = canvas
        self.doc = doc

    def draw(self):
        canvas = self.canvas
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#65717d"))
        canvas.drawString(55, letter[1] - 40, "KINNEWS  /  INDIVIDUAL NLP PROJECT")
        canvas.drawRightString(letter[0] - 55, 34, str(self.doc.page))
        canvas.restoreState()


def build_pdf(blocks: list[dict]) -> None:
    pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "Arial Bold.ttf")))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Arial-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#152238"), spaceAfter=11))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontName="Arial", fontSize=9, leading=13, textColor=colors.HexColor("#65717d"), spaceAfter=14))
    styles.add(ParagraphStyle(name="ReportHeading", fontName="Arial-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#156b78"), spaceBefore=16, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Arial", fontSize=9.2, leading=13.7, textColor=colors.HexColor("#26313b"), spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSmall", fontName="Arial", fontSize=8, leading=11, textColor=colors.HexColor("#4b5966"), spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportCaption", fontName="Arial", fontSize=8, leading=11, textColor=colors.HexColor("#4b5966"), spaceBefore=5, spaceAfter=9))
    path = REPORT / "report.pdf"
    document = BaseDocTemplate(str(path), pagesize=letter, leftMargin=55, rightMargin=55, topMargin=58, bottomMargin=50, title="Leakage-aware Kinyarwanda news topic classification", author="Christian Tonny")
    frame = Frame(55, 50, letter[0] - 110, letter[1] - 108, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates(PageTemplate(id="normal", frames=frame, onPage=lambda c, d: NumberedCanvas(c, d).draw()))
    story = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading" and block["text"] in ("7. Error Analysis and Discussion", "11. References"):
            story.append(PageBreak())
        if kind in ("title", "subtitle", "heading", "paragraph", "example", "reference"):
            name = {"title": "ReportTitle", "subtitle": "ReportSubtitle", "heading": "ReportHeading", "paragraph": "ReportBody", "example": "ReportSmall", "reference": "ReportSmall"}[kind]
            story.append(Paragraph(html.escape(block["text"]), styles[name]))
        elif kind == "link":
            if block.get("url"):
                label = html.escape(block["label"])
                url = html.escape(block["url"], quote=True)
                story.append(Paragraph(f"<b>{label}:</b> <link href='{url}' color='#156b78'>{url}</link>", styles["ReportSmall"]))
            else:
                story.append(Paragraph(f"<b>{html.escape(block['label'])}:</b> {html.escape(block['missing'])}", styles["ReportSmall"]))
        elif kind == "table":
            rows = [[Paragraph(html.escape(str(cell)), styles["ReportSmall"]) for cell in row] for row in block["rows"]]
            column_count = len(rows[0])
            widths = [105, 225, 85, 85] if column_count == 4 else ([140, 90, 90, 90, 90] if column_count == 5 else None)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0f1")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d6dfe1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([Paragraph(html.escape(block["caption"]), styles["ReportCaption"]), table]))
            story.append(Spacer(1, 9))
        elif kind == "figure":
            path = ROOT / block["path"]
            width, height = ImageReader(str(path)).getSize()
            desired_width = 465
            story.append(Image(str(path), width=desired_width, height=desired_width * height / width))
            story.append(Paragraph(html.escape(block["caption"]), styles["ReportCaption"]))
    document.build(story)


if __name__ == "__main__":
    REPORT.mkdir(exist_ok=True)
    blocks = build_blocks()
    (REPORT / "report_data.json").write_text(json.dumps(blocks, ensure_ascii=False, indent=2) + "\n")
    write_markdown(blocks)
    build_pdf(blocks)
