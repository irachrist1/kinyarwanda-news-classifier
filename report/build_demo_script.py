"""Build the timed presentation page from the final evaluation."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "Downloads" / "kinyarwanda_news_demo_script.html"
metrics = json.loads((ROOT / "results/test_metrics.json").read_text())
manifest = json.loads((ROOT / "results/data_manifest.json").read_text())
links = json.loads((ROOT / "report/links.json").read_text())

slides = [
    ("00:00–00:50", "The task", "Kinyarwanda news, 14 topics", "I built a classifier that suggests a topic for a Kinyarwanda news article. The goal is a useful first pass for a reader or an archive, rather than a final editorial decision. I used the public KINNEWS corpus. This recording walks through the data, the experiment choices, the held-out result, a few errors, and the working app. The repository and report links are on the final screen."),
    ("00:50–01:55", "The data", "Repeated articles changed the project", f"The source files contain {manifest['original_rows']:,} rows. I found five blank bodies, many repeated URLs, and 160 URLs with conflicting topic labels. The authors' cleaned train and test files also share 678 exact title and body pairs. I normalized the text, removed contradictory groups, and kept one copy of each article. That left {manifest['unique_articles']:,} unique articles. This matters because a model can look strong when it has already seen an article."),
    ("01:55–02:50", "The split", "A test set the model had not seen", f"I created a seeded, stratified split: {manifest['split_sizes']['train']:,} training articles, {manifest['split_sizes']['validation']:,} validation articles, and {manifest['split_sizes']['test']:,} test articles. The preparation code checks that title, body, and URL groups do not cross the split. I used validation macro F1 to choose the model and opened the test set only after that choice. All source hashes and class counts are recorded in the repository."),
    ("02:50–03:50", "The method", "Start with words and characters", "The majority class gives a floor, and a word-only linear classifier gives a meaningful baseline: it is cheap, strong for text, and easy to inspect. Input is an article body; output is one of 14 topic IDs. A TF-IDF vectorizer turns word or character counts into weighted sparse features. A linear support vector classifier learns one decision score per category and returns the category with the strongest score. I tested unigrams, then bigrams, then character groups, then both views. Each row in the log records one change and one hypothesis."),
    ("03:50–04:55", "The selection", "The simplest strong model won validation", "I also trained a bidirectional GRU and LSTM. TextVectorization maps tokens to integer IDs, a 64-dimensional embedding layer learns a vector per token, the recurrent layer reads the sequence in both directions, and a softmax maps its output to 14 classes. The runs use Adam, dropout, batch size 64, seed 42, and validation-loss early stopping. The combined word and character model reached 0.730 validation macro F1; the best recurrent run reached 0.447. More LSTM epochs helped, while doubling sequence length hurt this configuration. I selected the linear pipeline before opening the test set."),
    ("04:55–06:00", "The result", f"{metrics['selected']['accuracy']*100:.2f}% test accuracy", f"The selected model reached {metrics['selected']['macro_f1']:.3f} macro F1 on {metrics['test_rows']:,} held-out articles. Accuracy is the share of all correct predictions. Precision asks how often predicted topics are right; recall asks how many true articles in a topic are found. F1 balances those two, and macro F1 gives each topic equal weight despite uneven class sizes. The word-bigram comparison reached {metrics['baseline']['accuracy']*100:.2f}% accuracy and {metrics['baseline']['macro_f1']:.3f} macro F1. The improvement is small. A paired bootstrap interval for the macro F1 difference crosses zero, and the paired McNemar test gives p = {metrics['uncertainty']['mcnemar_exact_two_sided_p']:.3f}."),
    ("06:00–07:00", "The errors", "The average hides difficult topics", "Sport has a test F1 near 0.957, but history and education are near 0.52. Health articles were called politics 34 times. One history article about a former president was also called politics; its political vocabulary may outweigh the historical framing. Another article about a fashion event was called entertainment. Show one correctly classified sport example, then these two failures in the report. They suggest overlapping language and possibly broad publisher categories. These are hypotheses from the source labels and predictions, not judgments from a language annotator."),
    ("07:00–08:20", "The live demo", "Try three inputs", "Open the deployed Streamlit app and show the input, submit button, and returned Kinyarwanda and English labels. Paste each card text in turn. The saved model returns Sports for the first, Health for the second, and Entertainment for the fashion event. That last one is a useful failure: the corpus labels the full article as fashion, while the title alone triggers entertainment. The app loads the packaged scikit-learn pipeline directly; there is no outside model service. Finally submit an empty box to show the prompt. A short headline is less reliable than the articles used for training."),
    ("08:20–09:15", "Reproducibility", "The full path is in the repository", "Show src/data.py for normalization, conflict removal, deduplication, and split checks. Show src/linear.py and src/recurrent.py for the two model families, then src/evaluate.py for the held-out metrics and error examples. Open the experiment log's change and hypothesis columns. The single Colab notebook runs preparation, all nine tracked experiments, final evaluation, and inference. The report tables and figures are rebuilt from saved result files. Point out that the same final.joblib file drives both evaluation and the app."),
    ("09:15–10:00", "The takeaway", "Good evaluation changed the answer", "A compact classifier performed well for many Kinyarwanda news topics. It still struggles with smaller or overlapping classes, and its apparent edge over the simpler word baseline is uncertain. The strongest next steps are review of disputed labels by a Kinyarwanda speaker, a new publisher or time-based test collection, and out-of-domain checks before editorial use. That is the project and its current limit."),
]

demo_inputs = [
    ("Ikipe y’u Rwanda yatsinze umukino wa nyuma w’igikombe", "sport", "imikino · Sports"),
    ("Minisiteri y’ubuzima yatangaje gahunda nshya yo gukingira abana", "health", "ubuzima · Health"),
    ("Imyambarire idasanzwe yaranze ibirori bya Met Gala (Amafoto)", "entertainment", "imyidagaduro · Entertainment"),
]
model_check = __import__("joblib").load(ROOT / "artifacts/final.joblib")
labels = json.loads((ROOT / "artifacts/labels.json").read_text())
for demo_text, expected, _ in demo_inputs:
    predicted = labels[str(int(model_check.predict([demo_text])[0]))]
    assert predicted["english"] == expected


def slide(index: int, entry: tuple[str, str, str, str]) -> str:
    timing, eyebrow, heading, script = entry
    extra = ""
    if index == 8:
        cards = ''.join(f'<div><button class="copy" type="button" data-value="{html.escape(value, quote=True)}" aria-label="Copy demo input">{html.escape(value)} <b>Copy</b></button><small>Verified output: {html.escape(output)}</small></div>' for value, _, output in demo_inputs)
        extra = f'<div class="demo"><span>THREE INPUTS TO PASTE</span>{cards}</div>'
    if index == 10:
        live = links["live_system"] or "Add the public app link after deployment"
        app_link = f'<a href="{html.escape(live, quote=True)}">Live app ↗</a>' if links["live_system"] else f'<span>{html.escape(live)}</span>'
        extra = f'<div class="links"><a href="{html.escape(links["repository"], quote=True)}">Repository ↗</a>{app_link}</div>'
    return f'''<section class="slide snap" id="slide-{index}"><div class="inner"><div class="top"><span>{index:02d} / 10</span><span>{html.escape(timing)}</span></div><div class="body"><div class="eyebrow">{html.escape(eyebrow)}</div><h2>{html.escape(heading)}</h2><p class="script">{html.escape(script)}</p>{extra}</div><div class="foot"><span>Use ↓ or space for the next screen</span><div class="rail"><i style="width:{index*10}%"></i></div></div></div></section>'''


page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Kinyarwanda news classifier · 10-minute demo</title>
<style>
*{{box-sizing:border-box}}html{{scroll-snap-type:y mandatory;overflow-y:scroll;height:100%;scroll-behavior:smooth}}body{{margin:0;background:#f7f4ef;color:#1a1a1a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}.slide{{height:100vh;min-height:620px;scroll-snap-align:start;display:flex;align-items:center;padding:42px 6vw;border-top:1px solid #e0dbd2}}.inner{{width:100%;max-width:1020px;height:min(82vh,760px);margin:0 auto;display:flex;flex-direction:column;justify-content:space-between}}.top,.foot{{display:flex;align-items:center;justify-content:space-between;color:#8b8a80;font-size:14px;letter-spacing:.03em}}.body{{max-width:790px}}.eyebrow{{color:#156b78;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.16em;margin-bottom:24px}}h2{{font:400 clamp(50px,7.5vw,104px)/1.02 Georgia,serif;letter-spacing:-.045em;max-width:900px;margin:0 0 42px}}.script{{font-size:clamp(17px,1.8vw,22px);line-height:1.66;color:#4a4a4a;max-width:760px;margin:0}}.rail{{width:180px;height:3px;background:#e0dbd2;border-radius:8px;overflow:hidden}}.rail i{{display:block;height:100%;background:#156b78}}.demo{{margin-top:20px;border:1px solid #e0dbd2;background:white;border-radius:16px;padding:15px 20px;max-width:760px;display:flex;flex-direction:column;gap:10px}}.demo span{{font-size:11px;letter-spacing:.15em;color:#8b8a80;font-weight:700}}.demo>div{{border-top:1px solid #e0dbd2;padding-top:8px;display:flex;flex-direction:column;gap:3px}}button{{border:0;background:transparent;font:inherit;color:#1a1a1a;text-align:left;cursor:pointer;padding:0;display:flex;justify-content:space-between;gap:20px}}button b{{color:#156b78;font-size:14px}}small{{font-size:13px;color:#6b706d}}.links{{display:flex;gap:24px;margin-top:32px;font-size:16px}}a{{color:#156b78}}.slide .body{{opacity:0;filter:blur(7px);transform:translateY(18px) scale(.99);transition:opacity .7s,filter .7s,transform .7s cubic-bezier(.16,1,.3,1)}}.slide.visible .body{{opacity:1;filter:none;transform:none}}@media(max-width:680px){{.slide{{padding:25px 7vw;min-height:620px}}.inner{{height:88vh}}h2{{margin-bottom:25px}}.script{{font-size:16px}}.foot span{{display:none}}.rail{{width:100%}}.links{{flex-direction:column;gap:10px}}}}@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}.slide .body{{opacity:1;filter:none;transform:none;transition:none}}}}
</style></head><body>
{''.join(slide(i, entry) for i, entry in enumerate(slides, 1))}
<script>
const slides=[...document.querySelectorAll('.slide')];const observer=new IntersectionObserver(entries=>{{for(const entry of entries)if(entry.isIntersecting)entry.target.classList.add('visible')}},{{threshold:.36}});slides.forEach(s=>observer.observe(s));document.addEventListener('keydown',event=>{{if(!['ArrowDown','ArrowUp',' ','PageDown','PageUp'].includes(event.key))return;event.preventDefault();const current=Math.round(window.scrollY/window.innerHeight);const next=Math.max(0,Math.min(slides.length-1,current+(event.key==='ArrowUp'||event.key==='PageUp'?-1:1)));slides[next].scrollIntoView({{behavior:'smooth'}})}});document.querySelectorAll('.copy').forEach(button=>button.addEventListener('click',async event=>{{const value=event.currentTarget.dataset.value;try{{await navigator.clipboard.writeText(value)}}catch{{const field=document.createElement('textarea');field.value=value;document.body.appendChild(field);field.select();document.execCommand('copy');field.remove()}}event.currentTarget.querySelector('b').textContent='Copied';setTimeout(()=>event.currentTarget.querySelector('b').textContent='Copy',1400)}}));
</script></body></html>'''
OUT.write_text(page)
print(OUT)
