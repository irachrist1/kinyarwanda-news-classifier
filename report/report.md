# Leakage-aware Kinyarwanda news topic classification

Individual NLP summative project | Christian Tonny | 2 October 2026

**GitHub repository:** https://github.com/irachrist1/kinyarwanda-news-classifier

**Demo video:** Not yet recorded; add the direct video link before submission.

**Live system:** Public deployment pending; run locally with `streamlit run app.py` in the meantime.

## 1. Abstract

This study tests whether a compact text classifier can assign Kinyarwanda news articles to 14 topics. The KINNEWS source contains 21,268 published rows; after removing blank text, conflicting labels, and duplicates, 10,831 unique articles remained. A seeded, stratified split reserved 1,625 articles for held-out evaluation. Word and character TF-IDF features with a linear support vector classifier reached 81.05% test accuracy and 0.740 macro F1, compared with 80.49% and 0.733 for the word-bigram baseline. The paired 95% bootstrap interval for the macro F1 difference was [-0.008, 0.023], so this small improvement is uncertain. Recurrent models with learned embeddings performed worse under the tested training budget. Errors were concentrated in overlapping categories and smaller classes. The final classifier is packaged in a web interface and can be reproduced from the public repository.

## 2. Introduction

Kinyarwanda news publishers produce articles across politics, sport, health, culture, and other topics. Automatic topic assignment can help readers and archives route or search articles, especially when editorial labels differ across sites. KINNEWS provides a public benchmark for this language (Niyongabo et al., 2020). The intended users here are news readers and archive staff who need a first-pass topic suggestion, not an authoritative editorial decision.

The project asks three questions: (1) do character features add useful information to a word-based classifier; (2) can small GRU or LSTM models with learned embeddings match that baseline; and (3) which categories and examples remain difficult? The contribution is a reproducible, leakage-aware assessment and deployable classifier, not a claim that Kinyarwanda topic classification is new.

## 3. Related Work

Niyongabo et al. (2020) introduced KINNEWS and KIRNEWS with monolingual and cross-lingual news classification baselines, including TF-IDF and neural approaches. Their corpus supplies the articles and original topic labels used here. MasakhaNEWS extended African-language news topic benchmarking to 16 languages and evaluated classical and pretrained model approaches (Adelani et al., 2023). These studies show the value of local-language datasets while making source quality and evaluation design central concerns.

Linear support vector machines are a long-standing approach to sparse, high-dimensional text features (Joachims, 1998). We test word and character TF-IDF views because word phrases capture topic terms while character sequences may capture spelling and morphological variation. The latter is a hypothesis, not a guaranteed advantage. For a sequence-aware comparison, we use gated recurrent units introduced by Cho et al. (2014) and long short-term memory cells introduced by Hochreiter and Schmidhuber (1997). Published KINNEWS scores are not directly comparable here because this project removes repeated articles and uses a newly generated split.

## 4. Dataset and Data Preparation

The authors' raw KINNEWS CSVs provide numeric labels, Kinyarwanda and English label names, source URLs, titles, and article bodies (Niyongabo et al., 2020). The two files have 21,268 rows in total. Five bodies are blank. The published raw files repeat 10,179 URLs, and 160 URLs have conflicting topic labels. In the authors' cleaned split, 678 exact title-and-body pairs appear in both train and test. That overlap could let a classifier recognize articles rather than generalize.

The preparation script normalizes Unicode to NFC and collapses whitespace, removes blank text, discards title or body groups with conflicting labels, then keeps one record per normalized title and body. This leaves 10,831 unique articles. A fixed seed of 42 creates a stratified 70/15/15 split with 7,581 training, 1,625 validation, and 1,625 test articles. No title, body, or URL is shared across the resulting splits. Source file SHA-256 hashes and exact class counts are in the repository's data manifest.

![Figure 1. Topic counts after conflict removal and deduplication. Source: prepared KINNEWS data.](report/figures/class_counts.png)

*Figure 1. Topic counts after conflict removal and deduplication. Source: prepared KINNEWS data.*

The class distribution is uneven: the largest class has 2,070 articles and the smallest has 109. Median article length is 291 words. Several publishers dominate the corpus, and publication dates are absent. A random split therefore measures generalization to unseen articles from broadly similar sources, not to future time periods, new publishers, or conversational text.

## 5. Methodology

The baseline predicts the most frequent class. The main statistical baseline uses word-unigram TF-IDF and a linear support vector classifier. Controlled variations add word bigrams, replace words with character 3- to 5-grams, and add character 3- to 5-grams to word unigrams and bigrams. The vectorizers use a minimum document frequency of two and sublinear term frequency; word and character vocabularies are capped at 100,000 and 150,000 features. LinearSVC uses C = 1.0. TF-IDF and the classifier are fitted only on the training partition during model selection.

For the sequence comparison, a Keras text vectorizer keeps up to 20,000 tokens and pads or truncates to 160 tokens. A 64-dimensional embedding layer, bidirectional GRU or LSTM with 48 hidden units, dropout of 0.2, and a 14-way softmax form the network. Training uses Adam at 0.001, batch size 64, sparse categorical cross-entropy, seed 42, and validation-loss early stopping. The longer-run variant raises only the maximum epoch count from four to eight; another variant raises only sequence length from 160 to 320 relative to that run. The embeddings are learned from this project's training set rather than imported from a pretrained model.

Macro F1 averages class F1 values equally, so it is the selection metric for the imbalanced 14-class problem. Accuracy, macro precision, macro recall, and weighted F1 provide complementary views. The test split was left untouched until the combined word and character approach was selected on validation macro F1.

## 6. Experiments and Results

**Table 1. Linear-model validation experiments; each non-anchor row names one change from its reference.**

| Experiment | Single change | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| majority | Most frequent class | 19.1% | 0.023 |
| word_unigram | Word unigram TF-IDF with linear SVM | 80.6% | 0.718 |
| word_bigram | Add word bigrams | 81.3% | 0.719 |
| character | Replace word features with character 3–5-grams | 80.9% | 0.729 |
| word_character | Add character 3–5-gram features | 81.5% | 0.730 |

**Table 2. Recurrent-model validation experiments.**

| Experiment | Single change | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| gru | Bidirectional GRU with learned embeddings | 61.5% | 0.389 |
| lstm | Replace GRU cells with LSTM cells | 66.3% | 0.399 |
| lstm_8epochs | Raise maximum epochs from 4 to 8 | 69.9% | 0.447 |
| lstm_8epochs_320tokens | Raise input length from 160 to 320 tokens | 54.3% | 0.237 |

Adding word bigrams raised validation accuracy but barely changed macro F1. Character-only features had slightly stronger macro F1 than word features. Adding character features to the word-bigram model gave the highest validation macro F1 (0.730), with 81.5% accuracy. The GRU and LSTM did not match the linear models. Extending LSTM training helped, but doubling input length hurt validation performance; that run stopped early when validation loss rose. These outcomes support the simpler approach for this dataset and compute budget, while not ruling out better tuned or pretrained sequence models.

**Table 3. Held-out test results after refitting the two linear approaches on training plus validation data.**

| Model | Accuracy | Macro P | Macro R | Macro F1 |
| --- | --- | --- | --- | --- |
| Word bigram | 80.49% | 0.799 | 0.695 | 0.733 |
| Word + character | 81.05% | 0.788 | 0.708 | 0.740 |

Across 2,000 paired bootstrap resamples, the selected model's 95% accuracy interval was [79.1%, 82.9%] and its macro F1 interval was [0.701, 0.772]. The interval for the macro F1 difference included zero. The exact paired McNemar test on correctness gave p = 0.289, with 33 cases correct only for the combined model and 24 only for the baseline. The evidence supports similar test performance rather than a reliable win for the extra feature set.

## 7. Error Analysis and Discussion

**Table 4. Selected class-level test results; support is the number of true examples.**

| Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| sport | 256 | 0.964 | 0.949 | 0.957 |
| religion | 105 | 0.900 | 0.857 | 0.878 |
| politic | 310 | 0.731 | 0.816 | 0.771 |
| health | 203 | 0.746 | 0.709 | 0.727 |
| fashion | 19 | 0.786 | 0.579 | 0.667 |
| education | 25 | 0.647 | 0.440 | 0.524 |
| history | 34 | 0.625 | 0.441 | 0.517 |

![Figure 2. Held-out confusion matrix for the selected model. Darker cells contain more articles.](results/figures/confusion_matrix.png)

*Figure 2. Held-out confusion matrix for the selected model. Darker cells contain more articles.*

Sport is comparatively clear (F1 = 0.957), while history (0.517) and education (0.524) are harder. The largest confusion is health predicted as politics (34 articles). These class differences reflect both support and overlap in vocabulary, so a single aggregate score would hide useful information.

Error: 'Abandi bimukira bo muri Somalia, Eritrea, Sudan na Syria bageze mu Rwanda [ REBA AMAFOTO]' | source label: health; prediction: politic. The source assigns this migration story to health, while political terms may dominate the text; category boundaries or source labels may be broad.

Error: 'Imyambarire idasanzwe yaranze ibirori bya Met Gala (Amafoto)' | source label: fashion; prediction: entertainment. A fashion event also contains entertainment vocabulary; the topics overlap.

Error: 'BRD yatanze miliyoni 50 FRW muri #ConnectChallenge imaze gutangwamo telefoni zirenga 32 000' | source label: economy; prediction: technology. The article combines funding and mobile-phone terms, mixing economy and technology cues.

Error: 'Amwe mu mateka yaranze uwari Perezida wa Burkina Faso Braise Compaore' | source label: history; prediction: politic. A historical profile of a former president contains strong political vocabulary.

Correct: 'Abantu 35 bari guhugurirwa kuyobora abashaka kureba amoko y’inyoni mu Rwanda' | source label: tourism; prediction: tourism. The title contains bird-watching and guiding cues associated with tourism.

Correct: 'Abanyarwanda biga muri Congo bamaze ukwezi batajya ku ishuri' | source label: education; prediction: education. School attendance terms make the education topic clear.

Correct: 'Abanyamideli 10 batoranyijwe muri 534 bitabiriye ijonjora rya Kigali Fashion Week ya 2019 (Amafoto)' | source label: fashion; prediction: fashion. Fashion models and Fashion Week provide strong category cues.

The interpretations above are hypotheses based on article titles, source categories, and predicted labels; they were not adjudicated by a Kinyarwanda-speaking annotator. Some examples may expose broad or noisy publisher categories rather than a purely model-induced error. A manual review of full articles would be needed to separate those causes.

## 8. Deployment

The Streamlit interface accepts a Kinyarwanda headline or article and calls the packaged scikit-learn pipeline directly. It returns the Kinyarwanda topic name and an English gloss. No external model API is required, and decision margins are not presented as calibrated probabilities. The app was tested locally with an input-to-output workflow and an HTTP health check. The repository includes its exact model artifact and dependency list. The public live link will be inserted in the project links when deployment is complete; local execution remains an alternative demonstration path.

## 9. Limitations and Future Work

Deduplication removed nearly half of the published rows and all title/body groups with contradictory labels. This improves evaluation separation but changes the benchmark population. The random split still shares publishers and possible stylistic patterns across partitions. The corpus lacks dates, independent label adjudication, and broad representation of informal Kinyarwanda. A classifier trained on article bodies may be unreliable on a short headline or unrelated text, and it has no calibrated abstention mechanism.

Future work should include a Kinyarwanda-speaking review of disputed labels, a held-out publisher or time-based collection, and calibration or out-of-domain detection. A pretrained language model could be tested with enough compute and a single-change experimental design. Larger or longer recurrent models should be evaluated only after controlling token coverage and regularization. Any public deployment should be checked again close to assessment because free hosting may sleep or change.

## 10. Conclusion

A reproducible, article-level split of KINNEWS yielded a practical 14-topic Kinyarwanda news classifier. The combined word and character model achieved 81.05% held-out accuracy and 0.740 macro F1, but its advantage over the word-bigram baseline was not statistically clear. Recurrent models with learned embeddings underperformed in the tested configurations. The most useful lesson is that data duplication, label quality, and class-wise errors matter as much as model complexity for this task.

## 11. References

Adelani, D. I., et al. (2023). MasakhaNEWS: News topic classification for African languages. Proceedings of IJCNLP-AACL, 144-159. https://doi.org/10.18653/v1/2023.ijcnlp-main.10

Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. Proceedings of EMNLP, 1724-1734. https://doi.org/10.3115/v1/D14-1179

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. Proceedings of ECML. https://www.cs.cornell.edu/~tj/publications/joachims_98a.pdf

Niyongabo, R. A., Hong, Q., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. Proceedings of COLING, 5507-5521. https://doi.org/10.18653/v1/2020.coling-main.480

KINNEWS and KIRNEWS Corpus. (2020). Dataset, labels, and documentation. https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus

scikit-learn developers. (2026). TfidfVectorizer and LinearSVC documentation. https://scikit-learn.org/stable/modules/feature_extraction.html and https://scikit-learn.org/stable/modules/svm.html

TensorFlow developers. (2026). Keras recurrent layers and TextVectorization documentation. https://www.tensorflow.org/api_docs/python/tf/keras/layers

Streamlit developers. (2026). Streamlit documentation. https://docs.streamlit.io/
