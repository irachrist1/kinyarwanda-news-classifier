# KINNEWS data

The [KINNEWS corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus), introduced by Niyongabo et al. at COLING 2020, contains Kinyarwanda news articles collected from Rwandan news sites. Its repository publishes the data under an MIT license. This project uses the authors' raw CSV files, including the article URL, title, body, and topic labels. The CSV files remain outside Git because the original articles are third-party content.

## Quality audit

The two published files contain 21,268 rows, including five blank bodies. They contain 10,179 repeated URLs and 160 URLs associated with more than one label. In the authors' cleaned files, 678 identical title-and-body pairs occur in both the published train and test sets. Using that test set unchanged would overstate generalization.

The preparation script normalizes Unicode and whitespace, removes blank bodies and title/body groups with conflicting labels, then keeps one copy of each title and body. This leaves 10,831 unique articles. It creates a seeded, stratified 70/15/15 split: 7,581 training, 1,625 validation, and 1,625 test articles. No normalized title or body occurs in more than one split. The new test scores cannot be compared directly with published scores obtained from the original split.

The labels are imbalanced: the largest class has 2,070 articles and the smallest has 109. The median article has 291 words, while the longest has 11,583. Sources are concentrated in a few news sites, so the test split measures performance on similar publishers and may not represent new outlets or conversational text. Publication dates are not supplied, so a temporal holdout is not possible.

The exact source file hashes, class counts, and split sizes are recorded in [data_manifest.json](results/data_manifest.json). The first and second CSVs come from the authors' [download folder](https://drive.google.com/drive/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox).
