# Kinyarwanda news project submission tracker

Due **18 October 2026 at 11:59pm**. Upload the final **PDF** to the course submission page. The brief gives both 7–10 and 10–15 minutes for the video, so aim for exactly 10 minutes.

## Direct links

- Repository: https://github.com/irachrist1/kinyarwanda-news-classifier
- Live app: https://kinyarwanda-news-topics.vercel.app/
- Demo video: [paste the direct viewable URL]

## Before recording

- [ ] Open the live app in a private browser window to check public access.
- [ ] Read the report and make sure you can explain every result and model choice.
- [ ] Practise the ten-minute demo script in your own words.
- [ ] Show several app inputs and outputs, including a success and a failure.
- [ ] Explain the majority and word baselines, each experiment's change and hypothesis, and what the results mean.
- [ ] Explain TF-IDF, the linear classifier, embedding layer, GRU/LSTM comparison, training settings, macro F1, and test split.
- [ ] Explain the errors, uncertainty interval, limitations, and how the app loads the saved model.
- [ ] Record a ten-minute video with audible narration and readable screen text.
- [ ] Test the direct video link without being signed into its hosting account.

## Before uploading

- [ ] Insert the direct video link in `report/links.json` and the README.
- [ ] Rebuild and inspect the final PDF and Word report; verify all three direct links.
- [ ] Check that the public repository contains preparation, training, evaluation, inference, the Colab notebook, results, model, and setup steps.
- [ ] Check that the live app still classifies text at its public URL.
- [ ] Upload the final PDF before the deadline and confirm the submission receipt.

## Class reminders

- [ ] Complete every Intranet task; the lecturer said incomplete tasks can prevent grading (1 September).
- [ ] Keep attendance at or above 85%; the lecturer said lower attendance receives zero (1 September).
- [ ] Be ready to explain the code and decisions yourself in the demo or viva (29 September).

## Viva practice

- Why remove repeated articles and conflicting labels before splitting? ____________________
- Why use macro F1 for 14 uneven classes? ____________________
- How does TF-IDF feed a linear support vector classifier? ____________________
- What did character features change, and is the test gain reliable? ____________________
- How do embedding, GRU/LSTM, early stopping, and test isolation work here? ____________________
- Why might health be predicted as politics or fashion as entertainment? ____________________
- What does the app load, and what happens on an empty input? ____________________
