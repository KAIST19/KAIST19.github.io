---
title: "Glue Benchmark"
categories: NLP
---

[Glue Leaderboard](https://gluebenchmark.com/leaderboard/)

# Single sentence tasks

## CoLA (Corpus of Linguistic Acceptability)

- binary classification—acceptable/unacceptable
- Matthews correlation coefficient $MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

## SST-2 (Stanford Sentiment Treebank)

- binary classification—positive/negative

# Similarity and Paraphrase Tasks

## MRPC (Microsoft Research Paraphrase Corpus)

- binary classification—paraphrase or not for a pair of sentences
- F1-score and accuracy

## STS-B (Semantic Textual Similarity Benchmark)

- annotated similarity (1–5) by human
- Pearson correlation coefficient and Spearman correlation coefficient

## QQP (Quora Question Pairs)

- binary classification—paraphrase or not for a pair of sentences
- F1-score and accracy

# Inference Tasks\*\* (NLI, Natural Language Inference)

## MNLI (multi-genre NLI)

- derived from SQuAD
- labeled (entailment/contradiction/neutral)

## QNLI (question-answering NLI)

- predicting whether the sentence contains the information to answer the question

## RTE (recognizing textual entailment)

- binary classification (entailment/not-entailment)
- not-entailment means either neutral or contradiction

## WNLI (Winograd NLI)

- predicting which noun a pronoun corresponds to
