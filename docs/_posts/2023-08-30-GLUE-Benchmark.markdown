---
title: "Glue Benchmark"
categories: NLP
---

[Glue Leaderboard](https://gluebenchmark.com/leaderboard/)

# Single sentence tasks

## CoLA (Corpus of Linguistic Acceptability)

- Binary classification—acceptable/unacceptable
- Matthews correlation coefficient $MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

## SST-2 (Stanford Sentiment Treebank)

- Binary classification—positive/negative

# Similarity and Paraphrase Tasks

## MRPC (Microsoft Research Paraphrase Corpus)

- Binary classification—paraphrase or not for a pair of sentences
- F1-score and accuracy

## STS-B (Semantic Textual Similarity Benchmark)

- Annotated similarity (1–5) by human
- Pearson correlation coefficient and Spearman correlation coefficient

## QQP (Quora Question Pairs)

- Binary classification—paraphrase or not for a pair of sentences
- F1-score and accracy

# Inference Tasks (NLI, Natural Language Inference)

## MNLI (Multi-Genre NLI)

- Labeled (entailment/contradiction/neutral)

## QNLI (Question-Answering NLI)

- Derived from SQuAD
- Predicting whether the sentence contains the information to answer the question

## RTE (Recognizing Textual Entailment)

- Binary classification (entailment/not-entailment)
- Not-entailment means either neutral or contradiction

## WNLI (Winograd NLI)

- Predicting which noun a pronoun corresponds to
