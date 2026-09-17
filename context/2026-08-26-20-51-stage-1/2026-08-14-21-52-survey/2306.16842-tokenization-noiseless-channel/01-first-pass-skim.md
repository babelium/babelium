---
sop: first-pass-skim
tactic: keshav-three-pass
written_at: 2026-08-14T23:04:00+08:00
read_deeper: true
---

# First-pass skim — Tokenization and the Noiseless Channel

## Core claim

The paper argues that tokenizer quality can be measured intrinsically as efficient use of a noiseless communication channel, with Rényi efficiency of the induced unigram token distribution serving as a principled predictor of downstream machine-translation performance.

## Abstract's stated main result

> “In machine translation, we find that across multiple tokenizers, the Rényi entropy with $\alpha=2.5$ has a very strong correlation with Bleu: $0.78$ in comparison to just $-0.32$ for compressed length.”

## Structural signals

The paper moves from tokenization and noiseless-channel coding through Shannon's source coding theorem, then generalizes the efficiency measure using Rényi entropy, states a compression principle linking compression and learnability, and tests it in two experiments. The conclusion repeats the abstract's central numerical result and frames Rényi efficiency as an intrinsic tokenizer-evaluation metric, so there is no visible abstract–conclusion mismatch at this depth.

## Load-bearing figure/table titles

- Figure 1: Examples of unigram distributions with efficient and inefficient channel usage.
- Figure 2: Efficiency of sequence length and $\mathrm{H}_{2.5}/\mathrm{H}_{0}$ as predictors of MT performance.
- Table 1: Correlations between different predictors and MT performance (Bleu).
- Figure 3: Correlation of Rényi efficiency ($\mathrm{H}_{\alpha}/\mathrm{H}_{0}$) with Bleu in Experiment 1.
- Figure 4: Grid search over percentile-frequency predictor hyperparameters, with the highest Pearson correlation at the 3rd–83rd percentile range ($\rho=0.81$).
- Figure 5: Mean held-out log-likelihood change under linear models using different predictors.

## Read-deeper judgment

`true` — The paper is directly relevant to understanding why tokenization choices affect model performance, offers both an information-theoretic mechanism and empirical comparisons, and shows no fundamental flaw visible from the permitted material. A deeper pass should check the derivation and assumptions behind Rényi efficiency, how $\alpha=2.5$ is selected, whether correlations generalize beyond machine translation, and whether the percentile-frequency result weakens the claim that Rényi efficiency is the preferred intrinsic metric.
