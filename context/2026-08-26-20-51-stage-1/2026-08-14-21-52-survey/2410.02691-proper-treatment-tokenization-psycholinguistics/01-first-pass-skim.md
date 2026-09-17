---
sop: first-pass-skim
tactic: keshav-three-pass
read_deeper: true
---

# First-pass skim — On the Proper Treatment of Tokenization in Psycholinguistics

## Core claim

Token-level language models should be marginalized into character-level distributions before
computing surprisal for psycholinguistic regions, because experimental regions are arbitrary
character substrings and generally do not align with token boundaries.

## Abstract's stated result

> Our proposal of marginalizing a token-level model into a character-level one solves this
> misalignment issue independently of the tokenization scheme.

The paper also reports that surprisal over several alternative “focal areas” predicts reading
measures better than surprisal over the nominal region of interest.

## Structural signals

The paper proceeds from a character-level formalization of stimuli and regions, through focal
areas and surprisal, to token-string marginalization and an empirical comparison on reading
datasets. The conclusion matches the abstract.

## Read-deeper judgment

`true` — it is the concrete psycholinguistic consequence of the decoder-fiber marginalization
principle identified in arXiv:2407.11606 and contains both a formal correction and empirical
evidence about predictor choice.
