---
sop: first-pass-skim
tactic: keshav-three-pass
written_at: 2026-08-14T22:06:59+08:00
read_deeper: true
---

# First-pass skim 鈥?The Foundations of Tokenization

## Core claim

The paper models a tokenizer as a pair of composable stochastic maps and claims this framework yields necessary and sufficient conditions for preserving estimator consistency, while also formalizing ambiguity, finiteness, and sequentiality.

## Abstract's stated main result

> “Based on the category of stochastic maps, this framework enables us to establish general conditions for a principled use of tokenizers and, most importantly, the necessary and sufficient conditions for a tokenizer model to preserve the consistency of statistical estimators.”?
## Structural signals

The progression is explicit: preliminaries on formal languages, estimators, and stochastic maps; a formal tokenizer framework culminating in a “Fundamental Principle of Tokenization”? then separate statistical and computational analyses of inconsistency, ambiguity, finiteness, and sequentiality. The conclusion matches the abstract's scope and central claim.

## Load-bearing figure/table titles

- Figure 1: 鈥淓xample of an inconsistent tokenizer鈥?- No table captions are indexed.

## Read-deeper judgment

`true` 鈥?The paper directly addresses the survey's missing theoretical/formal layer, states a sharp necessary-and-sufficient result, and shows no abstract–conclusion mismatch visible at this depth. The next pass should verify the exact consistency definition, theorem assumptions, and whether the formalism genuinely covers practical tokenizers.

