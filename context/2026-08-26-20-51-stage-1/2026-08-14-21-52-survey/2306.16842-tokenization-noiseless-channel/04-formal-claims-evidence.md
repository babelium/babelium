# Formal claims and evidence

## Core setup

- The analysis restricts tokenization to invertible deterministic functions: `source.md:35-41`.
- The tokenizer is interpreted as a dictionary code sent through a noiseless channel:
  `source.md:45-50`.

## Rényi efficiency

- Rényi entropy is introduced as the coding quantity associated with discounted expected code
  length: `source.md:147-192`.
- Rényi efficiency is defined and bounded in `source.md:199-213`.
- The correct parameter relation is `s = alpha^{-1} - 1` (`source.md:183`, `source.md:203`).
  The `+1` occurrence at `source.md:179` is inconsistent with both the preceding identity
  `alpha=(1+s)^{-1}` and Theorem 4.2.

## Compression Principle

The claim that efficiency predicts downstream quality is explicitly labeled a hypothesis:
`source.md:217-235`.

## Empirical support and qualifications

- Selected Rényi efficiency at `alpha=2.5` correlates with BLEU at `0.78`: `source.md:21`,
  `source.md:257-277`.
- The experiments approximate the theoretical expression by setting a covariance term to zero:
  `source.md:241`.
- Tokenizer family provides predictive information beyond Rényi efficiency: `source.md:267-271`.
- Generalization beyond the tested language direction, model, and data is unresolved:
  `source.md:279-281`.

