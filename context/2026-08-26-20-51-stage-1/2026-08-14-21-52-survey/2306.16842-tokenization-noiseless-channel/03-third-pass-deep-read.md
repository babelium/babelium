---
sop: third-pass-deep-read
tactic: keshav-three-pass
paper: arXiv:2306.16842
---

# Third-pass deep read — Tokenization and the Noiseless Channel

## Formal reconstruction

The paper's mathematical chain is:

1. Restrict to an invertible deterministic tokenizer `t`.
2. Compute the induced token unigram law `p_Delta`.
3. Regard tokens as symbols sent through a noiseless prefix-code channel.
4. Use Shannon or Campbell coding bounds to associate token-frequency balance with code cost.
5. Normalize by a uniform-code baseline, yielding an efficiency score.
6. Hypothesize that this intrinsic score predicts neural-model performance.

The coding theorems justify the relation between Rényi entropy and a nonlinear code-length cost.
They do not prove the final step from coding efficiency to neural learnability.

## Most important judgment

The paper offers a useful intrinsic statistic, not a general theory of why tokenizers work. Its
strong empirical result is that one scalar summary of the unigram frequency distribution predicts
BLEU within the tested design. The noiseless channel is an explanatory analogy: neural models use
fixed-width token embeddings rather than the prefix codes constructed in the theorem.

## Boundaries and hidden assumptions

- The tokenizer must be deterministic and invertible. Stochastic segmentation, lossy normalization,
  unknown-token collapse, and decoder-fiber marginalization are outside the formal setup.
- The score uses unigram frequencies and discards token order, contextual dependence, morphology,
  sequence length interactions, and the geometry of learned representations.
- Experiments set the code-length/sequence-length covariance term to zero. The paper says the term
  is empirically small and negative, but the reported metric is still an approximation to the
  derived efficiency bound.
- `alpha=2.5` is selected by grid search and is explicitly model-dependent in the Compression
  Principle. It should not be treated as a universal tokenizer constant.
- The empirical evidence is mainly one MT language pair, dataset scale, and model family. The paper
  itself lists language direction, model, and training data as possible hidden effects.
- Experiment 2 shows tokenizer identity adds information orthogonal to Rényi efficiency, directly
  demonstrating that the metric is incomplete.
- The text immediately before Theorem 4.2 contains a sign typo, writing
  `s = alpha^{-1} + 1`; the surrounding derivation and theorem correctly require
  `s = alpha^{-1} - 1`.

## Interpretation of alpha

`H_alpha/H_0` measures effective support relative to vocabulary support. For `alpha>1`, Rényi
entropy is dominated by high-probability tokens; a low score therefore exposes distributions whose
mass is concentrated in a few extremely frequent tokens. The observed positive association with
BLEU supports “avoid excessive head concentration,” not the broader claim that compression alone
causes learnability.

## Final assessment

Keep the Rényi-efficiency proposal as an empirically promising frequency-balance diagnostic. Do not
present it as a proven optimality criterion. Its theory explains the metric's coding meaning; its
connection to downstream quality remains task-conditioned and correlational.

