---
sop: third-pass-deep-read
tactic: keshav-three-pass
paper: arXiv:2410.02691
---

# Third-pass deep read — Proper Treatment of Tokenization in Psycholinguistics

## Formal reconstruction

Let `p_Delta` be a token-string LM and `kappa` its deterministic decoder. The induced character
probability is the decoder pushforward

`p_Sigma(sigma) = sum_{delta: kappa(delta)=sigma} p_Delta(delta)`.

Character-prefix probabilities likewise sum over token strings whose decoded outputs cover the
prefix. A focal-area surprisal is then a character-level conditional probability obtained from the
ratio of relevant prefix probabilities. This makes the predictor invariant to whether the focal
area happens to coincide with token boundaries.

## Connection to the foundations paper

The paper assumes exactly the practically important special case isolated by arXiv:2407.11606:
`kappa(tau(sigma))=sigma`, while `tau(kappa(delta))=delta` need not hold outside the canonical image.
The many noncanonical tokenizations outside that image are the source of decoder-fiber ambiguity
and the need to marginalize.

## Most important judgment

The paper is correct that whitespace and token-boundary rules should not define the linguistic
quantity being tested. Its strongest conceptual contribution is to make the character substring,
not the token sequence, the level at which surprisal theory is stated. The focal-area framework then
turns several reading-process hypotheses into directly testable predictors.

## Boundaries and assumptions

- The tokenizer is assumed deterministic, exact, and multiplicative. Stochastic tokenizers and
  lossy normalization require a more general stochastic-decoder treatment.
- Exact marginalization may be #P-hard; experiments use beam summing with beam size 5. Approximation
  error is not propagated into confidence intervals or compared across focal-area lengths.
- Results use GPT-2 small only. A different model may distribute probability across noncanonical
  tokenizations differently and change focal-area rankings.
- The empirical study is restricted to English L1 eye tracking. The authors do not test other
  languages, self-paced reading, or participant-level mixed effects.
- Measurements are averaged across participants. This removes individual variation that may
  interact with perceptual span and focal-area choice.
- The 2-by-10 focal-area design explores many predictors. Cross-validation and permutation tests
  support predictive differences, but the results remain model-selection evidence rather than a
  causal demonstration of the underlying reading mechanism.

## Key empirical interpretation

The first-three-character result for skipping is theoretically coherent: the decision to skip must
occur before the reader fixates the region, so full-region surprisal includes information not yet
available at decision time. Look-ahead improvements likewise indicate parafoveal or integration
effects. These findings are stronger than a tokenizer correction; they show that choosing the
linguistically and temporally appropriate character window changes psycholinguistic conclusions.

## Final assessment

This paper should be treated as both a methodological correction and a new predictor-design
framework. Its slogan that tokenization is “irrelevant” is true at the theoretical target level,
but only after a computationally nontrivial marginalization step whose approximation quality still
matters empirically.

