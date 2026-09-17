---
sop: second-pass-grasp
tactic: keshav-three-pass
---

# Second-pass grasp — On the Proper Treatment of Tokenization in Psycholinguistics

The paper separates psycholinguistic theory from tokenizer mechanics. Stimuli and experimental
regions are character intervals chosen independently of any LM vocabulary. A token LM therefore
cannot directly supply the probability of an arbitrary region or substring when token boundaries
do not align with it.

The proposed correction is to marginalize the token-string distribution into a character-string
distribution. Under an exact, multiplicative tokenizer, a character probability sums the
probabilities of every token sequence decoding to that character string. Prefix probabilities can
be expressed through a finite prefix cover, but the cover may be exponentially large, so the paper
uses Vieira et al.'s beam-summing approximation with beam size 5.

The paper introduces a focal area: any character interval overlapping a region of interest. This
lets a psycholinguistic predictor target the characters plausibly available to a reader rather than
being forced to use the whole region or tokenizer-dependent whitespace conventions. Candidate focal
areas include the first three characters, perceptual-span-dependent prefixes, and look-ahead into
the following region.

Across four English eye-tracking datasets, whole-region surprisal is rarely uniquely best. On CELER,
the first three characters give the strongest skip-rate predictor and roughly double the incremental
`R^2` of full-region surprisal. In other datasets, prefix-based focal areas match full-region
surprisal, while look-ahead improves several UCL and CELER reading-time predictions.

The paper's substantive position is that tokenization should be theoretically irrelevant to the
definition of psycholinguistic surprisal. It remains computationally consequential because exact
marginalization over noncanonical tokenizations is expensive.

