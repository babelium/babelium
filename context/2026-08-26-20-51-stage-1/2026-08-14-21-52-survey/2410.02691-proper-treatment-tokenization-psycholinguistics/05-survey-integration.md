# Survey integration: arXiv 2410.02691

## Bottom line

The paper operationalizes the decoder-fiber marginalization demanded by the formal foundations of
tokenization. Psycholinguistic surprisal should be defined over character substrings, not canonical
token sequences. Once character probabilities are available, the relevant predictor can be any
experimentally motivated focal area rather than a tokenizer-aligned region.

## What to retain

1. Canonical tokenization probability is generally not the probability of a character substring.
2. Character probability requires summing all token strings decoding to the target or covering its
   prefix.
3. Tokenizer-specific whitespace conventions should not dictate psycholinguistic ROI definitions.
4. Focal areas express which characters are plausibly available at the time a reading decision is
   made.
5. The first three characters outperform full-region surprisal for skip prediction on CELER, while
   look-ahead helps several reading-time measures.

## Required qualification

The experiments use approximate marginalization with GPT-2 small, English-only eye-tracking data,
and averaged participant measurements. Thus the methodological correction is general, whereas the
ranking of focal areas is model-, language-, and paradigm-dependent.

## Ready-to-use survey paragraph

Vieira et al. argue that psycholinguistic stimuli and regions of interest are inherently character-
level objects, whereas modern LMs define probabilities over token strings. They therefore recommend
marginalizing the token LM over every token sequence compatible with a character prefix before
computing substring surprisal. This removes token-boundary and whitespace artifacts and permits
surprisal predictors over arbitrary “focal areas.” Using approximate beam summation with GPT-2 on
four English eye-tracking datasets, they find that full-region surprisal is rarely uniquely optimal:
the first three characters best predict skipping on CELER, and prefix or look-ahead focal areas
match or improve several reading-time predictors. The character-level prescription is principled,
but exact marginalization may be intractable and the empirical focal-area rankings remain dependent
on the LM, approximation, language, and experimental paradigm.

## Relation to the other two papers

- `2407.11606` supplies the general stochastic-map language and explains why decoder fibers must be
  marginalized. This paper instantiates that principle for psycholinguistic surprisal.
- `2306.16842` studies whether the induced token distribution is efficient for learning. This paper
  instead removes tokenization from the definition of the evaluated linguistic probability.
