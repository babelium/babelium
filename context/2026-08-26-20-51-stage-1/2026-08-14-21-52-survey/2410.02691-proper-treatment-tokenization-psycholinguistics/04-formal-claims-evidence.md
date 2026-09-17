# Formal claims and evidence

## Character-level target

- Arbitrary psycholinguistic substrings generally do not align with tokens and require
  marginalization: `source.md:15-23`.
- The paper explicitly argues that tokenization is irrelevant to surprisal theory once stimuli and
  regions are formulated as character strings: `source.md:19-21`.

## Token-to-character marginalization

- The paper assumes exactness without global bijectivity:
  `kappa(tau(sigma))=sigma`, but not necessarily `tau(kappa(delta))=delta`:
  `source.md:169-181`.
- Prefix-cover marginalization can be exponential and is approximated with beam summing:
  `source.md:183-191`.
- Noncanonical tokenizations make exact character surprisal #P-hard in general:
  `source.md:221-223`.

## Focal-area findings

- The first three characters are motivated as information available before a skip decision:
  `source.md:141`, `source.md:159`.
- On CELER, their surprisal is the strongest skip-rate predictor; full-region surprisal is the
  weakest and has about half the incremental `R^2`: `source.md:201`.
- Look-ahead and prefix focal areas improve or match full-region predictors across datasets:
  `source.md:203-205`.

## Experimental limitations

- GPT-2 small and beam size 5 are used throughout: `source.md:253-255`.
- The study uses twenty ROI/focal-area constructions and cross-validated regressions:
  `source.md:257-263`.
- Language, modality, functional form, and participant variation limitations are stated in
  `source.md:211-215`.

