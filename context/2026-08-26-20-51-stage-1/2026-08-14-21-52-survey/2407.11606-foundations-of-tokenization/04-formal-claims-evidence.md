# Formal claims and evidence

## 1. Tokenizer object

The paper defines a tokenizer as a pair of stochastic maps, an encoder
`tau: Sigma* ~> Delta*` and decoder `kappa: Delta* ~> Sigma*`.

Evidence: `source.md:73`.

> A tokenizer model ... is a pair of stochastic maps ... respectively called the encoder and the decoder.

## 2. Main theorem

Given `q* = tau p*` and a token-level consistent estimator `q_n -> q*`, decoding is
consistent for the original distribution exactly when

`kappa tau p* = p*`.

Evidence: `source.md:101-103`.

This is a fixed-point condition for the round-trip kernel `K = kappa tau`. It assumes
token-level consistency; it does not prove that a neural LM learns `q*`.

## 3. Distribution-relative consistency versus exactness

- Consistency relative to `p`: `kappa tau p = p` (`source.md:109`).
- Exactness: `kappa tau = id` (`source.md:113`).
- Exactness is equivalent to consistency for every input distribution (`source.md:119`).
- For an exact tokenizer, the decoder is deterministic on the encoder's support (`source.md:129`).

Therefore stochastic segmentation is compatible with exactness: one text may have several
tokenizations, but every tokenization used by the encoder must decode back to that text.

## 4. Ambiguity

The decoder may be many-to-one outside the canonical encoder image. If a token LM assigns
probability outside that image, multiple token sequences contribute to the same text and
the character-level probability must marginalize over that decoder fiber.

Evidence: `source.md:139`, `source.md:159-161`.

## 5. Computational scope

For a deterministic multiplicative decoder with trivial kernel, any token sequence decoding
to a text has no more tokens than the text has characters (`source.md:177-187`). The resulting
preimage is finite but may be exponentially large (`source.md:193`). Multiplicativity implies
bounded variation (`source.md:217`), but finite-state conclusions still need the additional
rational-set-preserving assumptions discussed by the paper.

## Important qualifications found in the deep read

- `kappa tau p* = p*` is distributional preservation, not necessarily per-sample recovery.
- BPE/WordPiece are bijective only between texts and the canonical encoder image, not over all
  of `Delta*`.
- Section 5 writes the decoder as an ordinary string function, so its finiteness and
  sequentiality results do not directly cover stochastic decoders.
- The stated preimage bound starts at token length 1 and omits the empty-string case; under
  trivial kernel, the empty text has the singleton preimage containing the empty token string.

