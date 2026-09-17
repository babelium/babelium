# Survey integration: arXiv 2407.11606

## Bottom line

This paper supplies the missing formal language for discussing tokenization bias. Its central
object is the round-trip Markov kernel `K = kappa tau`; a token-level consistent estimator
recovers the intended character distribution if and only if the latter is a fixed point of
`K`. The result is sharp but deliberately conditional: it characterizes asymptotic distortion
after tokenization, not tokenizer learnability, finite-sample behavior, or neural optimization.

## What the survey should retain

1. A tokenizer is not merely a deterministic segmentation algorithm. It can be modeled as a
   stochastic encoder-decoder pair between character and token strings.
2. The correct consistency test is distribution-relative: `kappa tau p* = p*`.
3. Universal preservation is stronger and is called exactness: `kappa tau = id`.
4. Stochastic tokenization can be exact. Randomness is harmless when tokenizations assigned to
   different texts have disjoint support and decode deterministically on that support.
5. Token-sequence probability is not generally character-string probability. Character
   probability requires marginalizing over every token sequence that decodes to the string.
6. Finiteness of that marginalization does not imply tractability; the decoder fiber can still
   grow exponentially.

## Judgment and limitations

- Strong contribution: a clean necessary-and-sufficient asymptotic condition and a unified
  account of inconsistency and ambiguity.
- Main limitation: token-level consistency is assumed. The theorem cannot rank tokenizers by
  sample efficiency, optimization difficulty, sequence length, or inductive bias.
- Terminology warning: “bijective BPE/WordPiece” means bijective on the canonical encoder image;
  the ordinary concatenation decoder remains many-to-one over the full token-sequence space.
- Scope warning: the paper begins with stochastic maps, but its Section 5 computational results
  are effectively restricted to deterministic decoders.

## Useful derived decomposition

The paper's argument immediately gives the finite-error bound

`||kappa q_n - p*||_1 <= ||q_n - q*||_1 + ||kappa tau p* - p*||_1`.

The first term is token-LM estimation error; the second is irreducible tokenizer round-trip
bias. This inequality is a derived consequence, not a theorem stated verbatim by the paper.

## Ready-to-use survey paragraph

Gastaldi et al. formalize a tokenizer as a pair of stochastic maps between character strings
and token sequences. Their Fundamental Principle of Tokenization shows that, assuming the
token-level estimator converges to the encoded data distribution, decoding preserves
statistical consistency exactly when the true character distribution is a fixed point of the
encoder-decoder round trip, `kappa tau p* = p*`. Exact tokenizers strengthen this condition to
`kappa tau = id`, while still allowing stochastic segmentation provided all tokenizations in
the encoder's support decode to the same originating text. This framework also exposes why
canonical token-sequence scores are generally not character-level probabilities: probability
must be marginalized over all token sequences sharing a decoded string. The result provides a
precise theory of asymptotic tokenization bias, but it assumes token-level consistency and does
not address finite-sample learnability, optimization, or model misspecification.

## Links to the next two papers

- `2306.16842`: compare the fixed-point/round-trip view here with its noiseless-channel and
  Renyi-efficiency view. The former asks whether probability is preserved; the latter should
  quantify coding efficiency.
- `2410.02691`: its token-to-character marginalization should instantiate this paper's decoder-
  fiber principle in psycholinguistic surprisal estimation.
