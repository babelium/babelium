---
sop: second-pass-grasp
tactic: keshav-three-pass
written_at: 2026-08-14T22:11:58+08:00
---

# Second-pass grasp — The Foundations of Tokenization

The paper’s main move is to model a tokenizer as two composable stochastic maps. For a character alphabet $\Sigma$ and token vocabulary $\Delta$, the encoder $\tau:\Sigma^*\rightsquigarrow\Delta^*$ assigns token-sequence probabilities to each text, and the decoder $\kappa:\Delta^*\rightsquigarrow\Sigma^*$ assigns text probabilities to each token sequence. Deterministic BPE or WordPiece are point-mass special cases; stochastic segmentation such as regularized Unigram fits directly. The abstraction matters because character distributions, token distributions, encoders, and decoders can all be composed in the same calculus.

Assume a true character-level distribution $p^*$ and define the token-level target as its encoder image, $q^*=\tau p^*$. If token-level estimates $q_n$ converge pointwise to $q^*$, decoding gives $p_n=\kappa q_n$. A preliminary lemma shows that a fixed stochastic map preserves this convergence on countable spaces, so $\kappa q_n$ necessarily converges to $\kappa\tau p^*$. The Fundamental Principle of Tokenization therefore states

$$
\kappa q_n\to p^* \quad\Longleftrightarrow\quad \kappa\tau p^*=p^*.
$$

The intuition is sharp: consistent estimation in token space only recovers the intended character distribution when the encoder–decoder round trip preserves that distribution. This is a necessary-and-sufficient asymptotic statement, conditional on the token estimator already being consistent. It says nothing by itself about finite samples, rates, neural optimization, model misspecification, or whether training actually produces a consistent $q_n$.

The paper calls $\kappa\tau p=p$ consistency with respect to $p$, and calls $\kappa\tau=\mathrm{id}_{\Sigma^*}$ exactness. A non-exact tokenizer may preserve a special distribution accidentally, but exactness is equivalent to preserving every distribution. Exactness also clarifies the stochastic case: an encoder may assign several tokenizations to one text, but the output supports for different texts must be disjoint, and the decoder must deterministically recover the originating text on the encoder’s support. Thus stochastic segmentation can remain lossless.

Encoder noninjectivity is the main source of inconsistency. If different texts are collapsed by lowercasing, accent or punctuation removal, whitespace normalization, or a shared `unk` token, information is lost before estimation and the original masses generally cannot be recovered. Noninjectivity need not fail for every specially chosen $p$, but it removes the distribution-independent guarantee of exactness.

Decoder noninjectivity produces ambiguity: several token sequences decode to one text. The paper distinguishes noncanonical segmentations receiving probability from the LM (“spurious ambiguity”), deliberately stochastic segmentations used for regularization, and potentially linguistic ambiguity. In all cases, a character-string probability requires marginalizing token probability over all sequences that decode to it. This is the key bridge from statistical correctness to computational cost.

For finiteness, the paper turns to deterministic string functions. A decoder is multiplicative when it respects concatenation, and has trivial kernel when no nonempty token sequence decodes to the empty string. Under these assumptions, any token sequence decoding to a text of length $n$ has at most $n$ tokens. Since $\Delta$ is finite, the text has finitely many preimages, bounded by $\sum_{i=1}^n|\Delta|^i$. Marginalization is therefore finite but potentially exponential. Deterministic canonical tokenization may be linear, while properly including probability on alternative segmentations remains hard.

Sequentiality asks whether tokenization can be implemented by finite-state machinery. Using longest-common-prefix distance, the paper defines bounded variation and invokes Choffrut’s result: a rational-set-preserving function with bounded variation is subsequential. Multiplicative functions have bounded variation, covering ordinary concatenative decoders. Encoders need separate arguments: WordPiece’s maximal-munch encoder is presented as subsequential due to bounded token-match length, while BPE has finite-state realizations only under cited conditions on its merge rules.

The paper’s contribution is therefore a clean separation: $\kappa\tau p=p$ gives distribution-relative statistical soundness; exactness gives a universal guarantee; encoder injectivity diagnoses information loss; decoder noninjectivity creates the need for marginalization; multiplicativity plus non-erasure makes the marginal finite; bounded variation connects maps to sequential computation. It is a foundational paper, not an empirical one.

## Third-pass checks

- Verify Lemmas 3.1–3.2: pointwise convergence of probability mass functions on a countable set is upgraded to $\ell_1$ convergence, which justifies passing the limit through an infinite stochastic map.
- Check the theorem’s exact assumptions and quantifiers: finite alphabets, countable free monoids, total fixed stochastic maps, $q^*=\tau p^*$, and token-level consistency taken as a premise.
- Verify Proposition 3.2: exactness should force deterministic decoding on $\operatorname{supp}(\tau)$, disjoint encoder supports, stochastic-map injectivity of $\tau$, and surjectivity of $\kappa$.
- Clarify the restricted use of “bijective.” For BPE/WordPiece, $\kappa=\tau^{-1}$ only on the encoder image; globally, several noncanonical token sequences can decode to the same text.
- Check the shift from stochastic maps to ordinary functions in Section 5. Multiplicativity, kernels, preimages, bounded variation, and subsequentiality appear to cover deterministic maps only.
- Verify the finiteness bound, empty-string case, and Choffrut prerequisites. Bounded variation alone is insufficient; preservation of rational sets is also required.
- Reconstruct Example 4.1 from the PDF/TeX because the extracted text appears to contain an index typo in one decoded probability.
