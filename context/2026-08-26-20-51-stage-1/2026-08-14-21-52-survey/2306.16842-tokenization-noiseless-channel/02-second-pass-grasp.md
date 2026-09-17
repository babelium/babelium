---
sop: second-pass-grasp
tactic: keshav-three-pass
---

# Second-pass grasp — Tokenization and the Noiseless Channel

The paper treats a deterministic, invertible tokenizer as a dictionary code. A text distribution
is pushed through a tokenization function `t: Sigma* -> D subset Delta*`, inducing a token-string
distribution and, from it, a unigram token distribution. Tokens are then assigned prefix-free
base-`b` codewords. The noiseless-channel metaphor identifies good tokenization with efficient
use of this hypothetical code.

For Shannon coding, the entropy of the token unigram distribution bounds the expected code length.
The paper normalizes optimal code length by the cost of a uniform code to obtain an efficiency
score comparable across vocabulary sizes. Shannon entropy, however, only linearly prices long
codewords and may tolerate many extremely rare tokens.

The main extension replaces ordinary expected code length with Campbell's discounted expected
length. Its coding bound is controlled by Rényi entropy, with `s = alpha^{-1} - 1`. Normalized
Rényi efficiency is approximated by `H_alpha / H_0`; changing `alpha` changes which part of the
frequency distribution dominates. In their interpretation, the best empirical value `alpha=2.5`
indicates that over-concentration in very frequent tokens is especially harmful.

The theoretical coding statements support a proposed Compression Principle: for a model and task,
some Rényi order should make tokenizer efficiency predictive of downstream performance. This is a
hypothesis connecting coding balance to learnability, not a consequence of the source-coding
theorems.

On English–German machine translation, the selected `H_2.5/H_0` score correlates with BLEU at
`0.78`, compared with roughly `-0.32` for tokenized sequence length. Across BPE variants and several
tokenizer families, Rényi efficiency predicts performance better than tokenizer identity alone,
but a model using both performs best. Thus the metric captures a real frequency-balance effect but
does not exhaust tokenizer quality.

