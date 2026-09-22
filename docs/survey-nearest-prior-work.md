# Nearest Prior Work: Checking the Displacement Criterion Against 2026 KV-Cache Quantization Theory

The displacement criterion developed in [Phi and the Displacement Criterion](theory-criterion.md) rests on one core idea: cost should be measured in the coordinates downstream computation actually reads, declared from the architecture before any measurement, rather than in some architecture-agnostic reconstruction metric. Before treating that idea as a novel contribution, it needs to be checked against whatever the literature has already done with it. This note reports that check, run against four definitional/theorem-level readings (not abstracts) of papers in the KV-cache-quantization literature, plus two adjacent papers initially suspected of being rivals. The honest result: **the core idea — "measure cost in the coordinates downstream actually reads" — was independently and formally derived in 2026 within the KV-cache quantization literature, ahead of this project's own arrival at it.** What survives as a genuine increment after this check is narrower than what was originally hoped for, and is stated precisely in section 6 below.

## 1. HeadQ (arXiv:2605.03562v2, May 2026) — the closest prior instance

HeadQ's own framing states directly: "persistent cache error should be measured in model-visible coordinates." Its **Theorem 1 (Fixed-query representation of visible key error)** is an instance of the zero-kernel theorem from [Phi and the Displacement Criterion](theory-criterion.md), section 2, specialized to attention keys. It defines the null set of key perturbations that do not change the attention distribution for a fixed query q as the set of matrices of the form a rank-one term (a constant vector times the all-ones vector, transposed) plus a component N with q-transpose times each row of N equal to zero. Two key perturbations differing only within this null set induce identical attention distributions, so the "behaviorally visible" perturbation space is exactly the quotient of the full perturbation space by this null set — a direct instance of the same "quotient by what downstream cannot see" construction this project's criterion is built around.

The correspondence, term by term:

| HeadQ | This project's terms |
|---|---|
| the null set of key perturbations | the zero kernel |
| the behaviorally visible perturbation space | the part Phi can actually see |
| taking the quotient of the perturbation space by the null set | the quotient map; position within the fiber is invisible downstream |
| a softmax-null or attention-inert perturbation | a perturbation falling inside the zero kernel |

HeadQ's **Corollary 1 (Storage MSE does not order key behavior)** is precisely the argument that cost is relative to Phi, and it is **stronger than what this project had planned to establish**: it proves that there exist perturbations with arbitrarily large storage MSE that leave attention completely unchanged, so storage MSE cannot, in general, be used to rank behavioral damage at all — not merely "storage MSE is an imperfect proxy," but a proof that it fails to even preserve ranking order in some cases.

HeadQ's **Proposition 2 (Fixed-attention value surrogate)** is the value-side counterpart of a Phi-aligned quantization scheme: starting from what attention actually reads out, it derives a token distortion weighted by the square of the attention weight (the sum, over tokens j, of the squared attention weight times the squared norm of the value perturbation, divided by dimension), and uses this to set a value-quantization policy.

**The experiment this project had planned to run has already been run.** HeadQ reports null-space interventions, same-budget counterexamples, and cache-only matched-MSE panels across six models (from GPT-2 124M to Mistral 1.1B), concluding that "downstream damage follows visible burden rather than storage burden."

**What HeadQ does not have**: it covers only one interface (the KV cache); its query basis comes from **calibration-learned query PCA**, not from architecture weights alone; there is no cross-system comparison; no composition operators (in this project's terms, no serial/parallel assembly calculus); and no treatment of tokenization, VQ, or autoregressive sampling.

## 2. Attention-Preserving Transforms for KV Cache VQ (arXiv:2608.04074) — a direct precedent for the Phi-aligned VQ idea

This paper writes KV-cache quantization as transform coding, with distortion defined as the error in the attention product rather than reconstruction error: the squared Frobenius norm of the difference between the true score matrix and the quantized one decomposes into a sum, over tokens j, of the quantization error on key j, quadratically weighted by M_q equal to Q-transpose times Q (the query second-moment matrix). The authors' own name for this: "input-weighted quadratic distortion with constant sensitivity matrix." **This is exactly the "quantization error weighted by downstream-readout sensitivity" idea that this project's Phi-aligned VQ proposal was aiming at — and this paper already supplies the closed-form optimal transform for it.**

**They also already have the zero-gain condition.** Their own statement: "Orthogonal transforms are suboptimal for the high-resolution regime unless M_q is proportional to the identity." **This is precisely the preregistered condition this project intended to state independently (that the gain from a Phi-aligned scheme should vanish when the relevant sensitivity is isotropic).**

**The key difference is where the transform comes from.** Every component of their transform is derived from calibration statistics: the key mean, the key covariance, the query second moment M_q, and the score second moment M_s. The projection weight matrices for query, key, and value are used only to *define* the query, key, and value vectors themselves; they are explicitly **not used to construct the transform**. Their theoretical foundation draws on classical rate-distortion theory (Berger, 1971), transform coding (Goyal, 2001), Zador (1982), and — worth flagging specifically because this project's own evidence chain had not previously located it — **Linder's (1999) work on non-difference distortion and companding**.

**What this paper does not have**: it covers only the KV cache; there is no weight-level derivation (everything runs through calibration); no cross-interface treatment; and the authors themselves are explicit that at 2-bit precision they "use the theory to guide our design rather than as a guarantee," an honest limit on how far their own closed-form result actually reaches.

## 3. V-information (arXiv:2002.10689, ICLR 2020) — a real but resolvable near-collision

V-information's **Definition 1 (Predictive Family)** requires "optional ignorance" of a model family V; it defines a V-conditional entropy as the infimum, over predictive models f in V, of the expected negative log-likelihood the model assigns; and it defines V-information as the difference between the V-entropy of Y given nothing and the V-entropy of Y given X.

**Where V comes from is explicitly a modeling choice, and the paper treats this as a selling point rather than a limitation.** Their own words: V is "a set of predictive models the agent is allowed to use, e.g., due to computational or statistical constraints," and the paper states it "is explicit about the assumptions (as a feature instead of a bug)." In practice, V is chosen per application (by edge in structure learning, by third-order polynomial for gene networks, by PixelCNN++ for video) — a human or task-specific choice, not something read directly off a fixed architecture.

**The zero characterization is about the data, not about the architecture** — this is the substantive point of difference. Their Proposition 2.3 states V-information is zero when X and Y are statistically independent. **The zero-kernel theorem in this project's criterion says something different in kind: it is that Phi cannot see the fiber — a property of the architecture, independent of the data distribution.** This is a real, structural distinction, not a semantic one.

Two further points run in opposite directions from this project's own results. V-information can be *created* by computation (it can violate the data-processing inequality in a way ordinary mutual information cannot), whereas this project's serial-composition result (sub-additivity, from [The Mechanism Layer](theory-mechanisms.md), section 1.4) runs the other direction. And the V-information paper, in its entirety, contains no treatment of quantization, discretization, or tokenization.

**So the honest answer to the natural reviewer question "isn't this just V-information again": V-information asks how much information a *constrained observer* can extract; this project's criterion asks whether *a specific quotient map* erases something downstream can read. The family in the first case is exogenous, chosen by the modeler; the family in the second is enforced by the architecture itself.** This distinction holds up under scrutiny and is worth stating explicitly rather than assuming it is obvious.

## 4. The IPM family (arXiv:0901.2698)

Cost_Phi, as defined in [Phi and the Displacement Criterion](theory-criterion.md), section 2, is exactly an integral probability metric (IPM) with the test-function class set to Phi, evaluated between the pushforward distribution kappa*tau*p and the original distribution p. Standard special cases: bounded functions give total variation; 1-Lipschitz functions give Wasserstein-1; bounded Lipschitz functions give the Dudley metric; the unit ball of a reproducing-kernel Hilbert space gives MMD. This paper's own main result is that total variation is the unique quantity that is simultaneously a phi-divergence and an IPM.

**By the ordinary monotonicity of a supremum over a larger set, a smaller test-function class gives an IPM value no larger than a bigger one containing it. So the claim in this project's own criterion note that "when Phi contains indicator functions, the supremum equals total variation exactly" is a standard special case of this fact, and should be cited as such rather than presented as a novel derivation.**

**The IPM literature itself never discusses how to choose the test-function class — it treats that class as given.** So "Phi is derived from the architecture rather than chosen freely" is genuinely a gap in the IPM literature specifically, not a point of overlap with it.

## 5. Two papers initially suspected of conflict, and found instead to be allies (or clean non-collisions)

**"Is Hierarchical Quantization Essential?" (arXiv:2601.22244).** Single-level VQ, at a matched budget, matches hierarchical VQ's reconstruction quality. But this paper tests **multi-scale concatenation, not residual VQ**, and contains **no theory at all of per-level error versus total error**. Its core argument — that an upper level is a deterministic function of a lower level, and therefore supplies no additional reconstruction information — is a special case of zero-kernel reasoning, and its conclusion runs in the same direction as this project's tower-composition result (that a tower need not cost more than its parts, derived in [The Mechanism Layer](theory-mechanisms.md), section 1.4). **This is a genuine ally, not a collision.** Its own scale: 256-by-256 ImageNet, 1.1 to 1.7M parameters, 4x A100.

**"Task-Driven Semantic Quantization" (arXiv:2502.17842).** Uses divergence in a frozen OneFormer segmentation map as a substitute for pixel MSE. **The weighting comes from backpropagating a forward task loss, not from architectural structure**; there is no theoretical characterization of a zero-gain condition; and the readout used is a *separate* task head, not the next layer of the same model that produced the representation being quantized. **The distinction from this project's Phi-aligned VQ proposal is clean and does not require further argument.**

## 6. Direct consequences for the token theory's later material

**One.** [The Mechanism Layer](theory-mechanisms.md), section 3.4's main result must be restated with a narrower scope. The argument "derive the visible quantity from the architecture, then show a naive metric cannot rank outcomes correctly" has already been carried out, as a formal theorem, on the KV cache, by HeadQ's Corollary 1. What survives as an increment can only be: the **same derivation, run across several interfaces at once**, plus using **only weights, never calibration data**.

**Two.** The zero-gain condition proposed for Phi-aligned VQ cannot be claimed as new. The specific condition (no gain when the relevant sensitivity matrix is proportional to the identity) is already established in arXiv:2608.04074. This project's version of it can only be pitched as either a cross-interface generalization of that result, or as a demonstration that a weight-level substitute for their calibration statistic reproduces the same condition.

**Three.** The earlier informal claim that "when Phi contains indicator functions, the criterion's supremum reduces to total variation" must be restated as a citation to the standard IPM result, not presented as an original derivation.

**Four.** The rate-distortion lineage (Berger 1971, Zador 1982, and especially Linder's 1999 work on non-difference distortion / companding) needs to be positioned explicitly in this project's own literature account. **This is the entire theoretical foundation of arXiv:2608.04074, and it had not previously appeared anywhere in this project's evidence base.**

**Five.** What remains genuinely distinctive, after this check, narrows to three things:

- **The same account, applied across AR sampling, VQ, tokenization, retrieval, and hierarchical granularity at once.** Every prior instance found is locked to a single interface (the KV cache specifically, or a single quantization scheme).
- **The zero kernel, characterized as a general structural fact independent of both the carrier and the data**, together with the composition operators from [The Mechanism Layer](theory-mechanisms.md) (sub-additivity under serial composition, proven; the parallel case, not yet proven). Every prior instance is a single-interface application, with no assembly calculus around it.
- **Phi derived from architecture weights alone, never from calibration data.** This is the single point of difference from *every* precedent found in this search.

**The third point is simultaneously the largest remaining risk.** If a weight-level derivation cannot actually predict the empirically measured ordering of costs across readout sites, then this project has not added a discipline on top of the calibration-based approach — it has simply used a weaker information source to arrive at a worse answer, which is not a contribution at all.

## 7. A full sweep of the 2026 KV-cache-quantization literature

This is not two papers; it is an active subfield. What has been located so far:

| Paper | Date | Distortion defined as | What the derivation is built from |
|---|---|---|---|
| HeadQ (2605.03562) | 2026-05 | key error visible to the attention score, modulo a constant shift | **calibration-learned query PCA** |
| Runtime-Certified Bounded-Error Quantized Attention (2605.20868) | 2026-05 | a runtime certificate on attention approximation error | — |
| Sound Runtime Risk Observability (2607.28699) | 2026-07 | runtime risk gating | — |
| Attention-Preserving Transforms (2608.04074) | 2026-08 | attention-product error, with a closed-form optimal transform | **calibration statistics** |
| Through the Lens of Transform Coding (2608.14191) | 2026-08 | attention-aware distortion, additively decomposable | **bit allocation fit on a calibration set** |

**The pattern is consistent, not coincidental: every one of these uses calibration data or activation statistics as the input to its derivation.** arXiv:2608.04074 states explicitly that its query, key, and value projection weight matrices are used only to *define* the q/k/v vectors, and are explicitly **not** used to construct the transform itself; arXiv:2608.14191's bit allocation is likewise fit on a calibration set.

**A separate, adjacent line uses weights alone, but does not read downstream at all.** "SVD-Based Weight Preservation" (arXiv:2512.01343) uses an SVD of the weights themselves to decide which weights deserve high precision, and describes its own method as "data-free, structure-aware." **But it measures the importance of a weight, not what downstream actually reads from it**; it offers no bound of any kind; it makes **no claim about a null space or an invisible direction** (a low-priority weight is simply deprioritized, not shown to be harmless by construction); and its validation is accuracy on three GLUE tasks, a correlational result rather than a derivation.

**A cross-modal line (UniAR, MergeTok, UniTok, Kelix, and similar systems) builds working systems rather than a cost theory, and does not constitute a collision with this project's theoretical claims at all.**

**So the gap this search was checking for is confirmed to exist**: the specific combination of "downstream readout structure derived from weights, plus a zero-kernel characterization, plus coverage across more than one interface" does not currently exist anywhere in the literature located by this search.

## 8. What survives, stated plainly

| | Prior work | This project |
|---|---|---|
| Measuring cost in visible coordinates | **already exists** (several 2026 papers) | cannot be claimed as new |
| A single-interface zero-kernel / quotient characterization | **already exists** (HeadQ, Theorem 1) | cannot be claimed as new |
| Deriving a readout-weighted quantization scheme from the readout operator | **already exists** (arXiv:2608.04074, closed form) | cannot be claimed as new |
| A zero-gain condition | **already exists** (isotropy of the sensitivity matrix) | cannot be claimed as new |
| **The same account, across AR sampling / VQ / tokenization / retrieval / hierarchical granularity** | none found (every instance is locked to one interface) | **a real increment** |
| **The zero kernel as a general structure, independent of carrier and data, with composition operators** | none found (every instance is a single-interface application) | **a real increment** |
| **Phi derived from weights alone, not calibration data** | none found (every instance uses calibration) | **a real increment, but see the risk below** |

**The risk on the third point has now been made concrete, not merely flagged.** A weight-level derivation, from this point forward, needs a control arm to be a genuine finding rather than a hopeful assumption: **run the weight-level derivation against a calibration-level derivation and against two null models (a shape-null and a spectral-null baseline) side by side, and report the predictive performance of all four together.** Under this design, even if the weight-level derivation turns out to underperform the calibration-level one, the result is still a real, contentful finding — namely, *how much* of Phi's visibility structure can be read off the architecture alone, without ever looking at data — rather than simply a failed attempt at besting an existing method. This control arm fell directly out of doing this literature check; it was not part of the original experimental design, and its absence would have made any single-arm result uninterpretable either way.

## What remains unread

The primary rate-distortion sources themselves (Berger 1971, Zador 1982, Linder 1999) have been identified as load-bearing for arXiv:2608.04074 but not yet read directly. The undertrained-token detection literature (relevant to the reachability predictions discussed in [Phi and the Displacement Criterion](theory-criterion.md), section 7.6, and needing a check for circularity with the norm-based measurements already used elsewhere in this project) has not yet been searched. The 2026 KV-cache-quantization literature beyond the five papers located in section 7 is still an active area and this sweep should be considered a first pass, not a complete one.

## References

"HeadQ," arXiv:2605.03562v2, May 2026.

"KV Cache Vector Quantization with Attention-Preserving Transforms," arXiv:2608.04074, August 2026.

Xu, Zhao, and Liang, "A Theory of Usable Information Under Computational Constraints" (V-information), ICLR 2020, arXiv:2002.10689.

Sriperumbudur, Fukumizu, Gretton, Schölkopf, and Lanckriet, "On Integral Probability Metrics, phi-Divergences and Binary Classification," arXiv:0901.2698.

"Is Hierarchical Quantization Essential for Learned Image Compression?", arXiv:2601.22244.

"Task-Driven Semantic Quantization," arXiv:2502.17842.

"Runtime-Certified Bounded-Error Quantized Attention," arXiv:2605.20868, May 2026.

"Sound Runtime Risk Observability for Quantized Attention," arXiv:2607.28699, July 2026.

"Through the Lens of Transform Coding: Attention-Aware Bit Allocation for KV Cache Quantization," arXiv:2608.14191, August 2026.

"SVD-Based Weight Preservation for Data-Free Quantization," arXiv:2512.01343.

Berger, "Rate Distortion Theory," 1971. Zador, "Asymptotic Quantization Error of Continuous Signals and the Quantization Dimension," 1982. Linder, "On the Training Distortion of Vector Quantizers" (non-difference distortion / companding), 1999.
