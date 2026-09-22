# The Mechanism Layer: A Composition Calculus and Four Derivations

This note continues [The Conversion Signature](theory-signature.md) and [Phi and the Displacement Criterion](theory-criterion.md). It covers the layer the theory calls the mechanism-family layer: the piece meant to derive *architecture-dependent* differences rather than architecture-independent structure. Before this layer, the theory had only a handful of scattered conclusions sharing one shape ("several known mechanisms turn out to be one fact"). This note turns that shape into a small calculus — three composition operators and five inference rules — that lets "one placement" become a written expression, "one architecture" become a composition of expressions, and "the difference between two architectures" become a readable gap between the costs of two expressions. Four derivations follow from running the calculus, two of which produce genuinely new conclusions.

One honest qualifier up front, and it should not be forgotten while reading the rest: the calculus at this stage is **notation plus five rules**, not a formal system in the sense of Lean. It can write a placement down, test whether two placements are isomorphic, and bound the cost of a composition. It cannot automatically search or synthesize anything. The theory's original ambition — a formal system playing the role for token schemes that Lean plays for mathematics — is still a long way off; what is established here is only that placements can be written, aligned, and compared. Whether the resulting predictions actually hold is a separate, later question.

## 1. Extending the signature: a pair of anchors, and a graph of quotients

Two gaps in the signature from the earlier note turn out, on inspection, to be the same mistake showing up twice: something that should have been written as a **structure** was written as a single **slot**.

### 1.1 The gap, in one shape

| Source | Gap | What was written as one thing |
|---|---|---|
| Untied embeddings | When weights are not tied, the anchor used for scoring and the anchor handed back to downstream are not the same map | two anchor maps |
| AlphaFold2 | The same placement depends simultaneously on a finite quotient (residue type) and a continuous quotient (torsion angle) | two quotients in parallel |
| Canonical vs. designed quotient | The quotient a tokenizer is designed to produce and the quotient it actually manages to produce (the canonical segmentation set) can differ | two quotients in series |

The first two were already flagged as "something written with one slot that needed several." Adding the third changes the diagnosis: it is not merely "several," it is **structured plurality**. A double anchor is a *pair* (two maps working in coordination); multiple quotients come with *two distinct kinds of composition* (parallel and serial); simply changing a slot into a list would capture neither the coordination nor the composition.

### 1.2 The anchor map splits into a pair

anc splits into (anc_scr, anc_ret): anc_scr maps each address into the carrier used for scoring, anc_ret maps each address into the carrier handed back downstream. The conversion becomes:

tau(x) = sel(argmax over a of c(x, anc_scr(a))), and kappa = anc_ret.

**Tying is not a default, it is a decidable condition**: tying holds exactly when the two carriers coincide and anc_scr equals anc_ret. VQ always ties (the same codeword enters both the distance computation and the input handed to the decoder); a language model with tied input/output embeddings ties; an untied language model does not; the identity-native side (text) ties by construction, since both roles are filled by the same surface form.

This split immediately changes how the section property (from [Phi and the Displacement Criterion](theory-criterion.md), section 7) must be read. Writing tau-kappa = identity out with the split anchors gives: for every address a, the address that scores highest against anc_ret(a) under anc_scr must be a itself. **Both anchor systems now appear in the condition.** So there are two independent sources of section failure: the vocabulary itself (discussed in the criterion note), and **misalignment between the two anchor systems** — a failure mode that did not exist before the split, because there was nowhere to write it down.

**A new, cheap measurable quantity: the anchor alignment rate.**

AAR = the fraction of addresses a for which the highest-scoring anc_scr(b) against the vector anc_ret(a) is b = a.

When tied, AAR = 1 follows from the Cauchy-Schwarz inequality exactly when anchor norms are uniform; when norms are not uniform, tying does not automatically force AAR to 1, so AAR inherits the earlier norm-dispersion question and turns it into something directly measurable, not merely a geometric aside. When untied, AAR is a purely empirical question with no prior to lean on. **The cost of measuring it is one matrix multiplication — zero forward passes, zero training, minutes on a single GPU.** A preregistered prediction follows directly: AAR(tied) should exceed AAR(untied) and sit close to 1; the kill condition is that a tied model's AAR falls significantly below 1 (say, under 0.95), which would mean "tying implies anchor alignment" is the wrong reading and the norm-dispersion issue needs to be re-derived.

### 1.3 The address set splits into a graph of quotients

A single address set A = Omega-plus modulo an equivalence relation becomes a finite directed graph: nodes are quotients, edges are maps between quotients, admitting two composition primitives.

**Parallel (product).** Omega decomposes as a product of components, each taking its own quotient independently: A = the product of the A_i, each A_i = Omega_i modulo its own equivalence relation. Instances: AlphaFold2 (residue type times torsion angle), product quantization, multi-head or multi-codebook VQ.

**Serial (tower).** Omega maps to A_1, which is itself further quotiented into A_2, and so on. Instances: the canonical-segmentation construction from the criterion note (the full address-sequence set further quotiented by the R-orbit relation), residual VQ, hierarchical granularity (byte to patch to word), and MLA (which inserts one extra compression beyond ordinary commitment — that extra compression is itself a quotient).

This split explains the shape of AlphaFold2's earlier "exception": it is not an exception, it is the **necessary consequence of parallel composition** — parallel components are independent, so a property can take any combination of values across the components. Consequently, "a given property holds on a given placement" is, on multi-quotient placements, an incomplete sentence unless it also states **which quotient in the graph** is meant, alongside the side and direction distinctions already required by the criterion note.

### 1.4 Cost along a tower: an inequality, and a monotonicity guess that had to be retracted

Record first a guess that seemed obvious, was written down, and then had to be struck out. The intuitive expectation was that cost along a tower should be **monotonic**: whatever a fiber loses earlier in the tower should not come back later. **This is false.** The counterexample is a degenerate quotient: collapse everything to a single point, and the second stage of the tower trivially preserves any algebraic structure (it is a homomorphism onto a point), regardless of whether the first stage preserved anything. So "cost along a tower is non-decreasing" is false in general and must not be written into the theory as a law.

**What does hold is sub-additivity.**

**Proposition.** Along a tower where tau equals tau_2 composed with tau_1, for the same Phi:

Cost_Phi(tau_2 composed with tau_1) is at most Cost_Phi(tau_1) plus Cost_Phi(tau_2).

**Proof.** Insert the intermediate point kappa_1(tau_1(x)) into the displacement for each test function phi, apply the triangle inequality, and take the supremum.

Three consequences follow. **The direction runs opposite to intuition: a tower is not more expensive than the sum of its parts, it can be cheaper.** Inserting an additional quotient layer can *lower* total cost, if the second layer erases exactly the differences the first layer created — a directly falsifiable claim, and it lands squarely on MLA (see section 3.4 below): if MLA's total cost is no higher than a same-family model without it, this proposition gets a first piece of positive evidence. Second, the commutator family (structure-preservation, from the criterion note) does not obey this inequality — a commutator is not a displacement, so inserting an intermediate point does not give a triangle inequality for it; **displacement and commutator behave differently under composition**, a second, independent instance of the fact that the two are not the same kind of quantity. Third, on parallel composition there is currently no known inequality at all — only a per-component table. If Phi happens to decompose as a sum of per-component functions, the total cost is at most the sum of the per-component costs; otherwise there is no known bound. **This is logged as unproven and must not be used as a conclusion.**

### 1.5 The expanded signature

The conversion signature becomes the tuple (Omega-plus, Q, X, c, anc_scr, anc_ret, sel, omega), where Q is the quotient graph (nodes are quotients, edges are quotient maps, admitting both parallel and serial composition), reducing to the earlier single-node signature as a special case. The count of fields grows from seven to eight, but **what was added is structure, not another slot**: Q is a graph, not a set; the anchor pair is a coordinated pair, not a list.

Three columns of this expanded signature are each independently non-trivial and, taken together, classify a real model: whether the anchor is tied (binary, read directly from a model's tie-embeddings configuration flag); the shape of the quotient graph (single node, tower, or parallel, read from whether the model uses MLA or multiple codebooks); and AAR (continuous-valued, computed by the matrix multiplication of section 1.2). **A table along these three columns can actually be filled in for real models** — which is the acceptance test for whether the expansion was worth making: everything that can be measured can now be written down.

## 2. A composition calculus: three operators, five rules

### 2.1 Objects and assembly operators

**Object**: an interface, the expanded signature tuple of section 1.5.

**Three assembly operators:**

| Operator | Notation | Definition | Instances |
|---|---|---|---|
| Serial | I_2 composed with I_1 | the first interface's A becomes the second interface's Omega | byte to patch to word; residual VQ; MLA stacked on top of ordinary commitment |
| Parallel | I_1 tensor I_2 | the carrier decomposes as a product; each factor takes its own quotient | AlphaFold2; product quantization; multiple codebooks |
| Dual | I-dagger | omega flips; anc and tau exchange roles | input side versus output side |

**One architecture equals one expression.** A standard language model is I_out composed with [computation] composed with I_in, where I_out equals the dual of I_in in the sense that their omega flags are opposite (established already in the signature note, section 4.3). A VQ-VAE is I-dagger composed with I.

### 2.2 Five rules

**R1 (duality always flips).** A paired interface's two omega flags are necessarily opposite, so each side calls sel exactly once, at opposite ends. Source: the signature note, section 4.3. Falsifiable: if some real instance has the same omega on both sides, the signature is wrong.

**R2 (serial cost is sub-additive).** Cost_Phi of a serial composition is at most the sum of the two individual costs. Established in section 1.4 above. Not monotonic — the degenerate-quotient counterexample stands.

**R3 (zero kernel).** Cost_Phi is identically zero if and only if every function in Phi is invariant along the quotient, if and only if the fiber is invisible. Established in the criterion note, section 2, with the correction from the criterion note's section 7: invariance must hold along the **effective** quotient (every layer of a tower included), not merely the nominally designed one.

**R4 (double collapse).** Every application of tau contains exactly two collapses: quotient collapse (position within the fiber) and selection collapse (the distribution over the fiber). The two have different zero-kernels and cannot be merged. Established in the criterion note, sections 5-6.

**R5 (a commutator per structure).** Every additional structure the carrier carries gives rise to one commutator; commutators do not obey R2. The sequential-structure instance has an explicit formula and a strong/weak split (prefix preservation versus concatenation preservation); the convex/smooth-structure instance has not been written, and copying the strong/weak split onto it without proof is explicitly disallowed, since differentiability is a local property and preserving convex combinations is a global one, and the two do not stand in the containment relation the sequential case relies on.

**The boundaries of these rules must be recorded honestly.** R1 has not been checked against every known instance. R5 has only one structure with an actual formula, so "a family of commutators" currently means one member plus a shape, not yet a family.

## 3. Four derivations

### 3.1 Derivation one: the single-step cost gap between AR and diffusion

The range of sel decides whether commitment is reversible. AR's sel lands in A (a discrete point); once committed, the position within the fiber cannot be recovered, so it is **cacheable** — the first collapse of R4 has already been paid, and nothing later needs to be recomputed. Diffusion's sel degenerates to the identity at intermediate steps (DDPM's own description of its final step, displaying the mean "noiselessly"); the fiber never collapses at all, so it is **not cacheable** — every step must recompute everything.

**So: cacheable equals irreversible commitment, and the single-step cost gap is its direct consequence.** This gives an established observation (that caching explains the AR/diffusion cost gap) an actual derivation (via R4 and the range of sel), rather than leaving it as an unexplained empirical pattern.

**A new sentence falls out of running this derivation carefully.** KV-cache legitimacy additionally requires the **weak** tier from the sequential-structure commutator (prefix preservation, from the criterion note's discussion of the essay's chapter 6). So the complete condition for cacheability is **two conditions, not one**: sel's range is discrete (commitment is irreversible), **and** tau preserves prefixes (the causal mask holds). **Either condition failing makes caching illegitimate**, and this explains why a bidirectional model cannot cache incrementally even though its commitment is discrete — it fails the second condition, not the first. Earlier project records had cacheability resting on only one condition; this derivation corrects that.

### 3.2 Derivation two: abstention and the sink

Abstention equals the capacity to produce a Phi-zero output. If sel's range contains no "empty" value, then every invocation must hand back some real address a — **abstention is inexpressible**. Three implementations (a register token, an attention bias, a null codeword) are all the same fact in different clothes: **an extra point bolted onto A** — which is precisely the Omega-plus = Omega disjoint-union-with-a-null-point step from the signature note.

**So the sink's necessity is a corollary of the signature, not an empirical regularity**: Omega-plus contains the extra point, but if A's construction does not include the image of that extra point, it has nowhere to land, and its mass must fall onto some genuine token — producing a content sink. An earlier correction (that the pathology is not the normalization constraint itself, but the absence of anything the constraint can be satisfied by) receives a formal statement here: **the pathology is that A is a quotient of Omega alone, not of the full Omega-plus.**

### 3.3 Derivation three (new): unreachable tokens are a necessity of vocabulary construction

Section 7 of the criterion note introduced the unreachable-token set (tokens that can be produced by the model but never by any text input passing through the tokenizer). Read through the calculus: vocabulary construction (BPE training) fixes the equivalence relation on Omega, while reachability is fixed by **sel**, specifically tau's greedy or priority-order rule. **These are different fields of the signature** — the equivalence relation is set by A's construction, reachability is set by sel — so there is no reason to expect them to align. **The existence of unreachable tokens is a necessary consequence of "the quotient and the selection are decided by different processes," not an implementation flaw.**

A falsifiable corollary follows: **a tokenization scheme where the quotient and the selection share a common source should not have unreachable tokens.** Unigram tokenization (the maximum-likelihood segmentation, where quotient and selection both derive from a single likelihood) should therefore show a lower unreachable-token rate than BPE (whose quotient comes from merge frequency and whose selection comes from merge order — two different sources). This is a direct, checkable comparison **across tokenizer families**, not a comparison of degree within one family.

### 3.4 Derivation four (new): MLA is a tower, so its cost is bounded by the tower inequality

MLA inserts one additional compression beyond ordinary commitment (compressing the key/value cache through a low-rank bottleneck). Under section 1.3's classification, this is one more layer of a tower. R2 gives Cost(MLA composed with commitment) is at most Cost(commitment) plus Cost(MLA), and since R2 is not monotonic, **total cost can come out lower than a model without MLA.**

This turns "Phi must be written separately for each architecture family" from an inconvenience into material: the MLA family's Phi reads post-compression quantities, so its zero kernel is **larger** (more test functions are invariant along the quotient) — by R3, MLA's quotient-collapse term should therefore be **smaller**. **A preregistered prediction: at matched scale, the quotient-collapse term of Cost_Phi is lower for an MLA architecture than for a standard MHA or GQA architecture.** Kill condition: if it is significantly higher, the reading "compression enlarges the zero kernel" is wrong.

## 4. Ten predictions this layer commits to in advance

All ten below are stated with their test method and kill condition fixed before any measurement, following this project's rule that a prediction not stated this precisely does not count as preregistered.

| # | Prediction | Source | Test method | Kill condition |
|---|---|---|---|---|
| P1 | AAR(tied) is close to 1 and exceeds AAR(untied) | section 1.2 | two matrix multiplications, zero GPU | tied-model AAR falls below 0.95 |
| P2 | The unreachable-token set overlaps heavily with the independently identified undertrained-token set | criterion note, section 7 | corpus scan plus literature cross-reference | overlap not significant against a frequency-matched control |
| P3 | Unigram's unreachable rate is lower than BPE's | section 3.3 | cross-tokenizer scan | reversed or no difference |
| P4 | Non-canonical mass from self-generated continuation grows with sequence length | criterion note, section 7 | generation plus re-tokenization, run jointly | does not grow with length |
| P5 | The selection-collapse term (cost_sel) falls with model scale | criterion note, section 6 | a scale ladder of models | does not fall, or rises |
| P6 | The MLA family's quotient-collapse term is lower than a same-scale MHA/GQA baseline | section 3.4 | cross-architecture comparison at matched scale | significantly higher |
| P7 | Embedding-norm dispersion makes the argmax diverge from the nearest-neighbor rule | criterion note (background) | row-norm inspection, zero GPU | norms are in fact uniform |
| P8 | AR's residual-to-embedding gap is unconstrained during training; VQ's is actively suppressed | criterion note (background) | matched-scale comparison | AR turns out to be equally suppressed |
| P9 | A paired interface's two omega flags are always opposite | R1 | check every known instance | some instance has matching omega on both sides |
| P10 | A bidirectional model cannot cache incrementally, even with discrete commitment | section 3.1 | a structural check, zero GPU | a bidirectional model with legitimate incremental caching is found |

**Five of the ten (P1, P2, P3, P7, P9) require zero GPU, and one more (P10) is a pure structural check.** This is not "do the cheap ones first" as a matter of convenience — these six genuinely do not need hardware, and each is independent of P4, P5, P6, and P8. All six can be settled before any compute is provisioned.

## 5. What this layer has and has not established

The calculus is shaped to the point of "can be written, can be aligned, can be compared": three operators, five rules (R2 newly proven in this pass), four derivations (two of which, sections 3.3 and 3.4, are new conclusions rather than restatements), and ten preregistered predictions. It cannot search or synthesize; it remains far from the Lean-like ambition stated at the outset, and that gap is recorded here rather than glossed over.

Two substantive results are new as of this pass: **cacheability requires two conditions, not one** (section 3.1), and **unreachable tokens are a necessity of "the quotient and the selection are decided by different fields of the signature"**, which additionally supplies a cross-tokenizer-family comparison rather than a within-family one (section 3.3). Left untouched: the convex/smooth commutator formula (R5 still has only one member); the completeness of the structure list itself; and the cost relationship under parallel composition, which remains an open, unproven question.
