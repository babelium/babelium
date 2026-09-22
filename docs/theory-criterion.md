# Phi and the Displacement Criterion: Measuring Whether a Conversion Loses Anything

This note continues [The Conversion Signature](theory-signature.md). It develops the theory's central deliverable in full technical detail: a criterion for whether a given conversion (a placement of the tau/kappa/anc/sel machinery from the signature) loses information that downstream computation actually needs. The short version lives in [The Antinomy of the Token](token-antinomy.md), chapters 5 through 7. This note is the long version: the definitions, the failed attempts, the theorems, the places the criterion cannot reach, and the head-on conflict at the end that narrows its scope.

## 1. Phi must be declared before measurement, never picked after

The problem to solve first: if Phi (the family of test functions standing for "what downstream actually reads") could be picked after seeing the conversion result, any conclusion becomes achievable by picking a Phi that supports it. The theory would collapse into "everything depends on how you define downstream." This is not a defect to patch after the fact; it must be closed before any measurement is taken.

The rule: **Phi must be declared before measurement, and may only be read off the architecture, never picked from the result.**

"Read off the architecture" is a decidable operation, not a matter of taste: Phi is the set of quantities that this interface's output is actually consumed by in the next step of computation. This can be settled by reading code, independent of what anyone wants to prove.

Three instances, all forced by architecture:

| Interface | What the next stage receives | Phi |
|---|---|---|
| AR output sampling | one vector, the embedding of the sampled token | log-likelihood of the continuation / task metric |
| VQ encoding | a codeword index, which the decoder reconstructs from | reconstruction error + whatever the discriminator reads |
| Attention | the value vector after weighted summation | everything downstream of that layer |

If two people declare different Phi for the same interface, at least one of them did not read it off the architecture. This is what makes Phi arbitrable.

One immediate consequence falls out. The AR output interface's next stage receives only `emb(a_t)`, a function of the address alone. So the Phi-sensitivity **inside the fiber** of this interface is identically zero, by definition. The entire cost of this commitment cannot lie in "some hidden detail still inside the selected token"; it can only lie in the act of selecting this one and discarding all the rest. This corrects an earlier protocol: for interfaces of this shape, "swap in a different representative inside the fiber" measures nothing, since downstream cannot see any difference inside the fiber. What must be measured instead is swapping in **no commitment at all**:

cost_t = Phi(feeding the weighted-average embedding sum_a p_t(a) emb(a)) minus Phi(feeding emb(a_t))

The committed path feeds the embedding of the selected address; the uncommitted path feeds the p_t-weighted embedding average. Both live in the same embedding space; no architecture change and no retraining is required. The difference is precisely "was sel invoked or not." This is the first time the resampling protocol from the signature note reaches an executable form, and it is forced by the Phi rule, not chosen for convenience.

## 2. A theorem: zero cost is exactly "the fiber is invisible to Phi"

Write the object precisely, since earlier looser phrasings caused real errors later (section 7). Let p be a distribution on the carrier Omega, kappa*tau*p its pushforward under the round trip kappa-tau: Omega to Omega (both sides live in the same space; no external bridge is needed once anc is a genuine section, discussed further in section 4). Phi is the family of test functions declared per the rule of section 1.

**Definition.** Cost_Phi(p) = sup over phi in Phi of the absolute difference between the expectation of phi under kappa-tau-pushforward-p and the expectation of phi under p.

This is an integral probability metric (IPM) between p and kappa-tau-pushforward-p, with Phi playing the role of the test-function class. If Phi is the unit ball of bounded measurable functions, this is total variation; if 1-Lipschitz functions, Wasserstein-1; if the unit ball of an RKHS, MMD. **As mathematics this has no novelty at all** — IPMs are standard objects. The entire substantive content of the theory sits in one place: how Phi is chosen for each placement, per the rule of section 1.

**Theorem.** Suppose anc is a section of the quotient map, i.e. tau(anc(a)) = a for every address a (every anchor is caught by its own equivalence class). Then Cost_Phi(p) = 0 for all p if and only if every phi in Phi is invariant under the equivalence relation, i.e. phi = phi-bar composed with tau for some phi-bar.

**Proof.** (Sufficiency.) Suppose phi = phi-bar-of-tau. Then phi(kappa*tau*x) = phi-bar(tau*kappa*tau*x) = phi-bar((tau*kappa)(tau*x)) = phi-bar(tau*x) = phi(x), using tau*kappa = identity on A (the section property). This holds pointwise, so expectations agree under any p. (Necessity.) If some phi in Phi is not equivalence-invariant, there is an x with phi(x) not equal to phi(kappa*tau*x); take p to be the point mass at x, giving Cost_Phi(p) at least the nonzero gap.

Three points about this theorem deserve emphasis:

**It is independent of the data distribution p.** Whether the criterion holds depends only on what downstream reads, never on what data is actually fed to the model. This lines up with the Phi rule of section 1: nothing about the data can be leveraged to change whether the criterion holds.

**It is doing double duty on one structural fact.** The section property tau*kappa = identity on A gives idempotence of kappa*tau in the earlier round-trip argument, and here gives pointwise invariance. One fact doing two jobs is evidence it is the actual load-bearing assumption of this layer, not an incidental technical convenience. (Section 7 shows this same fact is also the theory's most fragile point.)

**Equivalence-invariant is exactly "the fiber is invisible."** phi = phi-bar-of-tau means phi depends only on the address, never on which specific content within the fiber produced it. So:

**Cost_Phi is identically zero if and only if the fiber is invisible to Phi.**

This resolves an earlier disordered finding (a table of six placements where four turned out to have identical Phi, with the real discriminating variable being something else): the something else was fiber visibility, and this theorem shows fiber visibility is exactly the zero-set of the criterion. It also explains why so many different conversion scenarios turn out, on close inspection, to declare similar-looking Phi: they are similar precisely because all of them fall inside the safe zone of "depends only on the address." What actually distinguishes placements is never Phi by itself, but whether something else exists that can pierce the safe zone and re-expose the details the conversion pressed away — tying together an observation from the diffusion/AlphaFold2 case study (a different ruler sees a different world) with this theorem in one formal statement.

**Non-vacuity, checked.** There is at least one placement where the criterion holds exactly and non-trivially: diffusion pixel binning. Taking Phi to be the family of bin-indicator functions, phi(kappa*tau*x) = phi(x) holds pointwise (a bin's center is inside its own bin), so the cost is exactly zero even though kappa*tau is not the identity map. The kernel of the criterion is non-trivial; the word "weak" in "weak substitute criterion" carries real content, not an empty qualifier.

## 3. The equality-form criterion is nearly vacuous, and must be recast as a functional

Before the theorem of section 2, an earlier attempt tried writing the criterion as a bare equality — does kappa*tau*p equal p, yes or no. This section records why that attempt failed and how failing pointed to the right fix, because the failure mode recurs (section 7 shows a version of the same mistake a third time).

Writing the equality out with Phi taken to be the full class of bounded measurable functions with sup-norm at most 1:

sup over such phi of the gap between the two expectations equals twice the total variation distance between p and kappa*tau*p.

So the equality-form criterion holds if and only if kappa*tau*p equals p as distributions. But **distributional equality does not imply the map is the identity.** A ready counterexample: on a two-point space with p uniform, let kappa*tau swap the two points; then kappa*tau*p equals p as a distribution while kappa*tau is nowhere equal to the identity map. If the criterion only recovers "measure-preserving," it fails to recover the already-proven lossless round-trip case as a special case, and the whole derivation would be untethered from the one fact it is supposed to generalize.

The gap is closed by the same structural fact used in section 2: anc is a section, so tau*kappa = identity on A, giving kappa*tau*kappa*tau = kappa*(tau*kappa)*tau = kappa*tau, i.e. kappa*tau is **idempotent**. The image of an idempotent map equals its fixed-point set, so measure-preservation plus idempotence forces p to place all its mass on the fixed-point set, i.e. kappa*tau equals the identity **p-almost surely**. The equality-form criterion is thereby shown to recover the already-proven case, but only in its almost-sure form, not its pointwise form — a weakening that must be stated permanently, since the already-proven case as originally stated was pointwise.

Substituting the seven placements into the equality-form criterion gives one true case (lossless tokenization itself) and five false, plus one undefined (retrieval, where kappa*tau is not even defined because no commitment happens). **One true out of seven, and the one true case is exactly the case the criterion was built to generalize.** The content of the statement "the criterion holds if and only if the round trip is lossless" is "no information was lost if and only if no information was lost" — a tautology dressed as a theorem, not a substantive test. This is the same disease as a constant column carrying no discriminating information, just with the constant landing on false rather than true.

**The fix: the criterion is a functional, not a predicate.** What actually carries content is not whether the equality holds, but by how much it fails — precisely the Cost_Phi of section 2. Once cast this way, the "does it hold anywhere non-trivially" and "does it fail anywhere it should" questions can be asked properly, and section 2's theorem is the answer to the first.

## 4. Test 1: consistency with the already-proven case, and the double role of one fact

With Cost_Phi established, three self-checks the criterion should pass are worth running explicitly.

**Test 1 (consistency).** Does the criterion, at Phi = all bounded measurable functions, recover the already-proven lossless case? Yes, as derived in section 3: the equality is exactly kappa*tau = identity p-almost surely, which is the already-proven case in its weakened (almost-sure, not pointwise) form. This weakening is defensible on its own terms, not merely a concession: downstream computation only ever sees inputs actually drawn from p, so what kappa*tau does on strings that never occur under p is irrelevant to any measurable downstream quantity.

Both sides of the round trip live in the same carrier Omega, so no external bridge between different spaces is required to state the criterion (this closes off a construction that once seemed necessary for AR specifically — see section 7's discussion of why AR was mistakenly thought to require a special bridge). The criterion is also insensitive to whether the scoring anchor and the returned anchor are literally the same map (the "double anchor" case of untied embeddings): the criterion only requires kappa*tau to be some well-defined map, not that it be a nearest-neighbor projection, so it remains well-defined regardless of tying. And it never invokes the word "nearest," so no separate assumption about anchor norms being equal is needed either — a concern that would matter for a geometric picture of the map but does not touch the criterion's definition.

## 5. Test 2 and Test 3: the criterion fails on AR, and the diagnosis is a missing term

**Test 2 (non-vacuity)** has already been shown to pass in section 2: diffusion pixel binning gives an exact, non-trivial zero.

**Test 3 (discriminating power)** asks whether the criterion, on a case already known to be problematic, correctly reports a problem. This is where it fails, and the failure is informative.

Apply the theorem of section 2 to AR. By the Phi rule of section 1, everything AR's next stage reads is a function of `emb(a_t)` alone, hence equivalence-invariant. So:

Cost_{Phi_AR} is identically zero.

**As a description of what downstream can detect, this is correct** — downstream genuinely cannot tell the selected token from the ones it displaced. **As a cost theory, this is absurd.** Tokenization and sampling plainly do carry costs; token healing from the opening of the essay is acknowledged, real, and already patched in production systems. The criterion reports zero exactly where it should raise an alarm.

**Test 3 fails, and it fails cleanly** — not everywhere (which would mean the criterion has no discriminating power at all), but precisely at the one case it should have flagged. A failure this specific is diagnosable.

**Diagnosis.** A standard conversion does two collapses at once, and the criterion measures only one of them.

The first collapse is **quotient collapse** (via anc): x is replaced by anc(tau(x)), erasing the specific position within the fiber. This is exactly what Cost_Phi measures.

The second collapse is **selection collapse** (via sel): the entire distribution p_t over candidate addresses collapses to the single address a_t actually chosen, and every other candidate, along with its probability weight, is discarded. **Cost_Phi cannot see this at all**, because by the time kappa*tau*p is formed, the argmax/sampling step has already thrown the distribution away; that discarding never enters the comparison between p and kappa*tau*p.

This explains three previously separate observations at once. **Why AR looked like an outlier**: its cost lies almost entirely in the second collapse, while the criterion only measures the first (an earlier mistaken diagnosis attributed this to AR living in a different space; that was independently refuted, and this is the correct diagnosis — a difference in *which kind of loss*, not a difference in space). **Why VQ's cost seems already accounted for by its own architecture**: the commitment loss term measures exactly the first collapse (the specific gap between the encoder output and the assigned codeword), which is squarely inside what the criterion covers. **Why the "average candidate vs. selected candidate" comparison from the diffusion/AlphaFold2 discussion is not a special case of the criterion but the missing term itself.**

## 6. The missing second term, and the three tests re-scored

The fix: add the missing term honestly.

cost_sel_t = sup over phi in Phi of the absolute difference between Phi(sum_a p_t(a) * anc(a)) and Phi(anc(a_t))

This term is measurable, and the uncommitted-path construction from section 1 is exactly its measurement device: feed the weighted-average anchor down one path, feed the actually-selected anchor down the other, using the same forward pass in both cases.

This term does **not** vanish under the same equivalence-invariance exemption that zeroes out the first term, because "the weighted-average anchor" is, almost surely, not equal to any actual anchor — so even a phi depending only on the address can still detect the gap between "the averaged candidate" and "the one actually selected." **The two terms have different, independent kernels; neither term can substitute for the other, and they cannot be merged into one.**

Checking across cases: retrieval never truly selects (marginalizing over top-k), so cost_sel is zero by construction; the last diffusion step displays the model's computed mean rather than sampling, so no real selection happens and cost_sel is zero there too; AR is the opposite case, cost_sel strictly positive while the quotient-collapse term is zero. Four situations — retrieval, the last diffusion step, AR, VQ — line up into four distinct combinations of the two terms, cross-checking each other.

**The three tests, re-scored:**

| Test | Single-term criterion | Two-term criterion |
|---|---|---|
| 1. Consistency | passes (via idempotence) | passes (the second term is also zero on a lossless round trip, since p_t is already a point mass) |
| 2. Non-vacuity | passes (non-trivial kernel, diffusion binning) | passes (both terms have non-trivial and *different* kernels) |
| 3. Discriminating power | **fails** (AR reports zero) | **untested pending measurement**, but four distinct value-combinations already appear across the four cases above, so the criterion is no longer trivially uniform |

So the criterion had to be reshaped **twice**, not once: predicate to functional (section 3), then single-term functional to two-term functional (this section). The second reshaping is not a correction term bolted on; it is an acknowledgment that anc and sel are two independent collapses in the signature, and the original criterion only wrote down one of them.

## 7. The head-on collision: the section property fails on exactly the case that matters most

Everything above depends on one structural fact: anc is a section of tau, i.e. tau(anc(a)) = a for every address a — every anchor, fed back through the conversion, returns to its own class. This section examines that fact directly, because it turns out to conflict with an independently derived conclusion, and the conflict has a real, already-named instance: token healing.

### 7.1 The conflict, stated precisely

The two round-trip composites break in opposite directions depending on placement (established independently in the signature note, section 3.5): in identity-native placements (text), kappa*tau = identity holds (the string round-trips losslessly) while tau*kappa is not guaranteed to equal the identity on addresses (a non-canonical address sequence can get re-cut differently). In similarity-native placements (VQ), it is the reverse.

But the theorem of section 2 (and the idempotence argument of section 3) needs precisely tau*kappa = identity on A — the section property — to hold **in identity-native placements**, which is precisely the placement where the independently derived conclusion says it is **not** guaranteed to hold.

**This is a direct, head-on conflict between two independently derived conclusions, both resting on solid ground within their own derivations.**

### 7.2 The conflict is not hypothetical: it already has a name

The conflict does not need to be settled by further mathematics, because one side of it already has a confirmed, named, already-patched instance: **token healing**, from the opening of the essay. A token's canonical surface string, tokenized on its own, may well be re-cut differently by a greedy longest-match tokenizer than it was cut when it appeared inside its original context. This is precisely "convert-back-then-convert-again fails to return the original address," happening in production systems, already named, already carrying a standard fix.

This is considerably harder evidence than an earlier suspected counterexample (dead codes possibly existing in a VQ codebook, where a code assigned to no training input might fail the section property and nobody would know). Dead codes are a theoretical worry, unverified; token healing is a confirmed, named, already-patched fact.

### 7.3 The correct level: the failure was diagnosed at the wrong granularity

Careful tracing shows the apparent contradiction, once the levels are separated correctly, dissolves — not by weakening either conclusion, but by recognizing they were talking about different objects.

Write the tokenizer's map as tau: strings to token *sequences* (mapping into A-star, not A), and kappa-star: A-star to strings as its concatenation extension. There are then **two** distinct round-trip maps, not one:

r: A to A-plus, r(a) = tau(kappa(a)) — the **token-level** round trip: take one token's own surface form and re-tokenize it alone.

R: A-star to A-star, R(w) = tau(kappa-star(w)) — the **sequence-level** round trip: take a full sequence, detokenize it to a string, and re-tokenize the whole string.

R extends r by concatenation only if r itself happens to be a monoid homomorphism, which it generally is not — and that failure is exactly what "prefix preservation holds but concatenation preservation fails" (from the signature note) describes at the sequence level. So three previously conflated statements must be kept separate:

| Statement | Content | Status |
|---|---|---|
| Token-level section | r(a) = (a) for every a | **false** — this is what token healing demonstrates directly |
| Sequence-level section | R restricted to the image of tau equals the identity | **true**, see 7.4 |
| Concatenation preservation | r-star equals R | false — this is the prompt boundary effect |

The independently derived conclusion that "kappa*tau = identity holds while tau*kappa does not" was stated about the **token level**. What the theorem of section 2 actually needs is the **sequence level**. These are not the same claim, and treating them as one claim is where the apparent conflict originated.

### 7.4 Lemma: the sequence-level section holds automatically, given losslessness

**Lemma.** Suppose tau is deterministic and lossless on strings, i.e. kappa-star composed with tau is the identity on Omega. Then R is idempotent, and its image equals its fixed-point set, equals the image of tau, i.e. the set of **canonical** token sequences the tokenizer can actually produce.

**Proof.** R(R(w)) = tau(kappa-star(tau(kappa-star(w)))) = tau(kappa-star(w)) = R(w), using kappa-star composed with tau equals identity in the middle step. An idempotent map's image equals its fixed-point set; and the image of R is contained in the image of tau by construction, with the reverse containment following from tau(s) = R(tau(s)) for any string s.

So kappa-star **is** a genuine section of tau, restricted to the canonical set of token sequences it can actually produce. This restores both earlier results, at the sequence level rather than the token level: the round-trip idempotence argument of section 3 and the zero-kernel theorem of section 2 both go through unchanged, once the carrier is switched from A to A-star and "anchor is a section" is switched from a property of the tokenizer to the corresponding property of R.

**The cost of restoring them: the assumption changes character.** It used to be a property of the tokenizer, checkable by inspecting the vocabulary. It is now a property of the **input distribution**: whether the actual distribution of token sequences the model encounters is supported on the canonical set. This is not sweeping the problem away — it relocates the problem to where it can actually be pinned down, as the next section shows.

### 7.5 The cost of non-canonical mass, and where it comes from

Let pi be the distribution of token sequences actually encountered, and let Phi-bar be the equivalence-invariant subfamily of Phi. Restricting the section-2 argument to this subfamily gives a **section cost**:

Cost_sec_{Phi-bar}(pi) = sup over phi-bar in Phi-bar of the absolute difference between the expectation of phi-bar under the pushforward of pi by R and the expectation of phi-bar under pi.

This is not a third term added on top of the two-term criterion from section 6; it is the piece of the *first* term that section 2's argument had previously certified as exactly zero. Cost_Phi restricted to the equivalence-invariant subfamily equals Cost_sec exactly when the section property holds (identically zero, by the argument of section 2); it becomes non-zero exactly when it fails. The term count does not change; the first term **splits into two pieces**.

Because the integrand vanishes pointwise on the canonical set (the fixed-point set of R), an upper bound follows immediately:

Cost_sec_{Phi-bar}(pi) is at most 2 times the sup-norm of Phi-bar, times pi(the non-canonical complement of the canonical token-sequence set)

**"Non-canonical mass" — the probability that an actually-encountered token sequence falls outside what the tokenizer could have produced on its own — is the quantity that must be measured, not "the fraction of tokens that individually fail to round-trip."** This changes what an experiment needs to compute in two ways: it must be weighted by how often each sequence actually occurs (not counted uniformly over vocabulary entries), and the bound is a genuine inequality, not a formality — mass can move between sequences with equal phi-bar values and pay zero cost even while non-canonical mass is positive.

**Where does pi fail to be supported on the canonical set?** By construction, pi is supported on it whenever sequences arise from applying tau to text (Lemma of 7.4). So a violation can only arise from a path that produces a token sequence **without** ever passing through tau. There are exactly three such paths, and they are three already-known, independently-documented pathologies:

| Violating path | Known phenomenon |
|---|---|
| Concatenating two separately tokenized passages | prompt boundary effect; instability of few-shot template concatenation |
| Autoregressive continuation (sampling addresses directly from A, never passing through tau) | generated text re-tokenizing differently from how it was produced; inconsistency between continuation and re-encoding |
| Cutting, truncating, or reusing cache state at a token boundary | exactly what token healing patches |

**Three independently recorded pathologies are three violation modes of one derived condition.** Token healing is not a patch bolted onto "the anchor is not a section"; it is **the projection operator that pulls pi back onto the canonical set** — that operator is precisely R itself. The correct formalization of healing is "apply R once."

### 7.6 Token-level failure reclassified, and a link to a known but separately-discussed phenomenon

The token-level map r remains a real, measurable object, but its role changes: it is no longer foundational, it is a **diagnostic on the vocabulary itself**. Classify by the shape of r:

| Class | Condition | Meaning |
|---|---|---|
| Intact | r(a) = (a) | this token is the canonical segmentation of its own surface form |
| Fragmented | length of r(a) exceeds 1 | this token, in isolation, would never be produced; it only appears embedded in longer context |
| Renamed | length of r(a) equals 1 but r(a) is not a | a different token encodes the same string and takes priority |

**Lemma (renaming terminates in one step).** If r(a) = (b) with b not equal to a, then kappa(b) = kappa-star(r(a)) = kappa(a) by losslessness, so r(b) = tau(kappa(b)) = tau(kappa(a)) = (b), i.e. b is intact. Consequently there are no cycles longer than one step; there is no such thing as "repeated re-tokenization oscillation."

**Consequence: unreachable tokens.** Let A_reach be the set of tokens that appear in some canonical segmentation. A token a not in A_reach can be **output** by the model, but no text input can ever produce it. Such a token never occurs as training input and never occurs as a training target, so its input embedding and output row receive gradient signal only from negative examples. **The theory here derives an independently observed phenomenon**: a subset of vocabulary entries that are "write-only," whose embeddings stay near initialization and whose behavior is anomalous — this is the glitch / undertrained token family (the SolidGoldMagikarp class of reports). A concrete, falsifiable prediction follows: the set of tokens outside A_reach should overlap heavily with the set independently identified as undertrained by embedding-norm or logit-based detectors; the kill condition is that this overlap is not significantly above chance under frequency-matched controls.

### 7.7 The verdict, and what changes as a result

**The verdict: neither conclusion was wrong; they were about different levels.** The theorem of section 2 is not weakened in scope — restated at the sequence level, it holds without qualification; the section property is not a property of the tokenizer that can fail on text input, it is a property of the input distribution's support, and its failure points are exactly the three already-known pathologies of 7.5. What must change is bookkeeping, not scope: statements about "the anchor is a section" must specify whether they mean the token level (generally false, and token healing is the direct demonstration) or the sequence level (true whenever the input distribution is itself produced by the tokenizer, which is the normal case, and false only along the three violating paths above).

This is a genuinely different resolution from "restrict the theorem's domain to exclude text input," which was the first response the conflict suggested. The theorem's domain is not restricted; what is added is an explicit account of what happens exactly where the naive statement would have predicted trouble — and that account turns out to reproduce, independently, three phenomena that were already known and already named by engineers who had never heard of this framework.

## 8. Where the criterion still cannot reach: process versus outcome

One class of question survives every repair above and is not fixable by adding more terms: questions about **how a given conversion rule was produced**, as opposed to what the rule looks like once produced. Whether a gradient in VQ's training pulled the fiber-side of the loss or the anchor-side (the two `stop-gradient`-separated pulls in the commitment loss); whether an update came from backpropagation or from a non-gradient statistical rule (an EMA update to a codebook); whether a given commitment ever actually entered the training graph at all — every one of these is a question about **provenance**, and Cost_Phi and cost_sel are both functions purely of the pushforward pair (p, kappa*tau*p), never of anything upstream of that pair.

This is a stronger claim than an earlier, more technical objection (that a criterion built from integrals cannot see behavior on a measure-zero set, since changing tau there does not change any integral — true, but open to the objection that no one actually perturbs tau on a measure-zero set in practice). The provenance claim needs no such technical maneuver: **the update rule used during training is simply not a property of the resulting conversion map.** Two entirely different training regimes can converge, by entirely different roads, to the same final tau, and the criterion, evaluated on that final tau, cannot distinguish which road was taken. This is not a contrived edge case; it is the ordinary structure of the situation (gradient descent versus an exponential-moving-average codebook update are both real, common training regimes that can land on the same codebook).

This is not a failure of the criterion, and stating it precisely is what keeps it from being oversold. What Cost_Phi and cost_sel measure is exactly **the bill training has to pay** for a given conversion — a quotient-collapse charge and a selection charge. The criterion **tracks the ledger; it does not track who paid it, when, or by what means.** This is a correct division of labor. But it forecloses one sentence that would otherwise be tempting to write: this criterion does not "settle the matter of commitment" in general — it settles how large commitment's cost is, not how a given commitment concretely came to exist.

## 9. Current state: the shape is fixed, three concrete things remain

Pulling the preceding sections together into an honest accounting, with no item left as a vague "needs more research":

**The shape is fixed.** Not the originally hoped-for single unified criterion, but a **two-term displacement criterion** (Cost_Phi for quotient collapse, cost_sel for selection collapse) — plus, from the essay's chapter 6, **a family of commutators** for the structural properties (differentiability, cacheability, composability) that a distributional comparison structurally cannot reach, of which only the sequential-structure instance (prefix versus concatenation preservation) has an actual computable formula so far. Displacement and commutator are different kinds of quantity (one compares a point to its image along a single path; the other compares two paths from the same input) and cannot be merged into one equation without inventing an operation that does not exist. The goal of "one criterion for everything" has genuinely failed, and the manner of the failure is precise rather than vague.

**What is missing, stated so each item has a concrete completion condition:**

1. The uncommitted-path experiment of section 1 (feeding the weighted-average anchor versus the actually-selected anchor) has been designed but not yet run to produce an actual measurement.
2. The section-property conflict of section 7 leaves one open computation: given the section cost formula of 7.5, an explicit numerical account of non-canonical mass — and its breakdown across the three violating paths — has not yet been produced for a real model and a real corpus.
3. The commutator framework has only one structural instance (sequential structure) worked into a formula; a second instance for a different structure (smooth structure, say) is required before the framework can be judged a genuine theory rather than a suggestive analogy. A tempting further generalization — treating interpolability and differentiability as a similar strong/weak pair — was deliberately not attempted, because differentiability is local and interpolability (preserving convex combinations) is global, and the two properties do not stand in the containment relation the sequential case relies on; copying the pattern across would produce a conclusion that looks similar but does not actually hold.

## 10. Where the criterion's blind spot lands: back on the signature's one free slot

One correspondence across two independent lines of derivation is worth stating explicitly, because it was not designed in.

The signature (section 4.2 of the earlier note) singled out sel as the **one slot with real freedom** in the entire seven-part signature — the other six are essentially pinned once the architecture is fixed. This section's diagnosis (section 5) singled out selection collapse as **the one thing the original criterion could not measure at all**, requiring the addition of the cost_sel term.

**These are two entirely independent findings — one about where freedom sits in the notation, one about where measurement fails in the criterion — and they land on the exact same slot.** The part of the cost the criterion could not originally measure is located precisely at the one part of the signature that carries any freedom at all. Where the freedom sits happens to be exactly where measurement is hardest. This may not be a coincidence, but a structural fact about this system: freedom and measurability constrain each other.
