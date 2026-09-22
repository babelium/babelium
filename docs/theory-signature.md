# The Conversion Signature: Identity, Similarity, and Commitment

This note gives the formal ground layer of the token theory sketched in [The Antinomy of the Token](token-antinomy.md): the minimal structure each face of a token needs to carry, the quotient construction behind the identity face, the comparison-and-anchor construction behind the similarity face, and the seven-part (later eight-part) signature that packages both into one object comparable across architectures. Where the essay simplifies for readability, this note keeps the equations, the kill conditions, and the provenance.

## 1. What structure each face actually needs

The guiding question is not "what is a token" but "what is the minimal structure each face needs in order to support the capacities assigned to it." Giving a face more structure than it needs lets the theory quietly depend on a property (usually discreteness) that was never actually required; giving it less leaves the capacity unsupported. So each capacity is derived from what it needs, not asserted.

### 1.1 The identity face

The identity face is required to be countable, indexable, cacheable, reusable, and composable. Taking these in turn:

**Indexable** requires a set A and an addressing operation. Nothing more; A need not be ordered, finite, or discrete.

**Countable** looks like it requires a countable cardinality, but what is actually used is the ability to normalize a sum over a distribution on A. That is the use of countability, not countability itself. The honest requirement is a sigma-finite measure on A, not a presumed counting measure.

**Cacheable** requires being able to decide equality: given two addresses, whether they are the same one. This is the one piece of structure the identity face cannot give up, an equality relation whose outcome is binary.

**Reusable** requires the same thing as cacheable, plus one more condition: the equality judgment must be stable across context. Address a here and address a there must be judged the same address regardless of where each occurrence sits.

**Composable** requires a partial binary operation on A that concatenates addresses into address sequences. It is partial: not every pair of addresses may be concatenated (BPE merge rules, mesh adjacency constraints are both instances of this partiality).

Identity face = (A, equality relation, concatenation), equality relation binary and stable across context.

Discreteness never appears in this signature. It is one sufficient way to realize "equality is binary," not a necessary condition, consistent with the fact that randomized tokenization can still be exact, provided the supports of the different segmentations do not overlap (equality remains decidable). What is necessary is a decidable identity, not discreteness.

### 1.2 The similarity face

The similarity face is required to support differentiability, interpolability, and generalization.

**Interpolable** requires a path structure: given two points, a connecting path exists whose interior points also lie in the carrier. This calls for convexity or path-connectedness, not a metric.

**Differentiable** requires a smooth structure: the carrier is (locally) a manifold, and the quantities in use are differentiable on it.

**Generalizable** requires a neighborhood structure: the statement "nearby points behave alike" must be expressible. This needs a topology, plus something turning "nearby" into a comparable quantity.

The third item is the crux. "Comparable" need not be a metric, need not be symmetric, need not satisfy the triangle inequality (inner products, KL divergence, and cosine similarity are none of these, yet all function as comparisons in practice). The minimal requirement is a comparison function c on the carrier, together with the neighborhood family it induces.

Similarity face = (X, topology, comparison function c), X path-connected and locally differentiable, c graded.

### 1.3 The one structural gap between the two faces

Laying the two signatures side by side, there is exactly one point of difference, but it is the whole point:

| | Identity face | Similarity face |
|---|---|---|
| Range of comparison | two values | real numbers |
| Name of comparison | equality | comparison |
| Stability requirement | stable across context | locally continuous |

Crossing this gap is exactly what the conversion interface does: turning the graded comparison into the binary equality, or the reverse. Going from graded to binary requires a choice (threshold, argmax, nearest anchor, sampling; the mechanisms differ but each is a choice); going from binary to graded requires no choice, it is an embedding, assigning a point to an address.

Commitment sits exactly on this asymmetry: commitment is the event in which a graded comparison is collapsed into a binary equality judgment. This gives commitment a definition independent of any specific architecture.

### 1.4 Reusability belongs to the address, not the content

A direct corollary of "equality must be stable across context," worth stating on its own because it dissolves a common confusion: what is stable across context is the address, not the content the address denotes. Address 5 is address 5 in any context; the representation that address 5 evokes here can be entirely different from what it evokes elsewhere. The KV cache is legitimate precisely because of the former; "the same token means different things in different contexts" is a statement about the latter. The two do not conflict, they are two different layers. The identity face stability is a property of the addressing layer; the similarity face variability is a property of the content layer. The two-faced object is not two contradictory properties on the same layer; it is one property on each of two different layers.

### 1.5 Interfaces come in pairs

A standard language model contains not one conversion interface but two:

- Input side: signal to address (tokenization, a graded-to-binary collapse, a genuine commitment) to similarity (embedding lookup, binary-to-graded, not a commitment).
- Output side: similarity to address (logits to the selected token, a genuine commitment) to signal (detokenization lookup, not a commitment).

Each side carries exactly one commitment, and the two commitments sit at different points: earliest on the input side, latest on the output side. This immediately explains an established engineering fact: the input and output vocabularies can be sized independently, with no further assumption needed, since each side is an independent commitment event, nothing requires their address sets to be the same size or the same set.

### 1.6 Where the already-proven fixed point actually lands

One fact must be logged honestly, or later generalization starts from the wrong place. The one case where a full round trip is already known to be lossless (encoding and decoding text through an unambiguous tokenizer) is a round trip between two discrete address sets of different granularity: a character string is itself a discrete address set, just at finer granularity than the token string. What is proven is a round trip inside the identity face, between two levels of coarseness. It never touches a round trip between the identity face and the similarity face, which is the theory actual target. The similarity face, in a standard language model, appears only after embedding. The already-proven case is about coarse address versus fine address, not graded versus binary. Generalizing from it has to cross a wider gap than it first appears to.

## 2. The identity criterion: what counts as the same token

### 2.1 The address set is a quotient, not a given

The relation above needs its own account. The key repositioning: A is not given first and then used, it is the quotient of content by that relation:

A = Omega / equivalence.

where Omega is the content domain (strings, image patches, point-cloud fragments, residue configurations, documents, and so on). Fixing the equivalence relation is fixing A. Designing a tokenizer equals fixing an equivalence relation on Omega, one fact, two descriptions.

This placement has a consequence for the theory ambitions: if A were a precondition, the theory would have to follow every architecture specific vocabulary; since A is a quotient, the theory need only discuss what the equivalence relation looks like, and that relation lives on Omega, independent of whatever model happens to be trained on it. The quotient structure is the formal content behind "the theory sits above any particular architecture."

### 2.2 Three grades of identity criterion

What gets called "the same token" in practice is not one thing but three, ranked by how the equivalence relation is defined:

**Nominal grade.** Equivalence is address equality itself. Always decidable, always stable, but empty: it only says an address equals itself, saying nothing about when two pieces of content should share an address.

**Constructive grade.** Equivalence is defined by a procedure on Omega: BPE merge sequence, VQ nearest-neighbor rule, patch gridding, a hash. Two items are equivalent iff the procedure gives the same output. Decidable, computable, stable across context (the procedure does not look at context), this is the grade actually used in practice.

**Functional grade.** Equivalence is defined by "the system treats the two the same": for every context, the two items induce identical behavior. This is the grade that actually matches "means the same thing" semantically.

The functional grade here is exactly observational equivalence / contextual equivalence from type theory and process calculi (the bisimulation family), a case of an outside discipline entering under the theory single admission standard: type theory contributes an actual usable criterion for the identity face, not a stylistic affinity.

None of the three grades is entirely well-behaved:

| | Decidable | Stable across context | Has content | Compatible with concatenation |
|---|---|---|---|---|
| Nominal | yes | yes | no | trivially |
| Constructive | yes | yes | yes | no, see 2.3 |
| Functional | no | needs a universal quantifier over all contexts | yes | yes |

The functional grade is undecidable, it universally quantifies over all contexts, and its stability is bought at the price of that quantifier, that is, at the price of being uncomputable. No grade is free of a defect, and the two defects are not the same defect: this is the structural cost of the gap identified in 1.3, tokenization must make a choice, and the constructive and functional grades are two different ways of paying for that choice.

### 2.3 The constructive grade is not compatible with composition, a known pathology derived from the signature

Section 1.1 supplied concatenation (composability). Requiring equivalence compatible with concatenation means requiring equivalence to be a congruence: if x is equivalent to x-prime and y is equivalent to y-prime, then x-y is equivalent to x-prime-y-prime. When congruence holds, the quotient map is a homomorphism and composition can be carried out entirely at the address level.

Check standard tokenization. Let Omega be the free monoid of strings, A-star the free monoid on the vocabulary, tau the tokenizer mapping strings to token sequences, kappa its inverse (detokenization).

Kappa is a monoid homomorphism, it simply concatenates the surface forms of the addresses, so converting two pieces back separately and joining equals joining first then converting back, unconditionally.

Tau is not: converting a joined string is generally not the same as joining the separate conversions.

An entire family of known phenomena collapses into one statement: the prompt boundary effect, the leading-space problem, the patch that token healing applies, the fact that the same string is tokenized differently depending on where it sits, all of these are the single fact that tau is not a homomorphism. They are not independent engineering quirks; they are different surface appearances of the same algebraic defect.

This gives the asymmetry from 1.5 an independent, algebraic statement: the free direction is a homomorphism; the committing direction is not. Two different routes, capacity-driven derivation and algebraic structure, arrive at the same asymmetry, corroborating each other.

### 2.4 The gap between the constructive and functional grades is a measurable, directional defect

The quotient given by the constructive grade and the quotient given by the functional grade generally differ, and the gap has a direction, with each direction corresponding to a family of known pathologies.

**Constructive splits what functional would merge**: two addresses are constructively distinct, but the system treats them identically. VQ codebook collapse is exactly this shape: nominally K codewords, functionally far fewer are ever really distinguished. Near-synonym redundancy in a vocabulary is the same phenomenon.

**Constructive merges what functional would split**: one address is treated as different things by the system depending on context. Polysemous addresses are this shape.

Both directions are measurable: the first by testing whether distinct addresses induce distinguishable downstream behavior; the second by testing whether the same address is treated consistently across different contexts. This is a directional, two-term defect, matching what a later refinement of the criterion needs.

### 2.5 What is absent is not in Omega

The quotient construction has an immediate consequence: whatever is not in Omega cannot be reached by taking a quotient. "Nothing at all" is not the equivalence class of any content; it has to be bolted on separately, giving an extended content domain Omega-plus that is the disjoint union of Omega with one extra point standing for absence.

Register tokens, mask tokens, attention sinks, null tokens, padding, all of these are instances of this bolted-on extra point, not the address of some piece of content in Omega. They have no preimage in Omega; this is the formal reason for their anomalous behavior, not an implementation coincidence.

## 3. The similarity criterion: what counts as comparable

### 3.1 The comparison function does not live on Omega

The equality relation of section 2 lives on Omega. The comparison function does not, it lives on X, and in a standard text model X appears after the address: content maps to address via the equivalence relation, address maps to X via embedding, and comparison happens on X.

The similarity face does not compare content; it compares the images of addresses. Two pieces of content cannot be compared directly, they must first be quotiented into an address, then embedded, before comparison is possible. Graded comparison happens downstream of binary judgment.

This is not a universal fact; it is a fact about text. VQ runs the other way: the encoder first produces a continuous feature, which is then quantized into an address.

### 3.2 Which face is native depends on how the conversion is placed

| Placement | Order | Native face |
|---|---|---|
| Text input tokenization | content to address to similarity space | identity |
| AR output sampling | similarity space to address | similarity |
| VQ quantization | content to similarity space to address | similarity |
| Diffusion / flow trajectory | similarity space to similarity space, repeatedly | similarity |
| Hierarchical granularity (residue / atom / frame) | content to address to similarity space | identity |
| Retrieval (RAG / KG) | similarity space or address | either pole |

Which face is native is not a property of the token; it is a property of the placement. A diagnosis follows: text-input tokenization is the outlier among the six rows (sharing that status only with hierarchical granularity), and it happens to be the starting point for nearly all token theorizing. The root assumption "a token is a slice of the signal" comes from exactly this one identity-native placement. Switch placement, and the signal is not something being cut at all, it is something being snapped onto an anchor.

### 3.3 Three grades of the comparison criterion

**Stipulated grade.** The comparison is fixed by convention (cosine, dot product, squared distance). Computable, but offers no account of why this particular choice.

**Learned grade.** The comparison is shaped by the objective. Contrastive learning optimizes it directly; any trained embedding matrix has its geometry shaped this way. This is the grade actually used in practice.

**Functional grade.** The comparison between two points is large iff downstream computation treats the two points similarly. This is the grade that is semantically wanted.

The functional grade here and the functional grade for identity (2.2) are the graded version of the same thing: identity functional grade asks whether treatment is identical (binary); this functional grade asks how much treatment differs (graded). This is not a coincidence, it is the same gap from 1.3 recurring at the level of the criterion, as it will recur again at the level of the signature.

### 3.4 Anchor plus comparison: three mechanisms are one

Converting from the similarity space to the address set requires partitioning the similarity space into cells, one address per cell. The comparison function alone cannot do this; it needs a set of anchors, one representative content per address. The conversion selects the address whose anchor scores highest under the comparison function against the input.

Substituting three cases:

| Mechanism | Anchor | Comparison |
|---|---|---|
| AR argmax / sampling | each row of the unembedding matrix | dot product |
| VQ nearest neighbor | each codeword in the codebook | negative squared distance |
| Attention | each key | scaled dot product |

Three things treated as different mechanisms are three substitutions into the same expression. The anchor is the address representative in the similarity space; detokenization is the anchor map itself.

### 3.5 The two round trips break in opposite directions

There are two composites of the anchor map and the conversion map, and which one is the identity depends on which face is native:

**Identity-native (text).** Converting an address to content and back returns the original string (the round trip is lossless on strings); but converting content to an address and back to an address again does not return the original address on the address-sequence side (a non-canonical address sequence gets re-cut).

**Similarity-native (VQ).** Converting an address to content and back to an address returns the original address (a codeword quantizes back to itself); but converting content to an address and back to content does not return the original content (quantization error).

The two placements break the opposite composite.

This is already known algebraically from 2.3 (the return map is a homomorphism, the conversion map a section); this section supplies its dual (the return map is a section of the conversion map). The same pair of maps swaps algebraic identity depending on placement.

### 3.6 The fixed point is identically false when similarity is native, a wound in the core

The already-proven case (1.6) amounts to a round-trip fixed point on a starting distribution. In similarity-native placements, converting content to an address and back pushes any distribution on the similarity space forward into a discrete measure supported on finitely many anchors. If the starting distribution is continuous, the pushed-forward discrete measure can never equal it.

The fixed-point condition is not merely hard to satisfy, it is identically false. This is not something later generalization can patch to cover the similarity-native cases; there, the condition has no content to begin with.

But the same failure points precisely at the replacement. If a downstream readout is to give the same value before and after the discretization, that readout must be coarse enough not to see the discretization, that is, it must read only the expectation over a family of test functions. This is exactly the shape of weak convergence / moment matching: require the expectation of every test function in a declared family Phi to agree before and after the round trip, rather than requiring the two distributions to be identical pointwise.

A larger Phi makes the condition stronger, a smaller Phi makes it weaker, the theory substantive content becomes "what is Phi for each placement." This is taken up formally in [Phi: What Downstream Actually Reads](theory-phi.md).

## 4. The seven-part signature

Sections 1 through 3 accumulated six hard requirements: interfaces come in pairs (1.5); the two directions have different algebraic identity, and which direction is a homomorphism swaps with placement (2.3 / 3.5); an anchor map is required (3.4); a binary flag for which face is native is required (3.2); the address is a quotient (2.1); absence must be bolted on (2.5). One signature satisfying all six:

The conversion signature is the tuple (Omega-plus, A, X, c, anc, sel, omega), where:

- Omega-plus is the content domain with absence already bolted on (2.5).
- A equals Omega-plus modulo the equivalence relation, the address set as quotient (2.1).
- X is the similarity carrier, path-connected and locally differentiable (1.2).
- c is the graded comparison function (1.2 / 3.3).
- anc maps each address to a representative in some carrier C, the anchor map (3.4).
- sel is the selection function, defined on fibers (detailed in section 5).
- omega is a binary flag in {identity-native, similarity-native}, recording which face is native (3.2).

Omega determines C, and in turn which map is given and which is derived:

| | C | Given | Derived |
|---|---|---|---|
| identity-native | Omega | anc is the address surface form | kappa is surface-form concatenation (a homomorphism) |
| similarity-native | X | anc is the anchor vector | kappa equals anc (a section) |

The anc in both columns is the same underlying fact, only landing in a different carrier: in text, anc(a) is the string the address a spells out; in VQ, it is the codeword vector. Omega does not change the shape of the signature, only the range of anc. This is why all six requirements fit into one signature rather than two.

Tau has the same form in both columns:

tau(x) = sel(argmax over a in A of c(x, anc(a)))

When identity-native, c is a match/no-match test (binary), and the argmax gives every legal segmentation; when similarity-native, c is graded, the argmax generally gives a single point, but the fiber is still a continuous set. Both columns need sel, for different reasons, taken up in section 5.

### 4.1 A correction: commitment is not "graded collapsed to binary"

Section 1.3 located commitment on the graded-to-binary gap. Writing the signature out shows that placement is not quite tight enough, and needs sharpening.

On the identity-native side, c is already binary (string matching), with no grade to collapse. But tokenization genuinely does commit: the same string may have several legal segmentations, and which one is chosen is not decided by matching alone. BPE decides by merge priority order; greedy decoding decides by longest match. Commitment sits on this ordering decision, not on the matching itself.

On the similarity-native side, the argmax selects a unique anchor, but the set of x mapped to that anchor forms an entire fiber, and only the anchor is returned as its representative. Commitment sits on "letting the anchor stand in for the whole fiber."

The shared structure across both sides is not graded-collapsed-to-binary, it is:

Commitment = selecting one representative from an underdetermined choice.

The underdetermination on the identity-native side is finitely many segmentations; on the similarity-native side it is one continuous fiber. The two differ only in the cardinality of the underdetermined set, not in kind.

This is a generalization of section 1.3, not a retraction of it. Graded-to-binary is one source of underdetermination (a continuous fiber), not the only one. The gap itself still stands (the table in 1.3 is unchanged), but commitment location moves from the gap to the underdetermination, a more general position, and one that now covers both placements under a single sentence. Commitment no longer depends on the condition "a continuous face is present," so purely discrete interfaces (knowledge-graph retrieval, character-to-word mapping) fall under it as well.

### 4.2 Sel is the one slot in the signature carrying real freedom

Omega-plus, A, X, c, anc, and omega all describe what the interface looks like; sel describes what the interface does. It is defined on fibers:

sel maps each fiber to either a single address in A (commitment is deterministic) or to a distribution over A (commitment is stochastic).

If the range is A, commitment is fixed; if the range is a distribution over A, commitment is random. The already-proven lossless case permits randomized tokenization while remaining exact precisely when sel takes distributional values and different fibers have disjoint support: randomness lives in sel and nowhere else.

Substituting the five placements: merge priority order / argmax or multinomial sampling / nearest neighbor / a chain of sel calls amortized along diffusion timesteps / the choice of granularity level / retrieval top-k truncation.

### 4.3 Interfaces come in pairs: two opposite-omega interfaces composed

Section 1.5 pairing can now be written explicitly. A standard language model: Omega maps via an input interface (omega = identity-native) into X, computation acts within X, then an output interface (omega = similarity-native) maps X back to Omega.

The two interfaces necessarily have opposite omega: the input side enters from content, the output side exits from representation. Consequences:

- Each side calls sel exactly once, at opposite ends (the pairing of 1.5 is now proven, not merely observed).
- The two sides address sets A are two independent quotients, with no requirement that they be the same (the empirical fact that input and output vocabularies can be sized independently).
- The two sides break the opposite composite (3.5), so their errors have different origins and cannot be summed into one quantity.

The VQ autoencoder is isomorphic to this: the encoder side is similarity-native, the decoder side identity-native, again opposite omega. "Interfaces come in pairs with opposite omega" is a testable structural prediction, checkable against any of the five placements.

## 5. Commitment: the event where sel is actually invoked

Section 4.1 fixed commitment as "selecting one representative from an underdetermined choice," folding it into sel. What remains is the event own account: when it is invoked, whether it can be observed, and whether it can be measured. Making commitment measurable is the theory central deliverable.

### 5.1 The event has a before and an after

Before sel is invoked, the fiber is open, multiple representatives are all still live. After, one representative has been fixed and the rest are unrecoverable. The substance of commitment is irreversible narrowing, not selection itself.

So "how large is a commitment" now has a shape: it measures how much was given up. But what makes "given up" large? There is a trap here that must be avoided.

### 5.2 Commitment is not entropy: an equation to avoid

The most natural quantity is the entropy of the distribution over the fiber before sel is invoked. BLT next-byte entropy is exactly this, and is the closest available proxy the evidence supports.

But the evidence chain also lists "entropy boundary equals semantic boundary equals commitment boundary" among the claims it cannot support. If commitment were defined as entropy, the theory would conflict with the evidence on day one.

Entropy asks "how uncertain am I." The cost of commitment is not uncertainty, it is irreversibility, and whether the representatives permanently given up in the fiber were expensive to give up depends on whether they carry weight downstream.

This is exactly Phi from the earlier discussion: the family of quantities downstream actually reads. So:

The cost of commitment equals the gap the fiber produces on Phi, not the entropy within the fiber.

Entropy measures dispersion inside the fiber; Phi-sensitivity measures whether that dispersion penetrates through to downstream. The two can be completely decoupled.

### 5.3 A two-dimensional classification: where the evidence chain disorder resolves

Entropy and Phi-sensitivity are two independent axes, giving four cells:

| | Phi flat (representatives within the fiber are downstream-equivalent) | Phi steep (differences among representatives reach downstream) |
|---|---|---|
| Low entropy (little real choice) | forced and harmless, nominal commitment | forced and consequential, segmentation of rare content |
| High entropy (many live candidates) | free and harmless, where registers / sinks form | free and consequential, a genuine semantic decision point |

Three pieces of evidence fall immediately into place:

**Why the first token is nearly a no-op.** Position zero fiber is Phi-flat. Removing it does not change what downstream reads, so its commitment cost is near zero, whatever its entropy happens to be. Bottom-left cell.

**Why the P0 sink can form for purely structural reasons.** A necessary consequence of the bottom-left cell: when a fiber is Phi-flat, which representative gets chosen is not constrained by downstream at all, so the choice is settled by structure (the causal mask, permanent visibility, pre-normalization) rather than content. Harmlessness is the reason it can be settled structurally, not two separate facts.

**Why small-scale removal of diffusion sinks leaves semantic alignment intact but visibly changes the rendered image.** The same commitment is flat under a semantic Phi and steep under a pixel-level Phi. Cost is not a property of the commitment alone, it is a property of the pairing between the commitment and the chosen Phi. Different papers measure different Phi, hence see different pictures, an outcome the evidence chain reached separately as "layered rather than single-factor," now explained rather than merely recorded.

**And why entropy falls short is now a corollary, not an assertion.** Entropy reads only one axis of a two-axis table. It is a one-dimensional proxy for a two-dimensional quantity, so "entropy boundary equals semantic boundary" must be false, not by empirical accident but for lack of dimensionality.

### 5.4 "Selection mechanism" and "carried content" are two different slots in the signature

The evidence chain most structural conclusion needs no extra memorization once written into the signature:

- Selection mechanism = what determines sel (structure, entropy, a learned router, a priority order).
- Carried content = what Phi reads off the result.

These are different fields of the signature; conflating them is a type error, not merely something to be cautious about.

### 5.5 Making it measurable: Phi-sensitivity under fiber resampling

The Phi-gap gives an executable protocol for measuring one commitment cost:

1. Fix Phi: state explicitly what downstream reads (task metric, probe-readable quantity, generation output, downstream computational benefit). This must be explicit, since section 5.3 shows cost varies with Phi.
2. Fix the fiber for this particular commitment: identify the representatives that sel narrowed away.
3. Resample within the fiber: swap in other representatives, holding everything else fixed. This is the new step.
4. Measure the change in Phi: the variance or worst-case gap is this commitment cost under this Phi.
5. Record entropy at the same time: for locating the cell in section 5.3 table, not for defining cost.

Step 3 requires intervention, not mere observation: the previously proposed observational protocol (jointly recording next-step entropy, representation change, and boundary decisions at fixed compute) is complementary but different, locating the row of the table; Phi-sensitivity locates the column.

This turns commitment from an explanatory concept into an operationally defined quantity: given Phi, the cost of commitment is the sensitivity of Phi under fiber resampling. Being relative to Phi rather than absolute is a feature, not a defect, since section 5.3 third piece of evidence shows the absolute version is necessarily self-contradictory.

### 5.6 When sel is invoked: atomic versus chained commitment

Sel can be invoked once, or as a chain that each only partially narrows the fiber. Under the Phi framework the difference is derivable: an atomic commitment pays the entire Phi-gap in one irrevocable step; a chained commitment pays only part of it at each step, and because the fiber is not yet fully closed, later steps can correct the gap left by earlier ones. The benefit of chained commitment is exactly that the fiber stays open, letting later steps correct the Phi-gap; the corresponding cost is that every step must recompare and recompute, so total compute rises with chain length. This is not "smoother" in some vague aesthetic sense, it is this specific structural fact.
