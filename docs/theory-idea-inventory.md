# Idea Inventory: Candidate Research Directions from the Token Theory

This is a full inventory of research ideas that grew out of the token theory in [The Antinomy of the Token](token-antinomy.md), [The Conversion Signature](theory-signature.md), and [Phi and the Displacement Criterion](theory-criterion.md), together with a stress-test scoring pass run against top-tier ML conference standards, assuming a fixed compute budget of four 8-GPU B200 nodes (32 B200s total, enough for genuine LLM-scale training experiments). The scoring is a rough triage of "value as paper material," not a verdict on whether any of these ideas are true, and not a prediction of acceptance probability. Nothing here has been elevated to an adopted research direction; it is raw material for a researcher choosing where to point limited compute.

## 1. Raw idea list

Each entry carries a status tag reflecting where it stood at the time of this inventory: **[main line]** (load-bearing to the theory as it stands), **[open]** (a real gap, direction not yet fixed), **[frozen]** (designed but deliberately not run), **[preregistered]** (design frozen before any measurement), **[survives]** (passed an earlier stress test), **[weakened]** / **[cut]** (demoted or dropped by later evidence), **[prior]** (inherited from work that predates this theory), **[DS]** (proposed in an external consulting pass, not yet vetted against this theory's own derivations), **[supplementary]**.

1. **Two-faced token theory [main line]**: a token is a conversion interface between discrete identity and continuous similarity, not simply a slice of a signal.
2. **Token as quotient structure [main line]**: a tokenizer is, in essence, an equivalence relation stipulating "what counts as the same" over a content space.
3. **The conversion-interface signature [main line]**: a unified description of different token systems via carrier, address, comparator, anchor, selector, and placement direction.
4. **An executable token calculus [open]**: a formal system in the style of Lean or PyTorch, letting token schemes from different domains be written down and compared uniformly.
5. **A formal definition of commitment [main line]**: commitment is the event of fixing one representative while multiple legal candidates are still live.
6. **Paired input/output interfaces [main line]**: a language model contains input and output conversion interfaces running in opposite directions, with errors originating from different sources on each side.
7. **AR, VQ, and attention as one structure [main line]**: all three reduce to "compare a vector against a set of anchors, then select a representative."
8. **A double-anchor system [open]**: the anchor used for scoring and the anchor handed to downstream may not be the same set, requiring an extension of the existing signature.
9. **Multi-quotient systems [open]**: one model may carry both a discrete quotient and a continuous quotient at once, requiring study of how multiple commitment costs combine.
10. **Dynamic address sets / signature families [open]**: how a token system migrates between signatures as its address set grows or is replaced.
11. **The architecture-declared downstream readout family Phi [main line]**: loss can only be measured in coordinates downstream actually reads, and Phi must be frozen before looking at the data.
12. **Phi-visible commitment cost [main line]**: information erased inside a fiber constitutes a real cost only if it is readable downstream.
13. **Commitment entropy times Phi-sensitivity [main line]**: uncertainty and consequence are two independent axes; entropy alone cannot represent commitment cost.
14. **The two-term criterion, quotient collapse and selection collapse [main line]**: distinguishing the loss from "flattening the candidate space" from the loss of "ultimately picking only one representative."
15. **The zero-kernel theorem [main line]**: if a fiber's variation falls into the null space of every downstream readout, commitment cost is zero.
16. **Whether the anchor is a section [open]**: whether encode(decode(token)) equals token can serve as a classification condition for token systems.
17. **The cost of a failed textual section [open]**: deriving the extra cost owed in the token-healing scenario, when the anchor does not return to itself.
18. **Separating the displacement criterion from the commutator [main line]**: round-trip error and structure-preservation error are two different kinds of object and cannot be forced into one formula.
19. **The two tiers, prefix-preserving and concatenation-preserving [main line]**: the former explains the KV cache, the latter explains the prompt boundary effect.
20. **A commutator for convex / smooth structure [open]**: building a second computable commutator instance, for interpolation and differentiability.
21. **A training-process-aware theory [open]**: the existing criterion only sees the final map and cannot distinguish gradient, EMA, frozen, or inference-time selection.
22. **A classification of training visibility [open]**: distinguishing a gradient-updated carrier, a gradient-updated representative, an EMA-updated representative, and a fully frozen one.
23. **Formalizing absent addresses [open]**: a unified account of registers, `[MASK]`, padding, attention sinks, and null slots.
24. **Stepwise compensation along a commitment chain [main line]**: a chained commitment allows later steps to correct earlier error, at the recurring cost of comparison.
25. **The no-commitment counterfactual experiment [frozen]**: comparing the argmax embedding against the probability-weighted average embedding, directly measuring selection-collapse cost.
26. **Predicting downstream cost ordering from weights alone [preregistered]**: predicting the ranking of commitment cost across different readout sites using only model weights.
27. **A four-arm predictive comparison [preregistered]**: comparing a shape-null model, a spectral-null model, a weight-level model, and a calibration-level model simultaneously.
28. **A cross-interface ordering law [preregistered]**: testing whether one weight-level derivation predicts ordering simultaneously across AR, VQ, and KV-cache interfaces.
29. **The Qwen3 per-head norm prediction [preregistered]**: Qwen3's q/k commitment cost should be lower than a same-scale Qwen2's, while v is unchanged.
30. **Paired MLA-vs-GQA pretraining [survives]**: controlling data, scale, and budget to isolate the causal effect of attention structure on commitment cost.
31. **A scaling law for commitment cost [demoted]**: studying whether cost changes with model scale along a self-trained scale ladder.
32. **TransMLA conversion as a control [supplementary]**: converting a same-lineage GQA model into MLA, as cheap evidence alongside paired pretraining.
33. **Phi-aligned VQ [survives]**: weighting quantization error along directions the decoder is actually sensitive to, rather than isotropic Euclidean distance.
34. **The zero-gain condition for Phi-aligned VQ [survives]**: when the decoder's readout is near-isotropic, the new objective should yield no gain.
35. **Sub-additivity of residual VQ [survives]**: testing the compositional law between per-level commitment cost and total cost.
36. **A cross-modal VQ unification experiment [survives]**: checking the same commitment law across image, audio, and speech models.
37. **Matched BPE / Unigram training [survives]**: controlling corpus, vocabulary, model, and budget while varying only the tokenizer family.
38. **A round-trip classification of tokenizers [survives]**: sorting tokens into intact, fragmented, renamed, and illegal-byte classes.
39. **Non-canonical mass and token healing [survives]**: measuring the non-canonical token mass produced by concatenation, self-generation, and boundary cutting.
40. **Corpus-absent tokens and undertrained tokens [open]**: whether tokens that are unreachable or rare as input develop glitch behavior.
41. **Anchor alignment and norm hijacking [survives]**: testing whether long-norm output vectors hijack the argmax away from short-norm tokens.
42. **AR hidden-state-to-embedding drift [open]**: whether, absent a commitment loss, AR's hidden states drift progressively away from the vocabulary codebook.
43. **Structure/semantics decoupling of the attention sink [main line]**: distinguishing the mechanism that forms a sink from the content it later comes to carry.
44. **Attention abstention mechanisms [survives]**: K-bias, a zero-value null key, and an output gate are all ways of providing the capacity to "select nothing."
45. **Sink is a necessary consequence of missing abstention [weakened]**: the direct necessity claim did not hold up; only a composite-mapping-level mechanistic account survives.
46. **Whether fine-tuning can repair the sink [cut]**: designed as an alternative explanation, now judged not worth a dedicated training budget.
47. **A dynamic computation-boundary experiment [open]**: at matched budget, comparing entropy-based, learned-router, random, whitespace, and hand-crafted semantic boundaries.
48. **A two-condition theory of caching [main line]**: discrete commitment alone does not guarantee incremental caching; the existing prefix must also remain unmodified by later content.
49. **Bidirectional models cannot cache incrementally [survives]**: directly comparing whether existing positions' key/value change after appending a token.
50. **Independent expansion of input/output vocabularies [open]**: the two sides are independent quotients, so input coverage and output expressiveness can be scaled up separately.
51. **A replaceable non-parametric address set [open]**: exploiting RAG's hot-swappable address set to study "updatable without retraining" token systems.
52. **Fixed to adaptive to learned boundaries [inherited prior]**: whether there is a shared evolutionary pattern for token boundaries across language, vision, audio, and 3D.
53. **Content-aware dynamic tokenization [inherited prior]**: deciding granularity and compute dynamically by local information density or task demand.
54. **An end-to-end tokenizer replacing the two-stage paradigm [inherited prior]**: jointly training boundary, representation, and task objective.
55. **MeshLex / geometry-aware mesh tokenizer [inherited direction]**: building a 3D mesh tokenizer combining curvature, topology, locality, and hierarchical patching.
56. **Cross-modal regularities in unified tokenizers [inherited direction]**: searching for shared mechanisms of boundary learning and granularity economy across text, vision, audio, and 3D.
57. **Dynamic commitment cost [external consult, overlaps with 24]**: reframing single-step loss as a trajectory quantity, "initial commitment cost minus subsequent compensating capacity."
58. **A temporary address pool [external consult]**: creating extra addresses at inference time to preserve information that a fixed token representation compressed away but is still needed later.
59. **Self-feedback training unifying training and inference [external consult]**: progressively feeding the model's own predictions back in as training proceeds.
60. **Reversible distance [external consult]**: building a cross-system unified criterion from post-compensation reconstruction error plus a commitment-distribution penalty.
61. **A dynamic VQ codebook [external consult]**: splitting, expanding, or reclaiming codewords by usage, replacing commitment loss with reversible distance.
62. **A reinforcement-learning address allocator [external consult]**: learning policies for creating, retaining, merging, and reclaiming temporary addresses.
63. **Commitment entropy and safety alignment [external consult]**: whether uncertainty before a discrete choice can predict or constrain unsafe behavior.

## 2. Stress-test triage scores (2026-09-17)

### 2.1 Scoring rubric

This pass triages against the standard of a high-scoring main-track paper at NeurIPS / ICML / ICLR / IJCAI / AAAI / KDD, with available compute fixed at four 8xB200 nodes (32xB200 total). The total score is "value as paper raw material," not acceptance probability, and not a verdict on whether the underlying theory is true.

- **Novelty moat, 30 points**: can it draw a hard boundary against existing tokenizer work, task-aware quantization, memory tokens, scheduled sampling, and KV-cache quantization?
- **Decisive evidence, 25 points**: can it name an experiment that would actually falsify the claim, rather than only report correlation or case studies?
- **Scope of impact, 20 points**: does the conclusion extend beyond one model, one benchmark, or one implementation detail?
- **Compute conversion, 15 points**: can 32xB200 buy controlled pretraining, scaling laws, multiple seeds, or training-time interventions that are hard for others to reproduce cheaply?
- **Maturity, 10 points**: are definitions, baselines, metrics, and an implementation path already clear enough?

Tiers: **S (90-100)** ready to anchor a strong paper on its own; **A (80-89)** has main-track potential; **B (70-79)** best combined with other material; **C (60-69)** suited as a mechanism, ablation, or theoretical support; **D (45-59)** background only, or needs a substantial rewrite; **X (below 45)** should not receive main-line resources right now.

### 2.2 Scores for all 63 items

01. **56 / D — Two-faced token theory**: strong narrative unifying power, but on its own lacks a new algorithm or decisive evidence; suited as shared framing across several papers.
02. **55 / D — Token as quotient structure**: mathematically clean, but reviewers will ask whether it is merely a restatement of quotient-space theory; needs a bound new theorem or design.
03. **60 / C — The conversion-interface signature**: usable as a cross-system comparison language; standing alone it reads as taxonomy and must produce a non-trivial prediction.
04. **47 / D — An executable token calculus**: huge engineering cost with unclear evaluation criteria; hard to reach a high-scoring main-track paper without existing users, a verifier, and new findings.
05. **54 / D — A formal definition of commitment**: necessary vocabulary, not a sufficient contribution on its own; its value depends on whether it derives a measurable quantity and an effective intervention.
06. **57 / D — Paired input/output interfaces**: structurally sound but the contribution is mostly explanatory; best in service of an input/output vocabulary decoupling experiment.
07. **61 / C — AR, VQ, and attention as one structure**: an appealing unifying view, but "all are anchor-selection" risks being seen as too abstract; needs a shared law and cross-domain evidence.
08. **66 / C — A double-anchor system**: can repair the existing signature and explain the tied/untied weight difference; still needs a phenomenon only the double-anchor theory predicts.
09. **62 / C — Multi-quotient systems**: the theoretical space is real, but lacks a clear composition law and a decisive instance; better suited as a follow-on extension.
10. **78 / B — Dynamic address sets / signature families**: can lead into dynamic vocabularies, external memory, and online adaptation; biggest risk is overlap with dynamic vocabulary, memory tokens, and RAG.
11. **76 / B — The architecture-declared readout family Phi**: preregistration discipline and architecture-derivation give it identity; prior-work collision is severe, must show it knows more than calibration statistics.
12. **73 / B — Phi-visible commitment cost**: connects theory to task consequences; risk of being folded into existing task-aware distortion / V-information frameworks.
13. **81 / A — Commitment entropy times Phi-sensitivity**: a two-dimensional mechanism that a controlled intervention can pierce, and it unifies dynamic computation with selection cost; must avoid degenerating into a mere two-dimensional correlation plot.
14. **70 / B — Quotient collapse and selection collapse as a two-term cost**: the decomposition has explanatory power; must show the two terms can be independently manipulated and each predicts a different failure mode.
15. **72 / B — The zero-kernel theorem**: a load-bearing piece behind several empirical threads; independent novelty is squeezed by IPM theory, task-aware quantization, and KV-cache work.
16. **60 / C — Whether the anchor is a section**: a clean, testable classification condition; too narrow as a sole main contribution, better in support of a tokenizer paper.
17. **74 / B — The cost of a failed textual section**: directly patches the theory's most visible hole; if it produces an estimable formula predicting healing gains, it could be promoted to a main line.
18. **49 / D — Separating displacement from the commutator**: important theoretical hygiene, mainly averting a wrong unification, not enough on its own to draw main-track reviewers.
19. **58 / D — The prefix / concatenation two-tier split**: unifies two engineering phenomena, but much of it is close to an established structural fact; needs a new algorithmic result to be promotable.
20. **42 / X — A commutator for convex / smooth structure**: currently only a formal wish, with no natural definition, non-trivial theorem, or experimental object yet.
21. **77 / B — A training-process-aware theory**: accurately targets the static criterion's blind spot and can be verified with training-scale compute; must avoid expanding without bound into "absorb all of optimization into the theory."
22. **59 / D — A classification of training visibility**: classification alone is not enough; if it yields an optimization-behavior prediction it could support item 21.
23. **70 / B — Formalizing absent addresses**: registers, sinks, and null slots share a real common problem; the strongest attack is that these objects have genuinely different functions, so unification might stall at the naming layer.
24. **85 / A — Stepwise compensation along a commitment chain**: dynamic, intervenable, and trainable, a direct response to the strongest external-consult attack; 32xB200 can establish multi-scale causal evidence.
25. **73 / B — The no-commitment counterfactual experiment**: a good measurement device; the averaged embedding falling off the data manifold is a fatal confound, must be resolved with training-time intervention or a matched-manifold control.
26. **79 / B — Predicting downstream cost ordering from weights alone**: preregistration, falsifiability, and pre-data prediction are all strong; the claim degrades noticeably if it only achieves correlation or loses to the calibration arm.
27. **63 / C — A four-arm predictive comparison**: a solid experimental design, but it is evidentiary structure, not a paper idea by itself.
28. **78 / B — A cross-interface ordering law**: would be strong if the same derivation held across AR, VQ, and KV cache; any single interface type failing exposes it as "unification in name only."
29. **60 / C — The Qwen3 per-head norm prediction**: a clean, specific prediction, good for one key figure; too narrow to carry a paper alone.
30. **87 / A — Paired MLA-vs-GQA pretraining**: 32xB200 can turn an otherwise unattributable question in public models into a real causal experiment; needs to raise the contribution to a general mechanism rather than an architecture horse race.
31. **71 / B — A scaling law for commitment cost**: large compute makes it feasible, but the scaling-law space is crowded; needs a theoretical shape, a turning point, or a successful extrapolation.
32. **62 / C — TransMLA conversion as a control**: cheap and diagnostic, but the conversion is asymmetric and cannot substitute for paired pretraining.
33. **86 / A — Phi-aligned VQ**: the theory produces a genuinely new design with a clear zero-gain condition; the score hinges on going beyond existing task-aware quantization boundaries.
34. **69 / C — The zero-gain condition for Phi-aligned VQ**: an important guard against "it always improves things"; best as the core validation of item 33, not a standalone paper.
35. **64 / C — Sub-additivity of residual VQ**: tests chained commitment but has a narrow scope of impact; better folded into a dynamic-commitment paper.
36. **72 / B — A cross-modal VQ unification experiment**: strengthens a unifying claim; if it is only the same gain reproduced across several datasets, it risks being seen as breadth without depth.
37. **82 / A — Matched BPE / Unigram training**: controlled pretraining can fill a long-standing causal gap in tokenizer research; must control actual training FLOPs, sequence length, and data exposure.
38. **55 / D — A round-trip classification of tokenizers**: provides a new measurement table, but risks becoming a descriptive audit; needs to show the classification predicts model behavior.
39. **74 / B — Non-canonical mass and token healing**: a clear engineering phenomenon with an intervenable endpoint; must show it is not a proxy for length, byte encoding, or boundary opportunity.
40. **67 / C — Corpus-absent and undertrained tokens**: the glitch-token direction has value but the literature is already crowded; needs causal manipulation of token reachability.
41. **66 / C — Anchor alignment and norm hijacking**: a concrete, cheap mechanism; limited standalone impact, suited as a mechanism chapter in a vocabulary-training paper.
42. **72 / B — AR hidden-state-to-embedding drift**: connects to training objective and generation error; a naive distance metric will be confounded by residual norm growth, anisotropy, and representation basis change.
43. **65 / C — Structure/semantics decoupling of the attention sink**: an important question but already has several mechanistic papers; needs a training-time causal manipulation, not another probe.
44. **72 / B — Attention abstention mechanisms**: three implementations unified under one functional requirement, and it admits controlled training; must show the gain comes from abstention capacity, not extra parameters or an optimization side effect.
45. **30 / X — Sink is a necessary consequence of missing abstention**: existing records have already weakened the necessity claim, and continuing to load weight on it invites a counterexample.
46. **18 / X — Whether fine-tuning can repair the sink**: this question has already been actively cut from the project, and its standalone contribution is insufficient; should not be revived just because compute is now available.
47. **86 / A — A dynamic computation-boundary experiment**: matched compute, causal manipulation, and training-scale comparison all fit main-track taste; must cleanly separate router gain from boundary semantics.
48. **60 / C — A two-condition theory of caching**: useful conceptual clarification, but the claim is easily seen as a direct corollary of the causal mask.
49. **35 / X — Bidirectional models cannot cache incrementally**: too self-evident to need verification; the information gain of the experiment is low, suited only as a sanity check.
50. **80 / A — Independent expansion of input/output vocabularies**: allows a real architectural intervention, measuring coverage, generation quality, and compute economics; must avoid degenerating into a plain vocab-size sweep.
51. **67 / C — A replaceable non-parametric address set**: connects to continual learning and knowledge updating, but as currently stated is close to RAG's already-known selling point.
52. **48 / D — Fixed to adaptive to learned boundaries**: a trend summary, not a falsifiable contribution; background use only.
53. **83 / A — Content-aware dynamic tokenization**: suited to a closed mechanism-and-gain loop via large-scale paired training; the track is crowded, needs a genuinely new boundary signal or theoretical target.
54. **78 / B — An end-to-end tokenizer replacing the two-stage paradigm**: high impact and trainable, but strong prior work already exists (BLT / H-Net); needs a precise new mechanism rather than repeating the trend.
55. **50 / D — MeshLex / geometry-aware mesh tokenizer**: could stand as an independent 3D paper, but connects only weakly to this project's current LLM compute advantage and to the token-theory main line.
56. **59 / D — Cross-modal regularities in unified tokenizers**: grand in scope but easily becomes a survey or a loose collection; needs one hard law first.
57. **82 / A — Dynamic commitment cost**: the right direction, and it absorbs item 24; the formula is currently unvalidated and, used alone, invites the criticism of renaming recoverable error.
58. **89 / A — A temporary address pool**: the raw material with the strongest advantage in both new architecture and available compute; must be directly compared against memory tokens, registers, KV cache, latent recurrence, and external memory.
59. **54 / D — Self-feedback training unifying training and inference**: very close to scheduled sampling / DAgger / professor forcing; insufficient novelty unless combined with a new commitment objective.
60. **58 / D — Reversible distance**: the current definition's inverse map, KL direction, and claim to be "the one universal criterion" are all unstable; only the motivation of "measuring subsequent recoverability" survives.
61. **65 / C — A dynamic VQ codebook**: codeword restart, splitting, EMA, and adaptive codebooks already exist as prior work; should serve as an implementation variant of item 33, not a standalone claim.
62. **72 / B — A reinforcement-learning address allocator**: can exploit long-horizon task reward to learn resource allocation; RL is not a necessary ingredient, and if supervised or differentiable control works equally well, the contribution disappears.
63. **45 / D — Commitment entropy and safety alignment**: potentially high impact but the causal chain is long and easily degenerates into post-hoc correlation; currently lacks a distinctive intervention or safety mechanism.

### 2.3 The five strongest paper skeletons after combination

01. **94 / S — Elastic Address LLM** (items 10 + 23 + 50 + 58, with 62 as an optional controller): train an LLM that can create, merge, and reclaim latent addresses per sequence on the fly, tested under strictly matched FLOPs / KV budget on long context, code completion, and boundary recovery. **Primary rejection attack**: that this is just memory tokens, register tokens, or external memory under a new name; must show that dynamic address identity, lifecycle, and recoverable information are all three simultaneously necessary. Target venues: NeurIPS / ICLR.
02. **93 / S — Controlled Learned Tokenization** (items 13 + 17 + 37 + 39 + 47 + 50 + 53 + 54): train fixed BPE, Unigram, entropy-boundary, learned-router, and dynamic input/output vocabulary schemes under identical data order, training FLOPs, parameter count, and multiple seeds, answering "why does the boundary help" rather than merely comparing tokenizers. **Primary rejection attack**: that variables remain unpinned and the gain is just sequence length or compute reallocation; must run strict compute matching and boundary-swap interventions. Target venues: NeurIPS / ICLR / KDD.
03. **92 / S — Dynamic Commitment and Recovery** (items 13 + 14 + 24 + 25 + 35 + 42 + 57): define commitment cost as a trajectory quantity varying by layer, timestep, and later repair, and train models with different compensation depths to test whether, when, and at what cost early loss can be recovered. **Primary rejection attack**: that this is just ordinary error propagation or deep-network denoising; must construct a causal contrast between models with the same initial commitment but different compensation paths. Target venues: ICML / ICLR / NeurIPS.
04. **90 / S — Architecture-Causal Commitment Scaling** (items 11 + 12 + 15 + 26 + 27 + 28 + 30 + 31, with 32 as a supplement): use paired MLA/GQA pretraining across multiple scales and seeds to test whether a weight-level zero-kernel prediction can forecast real commitment cost, compared directly against calibration-level methods. **Primary rejection attack**: that this is just a generalization of an existing KV-cache task-aware metric, and that MLA/GQA differ along more than one axis; must report strict parameter/FLOPs matching and either success or failure with the same formula across AR, VQ, and cache interfaces. Target venues: ICML / NeurIPS / ICLR.
05. **88 / A — Readout-Aligned Discrete Representation** (items 11 + 12 + 15 + 33 + 34 + 35 + 36, with 61 as an ablation): derive a quantization metric from actual downstream readout, train Phi-aligned VQ across image, audio, and language discrete representations, and predict in advance when it yields zero gain. **Primary rejection attack**: that task-aware quantization already exists; must make "Phi derived from architecture," "the zero-kernel condition," and "the same prediction holding across modalities" jointly irreplaceable. Target venues: ICLR / ICML.

### 2.4 Triage verdict

- **Priority: keep as a main line**: items 13, 24, 30, 33, 37, 47, 50, 53, 57, 58.
- **Priority: load-bearing in combination**: items 10, 11, 12, 14, 15, 17, 21, 25, 26, 28, 31, 36, 39, 42, 44, 54, 62.
- **Theoretical or experimental support only**: items 3, 7, 8, 9, 16, 23, 27, 29, 32, 34, 35, 38, 40, 41, 43, 48, 51, 61.
- **Not worth dedicated main-line investment**: items 1, 2, 4, 5, 6, 18, 19, 22, 52, 55, 56, 59, 60, 63.
- **Currently eliminated**: items 20, 45, 46, 49.

This pass is a rough triage only. No full novelty search was carried out, so every A/S-tier direction still requires a 2024-2026 prior-work collision check before formal commitment; items 33, 47, 53, 54, and 58 have the highest priority for that search.
