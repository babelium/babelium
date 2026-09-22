# Commitment and Uncertainty as Boundary Drivers: A Synthesis Across Seven Papers

This note synthesizes the seven papers covered individually in [Cross-Modal Tokenization](survey-cross-modal-tokenization.md)'s companion surveys — specifically [Dynamic Boundaries](survey-dynamic-boundaries.md) (BLT, H-Net), [Attention Sinks](survey-attention-sinks.md) (Why First Token, the P0-Sink Circuit, Massive Activations, and the diffusion causal analysis), and [Structural Slots](survey-structural-slots.md) (Catch-Tag-Release, ViT Registers) — into one evidence chain, addressing a single question directly relevant to the commitment framework in the token theory's companion notes: what actually determines where a transformer decides to concentrate scarce computation, representation, or attention mass, and how much of that determination is genuinely about semantic content or predictive uncertainty, as opposed to something more purely structural.

## Conclusion

These seven papers, together, support a **layered** rather than single-factor account. Transformers concentrate limited computation and representational capacity onto a small number of boundaries, positions, or dedicated slots, but the trigger signal falls into at least four distinct categories, and none of the four reduces cleanly to "semantics" or "predictive entropy" alone.

**One.** Predictive uncertainty can directly drive a computation boundary. BLT uses next-byte entropy from a separately trained model to decide patch boundaries — the most direct evidence available for this specific claim.

**Two.** A boundary can also be learned end-to-end from the task loss itself. H-Net learns dynamic chunking from the change between adjacent hidden-state projections, but what it measures is representational *dissimilarity*, not calibrated predictive uncertainty — a genuinely different quantity from BLT's entropy, even though both are called "boundary signals" in casual usage.

**Three.** A fixed sink can arise from purely structural causes with no semantic input at all. The P0-Sink Circuit, "Why do LLMs attend to the first token?", and the massive-activations paper together show that position zero's specialness originates first from the causal mask's forced self-attention, from continuous visibility, from normalization choice, and from the training-time context-length distribution — not from any property of the specific token occupying that position.

**Four.** Structural formation and semantic payload must be kept as two separate questions. Catch-Tag-Release and ViT Registers both show that a structurally produced slot can subsequently come to carry truth, sentiment, or global image information — real, decodable content — even though nothing about how the slot got selected in the first place required that content to be there.

**Five.** Without autoregression, sink-like concentration still appears, but its role changes. In diffusion transformers, the analogous phenomenon is dynamic, does not sit at a fixed position, and functions as a trajectory-routing point rather than a persistent anchor; removing it at modest budget noticeably changes the concrete visual realization of a generated image while leaving the semantic-alignment metrics tested essentially untouched.

The most defensible general statement these seven papers jointly support is:

**Models form boundaries or sinks at positions where new expensive computation is needed, where ineffective mixing must be blocked, or where global state needs a temporary place to live. Predictive uncertainty is one clearly usable trigger signal among these, but not the only one; semantics can influence or subsequently occupy these slots, but is not a necessary condition for any given slot's formation.**

**Commitment, in the sense this project uses the term, remains an explanatory construct that none of these seven papers directly measures.** BLT's next-byte entropy is the closest available operational proxy: low entropy roughly means the local continuation is already effectively settled, and high entropy roughly means additional expensive computation is warranted — but "roughly means" is doing real work in that sentence, and the gap between entropy and commitment cost is exactly the gap the companion theory notes (see [Phi and the Displacement Criterion](theory-criterion.md), section 5.2) formalize by separating entropy from Phi-sensitivity as two independent axes.

## The evidence, paper by paper

### 1. BLT: predictive entropy directly controls the computation boundary

BLT's boundary signal is the next-byte entropy of a separately trained, independent causal byte-level language model. A new patch opens either at a position of globally high entropy or at a position where entropy rises sufficiently relative to the previous position; predictable stretches are merged into longer patches, and the expensive latent transformer runs more densely at difficult positions.

This directly supports "predictive uncertainty drives computational resolution," but the scope of that support needs to be stated narrowly. The router is trained separately from the main model, not formed end-to-end with it. Its threshold is calibrated to hit a target average patch size on the training mixture, and can be, and in the paper's own main downstream comparison actually is, adjusted again at inference time. Repetition produces "entropy drift," forcing the authors to add corrections such as resetting the entropy context at newlines. Simple whitespace-based patching remains a closely competitive baseline. And the paper offers no evidence that its entropy boundary is equivalent to a semantic, reasoning-step, or cognitive-commitment boundary.

**Strength of conclusion: strongly supports "uncertainty is an effective proxy for allocating computation"; does not support "entropy reveals the one true semantic boundary."**

See [Dynamic Boundaries](survey-dynamic-boundaries.md), section 1, for the full reading notes.

### 2. H-Net: end-to-end boundaries come from representational change, not explicit entropy

H-Net uses the cosine dissimilarity between learned projections of the current and previous causal hidden states as a boundary score, hard-thresholded to select which positions pass into a deeper level of computation. The autoregressive prediction loss determines whether a given boundary is useful; a separate ratio loss constrains only the average compression rate.

This fills BLT's key gap: the boundary-forming mechanism and the main model are trained jointly, and the mechanism can be recursively nested into a multi-level hierarchy. On English text, the first level's learned boundaries often approximate word boundaries, and a second level occasionally forms phrase-like groupings; gains on Chinese, code, and DNA sequences show the mechanism is not simply rediscovering a whitespace rule under a different name.

But H-Net does not establish that its router score *is* predictive uncertainty. The score is a representational-dissimilarity quantity, and the paper explicitly states it is not used as a formal probability. The total computation budget is still externally specified through the ratio-loss target. The evidence for semantic boundary formation is mainly qualitative visualization, with no annotated boundary dataset and no intervention that independently varies semantic content while holding surface statistics fixed (or the reverse). And its performance comparisons against baselines are entangled with several other architectural choices at once — the choice of Mamba layers, normalization, width, depth, and parameter allocation.

**Strength of conclusion: strongly supports "the task loss can learn a useful, content-dependent boundary"; only weakly supports "these boundaries correspond to semantic shifts"; does not support "they are predictive entropy."**

See [Dynamic Boundaries](survey-dynamic-boundaries.md), section 2, for the full reading notes.

### 3. The P0-Sink Circuit: a fixed first-position sink can form from pure causal structure

The P0-Sink Circuit gives the most specific structural mechanism among the seven papers: the causal mask forces position zero to attend only to itself, giving it a permanent attention weight of exactly one on its own value, so its attention output is never mixed with any other position's content and carries different norm and variance statistics as a direct consequence. A subsequent MLP sublayer detects and amplifies this asymmetry, pushing position zero toward a stable, high-norm direction across different inputs, which downstream layers then use as a sink.

Removing the BOS token, LLaMA's sink reconstitutes within a shallow number of layers regardless — direct evidence that a fixed token's specific learned semantics is not a necessary condition for the mechanism to operate. Mistral's sliding-window attention, which breaks position zero's continuous visibility, is the mechanism's own boundary case: removing BOS there collapses the sink almost entirely, because that architecture never develops the purely positional version of the mechanism and instead depends on BOS's specific embedding alone — a genuine limiting condition, not a universal law across every architecture.

The paper's two proposed training interventions (TNLM loss and Force-sink masking) show that deliberately accelerating or hard-wiring this structural formation can improve results at an earlier training scale, but the performance evidence here is markedly weaker than the mechanism evidence: most experiments use a single random seed, both interventions simultaneously change either the training loss or the representational subspace (a genuine confound for any causal claim about "faster sink formation causes better performance"), and at a larger, more data-saturated training scale, not every variant of either method actually beats its baseline.

**Strength of conclusion: strongly supports "P0 sink formation requires no semantics"; only moderately supports "forming the sink earlier has training value."**

See [Attention Sinks](survey-attention-sinks.md), section 2, for the full reading notes.

### 4. Why First Token and Massive Activations: the structural sink's function is controlling mixing and routing

"Why do LLMs attend to the first token?" interprets the first-position sink as an approximate no-op: a head directs attention toward a low-value-norm first token specifically to minimize its own residual update, slowing the rate at which perturbations mix across a long, deep context — a phenomenon called over-mixing. Holding total training tokens fixed, sink strength grows systematically with context length; even absent a fixed BOS token, a sink still forms at whichever token happens to occupy the first position.

The massive-activations paper further **decouples** massive activation magnitude from attention sink strength, a genuinely separate finding from the sink-formation account above: changing the normalization scheme can remove the large activation spike almost entirely while leaving sink strength essentially unchanged; head dimension systematically controls sink ratio; a representation-conditioned gate can substitute for much of the sink's function; and restricting the training loss to only compute over long positions collapses sink ratio nearly to zero while perplexity stays roughly comparable.

Together, the two papers indicate that the first-position sink functions more as a piece of structural routing infrastructure a decoder-only model builds for managing local dependency, filtering out unhelpful long-range context, and implementing conditional no-ops, than as anything resembling a semantic boundary. The limitation worth preserving across both: directly removing BOS at inference time produces an input distribution the model never saw during training, so any resulting performance drop cannot be cleanly attributed to loss of the sink mechanism alone, as opposed to simple distribution shift.

See [Attention Sinks](survey-attention-sinks.md), sections 1 and 3, for the full reading notes.

### 5. Catch-Tag-Release: a structural slot can carry a written semantic tag

Catch-Tag-Release supplies a conclusion operating at a different level entirely. A prompt-dependent sink's value vector gets copied into every token that attends to it, forming a shared, low-dimensional "tag"; later layers can then use that tag to select, retrieve, or aggregate a particular group of tokens. A linear probe reads truth-value and sentiment out of that tag direction at high accuracy across several models, and a hand-constructed toy task demonstrates that a catch-tag-release strategy, using a `[SEP]`-style marker as a dynamic boundary, is at least sufficient to solve a simple sequence-averaging problem, with training experiments showing the same strategy can also emerge from gradient descent (though only in a minority of training runs).

This shows that "the sink token's own surface form carries no semantics" does not imply "the sink mechanism does nothing semantic." The more precise decomposition: a sink's *formation condition* can be purely positional, punctuation-based, or structurally boundary-based; the sink's *value vector, once formed*, can carry task-relevant semantic content; and these are two separate questions that should not be collapsed into one. The evidence in real language models remains, at this stage, mostly PCA-based, variance-decomposition-based, and probing-based; no experiment in this paper removes the tag in a real task and demonstrates the resulting behavioral change is causally necessary, and the theoretical toy construction establishes sufficiency, explicitly not necessity.

**Strength of conclusion: strongly supports "a sink can carry semantics"; only weakly supports "this semantic mechanism is indispensable to real-world reasoning."**

See [Structural Slots](survey-structural-slots.md), section 1, for the full reading notes.

### 6. ViT Registers: an internal computational slot does not require language or autoregression

Large vision transformers select a small number of low-information background patches, discard their local positional and pixel-level identity, and repurpose the resulting slot as a scratchpad for global image information. Adding explicit register tokens — carrying no input information and producing no task-relevant output — causes the high-norm behavior to move entirely into the registers, and the original patch-level artifacts disappear.

This is the critical counterexample to any claim that all register-like or sink-like behavior traces back to a causal, first-position privilege. The same underlying need — repurposing a token slot as an internal computation resource when no dedicated one is offered — appears under fully bidirectional visual attention, with no causal mask involved at all. It supports a more general resource-allocation principle: when a model needs a global workspace and the architecture provides no dedicated space for one, it will requisition whichever input position is cheapest to sacrifice.

This phenomenon is not measured on the same quantity as an LLM attention sink and should not be assumed to share a literal circuit with it; but it is sufficient to establish that register-like functionality does not depend on autoregression.

**Strength of conclusion: strongly supports "an internal scratchpad or register function exists across modalities, without requiring autoregression."**

See [Structural Slots](survey-structural-slots.md), section 2, for the full reading notes.

### 7. Diffusion Sinks: without autoregression, the sink shifts from a fixed anchor to a dynamic trajectory router

Diffusion transformers still show attention concentration, but of a different character from causal language models: it moves across layers and denoising timesteps, essentially never sits at index zero, and lands mainly on a small number of text-conditioning keys rather than a fixed structural position.

The authors run paired, training-free suppression on both the attention-score path and the value path. Standard top-1 suppression drives sink attention mass to near zero without reducing CLIP-T, ImageReward, or HPS-v2, the semantic-alignment and preference proxies tested — yet the generated output's layout, color, viewpoint, and style change substantially, by considerably more than an equal-budget random masking control produces. Under stronger, multi-position suppression, HPS-v2 begins to show a dose-dependent decline while CLIP-T remains comparatively robust.

So the sink in diffusion is not functionless — its role shifts from autoregression's persistent memory-and-no-op anchor to a replaceable, phase-dependent trajectory router. High incoming attention mass and necessity for coarse semantic alignment become genuinely decoupled quantities.

**Strength of conclusion: strongly supports "the sink phenomenon does not require autoregression"; strongly supports "the sink's function in autoregressive versus diffusion models differs"; does not support "a diffusion sink can be removed at any strength with no cost."**

See [Attention Sinks](survey-attention-sinks.md), section 4, for the full reading notes.

## A unified model: two independent axes

These seven results can be organized along two axes that are logically independent of each other: how a slot or boundary gets selected in the first place, and what it does once selected.

| Paper | How the slot / boundary is selected | What it does once selected |
|---|---|---|
| BLT | high or rising next-byte entropy | admits more expensive latent computation |
| H-Net | adjacent causal representational dissimilarity, plus task gradient | passes into a deeper level of computation |
| P0-Sink | position-zero asymmetry under the causal mask | establishes a stable sink / register |
| Why First Token | continuous first-position visibility, long-context mixing pressure | approximates a no-op, suppressing over-mixing |
| Massive Activations | pre-norm architecture, head capacity, short-context training | provides local routing, ignores unhelpful long-range context |
| Catch-Tag-Release | a prompt-dependent sink or structural boundary | writes, and lets later layers read, a semantic tag |
| ViT Registers | low local information, a sacrificial background patch | stores global image information |
| Diffusion Sinks | dynamic, timestep- and layer-dependent incoming mass | controls the concrete realization of the generative trajectory |

**The single most important distinction across all seven: selection mechanism is not the same thing as carried content.** A slot can form for purely structural reasons and only subsequently come to carry semantic content; or it can be selected by genuine predictive uncertainty and function purely as a computation-budget controller with no particular semantic payload of its own. Treating these as the same phenomenon simultaneously overstates the "semantic boundary" evidence and understates the purely structural mechanisms these seven papers document with considerably more rigor.

## What the current evidence does and does not support

**Supported.** Next-step predictive entropy is an effective signal for dynamic computation allocation. End-to-end training can learn dynamic chunking that transfers usefully across languages and modalities. A causal P0 sink's formation does not require a fixed BOS token's specific semantics. Register-like or sink-like slots can carry task-relevant semantic or global information. Register-like internal workspaces do not depend on autoregression. A sink's function changes across causal autoregressive decoding, bidirectional vision transformers, and diffusion inference.

**Not supported.** That every attention sink is a predictive-uncertainty or commitment boundary. That BLT's entropy boundary is equivalent to a semantic, word-level, reasoning-step, or cognitive-commitment boundary. That H-Net's router score is a calibrated uncertainty probability. That the P0 sink carries no decodable semantics whatsoever — the papers surveyed establish only that semantics is not necessary for its *formation*, which is a narrower claim. That Catch-Tag-Release has demonstrated real LLM inference *depends on* its tag mechanism, as opposed to merely permitting it to be read out. That a diffusion sink has no function, or can be removed at unlimited strength without cost. That patch boundaries, attention sinks, massive activations, and visual registers are instances of one single underlying circuit — the evidence instead supports several related but genuinely distinct mechanisms that happen to share a family resemblance.

## The experiment most needed to advance this further

To move this evidence chain toward a strong, unified "commitment drives the boundary" conclusion, the smallest experiment that would actually matter is this: within one end-to-end model, jointly record next-step entropy, representational change, the boundary decision actually taken, and the resulting downstream computational benefit — and separately, deliberately manipulate semantic content and surface-level predictability as two independent interventions, rather than letting them vary together as they do in every paper surveyed here. Compare entropy-based, learned-router, random, whitespace, and hand-annotated semantic boundaries under a strictly matched real computation budget, and test which boundary type best predicts the marginal benefit of investing one additional step of expensive, higher-level computation.

This would directly separate three explanations this evidence chain currently leaves entangled: predictive uncertainty, a genuine change in semantic state, and a purely structural need for a working-memory slot that has nothing to do with either.

## References

See the reference lists in [Dynamic Boundaries](survey-dynamic-boundaries.md), [Attention Sinks](survey-attention-sinks.md), and [Structural Slots](survey-structural-slots.md) for full citations to all seven papers synthesized here.
