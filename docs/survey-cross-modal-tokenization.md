# Cross-Modal Tokenization: How Text, Vision, and 3D Mesh Evolved

This survey traces how tokenization and patchification schemes evolved across three modalities — text, vision, and 3D mesh — and pulls out the regularities that recur across all three. It was written as background reading for the token theory developed in [The Antinomy of the Token](token-antinomy.md) and its companion notes; nothing here depends on that theory, and the theory does not depend on this survey being complete, but the two inform each other.

## 1. LLM tokenization: a history in three failures and four discoveries

### 1.1 From characters to subwords

The early history of NLP tokenization can be summarized as three failures. **Character level** makes sequences too long: an English sentence can run hundreds of characters, forcing the model to learn long-range dependencies over an unnecessarily long sequence at high compute cost. **Word level** runs into out-of-vocabulary tokens: English alone has hundreds of thousands of inflected forms, Chinese has no explicit word boundaries at all, and any fixed vocabulary eventually meets a word it has never seen. **Subword level** found the sweet spot. BPE (Sennrich et al., 2016) is almost embarrassingly simple: start from characters, repeatedly merge the most frequent adjacent pair in the corpus, until a target vocabulary size is reached. Frequent words stay whole ("the", "is"); rare words get split into meaningful pieces ("unhappiness" becomes "un" + "happiness").

Several variants followed without changing the core paradigm:

| Method | Year | Core difference | Representative models |
|---|---|---|---|
| BPE | 2016 | greedy merge by frequency | GPT-2/3/4, LLaMA |
| WordPiece | 2012/2018 | merge by mutual information, not frequency | BERT, DistilBERT |
| Unigram LM | 2018 | top-down pruning, not bottom-up merging | T5, ALBERT |
| SentencePiece | 2018 | language-agnostic framework, treats whitespace as an ordinary character | LLaMA, T5 |

GPT-2 introduced **byte-level BPE**: merging starts from the 256 UTF-8 bytes rather than characters, guaranteeing that any Unicode text can be encoded (eliminating out-of-vocabulary entirely) — now standard practice.

### 1.2 Four things that turned out to matter more than raw compression

Subword tokenization dominated NLP for nearly a decade before the field understood *why* it worked and how to do better. Several recent papers reveal deeper regularities.

**Compression ratio and downstream performance are not monotonically related.** Schmidt et al. (ACL 2024), "Tokenization Is More Than Compression," trained 64 language models (350M to 2.4B parameters) and built PATHPIECE — a tokenizer provably minimizing token count — specifically to test the assumption "more compression equals better performance." The relationship turned out to be inverted-U shaped, with a correlation coefficient of only 0.241; over-compression actively damages learnable structure. More importantly, pre-tokenization rules (preserving whitespace boundaries, digit boundaries) affect performance far more than compression ratio does, even when those rules reduce compression. A tokenizer is not merely a compressor — it is a **structure preserver**, and good tokenization has to trade off compression against preserving learnable structure.

**Optimal vocabulary size scales log-linearly with model size.** Tao et al. (NeurIPS 2024), "Scaling Laws with Vocabulary," found that the optimal vocabulary size N_v and the non-vocabulary parameter count N_nv satisfy N_v is proportional to N_nv to the power 0.83. This implies most current LLMs are undersized on vocabulary: LLaMA2-70B uses a 32K vocabulary against a predicted optimum near 216K, about seven times larger. In practice vocabularies have been growing fast — 32K for LLaMA 2, 128K for LLaMA 3, 262K for Gemini 3. More strikingly, Huang et al. (ICML 2025), "Over-Tokenized Transformer," found that **input and output vocabularies should be decoupled**: expanding the input vocabulary to 12.8M via n-gram embeddings lets a 400M model match a 1B baseline, a 2.5x effective size gain at nearly zero additional compute.

**Zipf's law is a signal of near-optimality.** He et al. (2025), "Pre-trained Models Perform Best When Token Distributions Follow Zipf's Law," found that as vocabulary size grows, the token-frequency distribution approaches Zipf's law (a straight line on a log-log plot), and downstream performance peaks exactly where Zipf alignment is highest (measured by the R-squared of a log-log fit). The regularity holds across NLP, genomics, and chemistry — a general principle giving a cheap vocabulary-selection criterion: check whether the token distribution is close to Zipf, without training a full model.

**An information-theoretic account of the tokenizer's role.** Erdogan et al. (Stanford, 2026) analyze tokenization as structured compression: a good tokenizer absorbs local statistical regularities into the token definitions themselves (raising unigram entropy H_1), freeing the language model to concentrate on modeling long-range dependencies (lowering conditional entropy at order 2 and above). They introduce a "capacity utilization" measure, unigram entropy divided by the log of vocabulary size, and show tokenizer-plus-LZ compression beats plain LZ compression by 10-20%, confirming tokenization's value as a preprocessing compressor.

### 1.3 The post-BPE era: three parallel tracks (2024-2026)

Subword tokenization's dominance is being challenged along three tracks at once.

**Learnable tokenization.** Dauncey and Wattenhofer (ETH Zurich, 2026), "You Can Learn Tokenization End-to-End with RL," model boundary selection as a stochastic policy trained with REINFORCE. Learned boundaries automatically align with semantic units — on code, the policy groups module names together, collapses whitespace runs, and skips boilerplate — at under 0.1% compute overhead. Rozental (2026)'s Zonkey goes further: a fully differentiable hierarchical diffusion language model with a learned segment splitter, from which word and sentence boundaries emerge with no inductive bias at all.

**Byte-level models.** Deng et al. (Amazon/Rice, 2026), ByteFlow Net, discard the tokenizer entirely and decide split points from the change in coding rate (information density) of a latent representation, using top-k selection to keep a static computation graph and stay hardware-friendly. The core insight: **boundaries should sit where information density changes, not where frequency statistics say to cut.** Zheng et al. (HKU, 2026), Proxy Compression, take a more pragmatic route: 90% of training uses compressed tokens for efficiency, 10% uses raw bytes for robustness, and inference runs entirely on bytes. Their key finding: **structured compression** (BPE, a neural compressor) transfers across representations, while **unstructured compression** (gzip) fails completely.

**Dynamic tokenization.** Feher et al. (Cambridge, ACL 2025) dynamically adjust BPE merge rules per batch at inference time, generating embeddings for new tokens on the fly via a hypernetwork. Sequence length drops 22-26% with only a 1.7-1.9% performance cost; a 350M model with a good dynamic tokenizer can match a 2.7B model saddled with a poor static one on translation.

**The core conclusion**: optimal tokenization is input-dependent and should be learned, not fixed. The question is no longer "which static tokenizer is best" but "how does tokenization become part of what the model learns to compute."

## 2. VLM patchification: a faster, more dramatic evolution

Visual tokenization evolved faster and more dramatically than text, because an image's spatial structure offers more exploitable prior structure than text's linear structure.

### 2.1 The fixed-grid era (2020-2021)

ViT (Dosovitskiy et al., 2020) transplanted NLP's tokenization logic directly onto images: cut an image into fixed 16x16 pixel patches, linearly project each into an embedding, and feed the sequence to a standard transformer. A 224x224 image becomes 196 tokens regardless of content — a blank white background and a busy street scene get exactly the same token count. Crude, but surprisingly effective. DeiT (Touvron et al., 2021) showed ViT could be trained well without JFT-scale data. BEiT (Bao et al., 2021) introduced masked image modeling using DALL-E's discrete VAE to produce masking targets — the first use of a **learned discrete visual vocabulary** as a ViT training signal. MAE (He et al., 2022) found that 75% of patches can be masked out and the model still reconstructs the image, revealing enormous redundancy in the fixed grid: most patches carry information inferable from their neighbors.

### 2.2 Discrete visual tokenizers (2021-2023)

In parallel, another line focused on building better visual tokenizers for generative models. VQ-VAE (van den Oord et al., 2017) is the starting point: encode an image into a discrete latent code via nearest-neighbor codebook lookup. VQ-VAE-2 (Razavi et al., 2019) introduced hierarchical (multi-scale) vector quantization — a top-level codebook capturing global structure, a bottom-level codebook capturing local detail — the first demonstration of multi-scale discrete visual tokenization. VQGAN / Taming Transformers (Esser et al., 2021) was a turning point, combining VQ-VAE with adversarial training (a PatchGAN discriminator) and a perceptual loss, substantially improving reconstruction quality and becoming the de facto standard tokenizer for autoregressive image generation.

Then a paper reshaped the field's understanding: Yu et al. (2023), "Language Model Beats Diffusion — Tokenizer is Key," introduced MagViT-v2 (lookup-free quantization, expanding the codebook to 2^18 entries) and showed that **given a sufficiently good tokenizer, language-model-style generation can match or beat diffusion models.** The core message — **tokenizer quality is the bottleneck on generation quality, not the generative architecture** — drove substantial subsequent investment in tokenizers, including GigaTok (Xiong et al., 2025), which scaled a visual tokenizer to 3 billion parameters.

### 2.3 Dynamic resolution (2023-2024)

Researchers began questioning the fixed-grid assumption itself. NaViT — "Patch n' Pack" (Dehghani et al., 2023, Google) — introduced sequence packing: patch sequences from images of different resolutions and aspect ratios are packed into the same batch, distinguished by an attention mask, removing the constraint that every image must be resized to a square. FlexiViT (Beyer et al., 2023, Google) trained a single ViT to run at multiple patch sizes (8x8, 16x16, 32x32) by randomizing patch size during training, letting inference trade compute against accuracy flexibly. Mixed-Resolution Tokenization (Ronen et al., 2023) went further still: **different regions of the same image use different patch sizes** — small patches (more tokens) for important regions, large patches (fewer tokens) for background — the first content-aware, variable-size segmentation within a single image.

### 2.4 Semantically aware tokenization (2024-2025) — the pivotal stage

A genuine paradigm shift happened here. EPOC (Chen et al., 2024, Meta/FAIR and HKUST) made the BPE analogy explicit: images should be segmented at the "subobject" level rather than into arbitrary fixed patches, just as BPE produces morphologically meaningful subwords, a visual tokenizer should produce semantically meaningful visual tokens. EPOC uses a lightweight 3.7M-parameter boundary detector plus watershed segmentation to produce semantically consistent tokens, reaching over 90% monosemanticity versus under 60% for fixed patches; VLMs trained on EPOC tokens converge faster, generalize better, and need fewer tokens.

dHT — Differentiable Hierarchical Visual Tokenization (Aasan et al., 2025, University of Oslo) — pushes this to the limit: a fully end-to-end differentiable tokenizer that builds a superpixel hierarchy and uses information criteria (AIC/BIC) to automatically select the optimal segmentation granularity per image, with **boundaries themselves learned via backpropagation**, and can be grafted directly onto a pretrained ViT. DART (Yin et al., 2025) scores image regions with a lightweight CNN and uses differentiable quantile splitting to produce variable-size patches — dense over important regions, coarse over background — gaining +1.6% accuracy on DeiT-Ti and cutting video-task FLOPs by 69%.

### 2.5 1D adaptive-length representations (2024-2026)

The most recent frontier abandons the 2D spatial grid assumption entirely. TiTok (Yu et al., 2024) demonstrated the counterintuitive fact that a single image can be reconstructed and generated from just 32 1D tokens, arranged with no spatial correspondence to any specific region — evidence that 2D spatial structure is not necessary in token space. ALIT — Adaptive Length Image Tokenization (Duggal et al., 2024, MIT CSAIL) — uses a recurrent encoder-decoder to iteratively distill 2D patch tokens into a variable number (32-256) of 1D latent tokens, with the count adapting to image complexity, familiarity, and downstream task. Its most striking finding: **tokens spontaneously bind to semantic objects/parts with no supervision at all** — individual tokens self-assign responsibility for specific semantic regions, echoed explicitly against LLM "thinking tokens." FlexTok (Apple, 2025) distills image features into a flexible-length 1D token sequence via cross-attention, letting a single model support different quality-compute tradeoffs. STAT — Soft Tail-dropping Adaptive Tokenizer (2026) — adaptively selects output token count by structural complexity, progressively dropping unimportant tokens via a tail-dropping mechanism.

### 2.6 Scaling laws and the pixel-level frontier (2025-2026)

"Scaling Laws in Patchification" (Wang et al., 2025, JHU/Berkeley) found a patchification scaling law: performance keeps improving as patch size shrinks, all the way to 1x1 (pixel-level tokenization), using a Mamba architecture (linear complexity) to handle 50,176 tokens per image. Three findings stand out: patch-size scaling is more cost-effective than parameter scaling — shrinking patches is a better deal than growing the model; at the pixel level, a complex decoder head becomes unnecessary; and the gains come from information gain, not merely from longer sequences. VAR — Visual AutoRegressive Modeling (Tian et al., 2024, PKU/ByteDance) — redefines autoregressive image generation itself: not next-token prediction but **next-scale prediction**, generating a coarse-to-fine sequence of multi-scale token maps, a genuinely different tokenization paradigm where generation order follows a resolution hierarchy rather than a raster scan.

### 2.7 Unified tokenizers and convergence (2025-2026)

The most recent trend unifies understanding and generation under one tokenizer: UniTok (2025) unifies visual generation and understanding under one tokenizer; AToken (2025) is the first unified visual tokenizer achieving both high-fidelity reconstruction and semantic understanding across images, video, and 3D; Cosmos Tokenizer (NVIDIA, 2024) is an industrial-grade image-and-video tokenizer; "Compression Tells Intelligence" (2026) is a theoretical paper connecting classical visual coding theory with MLLM tokenization, arguing compression efficiency and intelligence are correlated.

### 2.8 Summary of the VLM trajectory

| Stage | Period | Representatives | Core change |
|---|---|---|---|
| Fixed grid | 2020-21 | ViT, DeiT | fixed 16x16 patches |
| Discrete tokenizer | 2021-23 | VQGAN, MagViT-v2 | a learned discrete visual vocabulary |
| Dynamic resolution | 2023-24 | NaViT, FlexiViT | multi-scale, variable resolution |
| Semantically aware | 2024-25 | EPOC, dHT, DART | content-aware, differentiable boundaries |
| 1D adaptive | 2024-26 | TiTok, ALIT, FlexTok | breaking the 2D grid, variable-length sequences |
| Scaling law | 2025-26 | patchification scaling | pixel-level, information-theoretic grounding |

**Overall trend**: fixed to adaptive to learnable boundaries; 2D spatial grids to 1D latent sequences; uniform to content-aware to semantically aligned; a separate tokenizer stage to end-to-end joint learning.

## 3. Mesh tokenization: the youngest of the three, moving fastest

3D mesh tokenization is the youngest of the three modalities but has developed extremely quickly over the past two years.

### 3.1 Per-face / per-vertex methods (the "character level" of mesh)

This is the current mainstream paradigm, equivalent to character-level tokenization in text. MeshGPT (Siddiqui et al., 2023) pioneered this route: a VQ-VAE encodes face features, and a GPT-style decoder generates them autoregressively, with each face (three vertex coordinates, nine numbers) as one token. The problem is sequence length: an 800-face mesh needs roughly 800 tokens. Follow-up work optimizes within this same framework: MeshXL (2024) uses neural coordinate fields to scale to larger meshes; LLaMA-Mesh (2024) unifies mesh generation with an LLM; DeepMesh (2025) adds RL-enhanced autoregressive generation. Their shared bottleneck: **per-face tokenization's sequence length scales linearly with face count**, ruling out high-precision meshes.

### 3.2 Locality-aware methods

Nautilus (Wang et al., 2025) proposes a "nautilus shell" representation, organizing faces in shells around a center vertex. Comparative metrics:

| Metric | Nautilus | AMT | MeshGPT |
|---|---|---|---|
| Compression ratio | 0.275 | 0.462 | 1.0 |
| Locality ratio | 0.554 | 0.312 | 0.189 |
| Manifold ratio | 83.6% | 29.0% | 23.8% |

Nautilus's core finding: **preserving local dependency during tokenization, not just compressing, is critical for structural fidelity.** Attention maps show that under a locality-preserving tokenization, the transformer concentrates attention on neighboring tokens.

### 3.3 Topology-preserving methods

Mesh Silksong (Song et al., 2025) uses a vertex-layering method: each vertex visited exactly once (a 50% redundancy reduction), 4 tokens per vertex (2 coordinates, 1 intra-layer adjacency, 1 inter-layer adjacency), reaching a state-of-the-art 0.22 compression ratio while guaranteeing manifold topology, consistent normals, and watertightness detection, using a half-edge data structure for deterministic traversal.

### 3.4 Patch-level methods

MeshMosaic (Xu et al., 2025) segments a mesh into semantic patches using PartField at inference time or random Voronoi partitioning at training time, generates each patch autoregressively conditioned on the previous patch's boundary, encodes boundary conditions for the first 512 triangles via a GRU, quantizes locally per patch (rather than globally) using a 512-cubed grid, and scales to 100K-plus triangles, built on a 0.5B DeepMesh base. Its patches are unique per mesh, generated sequentially with boundary conditioning passed forward — closer to writing an essay paragraph by paragraph than to assembling from a shared, reusable vocabulary of parts.

### 3.5 Semantic-level methods

LoST — Level of Semantics Tokenization (Dutt et al., 2026) — orders tokens by semantic salience, so 1-4 tokens already decode into a recognizable shape, using RIDA (Relational Inter-Distance Alignment) for 3D semantic guidance; it reaches the effect of geometric level-of-detail methods with only 0.1-10% of the tokens. It operates on triplane latents rather than directly on the mesh.

### 3.6 Other notable methods

| Method | Core idea | Feature |
|---|---|---|
| TreeMeshGPT (2025) | tree-structured autoregression | hierarchical generation |
| FastMesh (2025) | component decoupling | efficiency optimization |
| ARMesh (2025) | next-level-of-detail prediction | multi-scale, similar to VAR |
| MeshRipple (2025) | sliding-window structured AR | locality plus efficiency |
| PrimitiveAnything (2025) | decomposition into geometric primitives | primitive-level tokenization |

### 3.7 A cross-modal mapping of mesh tokenization stages

Comparing mesh against text and vision, the progression maps roughly as follows:

| Text stage | Vision stage | Mesh stage | Representative |
|---|---|---|---|
| Character level | Pixel level | Per-face/vertex | MeshGPT, DeepMesh |
| Word level | Fixed patch | Fixed-size patch | early patch-based schemes |
| BPE subword | Semantic patch | Semantic / adaptive patch | MeshMosaic and successors |
| Learnable token | Differentiable segmentation | Learnable segmentation | not yet realized |

Mesh tokenization currently sits mostly at the fixed-patch stage — the mesh analogue of ViT's 16x16 grid — with the adaptive and learnable-boundary stages still largely open.

## 4. Classical mesh segmentation: decades of relevant methodology

Mesh segmentation is a classical computational-geometry problem with decades of accumulated methodology. These methods were not designed for tokenization, but they carry a deep understanding of "how to cut a mesh into good pieces."

**Variational Shape Approximation (VSA)** (Cohen-Steiner, Alliez and Desbrun, SIGGRAPH 2004) iteratively clusters faces to minimize the approximation error between the original surface and a piecewise-planar proxy — essentially k-means on face normals plus a geometric distortion measure. It directly optimizes reconstruction error (the same objective a VQ-VAE optimizes), producing near-planar patches: large patches over flat regions, small patches over high-curvature regions. Patch count is a free parameter converging via Lloyd iteration at O(kN) per iteration. Its limitation: planar proxies need many small patches to approximate curved features like cylinders or spheres.

**Curvature-driven segmentation** cuts along high-curvature edges, producing patches with low internal curvature variance. A modern unified framework, the Reeb-graph method (Beguet et al., 2024), uses a curvature-derived scalar (Shape Index) as input to a Reeb graph capturing the topological skeleton (minima, maxima, saddle points) of that scalar function, integrating both geometric and topological features at O(n log n) complexity. Curvature segmentation produces boundaries along natural geometric feature lines, but patch count is hard to control and flat regions can produce oversized patches.

**Spectral methods** use eigenvectors of the Laplace-Beltrami operator (LBO) to define a feature space for clustering. GeoTransformer (Farazi and Wang, 2024) directly compares two approaches: **RNS (Root-Node Selection)**, based on the algebraic multigrid literature, using k-medoid clustering on an anisotropic-geodesic distance matrix derived from the LBO, which **preserves the LBO spectrum** — the resulting partition retains the original eigenfunction structure, with patch size varying by underlying geometry (more patches over high curvature); versus **METIS**, a standard graph-partitioning algorithm that efficiently produces balanced patches but is **geometry-blind**, failing to approximate heat-kernel-signature and LBO eigenfunctions. The paper's conclusion: RNS substantially outperforms METIS at preserving geometric properties, a caution against defaulting to a partitioning algorithm chosen purely for balance.

**SDF segmentation.** The Shape Diameter Function measures local thickness by casting inward rays from each face; faces with similar SDF values are clustered, then refined by graph cuts. Neural ShDF (Roy, 2023) predicts SDF values via a GNN, a 10x speedup, reaching 97.1% on COSEG (close to MeshCNN's 97.3%) and 94.2% on Human Body (exceeding MeshCNN's 92.3%), with resolution-independence via downsampling plus full-resolution neighborhood queries. Segment Any Mesh (Tang et al., 2024) renders an SDF scalar as a 2D image and feeds it to SAM2 for zero-shot mesh segmentation, with SDF plus surface normals as multimodal input performing best. SDF segmentation produces semantically meaningful parts (arms, legs, handles) rather than geometric patches — sizes vary enormously and this is not directly suited to codebook learning.

**Convex-decomposition feature fields.** "Learning Convex Decomposition via Feature Fields" (Yang et al., 2026) reformulates convex decomposition as contrastive feature learning, with a self-supervised geometric loss based on the definition of convexity (a segment between two points inside a convex piece stays inside the piece's volume), a feed-forward model trained on 340K Objaverse shapes, outperforming V-HACD and CoACD, with multi-granularity control via a clustering threshold and 18-second inference (5s features, 13s clustering). Its key finding: **semantic segmentation features (from PartField) produce poor convex decompositions** — convexity-aware features and semantic features are fundamentally different things, a caution against assuming one notion of "good part" transfers to another.

**MeshCNN's edge features.** MeshCNN (Hanocka et al., 2019) operates on mesh edges, each carrying a 5D feature (dihedral angle, two interior angles, two edge-length ratios). Its pooling collapses edges with the smallest feature first, naturally simplifying the mesh and creating a multi-resolution hierarchy that preserves the most important geometric features at each level. The dihedral-angle feature in particular is a natural signal for patch-boundary detection: high-dihedral-angle edges are natural cut locations.

**Quad meshing.** "Learning Quadrangulated Patches" (Groueix et al., 2017/2019) learns a dictionary of surface patches, each a local parameterization of the surface, defined by a quadrilateral mesh structure aligned to principal curvature directions. This "learned patch dictionary" concept is close in spirit to codebook-based mesh tokenization, using quadrilateral rather than triangular patches, aligned along curvature to produce a geometrically natural decomposition.

## 5. Cross-modal regularities

Looking at all three modalities together, several clear regularities emerge.

**Regularity one: fixed to adaptive to learnable boundaries.** This is the clearest direction of travel across all three modalities — text moves from characters/words through frequency-driven BPE to RL-based and information-density-driven tokenization; vision moves from fixed 16x16 patches through FlexiViT and mixed-resolution schemes to fully differentiable, content-aware boundaries; mesh has largely only reached the fixed-partition stage, with the adaptive and learnable stages still open.

**Regularity two: Zipf's law is a general signal of near-optimality.** In text, token distributions closest to Zipf's law correlate with the best downstream performance (He et al., 2025). This suggests a general diagnostic: check how far a scheme's usage-frequency distribution sits from Zipf's law as a cheap proxy for whether the underlying tokenization is well-tuned, without training a downstream model to find out.

**Regularity three: tokenizer quality is the bottleneck for the whole system.** In text, vocabulary choice matters more than model size (the Scaling Laws with Vocabulary result); in vision, tokenizer quality sets the ceiling on generation quality regardless of generator architecture ("Language Model Beats Diffusion — Tokenizer is Key"). The same logic plausibly extends to any domain building an autoregressive or diffusion model over a learned discrete representation: improving the tokenizer can pay off more than improving the downstream generative model.

**Regularity four: content-aware segmentation strongly beats uniform segmentation.** In vision, EPOC's semantic tokens reach over 90% monosemanticity against under 60% for fixed patches. The general shape of the argument: uniform partitioning treats regions of very different structural complexity as equivalent, wasting capacity on simple regions and under-serving complex ones. A natural analogy carries across domains: simple, common structure should map to high-frequency vocabulary entries (like common characters), and complex, rare structure should be expressible as a composition of several simple entries (like rare characters built from common radicals) rather than requiring its own dedicated, rarely-used entry.

**Regularity five: the two-stage paradigm (segment, then quantize) is being challenged everywhere.** In text, ByteFlow Net removes the tokenizer entirely, deriving splits from information density; in vision, dHT learns segmentation boundaries end-to-end and differentiably. Both point toward joint optimization in any domain still using a separate segmentation stage ahead of quantization: boundaries determined by backpropagating the reconstruction loss through the whole pipeline, rather than fixed in advance by a separate algorithm.

**Regularity six: compression is not the only objective.** Schmidt et al. (2024) showed the inverted-U relationship between compression and performance in text; Nautilus (2025) found that preserving local dependency matters more than compression ratio in mesh tokenization. The general lesson: minimizing token count is the wrong sole objective; how well-suited the resulting token sequence is to the downstream autoregressive or attention-based model's own inductive biases matters independently.

## References

**LLM tokenization.** Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE), ACL 2016, arXiv:1508.07909. Kudo, "Subword Regularization" (Unigram LM), ACL 2018, arXiv:1804.10959. Kudo and Richardson, "SentencePiece," EMNLP 2018, arXiv:1808.06226. Mielke et al., "Between Words and Characters," TACL 2021, arXiv:2112.10508. Schmidt et al., "Tokenization Is More Than Compression," ACL 2024, arXiv:2402.18376. Tao et al., "Scaling Laws with Vocabulary," NeurIPS 2024, arXiv:2407.13623. Feher et al., "Retrofitting LLMs with Dynamic Tokenization," ACL 2025, arXiv:2411.18553. Huang et al., "Over-Tokenized Transformer," ICML 2025, arXiv:2501.16975. He et al., "Zipf's Law for Optimal Vocabulary," 2025, arXiv:2507.22543. Erdogan et al., "Information-Theoretic Perspective on Tokenizers," 2026, arXiv:2601.09039. Dauncey and Wattenhofer, "RL-based Tokenization," 2026, arXiv:2602.13940. Rozental, "Zonkey," 2026, arXiv:2601.21768. Zheng et al., "Proxy Compression," 2026, arXiv:2602.04289. Deng et al., "ByteFlow Net," 2026, arXiv:2603.03583. Lotz et al., "Beyond Text Compression," ACL 2025, arXiv:2506.03101.

**VLM patchification.** Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), ICLR 2021, arXiv:2010.11929. Touvron et al., "DeiT," ICML 2021, arXiv:2012.12877. Bao et al., "BEiT," ICLR 2022, arXiv:2106.08254. He et al., "MAE," CVPR 2022, arXiv:2111.06377. van den Oord et al., "VQ-VAE," NeurIPS 2017, arXiv:1711.00937. Esser et al., "Taming Transformers" (VQGAN), CVPR 2021, arXiv:2012.09841. Yu et al., "Language Model Beats Diffusion — Tokenizer is Key" (MagViT-v2), ICLR 2024, arXiv:2310.05737. Dehghani et al., "Patch n' Pack" (NaViT), 2023, arXiv:2307.06304. Beyer et al., "FlexiViT," 2023, arXiv:2212.08013. Chen et al., "Subobject-level Image Tokenization" (EPOC), 2024. Aasan et al., "Differentiable Hierarchical Visual Tokenization" (dHT), 2025. Yin et al., "DART," 2025. Yu et al., "TiTok," 2024. Duggal et al., "ALIT," 2024. Wang et al., "Scaling Laws in Patchification," 2025. Tian et al., "VAR," 2024. Bolya et al., "Token Merging" (ToMe), ICLR 2023, arXiv:2210.09461. Xiong et al., "GigaTok," 2025.

**Mesh tokenization.** Siddiqui et al., "MeshGPT," 2023, arXiv:2311.15475. Wang et al., "Nautilus," 2025, arXiv:2501.14317. Song et al., "Mesh Silksong," 2025, arXiv:2507.02477. Xu et al., "MeshMosaic," 2025, arXiv:2509.19995. Dutt et al., "LoST," 2026, arXiv:2603.17995.

**Mesh segmentation.** Cohen-Steiner et al., "Variational Shape Approximation" (VSA), SIGGRAPH 2004. Hanocka et al., "MeshCNN," SIGGRAPH 2019, arXiv:1809.05910. Farazi and Wang, "GeoTransformer," 2024, arXiv:2411.00164. Roy, "Neural ShDF," 2023, arXiv:2306.11737. Tang et al., "Segment Any Mesh," 2024, arXiv:2408.13679. Yang et al., "Convex Decomposition via Feature Fields," 2026, arXiv:2603.09285. Beguet et al., "Reeb Graph Segmentation," 2024, arXiv:2412.05335. Groueix et al., "Learning Quadrangulated Patches," 2017/2019, arXiv:1709.06868.
