# 跨模态 Tokenization 进化综述：从 LLM 到 VLM 到 Mesh

> **目的**：系统梳理文本、视觉、3D mesh 三个模态中 tokenization / patchification 方案的完整进化脉络，提炼跨模态的通用规律。
>
> **写作时间**：2026-04-24
>
> **阅读前提**：了解 BPE、VQ-VAE、ViT 的基本概念即可。

---

## 目录

1. [引言：为什么要做跨模态对比](#1-引言)
2. [LLM Tokenization 进化史](#2-llm-tokenization-进化史)
3. [VLM Patchification 进化史](#3-vlm-patchification-进化史)
4. [Mesh Tokenization 现状与竞品](#4-mesh-tokenization-现状与竞品)
5. [Mesh Segmentation 经典算法](#5-mesh-segmentation-经典算法)
6. [跨模态规律提炼](#6-跨模态规律提炼)
7. [参考文献](#7-参考文献)

---

## 1. 引言

MeshLex 的三阶段 pipeline — Patchification → Quantization → AR Reconstruction — 本质上是一个 tokenization 问题：怎么把连续的 3D mesh 切成离散的 token，让下游模型能高效地学习和生成。

这个问题在文本和视觉领域已经被研究了很多年，积累了大量经验教训。文本从字符级走到 BPE 再到可学习 tokenization，视觉从固定 16×16 patch 走到语义感知的自适应分割 — 每一步进化都伴随着显著的性能提升。

但 3D mesh 领域的 tokenization 研究才刚刚起步。大多数方法（MeshGPT、MeshXL、DeepMesh）还停留在 per-face 或 per-vertex 的"字符级"阶段，MeshLex 的 per-patch 方案已经是一个跳跃，但我们的 METIS 分割方案 — 按固定 face 数量均匀切割 — 本质上相当于 ViT 的固定 16×16 patch，是这条进化路线上的早期阶段。

本文的目标是：通过系统对比三个模态的 tokenization 进化史，找到通用的进化规律，然后把这些规律"翻译"到 mesh 领域，为 MeshLex 的下一代 patchification 方案提供方向。

---

## 2. LLM Tokenization 进化史

### 2.1 从字符到子词：基础范式的确立

NLP tokenization 的早期历史可以用三个失败来概括：

**字符级**的问题是序列太长。一个英文句子可能有几百个字符，模型需要在这么长的序列上建模长程依赖，计算成本高且效果差。

**词级**的问题是 OOV（Out-of-Vocabulary）。英语有几十万个词形变化，中文更是没有明确的词边界。任何固定词表都会遇到未登录词。

**子词级**找到了甜蜜点。BPE（Sennrich et al., 2016）的核心思想极其简单：从字符开始，反复合并语料中最高频的相邻 pair，直到达到目标词汇量。这样高频词保持完整（"the"、"is"），低频词被拆成有意义的子词（"unhappiness" → "un" + "happiness"）。

之后出现了几个变体，但核心范式没变：

| 方法 | 年份 | 核心差异 | 代表模型 |
|------|------|----------|----------|
| BPE | 2016 | 按频率贪心合并 | GPT-2/3/4, LLaMA |
| WordPiece | 2012/2018 | 按互信息合并（不是频率） | BERT, DistilBERT |
| Unigram LM | 2018 | 自顶向下剪枝（不是自底向上合并） | T5, ALBERT |
| SentencePiece | 2018 | 语言无关框架，空格当普通字符 | LLaMA, T5 |

GPT-2 引入了 **byte-level BPE**：从 256 个 UTF-8 字节开始合并，而不是从字符开始。这保证了任何 Unicode 文本都能被编码（彻底消灭 OOV），现在已经是标准做法。

### 2.2 深层规律：什么才是"好"的 tokenizer

子词 tokenization 统治了 NLP 将近十年，但直到最近几年，人们才真正理解它为什么好、怎样才能更好。几篇关键论文揭示了深层规律：

**发现一：压缩率和性能不是单调关系**

Schmidt et al. (ACL 2024) 的 **"Tokenization Is More Than Compression"** 是一篇里程碑式的工作。他们训练了 64 个语言模型（350M 到 2.4B 参数），还专门构建了 PATHPIECE — 一个可证明最小化 token 数量的 tokenizer — 来测试"压缩越好 = 性能越好"这个假设。

结果出人意料：**压缩率和下游性能的关系是倒 U 型的**，相关系数只有 0.241。过度压缩反而破坏了可学习的结构。更重要的发现是：预分词规则（pre-tokenization，比如保留空格边界、数字边界）对性能的影响比压缩率大得多 — 即使这些规则会降低压缩率。

这个发现的深层含义是：tokenizer 不仅仅是一个压缩器，它还是一个**结构保持器**。好的 tokenization 需要在压缩和保持可学习结构之间找到平衡。

**发现二：最优词汇量和模型大小呈对数线性关系**

Tao et al. (NeurIPS 2024) 的 **"Scaling Laws with Vocabulary"** 发现了一个优雅的 scaling law：最优词汇量 N_v 和非词汇参数量 N_nv 之间满足 N_v ∝ N_nv^0.83。

这意味着大多数当前 LLM 的词汇量偏小。LLaMA2-70B 用 32K 词汇，但预测的最优值是 ~216K（7 倍大）。实际上，词汇量确实在快速增长：LLaMA 2 的 32K → LLaMA 3 的 128K → Gemini 3 的 262K。

更有趣的是 Huang et al. (ICML 2025) 的 **"Over-Tokenized Transformer"** 发现：**输入和输出词汇表应该解耦**。通过 n-gram 嵌入把输入词汇量扩大到 12.8M，一个 400M 模型可以匹配 1B 基线 — 2.5 倍的等效大小提升，几乎零额外计算成本。

**发现三：Zipf 定律是最优性的信号**

He et al. (2025) 的 **"Pre-trained Models Perform Best When Token Distributions Follow Zipf's Law"** 可能是对 MeshLex 最有启发的一篇。他们发现：

- 随着词汇量增大，token 频率分布逐渐趋近 Zipf 定律（log-log 图上的直线）
- **下游性能的峰值恰好出现在 Zipf 对齐度最高的点**（用 log-log 拟合的 R² 衡量）
- 这个规律在 NLP、基因组学、化学领域都成立 — 是一个**通用原则**

这提供了一个廉价的词汇量选择标准：不需要训练完整模型，只需要检查 token 分布是否接近 Zipf。

**发现四：信息论视角 — tokenizer 是结构化压缩器**

Erdogan et al. (Stanford, 2026) 的信息论分析揭示了 tokenizer 的本质角色：

- 好的 tokenizer 把**局部统计规律吸收进 token 定义**里（unigram 熵 H₁ 上升），让语言模型专注于建模**长程依赖**（条件熵 H_{k≥2} 下降）
- 引入了"容量利用率" η = H₁/log₂(K) 来衡量词汇空间的使用效率
- Tokenizer + LZ 压缩比纯 LZ 压缩好 10-20%，确认了 tokenization 作为预处理压缩器的价值

### 2.3 后 BPE 时代：三条并行路线（2024-2026）

子词 tokenization 的统治地位正在被挑战，三条路线同时推进：

**路线一：可学习 tokenization**

Dauncey & Wattenhofer (ETH Zürich, 2026) 的 **"You Can Learn Tokenization End-to-End with RL"** 用 REINFORCE 把 token 边界选择建模为随机策略。学出来的边界自动对齐语义单元 — 在代码上，它会把模块名聚合在一起、折叠空白序列、跳过样板代码。计算开销 <0.1%。

Rozental (2026) 的 **Zonkey** 更激进：完全可微的层次化扩散语言模型，带学习的段落分割器。涌现出词和句子的边界，无需任何归纳偏置。

**路线二：字节级模型**

Deng et al. (Amazon/Rice, 2026) 的 **ByteFlow Net** 完全去掉 tokenizer，根据 latent 表示的**编码率**（信息密度）变化来决定分割边界。用 Top-K 选择保持静态计算图，硬件友好。核心洞察：**边界应该放在信息密度变化的地方**，而不是频率统计决定的地方。

Zheng et al. (HKU, 2026) 的 **Proxy Compression** 走了一条务实路线：90% 训练用压缩 token（高效），10% 用原始字节（鲁棒），推理时完全在字节上运行。关键发现：**结构化压缩**（BPE、神经压缩器）可以跨表示迁移，但**非结构化压缩**（gzip）完全失败。

**路线三：动态 tokenization**

Feher et al. (Cambridge, ACL 2025) 的动态 tokenization 在推理时对每个 batch 动态调整 BPE 合并规则，用超网络（hypernetwork）即时生成新 token 的嵌入。序列长度减少 22-26%，性能只降 1.7-1.9%。一个 350M 模型配合好的动态 tokenizer 可以在翻译任务上匹配 2.7B 模型配差的静态 tokenizer。

**核心结论**：最优 tokenization 是**输入依赖的、应该被学习的**，而不是固定的。问题不再是"哪个静态 tokenizer 最好"，而是"怎么让 tokenization 成为模型学习计算的一部分"。

---

## 3. VLM Patchification 进化史

视觉 tokenization 的进化比文本更快、更戏剧性，因为图像的空间结构比文本的线性结构提供了更多可以利用的先验。

### 3.1 固定网格时代（2020-2021）

**ViT (Dosovitskiy et al., 2020)** 把 NLP 的 tokenization 逻辑直接搬到了图像上：把图像切成固定的 16×16 像素 patch，每个 patch 线性投影成一个 embedding，然后用标准 Transformer 处理。一张 224×224 的图像变成 196 个 token，不管图像内容是什么 — 一张纯白背景和一张复杂街景得到完全相同数量的 token。

这个方案简单粗暴但出奇地有效。DeiT (Touvron et al., 2021) 证明了 ViT 不需要 JFT 级别的海量数据也能训练好。BEiT (Bao et al., 2021) 引入了 masked image modeling，用 DALL-E 的 dVAE 产生离散 token 作为 mask 预测目标 — 这是第一次把**学习到的离散视觉词汇表**用作 ViT 的训练信号。

MAE (He et al., 2022) 发现了一个关键事实：**75% 的 patch 可以被 mask 掉，模型仍然能重建图像**。这说明固定网格有巨大的冗余 — 大部分 patch 携带的信息是可以从邻居推断出来的。

### 3.2 离散视觉 Tokenizer（2021-2023）

与 ViT 并行，另一条线专注于为生成模型构建更好的视觉 tokenizer：

**VQ-VAE (van den Oord et al., 2017)** 是起点：把图像编码成离散 latent code，通过 codebook 最近邻查找。**VQ-VAE-2 (Razavi et al., 2019)** 引入了层次化（多尺度）向量量化 — 顶层 codebook 捕获全局结构，底层捕获局部细节。这是**多尺度离散视觉 tokenization 的首次展示**。

**VQGAN / Taming Transformers (Esser et al., 2021)** 是一个转折点：结合 VQ-VAE + 对抗训练（PatchGAN 判别器）+ 感知损失，大幅提升了离散 tokenizer 的重建质量。成为自回归图像生成的事实标准 tokenizer。

然后是一篇改变了整个领域认知的论文：Yu et al. (2023) 的 **"Language Model Beats Diffusion — Tokenizer is Key"**。他们引入了 MagViT-v2（lookup-free quantization，codebook 扩大到 2^18 个 entry），证明了：**只要 tokenizer 足够好，语言模型式的生成可以匹配甚至超越扩散模型**。

这篇论文的核心信息是：**tokenizer 质量是生成质量的瓶颈**，不是生成模型架构。这个发现驱动了后续对 tokenizer 的大量投入 — GigaTok (Xiong et al., 2025) 把视觉 tokenizer 扩大到 30 亿参数。

### 3.3 动态分辨率（2023-2024）

研究者开始质疑固定网格的假设：

**NaViT — Patch n' Pack (Dehghani et al., 2023, Google)** 是一篇里程碑论文。它引入了"序列打包"：把不同分辨率、不同宽高比的图像的 patch 序列打包到同一个 batch 里，用 attention mask 区分。消除了"所有图像必须 resize 到正方形"的限制。

**FlexiViT (Beyer et al., 2023, Google)** 训练了一个可以在多种 patch 大小（8×8、16×16、32×32）下运行的 ViT，通过在训练时随机化 patch 大小实现。推理时可以灵活地在计算量和精度之间权衡。

**Mixed-Resolution Tokenization (Ronen et al., 2023)** 走得更远：**同一张图像内不同区域用不同大小的 patch** — 重要区域用小 patch（更多 token），背景用大 patch（更少 token）。这是首次在单张图像内实现**内容感知的变尺寸分割**。

### 3.4 语义感知 Tokenization（2024-2025）— 最关键的阶段

这个阶段发生了范式转变：

**EPOC (Chen et al., 2024, Meta/FAIR + HKUST)** 是一篇范式转变的论文。它明确类比了 NLP 的 BPE tokenization：

> 图像应该按"子物体"（subobject）级别分割，而不是任意固定 patch。就像 BPE 产生有形态学意义的子词一样，视觉 tokenizer 应该产生有语义意义的视觉 token。

EPOC 用一个 3.7M 参数的轻量边界检测器 + 分水岭分割，产生语义一致的 token。关键指标：**单语义性（monosemanticity）>90%**，而固定 patch 只有 <60%。VLM 用 EPOC token 收敛更快、泛化更好、需要更少的 token。

**dHT — Differentiable Hierarchical Visual Tokenization (Aasan et al., 2025, U. Oslo)** 把这个思路推到了极致：端到端可微的 tokenizer，构建超像素层次结构，用信息准则（AIC/BIC）自动选择每张图像的最优分割粒度。**分割边界本身通过反向传播学习**。可以直接嫁接到预训练 ViT 上。

**DART (Yin et al., 2025)** 用轻量 CNN 对图像区域打分，然后用可微的分位数分割产生变尺寸 patch。重要区域密集分割、背景粗糙分割。DeiT-Ti 上 +1.6% 精度，视频任务上 FLOPs 减少 69%。

### 3.5 1D 自适应长度表示（2024-2026）

最新的前沿彻底打破了 2D 空间网格的假设：

**TiTok (Yu et al., 2024)** 证明了一个反直觉的事实：**一张图像只需要 32 个 1D token 就能重建和生成**。这些 token 不是空间排列的 — 它们是 1D latent 序列，没有对应到图像的特定区域。这说明 2D 空间结构在 token 空间中不是必需的。

**ALIT — Adaptive Length Image Tokenization (Duggal et al., 2024, MIT CSAIL)** 是最有启发性的工作之一。它用循环编码器-解码器把 2D patch token 迭代蒸馏成可变数量的 1D latent token（32-256 个）。Token 数量自适应图像复杂度、熟悉度和下游任务。

最惊人的发现是：**涌现出 token 自动绑定到语义物体/部件的现象** — 没有任何监督信号，个别 token 自发地"负责"特定的语义区域。论文明确类比了 LLM 的"思考 token"。

**FlexTok (Apple, 2025)** 用交叉注意力把图像特征蒸馏成灵活长度的 1D token 序列。单个模型支持不同质量-计算权衡。

**STAT — Soft Tail-dropping Adaptive Tokenizer (2026)** 根据结构复杂度自适应选择输出 token 数量，用"尾部丢弃"机制逐步丢弃不重要的 token。

### 3.6 Scaling Law 与像素级前沿（2025-2026）

**"Scaling Laws in Patchification" (Wang et al., 2025, JHU/Berkeley)** 发现了一个 **Patchification Scaling Law**：性能随 patch 大小减小而持续提升，一直到 1×1（像素级 tokenization）。用 Mamba 架构（线性复杂度）处理每张图 50,176 个 token。

三个关键发现：
1. **Patch size scaling 的性价比优于 parameter scaling** — 减小 patch 比增大模型更划算
2. 像素级 token 下，复杂的 decoder head 变得不必要
3. 收益来自信息增益，不仅仅是更长的序列

**VAR — Visual AutoRegressive Modeling (Tian et al., 2024, PKU/ByteDance)** 重新定义了自回归图像生成：不是 next-token prediction，而是 **next-scale prediction**。从粗到细生成多尺度 token map。这是一个根本不同的 tokenization 范式 — 生成顺序跟随分辨率层次，而不是光栅扫描。

### 3.7 统一 Tokenizer 与趋势收敛（2025-2026）

最新趋势是统一理解和生成的 tokenizer：

- **UniTok (2025)**：统一视觉生成和理解的 tokenizer
- **AToken (2025)**：首个在图像、视频、3D 上同时实现高保真重建和语义理解的统一视觉 tokenizer
- **Cosmos Tokenizer (NVIDIA, 2024)**：工业级图像+视频 tokenizer
- **"Compression Tells Intelligence" (2026)**：理论论文，连接经典视觉编码理论和 MLLM token 技术，论证压缩效率和智能相关

### 3.8 VLM 进化轨迹总结

| 阶段 | 时间 | 代表 | 核心变化 |
|------|------|------|----------|
| 固定网格 | 2020-21 | ViT, DeiT | 16×16 固定 patch |
| 离散 tokenizer | 2021-23 | VQGAN, MagViT-v2 | 学习离散视觉词汇 |
| 动态分辨率 | 2023-24 | NaViT, FlexiViT | 多尺度、变分辨率 |
| 语义感知 | 2024-25 | EPOC, dHT, DART | 内容感知、可微边界 |
| 1D 自适应 | 2024-26 | TiTok, ALIT, FlexTok | 打破 2D 网格、变长序列 |
| Scaling Law | 2025-26 | Patchification Scaling | 像素级、信息论基础 |

**核心趋势**：固定 → 自适应 → 可学习边界；2D 空间网格 → 1D latent 序列；均匀 → 内容感知 → 语义对齐；分离 tokenizer → 端到端联合学习。

---

## 4. Mesh Tokenization 现状与竞品

3D mesh 的 tokenization 是三个模态中最年轻的，但过去两年发展极快。

### 4.1 Per-Face / Per-Vertex 方法（"字符级"）

这是当前的主流范式，相当于文本 tokenization 的字符级阶段：

**MeshGPT (Siddiqui et al., 2023)** 开创了这条路线：用 VQ-VAE 编码 face 特征，然后 GPT 解码器自回归生成。每个 face 是一个 token（3 个顶点坐标 = 9 个数值）。问题是序列太长 — 一个 800 face 的 mesh 就需要 ~800 个 token。

后续工作都在这个框架内优化：
- **MeshXL (2024)**：神经坐标场，扩展到更大 mesh
- **LLaMA-Mesh (2024)**：统一 mesh 生成和 LLM
- **DeepMesh (2025)**：RL 增强的自回归生成

这些方法的共同瓶颈是：**per-face tokenization 的序列长度和 mesh 面数线性相关**，无法处理高精度 mesh。

### 4.2 局部性感知方法

**Nautilus (Wang et al., 2025)** 提出了"鹦鹉螺壳"表示：围绕中心顶点按壳层组织 face。关键指标：

| 指标 | Nautilus | AMT | MeshGPT |
|------|---------|-----|---------|
| 压缩率 | 0.275 | 0.462 | 1.0 |
| 局部性比 | 0.554 | 0.312 | 0.189 |
| 流形率 | 83.6% | 29.0% | 23.8% |

Nautilus 的核心发现：**在 tokenization 中保持局部依赖性，而不仅仅是压缩，对结构保真度至关重要**。Attention map 显示 transformer 在局部性好的 tokenization 下会集中关注邻近 token。

### 4.3 拓扑保持方法

**Mesh Silksong (Song et al., 2025)** 用顶点分层方法：每个顶点只访问一次（50% 冗余减少），4 个 token/顶点（2 坐标 + 1 层内邻接 + 1 层间邻接）。压缩率 0.22（SOTA）。保证流形拓扑、一致法向、水密检测。用半边数据结构实现确定性遍历。

### 4.4 Patch 级方法（MeshLex 所在的赛道）

**MeshMosaic (Xu et al., 2025)** 是 MeshLex 最直接的竞品：

- 用 PartField 语义分割（推理时）或随机 Voronoi（训练时）把 mesh 分成语义 patch
- 每个 patch 自回归生成，以前一个 patch 的边界为条件
- GRU 编码前 512 个三角形的边界条件
- 局部量化 512³/patch（vs 全局 512³）
- 扩展到 100K+ 三角形
- 基于 0.5B DeepMesh 模型

**MeshMosaic vs MeshLex 的关键区别**：MeshMosaic 的 patch 是每个 mesh 独有的（顺序生成，边界条件传递），MeshLex 的 patch 是跨 mesh 复用的（universal codebook）。MeshMosaic 更像"写作文"（逐段写），MeshLex 更像"拼积木"（从词汇表选取）。

### 4.5 语义级方法

**LoST — Level of Semantics Tokenization (Dutt et al., 2026)** 按语义显著性排序 token：1-4 个 token 就能解码出可识别的形状。用 RIDA（Relational Inter-Distance Alignment）提供 3D 语义引导。只需 0.1-10% 的 token 就能达到几何 LoD 方法的效果。但它操作在 triplane latent 上，不是直接操作 mesh。

### 4.6 其他值得关注的方法

| 方法 | 核心思路 | 特点 |
|------|----------|------|
| TreeMeshGPT (2025) | 树结构自回归 | 层次化生成 |
| FastMesh (2025) | 组件解耦 | 效率优化 |
| ARMesh (2025) | Next-level-of-detail 预测 | 类似 VAR 的多尺度 |
| MeshRipple (2025) | 滑动窗口结构化 AR | 局部性 + 效率 |
| PrimitiveAnything (2025) | 分解为几何基元 | 基元级 tokenization |

### 4.7 Mesh Tokenization 的进化阶段

对比文本和视觉，mesh tokenization 的进化可以这样映射：

| 文本阶段 | 视觉阶段 | Mesh 阶段 | 代表 |
|----------|----------|-----------|------|
| 字符级 | 像素级 | Per-face/vertex | MeshGPT, DeepMesh |
| 词级 | 固定 patch | 固定大小 patch | **MeshLex (当前)** |
| BPE 子词 | 语义 patch | 语义/自适应 patch | MeshMosaic, ? |
| 可学习 token | 可微分割 | 可学习分割 | **未来方向** |

MeshLex 当前处于"固定 patch"阶段 — 相当于 ViT 的 16×16。下一步应该跳到"语义/自适应 patch"甚至"可学习分割"。

---

## 5. Mesh Segmentation 经典算法

Mesh segmentation 是计算机图形学的经典问题，积累了几十年的方法论。这些方法虽然不是为 tokenization 设计的，但它们对"怎么把 mesh 切成好的 piece"有深刻的理解。

### 5.1 Variational Shape Approximation (VSA)

**VSA (Cohen-Steiner, Alliez & Desbrun, SIGGRAPH 2004)** 对 MeshLex 可能是最直接相关的经典方法。

核心思想：迭代聚类 face，最小化原始曲面和分段平面代理之间的逼近误差。本质上是 face 法向上的 k-means + 几何失真度量（L² 或 L^2,1 误差）。

VSA 的性质：
- **直接优化重建误差** — 代理曲面逼近原始曲面
- 产生**近平面**的 patch — 平面区域的 patch 大，高曲率区域的 patch 小
- Patch 数量是参数（k），通过 Lloyd 迭代收敛到局部最优
- 计算复杂度 O(kN) per iteration

**为什么 VSA 和 MeshLex 天然对齐**：VSA 的优化目标（最小化逼近误差）和 VQ-VAE 的优化目标（最小化重建误差）本质上是同一件事。VSA 产生的 patch 在平面区域大、在高曲率区域小 — 这正是我们想要的自适应行为。几何简单的 patch 更容易被 codebook 编码。

局限：VSA 的 patch 是平面代理，对于弯曲特征（圆柱、球面）可能需要很多小 patch 才能逼近。

### 5.2 曲率驱动分割

曲率分割的核心思想：在高曲率边缘处切割，产生内部曲率方差低的 patch。

**Reeb 图方法 (Beguet et al., 2024)** 提供了一个现代统一框架：
- 用 Shape Index（曲率导出的标量）作为 Reeb 图的输入
- Reeb 图捕获标量函数的拓扑骨架（极小值、极大值、鞍点）
- 同时整合几何特征（Shape Index）和拓扑特征（SDF）
- 复杂度 O(n log n)

曲率分割产生的 patch 边界沿着几何特征线（棱边、折痕），这在视觉上很自然。但问题是 patch 数量难以控制，且平面区域可能产生过大的 patch。

### 5.3 谱方法与 RNS

谱方法用 Laplace-Beltrami 算子（LBO）的特征向量定义特征空间，然后在这个空间中聚类。

**GeoTransformer (Farazi & Wang, 2024)** 对 MeshLex 有直接启示，因为它**正面对比了 METIS 和 RNS**：

**RNS (Root-Node Selection)**：基于代数多重网格文献，用 LBO 的各向异性边距离矩阵做 k-medoid 聚类。关键性质：**保持 LBO 的谱** — 分区后的 mesh 保留了原始的特征函数结构。Patch 大小根据底层几何变化（高曲率区域更多 patch）。

**METIS**：标准图分区算法。高效地创建平衡大小的 patch，但**不感知几何** — 在逼近 HKS 和 LBO 特征函数时失败。

论文的关键结论：**RNS 在保持几何性质方面显著优于 METIS**。这对 MeshLex 的含义是：我们当前的 METIS 分割可能在丢失重要的几何信息。

### 5.4 SDF 分割

Shape Diameter Function (SDF) 通过从每个 face 向内投射射线测量局部厚度。SDF 值相似的 face 被聚类，然后用图割算法细化边界。

**Neural ShDF (Roy, 2023)** 用 GNN 预测 SDF 值，速度提升 10 倍：
- COSEG 上 97.1%（接近 MeshCNN 的 97.3%）
- Human Body 上 94.2%（超过 MeshCNN 的 92.3%）
- 分辨率无关（通过降采样 + 全分辨率邻域查询）

**Segment Any Mesh (Tang et al., 2024)** 把 SDF 标量渲染成 2D 图像，喂给 SAM2，实现零样本 mesh 分割。SDF + 表面法向作为多模态输入效果最好。

SDF 分割产生语义有意义的部件（手臂、腿、把手），但这些是**语义部件**，不是几何 patch — 大小差异巨大，不适合直接用于 codebook 学习。

### 5.5 凸分解特征场

**Learning Convex Decomposition via Feature Fields (Yang et al., 2026)** 是最新的 SOTA：

- 把凸分解重新表述为**对比特征学习**
- 自监督几何 loss：基于凸性定义（凸对内的线段保持在体积内）
- 在 340K Objaverse 形状上训练的前馈模型
- 超越 V-HACD 和 CoACD
- 多粒度控制（通过聚类阈值）
- 推理 18s（5s 特征 + 13s 聚类）

关键发现：**语义分割特征（来自 PartField）产生的凸分解很差** — 凸性感知特征和语义特征是根本不同的。

**这个思路可以直接迁移到 MeshLex**：不学凸性特征，而是学"codebook-ability"特征 — 应该属于同一个 patch 的 face 有相似特征，因为它们可以被同一个 codebook entry 很好地表示。自监督 loss 可以是：同一 codebook entry 下的 face 对的特征距离应该小于不同 entry 下的。

### 5.6 MeshCNN 的边特征

**MeshCNN (Hanocka et al., 2019)** 在 mesh 的边上操作，每条边有 5D 特征：二面角、两个内角、两个边长比。

它的 mesh pooling 通过边折叠实现：优先折叠特征最小的边，自然地简化 mesh。这创建了一个**多分辨率层次结构**，每一层保留最重要的几何特征。

MeshCNN 的边特征（特别是二面角）可以用来指导 patch 边界检测 — 高二面角的边是自然的切割位置。

### 5.7 Quad Meshing 文献

**Learning Quadrangulated Patches (Groueix et al., 2017/2019)** 学习一个表面 patch 字典，每个 patch 是曲面的局部参数化。Patch 由四边形网格结构定义，沿主曲率方向对齐。

这个"学习 patch 字典"的概念本质上就是 MeshLex 在做的事 — 但用的是四边形 patch 而不是三角形 patch。Quad mesh patch 沿曲率方向对齐，产生几何上自然的分解。

---

## 6. 跨模态规律提炼

把三个模态放在一起看，有几个非常清晰的通用规律：

### 规律一：固定 → 自适应 → 可学习边界

这是最明确的进化方向：

| 模态 | 固定 | 自适应 | 可学习 |
|------|------|--------|--------|
| 文本 | 字符/词 | BPE（频率驱动） | RL tokenization, ByteFlow |
| 视觉 | 16×16 patch | FlexiViT, Mixed-Res | dHT, DART |
| Mesh | METIS（固定 face 数） | ? | ? |

Mesh 领域在"自适应"和"可学习"阶段都是空白。

### 规律二：Zipf 定律是最优性的通用信号

文本领域已经证明：token 频率分布最接近 Zipf 定律时，下游性能最好（He et al., 2025）。

MeshLex 当前的 codebook token 分布：lognormal，Zipf α = 0.34-0.50（弱幂律）。而文本的最优 α ≈ 1.0。这暗示我们的 patchification 没有产生足够的"频率分化" — 太多 patch 被使用得差不多频繁，codebook 的信息效率不高。

一个可能的解释：METIS 的均匀分割产生了过于相似的 patch，导致 codebook 使用过于均匀。如果改用自适应分割（大 patch 覆盖简单区域，小 patch 覆盖复杂区域），可能会自然产生更接近 Zipf 的分布 — 因为简单 patch 出现频率高（大面积平面区域），复杂 patch 出现频率低（局部高曲率区域）。

### 规律三：Tokenizer 质量是整个系统的瓶颈

- 文本："Scaling Laws with Vocabulary" — 词汇量选择比模型大小更重要
- 视觉："Language Model Beats Diffusion — Tokenizer is Key" — tokenizer 质量决定生成质量上限
- Mesh：同样的逻辑 — 改进 patchification 可能比改进 AR 模型获得更大收益

### 规律四：内容感知 >> 均匀分割

- 视觉 EPOC：语义 token 的单语义性 >90%，固定 patch <60%
- Mesh 类比：一个 35-face 的鞍点区域和一个 35-face 的平面区域，几何复杂度完全不同，但 METIS 把它们当作等价的 patch

你提到的"生僻字"类比非常精准：
- **简单 patch**（平面、圆柱）= 常用字 → 应该是 codebook 中的高频 entry
- **复杂 patch**（鞍点、高 genus）= 生僻字 → 应该被拆成多个简单 patch 的组合
- 在 3D mesh 语义下，生僻字可以被简单字组合替代（不像文本中"龘"不能用"龙龙龙"替代）

这意味着：最优 patchification 应该让复杂区域产生更多更小的 patch（每个都是"常用字"），简单区域产生更少更大的 patch。

### 规律五：两阶段范式正在被挑战

当前的标准范式是：先分割（tokenization）→ 再量化（VQ-VAE）。但：

- 文本：ByteFlow Net 完全去掉 tokenizer，从信息密度学习分割
- 视觉：dHT 端到端可微地学习分割边界
- 这暗示 mesh 上也可以做联合优化 — patchification 的边界由 VQ-VAE 的重建 loss 反向传播来决定

### 规律六：压缩不是唯一目标

Schmidt et al. (2024) 在文本上证明了压缩率和性能是倒 U 型关系。Nautilus (2025) 在 mesh 上发现"保持局部依赖性比压缩更重要"。

这意味着：MeshLex 不应该只追求最少的 patch 数量，还要确保 patch 之间的依赖关系对 AR 模型友好。

---

## 7. 参考文献

### LLM Tokenization

| 论文 | 会议/年份 | arXiv |
|------|-----------|-------|
| Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE) | ACL 2016 | 1508.07909 |
| Kudo, "Subword Regularization" (Unigram LM) | ACL 2018 | 1804.10959 |
| Kudo & Richardson, "SentencePiece" | EMNLP 2018 | 1808.06226 |
| Mielke et al., "Between Words and Characters" | TACL 2021 | 2112.10508 |
| Schmidt et al., "Tokenization Is More Than Compression" | ACL 2024 | 2402.18376 |
| Tao et al., "Scaling Laws with Vocabulary" | NeurIPS 2024 | 2407.13623 |
| Feher et al., "Retrofitting LLMs with Dynamic Tokenization" | ACL 2025 | 2411.18553 |
| Huang et al., "Over-Tokenized Transformer" | ICML 2025 | 2501.16975 |
| He et al., "Zipf's Law for Optimal Vocabulary" | 2025 | 2507.22543 |
| Erdogan et al., "Information-Theoretic Perspective on Tokenizers" | 2026 | 2601.09039 |
| Dauncey & Wattenhofer, "RL-based Tokenization" | 2026 | 2602.13940 |
| Rozental, "Zonkey" | 2026 | 2601.21768 |
| Zheng et al., "Proxy Compression" | 2026 | 2602.04289 |
| Deng et al., "ByteFlow Net" | 2026 | 2603.03583 |
| Lotz et al., "Beyond Text Compression" | ACL 2025 | 2506.03101 |

### VLM Patchification

| 论文 | 会议/年份 | arXiv |
|------|-----------|-------|
| Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT) | ICLR 2021 | 2010.11929 |
| Touvron et al., "DeiT" | ICML 2021 | 2012.12877 |
| Bao et al., "BEiT" | ICLR 2022 | 2106.08254 |
| He et al., "MAE" | CVPR 2022 | 2111.06377 |
| van den Oord et al., "VQ-VAE" | NeurIPS 2017 | 1711.00937 |
| Esser et al., "Taming Transformers" (VQGAN) | CVPR 2021 | 2012.09841 |
| Yu et al., "Language Model Beats Diffusion — Tokenizer is Key" (MagViT-v2) | ICLR 2024 | 2310.05737 |
| Dehghani et al., "Patch n' Pack" (NaViT) | 2023 | 2307.06304 |
| Beyer et al., "FlexiViT" | 2023 | 2212.08013 |
| Chen et al., "Subobject-level Image Tokenization" (EPOC) | 2024 | — |
| Aasan et al., "Differentiable Hierarchical Visual Tokenization" (dHT) | 2025 | — |
| Yin et al., "DART" | 2025 | — |
| Yu et al., "TiTok" | 2024 | — |
| Duggal et al., "ALIT" | 2024 | — |
| Wang et al., "Scaling Laws in Patchification" | 2025 | — |
| Tian et al., "VAR" | 2024 | — |
| Bolya et al., "Token Merging" (ToMe) | ICLR 2023 | 2210.09461 |
| Xiong et al., "GigaTok" | 2025 | — |

### Mesh Tokenization

| 论文 | 会议/年份 | arXiv |
|------|-----------|-------|
| Siddiqui et al., "MeshGPT" | 2023 | 2311.15475 |
| Wang et al., "Nautilus" | 2025 | 2501.14317 |
| Song et al., "Mesh Silksong" | 2025 | 2507.02477 |
| Xu et al., "MeshMosaic" | 2025 | 2509.19995 |
| Dutt et al., "LoST" | 2026 | 2603.17995 |

### Mesh Segmentation

| 论文 | 会议/年份 | arXiv |
|------|-----------|-------|
| Cohen-Steiner et al., "Variational Shape Approximation" (VSA) | SIGGRAPH 2004 | — |
| Hanocka et al., "MeshCNN" | SIGGRAPH 2019 | 1809.05910 |
| Farazi & Wang, "GeoTransformer" | 2024 | 2411.00164 |
| Roy, "Neural ShDF" | 2023 | 2306.11737 |
| Tang et al., "Segment Any Mesh" | 2024 | 2408.13679 |
| Yang et al., "Convex Decomposition via Feature Fields" | 2026 | 2603.09285 |
| Beguet et al., "Reeb Graph Segmentation" | 2024 | 2412.05335 |
| Groueix et al., "Learning Quadrangulated Patches" | 2017/2019 | 1709.06868 |
