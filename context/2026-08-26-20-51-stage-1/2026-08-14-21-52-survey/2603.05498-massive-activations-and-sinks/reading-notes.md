# The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks

## 论文身份

- arXiv:2603.05498
- 作者：Shangwen Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu
- 主题：massive activations 与 attention sinks 的机制关系及架构消融。

## 第一遍速读

核心判断：论文把 massive activation 与 sink 的共现解释为 pre-norm Transformer 的架构产物，并区分前者的全局隐式参数功能与后者的局部调制功能；这是纯结构解释的重要补强，`read_deeper = true`。

摘要的主结果原文：“We identify the pre-norm configuration as the key choice that enables the co-occurrence, and show that ablating it causes the two phenomena to decouple.”（PDF p.1，Abstract）

负重图表：Figure 1（spike 生命周期）；Figure 5–6（spike token 经 normalization 后的几何与 sink/non-sink 对照）；Tables 4–8（FFN、norm、head、gating、context length 消融）。结论明确 massive activations 是 global implicit parameters，而 sinks 是 local attention-head modulators。

## 第二遍理解

论文给出一条从 spike 到 sink、同时又证明二者非必要绑定的机制链。早期 block 将少数通道放大；pre-norm residual 让极值跨中层累积，末端 block 再以反号抵消。RMSNorm 把不同 spike token 压成稀疏、近常量的方向，其 key 因而成为跨 prompt 稳定的低维参照；只有 query 子空间与该 key 子空间对齐的 heads 才产生 sink。随后的从头训练消融表明，这只是标准 pre-norm 配方提供的一条便利路径：去除 massive activations 后 sink 可保留，而提供显式、当前表示条件化的 gate 后 sink 几乎消失。论文据此区分 massive activations 的全局“隐式参数”角色与 sink 的 head-local、input-conditioned routing / short-range bias 角色。

## 关键实验与限制

- 位置而非 token 语义触发 initial spike：把全词表 token 轮流放在位置 0，七个模型有 98.40%–99.93% 成为 spike token（PDF pp.5–6，Table 2）。delimiter 则通过 early-layer self-sinking 进入同一高增益方向。
- normalization 解耦：pre-norm baseline spike 3818、sink ratio 46.0%；Sandwich(QK) 将 spike 降至 92 但 sink ratio 仍 42.0%；DynamicTanh spike 153、sink ratio 61.0%，perplexity 未恶化（PDF p.9，Table 5）。
- head capacity：`d_head` 从 8 增至 128，sink ratio 从 4.1% 单调升至 46.0%；固定总容量时，少而宽的 heads 更容易形成 sink（PDF pp.9–10，Table 6）。
- 显式 gating：representation-conditioned per-channel / per-head gates 将 sink ratio 降至 4.5% / 6.4%，静态 position/token gates 仍为 41.1% / 31.1%（PDF p.10，Table 7）。
- context 干预：训练损失包含短位置时 sink ratio 约 42%–46%；只在 2048–4096 位置计算损失时降至 1.2%，而 perplexity 基本可比（PDF pp.10–11，Table 8）。
- 主要因果实验只在从头训练的 Llama-style 7B、DCLM 100B tokens 上；结果指标以 perplexity、sink ratio、最大 activation 为主，未测语义 probe、下游任务或跨模态模型。作者的“纯结构”结论对这套 decoder-only pre-norm 配方最强，不能直接泛化到非 AR Transformer。

## 统一问题与精确证据

### 1. sink 由什么触发，承担什么计算功能

initial sink 的触发主要是位置结构：第一个 token 在 causal mask 下只能自注意，因而每个 prompt 都经过同一静态线性轨迹；delimiter 通过强 self-attention 模拟这种隔离。sink 的功能不是携带全局内容，而是给特定 heads 提供 input-conditioned “dumping ground”，减少远处 token 的影响，使其偏向局部依赖。

> “This disparity confirms that the phenomenon is driven by architectural position rather than token semantics.”（PDF p.5，§3.1.3）

> “In this regime, the attention block applies a static linear transformation that is identical across all prompts, consistently steering the first tokens’ representations toward the trigger direction s⋆.”（PDF p.6，§3.1.3）

> “By dumping attention into the first token, the model can effectively ignore long-range context when it is not predictive.”（PDF p.11，§4.4）

### 2. 支持语义、不确定性还是纯结构解释

这篇证据强烈偏向纯结构与训练分布解释，不讨论预测熵或边界不确定性，也没有证明 sink value 含语义。位置 0 的全词表替换、normalization/head-dimension/gating/context-length 消融共同表明：语义身份不是 initial spike/sink 的必要触发因素，优化健康与短上下文需求会调节其强度。

> “For nearly all evaluated models, positional occupancy at the initial position induces massive activations in intermediate layers, independent of the token’s semantic identity.”（PDF p.6，Table 2 caption）

> “Their overlap in standard pretrained LLMs is best understood as a byproduct of the default normalization and training recipe, not a reflection of any underlying functional necessity.”（PDF p.11，§4.5）

### 3. 消融或干预因果证据

论文的核心价值正是干预链：改变 normalization 可单独去除 spike；改变 head dimension 系统控制 sink；加入动态 gate 可替代 sink；移除短位置训练目标可近乎消除 sink。它们在相近 perplexity 下成立，因此比仅观察预训练模型的共现更接近因果证据。

> “Sandwich normalization reduces spikes while preserving a sink ratio nearly identical to the baseline.”（PDF p.9，§4.2.2）

> “When the model has access to a dynamic, representation-conditioned gate, it can modulate attention routing on the fly, eliminating the structural need to maintain a spike token via large residual spikes.”（PDF p.10，§4.3.2）

> “Removing short contexts entirely—optimizing only over long-range positions—causes the sink ratio to collapse dramatically.”（PDF p.10，§4.3.3）

### 4. 对 AR 依赖与跨模态对照的意义

机制明确利用 causal decoder 的特殊首位置与 next-token 训练的 context-length 分布，因此支持“LLM 首 token sink 至少部分由 AR 结构诱导”。但论文没有 non-causal、diffusion 或 ViT 实验，不能证明所有 sink 都依赖 AR；尤其 explicit gate 可替代 sink，说明它更一般地属于路由需求，而非语言语义本身。

> “Since the first token only attend to itself, its output reduces to [a static linear map].”（PDF p.6，§3.1.3）

> “Attention sinks are fundamentally a byproduct of short-context training.”（PDF p.10，§4.3.3）

> “We study two phenomena that reliably co-occur in decoder-only, pre-norm Transformers.”（PDF p.1，Introduction）
