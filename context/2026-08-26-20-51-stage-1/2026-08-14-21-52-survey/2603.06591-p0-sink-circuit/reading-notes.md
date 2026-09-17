# What Makes Position Zero Special? A Mechanistic Study of Position-Zero Attention Sinks in LLMs

ArXiv: 2603.06591v2  
阅读材料：`source.txt` / `source.pdf`

## 第一遍：快速判断

### 只基于标题、摘要、标题层级、图表说明与结论

论文提出一个两层 Transformer block 的 **P0-Sink Circuit**：因果掩码使位置 0 只能注意自身，因而生成未混合且跨输入方向稳定的 attention 输出；随后 MLP 将该信号放大为高范数、固定方向的 P0 表示，进而触发后续层的 attention sink。摘要和结论明确声称该机制不依赖语义内容。

论文还提出两个无额外参数的方法：TNLM Loss 删除 `[BOS]` 位置的交叉熵项，使其成为专用 register；Force-sink Masking 将 P0 与其他位置的残差维度硬分区。图表显示作者检查了 `[BOS]` 消融、逐层范数与余弦相似度、PCA/降维聚类、学习率与数据规模、sink 形成进程及下游性能。

初步判断：这是“承诺/不确定性驱动边界”证据链中的结构机制核心，但其结论直接适用于 **causal LLM 的固定 P0 sink**，不能在第一遍就推广到动态 sink、非 P0 sink 或非自回归 Transformer。

### 是否继续深读

是。必须核查：

- “纯结构、无语义”究竟由哪些干预支持，还是主要来自机制解释。
- 两块 circuit 的证据是否包含真正的组件级因果干预。
- 早形成 sink 与性能改善是相关、受控干预结果，还是二者混合。
- Force-sink Masking 是否直接把待解释现象写入结构，从而只能验证效用，不能独立验证自然形成机制。

## 第二遍：主贡献、机制与实验

### 论文实际回答的问题

作者把问题限定为：在 causal LLM 中，为什么一个不依赖固定 `[BOS]` 语义的 P0 sink 能在很浅的层内重新出现，以及能否通过人为加速这一结构的形成改善训练。论文并不主张所有 attention sink 都有同一机制；它明确区分了始终可见的 full-context P0、会滑出窗口的 Mistral `[BOS]` sink、任意中间位置 sink 和 gated attention 的无 sink 替代机制。

### P0-Sink Circuit

电路包含两个连续作用：

1. **位置零识别。** causal mask 强制 P0 的注意力权重为 `p0=1`，所以它只能聚合自己的 value；其他位置混合多个 context vectors。作者用锥形 value-vector 模型说明，若不同 value 共享方向偏置 `α`，则 attention output 的期望平方范数为 `α² + (1-α²) E[Σ p_i²]`。随着可注意位置增多，`E[Σ p_i²]` 下降；P0 恒为 1，形成输出范数/方差不对称。
2. **固定方向放大。** 随后的 MLP 检测这一不对称，将 P0 推向跨输入一致的方向并放大范数。在 pre-norm 模型中，下游模块看到的是方向；高范数残差还使归一化后方向对训练扰动更稳定。局部线性化给出 `Norm(x+δx)-Norm(x) ≈ δx⊥/r(x)`。

作者用 LLaMA3.1-8B 的逐层范数、跨样本余弦相似度和 PCA/t-SNE 表明：去掉 `[BOS]` 后，P0 仍在两个 block 内与其他位置分离并成为高范数、固定方向表示。对 layer-0 head 29 的贡献做精确减法后，P0 在 MLP 中间状态中的聚类更清楚，但该 head removal 主要用于去除可视化混杂，并非完整的 circuit knockout。

### `[BOS]`、位置结构与架构条件

LLaMA 家族去掉 `[BOS]` 后，前两层 sink rate 下降，但从 layer 2 起恢复至接近原水平；这支持存在独立于 `[BOS]` embedding 的位置机制。Mistral 则是反例：去掉 `[BOS]` 后 sink rate 从约 87% 降至约 2%。作者解释为 sliding-window 使 P0 不再持续可见，模型没有发展稳定的 positional P0 sink，只依赖 `[BOS]` embedding。混合架构中 sliding-window 层无 P0 sink、full-attention 层有 P0 sink，也限定了该机制的适用域。

重复首 token 实验显示：LLaMA 中重复非 `[BOS]` 内容 token 带来的 loss 增幅小于重复 `[BOS]`，作者据此认为结构性 P0 是更稳定的“semantic vacuum”。不过这是功能性间接指标，不等价于直接证明 P0 表示不含任何可解码语义。

### 两个训练方法

- **TNLM Loss：** 删除 `[BOS]` 位置预测首个真实 token 的交叉熵。作者把 P0 梯度分为形成 sink 的分量与预测分量，认为固定 unigram 预测仍会干扰 sink circuit，去掉它可释放 P0 为 register。
- **Force-sink Masking：** 每层 pre-norm 前用固定二值 mask，把 P0 限制在 `k` 个通道，其他 token 限制在互补的 `d-k` 个通道。这样直接实现位置识别和非 sink 通道抑制，无需学习大幅 outlier rescaling。

### 训练与结果

主实验训练约 1B 参数模型、20B FineWeb-Edu tokens、3 个 peak learning rates，统一 seed 42；比较等参数 MLP baseline、Gated Attention、TNLM、Force-sink 及组合。主表中两种方法在 20B 设置平均下游分数均高于各自 baseline，Force-sink 多数设置接近或略高于 Gated Attention。Force-sink 显著提高 sink rate，同时降低 P0 激活 outlier，说明大激活不是 sink 的必要条件。

100B data-saturated 实验则更复杂：baseline 已有很高 sink rate，Gated Attention 的平均优势消失；TNLM 仅有小幅平均增益，Force-sink 的不同变体并非每个 checkpoint 都优于 baseline。因此论文最稳妥的性能结论是“加速未成熟 sink 的形成在 20B 设置中与更好结果共同出现”，而不是“sink rate 越高必然越好”。

### 第二遍结论

最强证据是结构边界：只要 P0 在 causal full-context attention 中始终存在，它拥有其他位置没有的 `p0=1` 混合不对称，且无 `[BOS]` 时仍可在浅层重建 sink。性能价值的证据弱于形成机制证据，因为方法同时改变损失或表示子空间，且主要训练比较只有单一 seed。

## 第三遍：机制核查、虚拟复现与改进

### 隐含假设与证据强弱

1. **锥模型是解释性近似，不是从模型中识别出的生成模型。** 它假设 value vectors 单位范数、与固定轴具有常数 cosine `α`、正交分量独立均匀；随后又借助“attention 随位置稀疏，因此 `E[Σp_i²]` 随长度单调下降”的经验判断。P0 的 `p0=1` 是严格结构事实，但“其他位置一定更低”依赖这些分布假设。
2. **高范数稳定性是局部一阶论证。** `δx⊥/r(x)` 推导说明归一化方向对同等大小正交扰动的敏感度随范数下降，但将真实 SGD 噪声视为与输入近似独立、把一次更新当作小扰动，不能单独证明训练动力学必然形成或维持 sink。
3. **“无语义”有两层不同含义。** 去掉 `[BOS]` 后 P0 sink 恢复，强力支持“不需要特定 token embedding/身份”；但“P0 表示完全不携带可解码内容”主要由高范数覆盖早期残差与重复-token loss 间接推出，未做 probing、patching 或 content-preservation knockout。
4. **自然 circuit 的必要性干预不足。** PCA、cosine 与精确移除 head 29 揭示表示分离；真正对 MLP 放大、特定方向或 attention-output variance 做逐组件 knockout 的结果没有在本文给出。Force-sink 是把两个功能结构化实现，证明这种替代机制可用，不等于证明自然模型只能通过所述两块 circuit 形成 sink。
5. **性能因果链存在共同干预混杂。** TNLM 同时删除一个训练目标；Force-sink 同时减少/隔离表示维度并改变优化几何。它们既改变 sink formation，也可能独立改善训练。因此“方法→更早 sink→更好性能”尚未被中介分析或 sink-matched control 隔离。作者在结论中使用的也是 “correlates”。
6. **训练统计不足以估计稳定性。** 主训练配置报告 seed 42，没有多 seed 方差。不同方法与 Gated Attention 的参数/FLOPs 处理较谨慎，但单 seed 下的 benchmark 差异不宜全部解释为可靠效应。
7. **`k` 是校准旋钮。** Force-sink 在 20B 用 `k=16`，100B 改用 `k=32`，作者明确承认无 principled criterion。该方法的普适性依赖模型宽度、训练量与最佳 `k` 如何缩放，尚未回答。

### 虚拟复现

1. **预训练模型机制复现。** 对 LLaMA3.1-8B、LLaMA3.2-1B/3B 和 Mistral，固定 1024 个未训练样本、长度 64；分别保留/删除 `[BOS]`，逐层保存 residual、attention output、MLP input/output。计算 `Sink^ϵ_0[l_start:]`、P0/其他位置范数和跨样本 mean-direction cosine。
2. **位置识别干预。** 除复现 PCA/t-SNE 外，逐 head 精确 subtract attention contribution，并对 layer-0 attention output 做 variance matching：把 P0 方差压到其他位置，或把任意位置方差抬到 P0 水平。若作者机制正确，前者应阻断浅层 P0 分离，后者应把 sink 转移到被处理位置。只缩放 residual norm 应不足以复制效果。
3. **两块 circuit knockout。** 在不改变 token embedding 的前提下，分别 patch layer-0 P0 attention output、layer-0/1 MLP output、以及 P0 direction；测后续 sink rate 和 loss。这样才能把“识别”和“放大”从观察性链条升级为组件级必要/充分证据。
4. **从头训练。** 严格复现 1B/20B、三学习率、seed 42，再至少增加 2–4 seeds；实现 TNLM 与每层 pre-norm Force-sink mask。除最终 benchmark 外，保存训练过程中 sink onset step、QK norm/conditioning、loss 与 P0 direction stability。做 `k∈{4,8,16,32,64}`、随机通道 mask、只在部分层 mask、相同有效宽度但不分区等 controls。

### 具体改进

- 增加 **sink-matched controls**：调节训练使不同方法最终 sink rate 相同，检验性能是否仍不同；或在相同性能 checkpoint 比较 sink onset，避免把相关当中介因果。
- 对自然模型做 activation patching/causal tracing，直接检验 layer-0 attention asymmetry 与后续 MLP fixed direction 的必要性和充分性。
- 用解码 probe、mutual information 或输入 token 替换测试“semantic vacuum”，把“不依赖 `[BOS]` identity”与“不携带语义”分开。
- 在 sliding-window 宽度、prefix-LM mask、双向 mask、局部+全局混合层之间连续扫描，找出 P0 持续可见性与 sink 形成的临界条件。
- 多 seed 报告 onset、loss 与下游分数置信区间；对 `k` 给出相对宽度 `k/d` 或预算约束下的缩放规律。

## 统一证据问题：QASPER 式回答

以下引文按 UTF-8 `source.txt` 核对；仅合并 PDF 换行。

### 1. sink/边界由何信号或结构触发

**回答：** 对本文的固定 P0 sink，触发源是 causal mask 的位置不对称：P0 只能注意自身，attention output 不发生上下文混合，因而比其他位置具有更稳定的方向与更高期望范数；后续 MLP 识别并放大这一信号。直接结构量是 `p0=1`，而不是 token 语义或仅仅 residual norm 大小。

> “The mechanism underlying this asymmetry is the causal attention mask. Under causal masking, all positions other than zero aggregate diverse context vectors, which reduces the consistency of any shared directional component in their attention outputs. Position zero, by contrast, attends only to itself, so its attention output remains unmixed and preserves its direction more reliably across inputs.”  
> — PDF p.5, §3.2

> “Position zero is the limiting case: with p0 = 1 enforced by causal masking, Σi p²i = 1 regardless of sequence length, yielding a strictly higher expected output norm than any other position. This norm asymmetry is the signal that MLP sublayers detect and amplify into the stable P0 representation.”  
> — PDF p.6, §3.2

### 2. 是否需要语义

**回答：** 不需要特定 `[BOS]` embedding 或内容身份，LLaMA 删除 `[BOS]` 后两层内重建 P0 sink；但“完全没有任何可解码语义”没有被直接测量。Mistral 还是重要边界：它的 sliding-window 条件下 `[BOS]` 是 sink 的唯一驱动，因此“无语义”不能无条件推广到所有架构。

> “Since LLMs without a [BOS] token also exhibit P0 sinks, there must exist at least one mechanism that gives rise to the P0 sink independently of the [BOS] embedding.”  
> — PDF p.3, §2

> “For the LLaMA family, the picture is different. Removing [BOS] affects only the first two layers, as shown in Table 1a, where Sinkϵ0[2:] recovers to near the w/ [BOS] level.”  
> — PDF p.3, §2

> “For Mistral, [BOS] is the sole driver of the sink: removing it collapses the sink rate greatly and directly reduces loss.”  
> — PDF p.3, §2

### 3. 哪些干预或消融支持因果解释

**回答：** 本文自己的主要干预是 `[BOS]` removal、单 head contribution 的精确移除、TNLM loss 和 Force-sink structural masking。前者支持 token identity 非必要；后两种从头训练显示人为实现/加速 P0 register 功能可改善 20B 设置。最关键的自然 circuit 组件 knockout 并未完整提供，所以形成机制的因果证据是“结构事实 + 消融 + 表征轨迹 + 工程性充分性”的组合，不是闭合的必要/充分证明。

> “This subtraction is exact: since attention head outputs are summed in the residual stream, individual head contributions can be isolated and removed precisely.”  
> — PDF p.5, §3.2

> “We therefore exclude the cross-entropy term at the <BOS> position from the training objective… This simultaneously removes the interfering gradient component and frees the position to act as a dedicated register that holds the sink.”  
> — PDF pp.7–8, §4.2

> “Applied at every layer, this keeps the P0 representation confined to the same k-dimensional subspace throughout the network, while every other token’s representation stays confined to the complementary (d−k)-dimensional subspace.”  
> — PDF p.8, §4.3

### 4. 是否依赖自回归目标

**回答：** 论文证明的 P0 circuit 直接依赖 causal attention mask 与 P0 持续可见性；训练目标影响形成速度。它没有证明 next-token cross-entropy 本身是 sink 存在的必要条件。相反，删除 P0 的一项交叉熵会加速其 register 化。sliding-window 层缺少稳定 P0 sink，说明关键是可见性/掩码结构，而非 RoPE。

> “Sliding window layers exhibit no P0 sink while full attention layers do, regardless of whether RoPE is employed.”  
> — PDF p.3, §2

> “Causal masking restricts the first token to attend only to its own value vector, making the QK interaction irrelevant there; both models reduce to a pointwise function, but with different targets.”  
> — PDF p.7, §4.1

### 5. 在 diffusion 中 sink 的功能发生何变化

**回答：** 本文未研究 diffusion，不能从其自身原文得出功能变化。它的证据域明确是 causal LLM；最多提供一个待对照的 AR 基线：固定 P0 被建模为可供 heads 指向的稳定 reference/register。diffusion 中是否仍如此必须由另一篇论文回答。

> “Causal large language models reliably form one at position zero, though its role remains debated.”  
> — PDF p.1, Abstract

> “We formalize the P0-Sink Circuit, a two-block subnetwork that exploits the asymmetry of the causal attention mask to generate a high-norm, fixed-direction representation at position zero, providing a consistent reference point for attention heads throughout the network.”  
> — PDF p.2, Contributions

## 最终证据定位

这篇论文最可靠地支撑：**固定 P0 边界可由 causal mask 的纯位置不对称启动，不需要特定 `[BOS]` 语义；MLP 随后把它固化成高范数、固定方向 register。** 它较弱地支撑：**更早形成 P0 sink 本身导致更好性能。**
