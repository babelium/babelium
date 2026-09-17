# 承诺／不确定性驱动边界：核心证据链

## 结论

这批论文支持一个分层而非单因子的解释：Transformer 会把有限计算与表示空间集中到少数边界、位置或专用槽位，但触发信号至少有四类，不能全部归结为语义或预测熵。

1. **预测不确定性可以直接驱动计算边界。** BLT 用下一字节熵决定 patch 边界，是目前最直接的证据。
2. **边界也可以由任务损失端到端学出。** H-Net 根据相邻隐藏状态的变化学习动态 chunk，但它测量的是表示差异，不是校准后的预测不确定性。
3. **固定 sink 可以纯粹由 causal 结构产生。** P0-Sink、Why First Token 和 Massive Activations 表明，位置 0 的特殊性首先来自 causal mask、持续可见性、normalization 和训练上下文分布，而非 token 语义。
4. **结构形成与语义载荷必须分开。** Catch–Tag–Release 和 ViT Registers 显示，一个结构性产生的槽位随后可以承载真假、情感或全局图像信息。
5. **没有自回归时 sink 仍会出现，但角色改变。** Diffusion Transformer 中的 sink 是动态、非 P0 的轨迹路由点；小预算删除会显著改变图像实现，却不损害所测语义对齐指标。

因此，当前最稳妥的总命题是：

> 模型在“需要新增昂贵计算、需要阻止无效混合、或需要临时存放全局状态”的位置形成边界或汇点。预测不确定性是其中一种明确可用的触发信号，但不是唯一原因；语义可以影响或进入这些槽位，却不是所有槽位形成的必要条件。

“承诺”仍是解释性概念。没有一篇论文直接测量了模型的承诺状态。BLT 的下一字节熵是最接近的操作化代理：低熵意味着局部续写已基本确定，高熵意味着应增加昂贵计算。

## 证据链

### 1. BLT：预测熵直接控制计算边界

BLT 的边界信号是独立 causal byte LM 给出的下一字节熵。全局高熵或相对熵上升的位置开启新 patch；可预测片段被合并，昂贵 latent Transformer 在困难位置运行得更密。

这直接支持“预测不确定性驱动计算分辨率”，但证据边界需要收窄：

- router 与主模型分开训练，并非端到端形成；
- 阈值由目标平均 patch size 校准，推理时还能另行修改；
- repetition 会造成 entropy drift，迫使作者加入 newline reset 等修正；
- space patching 是相当接近的竞争基线；
- 论文没有证明熵边界等同于语义、推理步骤或认知承诺边界。

结论强度：**强支持“不确定性是有效计算分配代理”；不支持“熵揭示了唯一或真实语义边界”。**

详见 [BLT 阅读笔记](2412.09871-blt/reading-notes.md)。

### 2. H-Net：端到端边界来自表示变化，而非显式熵

H-Net 用当前与前一 causal hidden state 的投影余弦差异作为 boundary score，并通过硬阈值选取进入更深层级的位置。自回归预测损失决定边界是否有用，ratio loss 则约束平均压缩率。

它补上了 BLT 的关键缺口：边界模型与主模型联合训练，且可递归形成多层 hierarchy。英语中第一层常接近词边界，第二层偶尔形成短语；中文、代码和 DNA 上的收益说明它不只是复刻空格规则。

但 H-Net 没有证明 router score 是预测不确定性：

- score 是表示差异，论文明确没有把它当作正式概率；
- 总计算预算仍由 ratio target 外部指定；
- 语义边界证据主要是可视化，没有标注边界或控制语义／表面统计的干预；
- 性能比较同时包含 Mamba、normalization、宽深度和参数分配变化。

结论强度：**强支持“任务损失能学出有用的内容依赖边界”；仅弱支持“这些边界对应语义变化”，不支持“它们就是预测熵”。**

详见 [H-Net 阅读笔记](2507.07955-h-net/reading-notes.md)。

### 3. P0-Sink：固定首位汇点可由 causal mask 纯结构地产生

P0-Sink Circuit 给出最具体的结构机制：causal mask 使位置 0 只能注意自身，始终有 `p0=1`，因此其 attention output 不混合其他位置，并具有不同的范数／方差统计。随后 MLP 检测并放大这一不对称，把 P0 推向跨输入稳定的高范数方向，触发后层 sink。

去掉 `[BOS]` 后，LLaMA 的 sink 在浅层重新形成，说明固定 token 语义不是必要条件。Mistral 的 sliding window 会破坏 P0 的持续可见性，移除 `[BOS]` 后 sink 基本消失，反过来限定了机制：它依赖 full-context causal visibility，不是任意架构的普遍定律。

TNLM Loss 和 Force-sink Masking 表明人为加速／写入这一结构在较早训练阶段可改善结果，但性能证据弱于形成机制证据：实验多为单 seed，且方法同时改变损失或表示子空间；100B-token 设置也不是所有变体都优于 baseline。

结论强度：**强支持“P0 sink 的形成无需语义”；中等支持“早形成 sink 有训练价值”。**

详见 [P0-Sink 阅读笔记](2603.06591-p0-sink-circuit/reading-notes.md)。

### 4. Why First Token 与 Massive Activations：结构性汇点的功能是控制 mixing 和 routing

Why First Token 把首位 sink 解释为近似 no-op：head 将注意力投向低 value-norm 的首 token，可减少 residual update，避免深层长上下文中的 over-mixing。相同训练 tokens 下，context 越长，sink 越强；没有固定 BOS 时，sink 仍会落到第一个实际 token。

Massive Activations 进一步把 massive activation 与 attention sink 解耦：

- 更改 normalization 可以大幅消除 spike，同时保留 sink；
- head dimension 会系统控制 sink 比例；
- representation-conditioned gate 可近乎替代 sink；
- 只在长位置计算训练损失时，sink ratio 几乎坍缩，而 perplexity 可保持接近。

两篇合起来说明，首位 sink 更像 decoder-only 模型为局部依赖、无效远程上下文过滤和条件 no-op 建立的结构路由设施，而不是语义边界。

需要保留的限制是：推理时直接删除 BOS 会造成训练分布外输入，性能下降不能完全归因于 sink 本身。

详见 [Why First Token](2504.02732-why-first-token/reading-notes.md) 与 [Massive Activations](2603.05498-massive-activations-and-sinks/reading-notes.md)。

### 5. Catch–Tag–Release：结构槽位可以写入语义标签

Catch–Tag–Release 给出不同层次的结论。prompt-dependent sink 的 value 会被复制到关注它的 token，形成共享低维 tag；后层可以依据 tag 选择、检索或聚合某组 token。probe 能从 tag 中高精度读出真假和情感，玩具任务也能训练出以 `[SEP]` 为动态边界的 catch–tag–release 方案。

这说明“sink token 表面无语义”不等于“sink 机制不处理语义”。更准确的拆分是：

- sink 的形成条件可以是位置、标点或结构边界；
- sink 的 value/tag 可以承载任务相关语义；
- 结构触发与语义载荷是两个不同问题。

真实 LLM 的证据仍以 PCA、方差解释和 probing 为主。论文没有在真实任务中删除 tag 后证明其因果必要性，理论构造也只证明充分性。

结论强度：**强支持“sink 可承载语义”；弱支持“该语义机制对真实推理必不可少”。**

详见 [Catch–Tag–Release 阅读笔记](2502.00919-catch-tag-release/reading-notes.md)。

### 6. ViT Registers：内部计算槽位不依赖语言或自回归

大型 ViT 会挑选少量低信息背景 patch，丢弃其局部位置与像素信息，并将其改作全局图像信息的 scratchpad。加入不携带输入、输出也不参与任务的显式 register 后，高范数行为完全转移到 register，原 patch 伪影消失。

这是对“所有 register/sink 都来自 causal P0”的关键反例。双向视觉 attention 中同样存在把冗余 token 改作内部计算槽位的需求。它支持更一般的资源分配原则：当模型需要全局工作空间而架构没有专用槽位时，会征用低价值输入位置。

它与 LLM attention sink 的测量对象并不相同，不能证明二者具有同一电路；但足以证明 register-like 功能不依赖 AR。

结论强度：**强支持“内部 scratchpad/register 功能跨模态、非 AR 存在”。**

详见 [ViT Registers 阅读笔记](2309.16588-vit-registers/reading-notes.md)。

### 7. Diffusion Sinks：无 AR 时 sink 从固定锚点变成动态轨迹路由

Diffusion Transformer 中仍存在 attention concentration，但它与 causal LLM 不同：sink 随 layer 和 denoising timestep 变化，几乎不在 index 0，并主要落在少量 text-conditioning keys 上。

作者对 score path 和 value path 做配对、训练外抑制。标准 top-1 删除将 sink attention mass 降到接近零，却不降低 CLIP-T、ImageReward 或 HPS-v2；但输出的布局、颜色、视角和风格会显著变化，而且变化远大于等预算随机 masking。更强 top-k 干预下 HPS-v2 开始出现 dose-dependent 下降，CLIP-T 仍较稳健。

因此 sink 在 diffusion 中并非无功能，而是其功能从 AR 的持久 memory/no-op anchor 转为可替代的轨迹路由。高 attention mass 与粗粒度语义对齐的必要性发生了解耦。

结论强度：**强支持“sink 现象不依赖 AR”；强支持“AR 与 diffusion 中的 sink 功能不同”；不支持“diffusion sink 可以任意删除而无代价”。**

详见 [Diffusion Sinks 阅读笔记](2605.09313-sinks-in-diffusion-transformers/reading-notes.md)。

## 统一模型

这些结果可以放在两个相互独立的轴上理解。

| 论文 | 槽位／边界如何被选中 | 被选中后做什么 |
|---|---|---|
| BLT | 下一字节高熵或熵上升 | 增加昂贵 latent compute |
| H-Net | 相邻 causal 表示差异 + 任务梯度 | 进入更深层级计算 |
| P0-Sink | causal mask 的位置零不对称 | 建立稳定 sink/register |
| Why First Token | 首位持续可见、长上下文 mixing 压力 | 近似 no-op，抑制过度混合 |
| Massive Activations | pre-norm、head capacity、短上下文训练 | 局部 routing，忽略无用远程上下文 |
| Catch–Tag–Release | prompt-dependent sink／结构边界 | 写入并读取语义 tag |
| ViT Registers | 低局部信息、可牺牲背景 patch | 存储全局图像信息 |
| Diffusion Sinks | timestep/layer 动态 incoming mass | 控制生成轨迹的具体实现 |

最重要的区分是：**选择机制不等于承载内容。** 一个槽位可以由纯结构原因形成，随后承载语义；也可以由预测不确定性选择，但只作为计算预算控制器。把两者混为一谈，会同时高估“语义边界”证据并低估结构机制。

## 当前证据能支持与不能支持的命题

### 可以支持

- 下一步预测熵是有效的动态计算分配信号。
- 端到端训练可以学出跨语言和模态适用的动态 chunking。
- causal P0 sink 的形成不要求固定 BOS 语义。
- register/sink-like 槽位可以承载任务相关语义或全局信息。
- register-like 内部工作空间不依赖自回归。
- sink 的功能随 causal AR、双向 ViT 和 diffusion inference 改变。

### 不能支持

- 所有 attention sink 都是预测不确定性或承诺边界。
- BLT 的熵边界就是语义、词、推理步骤或认知承诺边界。
- H-Net router score 是校准的不确定性概率。
- P0 sink 完全没有任何可解码语义；论文只证明语义不是其形成所必需。
- Catch–Tag–Release 已证明真实 LLM 推理依赖 tag。
- diffusion sink 没有功能，或可以无限强度删除而不影响质量。
- patch boundary、attention sink、massive activation 和 visual register 是同一种底层电路。

## 最需要补的实验

若要把这条证据链推进到“承诺驱动边界”的强结论，最小且关键的实验是：在同一端到端模型中同时记录下一步熵、表示变化、边界决策和后续计算收益，并分别操纵语义变化与表面可预测性。在严格相同的实际计算预算下比较 entropy、learned router、random、space 和人工语义边界，测试哪种边界最能预测“增加一次高层计算的边际收益”。

这会直接区分三种目前仍混在一起的解释：预测不确定性、语义状态变化，以及纯结构性的工作空间需求。
