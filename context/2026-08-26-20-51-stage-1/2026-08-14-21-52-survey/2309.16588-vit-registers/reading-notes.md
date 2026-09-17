# Vision Transformers Need Registers

## 论文身份

- arXiv:2309.16588；ICLR 2024
- 作者：Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski
- 主题：视觉 Transformer 的高范数伪影 token、内部计算复用与 register token 干预。

## 第一遍速读

核心判断：论文发现 ViT 会把低信息背景 patch token 改作内部全局计算槽位；显式加入 register token 后伪影消失且密集预测改善，是无语言、无 AR 条件下结构性“空闲槽位”机制的关键对照，`read_deeper = true`。

摘要的主结果原文：“The artifacts correspond to high-norm tokens appearing during inference primarily in low-informative background areas of images, that are repurposed for internal computations.”（PDF p.1，Abstract）

负重图表：Figures 3–5（高范数 token 的出现条件、局部信息丢失）；Table 1（全局信息 probe）；Figure 6–8（显式 registers 及数量消融）；Table 2–3（下游与 object discovery）；Figure 9（register attention）。结论报告显式附加、无需输出的 token 会完全移除伪影。

## 第二遍理解

论文发现大而训练充分的 ViT 会在中层挑选约 2% 的 patch token，将其局部位置/像素信息抹去并写入全局图像信息；这些 token 多来自与邻域相似的背景区域，输出范数约为普通 token 的十倍。作者把它解释为模型借用“可牺牲”的输入槽位做内部 scratchpad。显式加入不携带输入信息、输出也不用于任务的 learnable register tokens 后，高范数行为从 patch token 完整转移到 registers，原 patch 伪影消失，密集预测和 object discovery 多数改善。这一干预说明内部计算槽位是真实需求，但槽位落在背景 patch 上只是缺少专用空间时的副作用。

## 关键实验与限制

- 形成条件：DINOv2-g 的 outliers 约占 2.37%，约在 40 层模型的第 15 层分化、训练三分之一后出现，且仅 ViT-Large 及以上出现（PDF pp.3–4，Figures 3–4）。
- 信息 probe：outlier 的位置预测 top-1 为 22.8（normal 41.7），平均位置误差 5.09（normal 0.79），像素重建误差 25.23（normal 18.38）；但单 token 图像分类在多数数据集远强于 normal，接近 `[CLS]`（PDF pp.4–5，Figure 5/Table 1）。
- 核心干预：1 个 register 已使可见伪影消失，4 个为默认；register 模型的高范数完全转移到 registers，Aircraft global probe 为 71.1，接近原 outlier patch 的 73.3（PDF p.7 Figure 8；pp.14–15 Figure 15/Table 4）。
- 跨训练范式：DeiT-III、OpenCLIP、DINOv2 均消除 norm outlier；DINOv2 dense task 与 LOST 大幅改善，但 OpenCLIP 的 LOST 反而略降，说明“去伪影”不保证每个下游指标改善（PDF pp.6–8，Tables 2–3；p.13 Appendix C）。
- MAE 无此伪影，作者推测因其 patch-local reconstruction loss 不需要全局聚合；这是观察性对照，且 MAE 的冻结表示性能较低（PDF p.16，Appendix E）。
- 触发原因未被完全确定；阈值 150 是手选且跨模型变化。低信息背景只是统计倾向，输入位置还受 object-centric 数据与位置插值伪影影响（PDF p.3，§2.1；p.5，§2.2；p.12 Appendix A）。

## 统一问题与精确证据

### 1. register / sink-like token 由什么触发，承担什么计算功能

隐式 register 倾向在与邻域高度相似、局部信息冗余的 patch 上形成；还需要足够模型容量、训练时长，以及要求全局信息聚合的目标。其功能是丢弃该 patch 的局部身份，充当跨图像 token 的全局信息存储、处理与检索槽位。

> “The model learns to recognize patches containing little useful information, and recycle the corresponding tokens to aggregate global image information while discarding spatial information.”（PDF p.3，Introduction）

> “Large, sufficiently trained models learn to recognize redundant tokens, and to use them as places to store, process and retrieve global information.”（PDF p.5，§2.2）

### 2. 支持语义、不确定性还是纯结构解释

不支持预测熵或不确定性触发。证据是“局部冗余/低信息”与“全局可读信息”的组合：patch 的具体视觉语义被牺牲，但槽位承载图像级语义。机制本身是结构性的资源分配，写入内容则具有全局语义。

> “Outlier tokens have much lower scores than the other tokens, suggesting they are storing less local patch information.”（PDF p.4，Figure 5 caption）

> “We see that outlier tokens have a much higher accuracy than regular ones, suggesting they are effectively storing global image information.”（PDF p.5，Table 1 caption）

### 3. 消融或干预因果证据

显式 registers 是强干预：它们不提供输入信息、其输出不参与任务，却吸收全部高范数行为，使原 patch outliers 消失；数量消融显示一个槽位已足够去除伪影。这支持“模型需要内部计算槽位”而非“特定背景语义导致异常”的因果解释。

> “The tokens we add to the sequence add no information, and their output value is not used for any purpose. They are simply registers where the model can learn to store and retrieve information during the forward pass.”（PDF p.9，§4）

> “With registers, the norms of patch tokens do not contain outliers anymore, and the high-norm tokens are entirely contained in the set of registers.”（PDF p.14，Appendix D.1）

> “As a result, we conclude that the behavior leading to high-norm outliers in the model is effectively absorbed in the registers.”（PDF p.14，Appendix D.1）

### 4. 对 AR 依赖与跨模态对照的意义

这是关键非 AR 对照：双向视觉 self-attention、监督/文本监督/自监督三类训练都出现隐式计算槽位，因此“把低价值 token 改作 scratchpad/register”不依赖自回归或语言序列。它与 LLM attention sink 不是同一测量对象，不能证明所有 sink 同源；但足以否定“内部寄存器功能必然来自 causal first-position 特权”的强说法。

> “The mechanism implemented through memory tokens already appears naturally in Vision Transformers; our study shows that such tokens allow us not to create but to isolate this existing behavior.”（PDF p.9，§4）

> “We try it on three different state-of-the-art training methods for supervised, text-supervised, and unsupervised learning.”（PDF p.6，§3.1）

> “We have not been able to fully determine which aspects of the training led to the appearance of artifacts in different models.”（PDF p.5，§2.2）
