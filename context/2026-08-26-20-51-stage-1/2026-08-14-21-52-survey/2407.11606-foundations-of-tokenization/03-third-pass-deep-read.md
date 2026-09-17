---
sop: third-pass-deep-read
tactic: keshav-three-pass
paper: arXiv:2407.11606
---

# Third-pass deep read — The Foundations of Tokenization

## 一句话重构

令字符空间为 $X=\Sigma^*$、token 序列空间为 $Y=\Delta^*$，编码器和解码器是 Markov kernel

$$
\tau:X\rightsquigarrow Y,\qquad \kappa:Y\rightsquigarrow X.
$$

真正控制统计结果的不是二者各自，而是字符空间上的往返 kernel $K=\kappa\tau$。训练目标分布是 $q^*=\tau p^*$；只要 token 级估计 $q_n\to q^*$，解码结果就必然收敛到 $Kp^*$。因此

$$
\kappa q_n\to p^* \iff Kp^*=p^*.
$$

全文最准确的解释是：**tokenization 的渐近偏差等于 encoder–decoder 往返对真实分布造成的偏移**。所谓 exactness，即 $K=I$，只是把“对这个 $p^*$ 无偏”加强为“对所有可能分布无偏”。

## 定理链复核

1. Lemma 3.1 正确。在可数空间上，概率质量函数逐点收敛到一个仍归一化的概率质量函数，会推出 $\ell_1$（即 total variation）收敛。这本质上是离散版 Scheffé 引理，附录的 Fatou 证明成立。
2. Corollary 3.0.1 正确，因为 $\|p_n-p\|_\infty\leq\|p_n-p\|_1$。
3. Lemma 3.2 正确。Markov kernel 是 $\ell_1$ 收缩映射，所以固定 stochastic map 保持这种一致性。
4. Theorem 3.1 正确，但它是一个条件化很强的 transport 结论：token 级估计已经被假定一致，定理只识别解码后留下的渐近偏差，不证明现实神经 LM 会得到一致的 $q_n$。
5. Proposition 3.1 正确。若 $Kp=p$ 对所有概率分布成立，对每个点质量分布取值即可推出 $K=I$。附录中把 stochastic map 写得近似普通函数，记号不严谨，但结论不受影响。
6. Proposition 3.2 正确且比正文摘要更重要：若 $\kappa\tau=I$，那么 $\tau(\cdot\mid x)$ 所使用的每个 token 序列都必须被 $\kappa$ 以概率 1 解码回同一个 $x$。因此不同文本的 encoder 支持集互不相交；$\tau$ 可在同一文本的多个合法切分间随机，但不能跨文本混合。

由此得到 exact tokenizer 的完整结构：$Y$ 中被 encoder 使用的部分被分成按文本索引的互不相交纤维；$\tau$ 在每个纤维内任选分布；$\kappa$ 在这些纤维上确定性地返回文本。$\kappa$ 在 encoder 支持之外可以任意定义，这正是有限样本下 spurious ambiguity 的入口。

## 容易误读的关键点

### 1. consistency 不是 sample-wise losslessness

$Kp^*=p^*$ 只要求总体分布不变，不要求每个输入文本被还原。一个非注入 encoder 可以把多个文本压到同一 token 序列，再让 stochastic decoder 按先验比例随机生成这些文本，从而恢复 $p^*$ 的边缘分布。它在论文定义下“consistent”，却没有保存单个样本的信息。

所以应区分：

- distributional consistency：$Kp^*=p^*$；
- lossless coding / exact reconstruction：对每个文本往返都回到自身，即 $K=I$；
- identifiability：仅由 $q^*=\tau p^*$ 能否唯一确定 $p^*$。

若 $\tau$ 非注入，通常存在 $p_1\neq p_2$ 但 $\tau p_1=\tau p_2$，任何固定 decoder 都不可能同时恢复二者。distribution-relative consistency 可以依赖已知先验“猜回”边缘分布，却不能解决这个不可辨识性。

### 2. “bijective BPE/WordPiece”只是受限说法

论文把 deterministic、exact tokenizer 称为 bijective，但 $\kappa=\tau^{-1}$ 只在 canonical encoder image 上成立。全空间 $\Delta^*$ 上，`t|h|e`、`t|he`、`th|e` 可以都解码成 `the`，所以常见拼接 decoder 全局上仍是多对一。正确表述应是：encoder 与其 image 之间双射；decoder 在 image 外仍可产生 alternative segmentation。

### 3. pointwise consistency 在这里并不弱

正文称逐点收敛是较弱的选择，但由于状态空间可数、极限仍是完整概率分布，Lemma 3.1 已证明它等价于 $\ell_1$/TV 收敛。它确实弱于 KL 收敛，却不是通常直觉中的“仅逐坐标、可能丢失质量”的弱保证。

### 4. 核心定理没有覆盖 tokenization 对可学习性的影响

定理不比较不同 tokenizer 的样本复杂度、模型容量、优化难度、序列长度或归纳偏置。它把这些全部压进“$q_n$ 已一致”的前提。因此它能回答“若 token 模型最终学对了，解码会到哪里”，不能回答“哪个 tokenizer 更容易让模型学对”。

正文还笼统声称 MLE/交叉熵最小化会给出 consistent estimator；这需要可实现性、模型类、可识别性、优化达到全局解及采样条件，不能对任意神经 LM 直接成立。

## ambiguity 的实际含义

字符概率必须按

$$
(\kappa q)(x)=\sum_{y\in Y}\kappa(x\mid y)q(y)
$$

计算。确定性拼接 decoder 下，就是对所有能拼成 $x$ 的 token 序列求和。只计算 canonical tokenization 的概率，一般不是字符级概率。

论文区分的三类 ambiguity 可以统一为 token 质量分散在同一字符文本的多个纤维元素上：

- deterministic tokenizer 的 LM 给 encoder image 外的切分分配质量：spurious ambiguity；
- stochastic segmentation 主动给多个合法切分分配质量：regularization ambiguity；
- token 边界被赋予语言学解释：linguistic ambiguity。

其中“softmax 无法产生零概率”只适用于未加结构 mask 的普通 softmax 模型；有限状态约束、mask 或 constrained decoding 可以把非 canonical 序列概率精确置零。

## 计算部分复核

### 确定性范围被悄然收窄

Section 3 的 $\tau,\kappa$ 都允许 stochastic map；Section 5 的 multiplicativity、kernel、preimage、bounded variation 和 subsequentiality 却把 $\kappa(\delta)$ 当作单一字符串，因此这些结果实际只覆盖**确定性 decoder 函数**。对 stochastic decoder，需要重新定义随机转导、支持关系及求和复杂度，当前证明不能直接沿用。

### finiteness

对有限词表、multiplicative 且 non-erasing 的确定性 decoder，每个 token 至少输出一个字符，所以

$$
\kappa(\delta)=\sigma\implies |\delta|\leq|\sigma|.
$$

因此每个非空文本的 preimage 有限，粗上界为 $\sum_{i=1}^{|\sigma|}|\Delta|^i$。结论正确，但只是“有限”，仍可能指数大。

空串是正文边界遗漏：multiplicativity 推出 $\kappa(\varepsilon_\Delta)=\varepsilon_\Sigma$，trivial kernel 又排除任何非空序列解码为空，因此 $\kappa^{-1}(\varepsilon_\Sigma)=\{\varepsilon_\Delta\}$。论文给出的从 $i=1$ 开始的上界在 $|\sigma|=0$ 时为 0，应单独处理或从 $i=0$ 开始。

特殊 token 若解码为空字符（控制标记、某些 BOS/EOS 处理）会破坏 trivial-kernel 假设；重复插入它们可使同一文本拥有无限多个 preimage。

### sequentiality

Proposition 5.2“multiplicative function 有 bounded variation”成立。要由 bounded variation 推出 subsequential，还需要论文明确提到的 rational-set-preserving 前提；不能单凭 bounded variation 得出有限状态实现。

对普通 deterministic decoder，结论其实更直接：有限词表中每个 token 对应固定有限字符串，本身就是一个简单 sequential transducer。真正困难的是 encoder。WordPiece 的有限最大匹配长度支持 bounded-lookahead 实现，但需要完整的 fallback/unknown 规则；BPE 的 finite-state 性则依赖 merge 规则条件，并非对所有变体无条件成立。

## 虚拟重建后暴露的隐含假设

- $\Sigma$ 与 $\Delta$ 有限，$\Sigma^*,\Delta^*$ 可数；encoder/decoder 是固定、总定义、归一化的 kernel。
- “语言模型”必须对所有有限字符串形成总质量 1 的分布，等价于 autoregressive 生成过程几乎必然终止；允许无限序列或非终止质量的模型不在框架内。
- $q^*=\tau p^*$ 被当作实际训练目标。对 stochastic tokenizer，这要求训练时的切分采样确实服从固定 $\tau$，且采样过程足以在极限中再现该 mixture。
- decoder 不随样本量、训练数据或模型参数变化；学习式 decoder 不在定理的直接设定内。
- token LM 是对完整 token 序列的归一化分布。实际按长度归一化的分数、未归一化能量、per-token perplexity 不能直接代入。
- exactness 定义在所有 $\Sigma^*$ 上，而实际 tokenizer 常只保证训练字符集或合法 Unicode/byte 输入上的往返。
- Section 5 还额外假设 decoder 确定、词表有限、每个 token 输出非空字符串。

## 可直接补出的有限样本结论

利用 Markov kernel 的 $\ell_1$ 收缩性，可以把论文的纯渐近定理加强成简单误差分解：

$$
\|\kappa q_n-p^*\|_1
\leq
\|q_n-q^*\|_1
+
\|\kappa\tau p^*-p^*\|_1.
$$

第一项是 token LM 的估计误差，第二项是 tokenizer 的不可消除往返偏差。exact tokenizer 使第二项为零；非 exact tokenizer 即使 token LM 无限好也留下该 bias。这个式子比只陈述 iff 更适合综述使用，也给出了可实证估计的分解。

## 具体改进点

1. 把“总体分布不变”和“逐样本可逆”明确拆开，避免把先验驱动的随机重建也称为 lossless。
2. 给出上述有限样本误差分解，并进一步讨论 KL、cross-entropy 或前缀概率下的对应界。
3. 把模型错设与训练过程纳入：分析 $q_n$ 收敛到模型类中的投影而非 $q^*$ 时，tokenizer bias 和 model bias 如何叠加。
4. 为 stochastic decoder 单独建立计算理论；当前 Section 5 从 stochastic framework 切到普通函数，适用范围没有醒目标注。
5. 用“canonical-image bijection”代替对 BPE/WordPiece 的全局 bijective 称呼。
6. 修正空串上界，并显式处理会被 decoder 擦除的 BOS/EOS/control token。
7. 对 WordPiece/BPE 的 subsequentiality 分开列出充分条件，不把 bounded variation、bounded lookahead 与 finite-state realizability 混成一步。

## 第三遍最终判断

这篇论文的价值不在于给出复杂新算法，而在于把 tokenization 的统计问题压缩成一个清楚的算子条件：真实分布必须是 $K=\kappa\tau$ 的不动点。它也准确揭示了 canonical tokenization 概率不等于字符概率，完整字符概率需要对 decoder 纤维边缘化。

但“Foundations”应谨慎理解。主定理是建立在 token 级一致性已成立之上的渐近恒等式；最困难的现实问题——模型错设、优化、有限样本、前缀/条件概率、随机转导计算——仍未解决。对综述最值得保留的判断是：**该文提供了 tokenization 的无偏性判据和 ambiguity 的统一语言，而不是 tokenizer 优劣或可学习性的完整理论。**

## 提取校勘

`source.md` 的 Example 4.1 最后一项把 $\sigma_3$ 错抽成了 $\sigma_1$。HTML/PDF 图中正确值是

$$
\kappa\tau p^*(\sigma_3)=0.8\neq0.4.
$$
