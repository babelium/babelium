# G3 行:diffusion 步进

**取材方式**:论文原文式子。Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239。式号为原文所印。

**本行的判断点**(填表前定的):每步交出的是连续态还是地址,这决定它算不算一次换算。**判定结果:步进不是一次换算。** 下面是理由与代之而来的定位。

---

## 六格(按「步进」读)

### 格 1 — 候选换算点

原文式 (11) 给出均值,实际抽取写在 3.2 节正文与 Algorithm 2 第 4 行:

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\Big) + \sigma_t \mathbf{z},\qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$

(Algorithm 2 写 $1-\alpha_t$,式 (11) 写 $\beta_t$,同一量,因原文定义 $\alpha_t := 1-\beta_t$。)

### 格 2 — 交给谁

下一步的 $\boldsymbol\epsilon_\theta(\cdot, t-1)$,即同一个网络。

### 格 3 — 交出什么(判定所在)

**同空间的连续张量。** 原文:"$\mathbf{x}_1,\dots,\mathbf{x}_T$ are latents of the same dimensionality as the data $\mathbf{x}_0$",以及 "This ensures that the neural network reverse process operates on consistently scaled inputs"。

关键否定事实:这里**没有地址集,没有 $\sim$,没有 $\mathrm{anc}$**。$\mathbf{x}_{t-1}$ 不是某一类等价内容的代表,它就是内容本身。$\sigma_t\mathbf{z}$ 是随机性,但不是「在若干合法代表中取定一个」——没有若干代表可取。

于是:**步进缺 $\mathcal{A}$、$\sim$、$\mathrm{anc}$ 三个分量,不是签名 $\mathcal{I}$ 的一次实例。** 我原以为它是「交出连续量因而纤维部分可见」的一行,这个说法是错的:纤维部分可见的前提是有纤维,而无商则无纤维。

### 格 4/5/6 — 不适用

无 $\mathcal{A}$ 则格 5 的 $\Phi$ 无从定义(「下游读什么」问的是「下游能否分辨同一地址下的不同内容」,而这里两点不同就是不同),格 6 同理。三格留空。

---

## 换算实际落在哪里

同一篇论文里有一处**是**换算,原文式 (13),$t=1\to 0$ 的离散解码器:

$$p_\theta(\mathbf{x}_0\mid\mathbf{x}_1) = \prod_{i=1}^{D}\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)}\mathcal{N}\big(x;\mu_\theta^i(\mathbf{x}_1,1),\sigma_1^2\big)\,dx$$

边界 $\delta_+(x)=\infty$ 若 $x=1$ 否则 $x+1/255$;$\delta_-(x)=-\infty$ 若 $x=-1$ 否则 $x-1/255$。原文称其为 "an independent discrete decoder derived from the Gaussian"。

按六格重读这一处:

| 格 | 内容 | 出处 |
|---|---|---|
| 1 换算点 | 式 (13) 的分箱 | 原文式 (13) |
| 2 交给谁 | 似然项 / 最终输出 | 式 (13) |
| 3 交出什么 | $\{0,\dots,255\}$ 中的整数 | 原文 "image data consists of integers in $\{0,\dots,255\}$ scaled linearly to $[-1,1]$",反向即分箱 |
| 4 接收方读什么 | 落在该箱内的全部概率质量(积分) | 式 (13) 的积分上下限 |
| 5 $\Phi$ | 箱指标的函数,**外加**箱内积分值 | 由 3、4 推 |
| 6 纤维可见性 | **可见**,通道是式 (13) 的积分限:整条箱区间被读入 | 由 3 推 |

$[-1,1] \to 256$ 箱是真正的商,纤维是箱区间,而目标函数读的不是箱心而是整箱的积分——纤维**完全可见**,不是部分可见。

另有一个附带事实值得登记:原文说最终展示 "we display $\boldsymbol\mu_\theta(\mathbf{x}_1,1)$ noiselessly",即最后一步**不抽样**。$\mathrm{sel}$ 在这里退化成恒等。

---

## 本行结论

- 步进层面:不是签名的实例,缺三个分量。三格留空。
- diffusion 的换算在式 (13) 的像素分箱,纤维**完全可见**(积分而非取箱心),这是 G1(完全不可见)、G2(分裂)之外的第三种答案。
- 这一行是我预判错的第二处:预判「diffusion 与 RAG 同属交出连续量、纤维部分可见」,实际是「步进无纤维 + 分箱纤维全见」。

## 与理论的关系

diffusion 并非不含签名实例——它的 attention 层依然是 $\tau$ 的一次代入(锚点 = keys),ICML 2026 那批 diffusion sink 证据落在那里,不在步进。所以本行的空格不削减覆盖面,只是把 diffusion 的换算位置从步进移到两处:分箱,与层内 attention。

## 待钉

- 式 (11)(13)、Algorithm 2 第 4 行经 ar5iv 渲染版读取,S6 引用前需回 PDF 核印刷式号。
