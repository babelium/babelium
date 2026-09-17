# FIB-1 · 最近先行工作对账(第一批)

> 2026-08-26 · 读了四篇的定义与定理陈述(PDF 抽文,非摘要)

## 结论先给

**「代价应当在下游真正读取的坐标里度量」这个想法,2026 年正在 KV cache 量化那一支被独立地、形式地做出来。我们两个主设计各有直接先例。**

存活的增量只剩三条:跨系统的同一套账、零核作为一般结构刻画、$\Phi$ 从**权重**导出而非从校准数据导出。**第三条是与全部先例的唯一区别,而它恰好是 F0 纪律的内容。**

---

## 一、HeadQ(arXiv 2605.03562v2,2026-05,独立作者)—— 最近的一篇

> "persistent cache error should be measured in model-visible coordinates"

**Theorem 1(Fixed-query representation of visible key error)** 就是零核定理在注意力键上的实例:

$$N_q=\{\mathbf{1}\mu^\top+N:\ q^\top n_j=0\ \forall j\}$$

两个键扰动 $\delta K,\delta K'$ 若差在 $N_q$ 内则诱导同一注意力分布,故「行为可见的扰动空间」是商 $(\mathbb{R}^d)^n/N_q\cong\mathbb{R}^n/\mathrm{span}\{\mathbf 1\}$。

**逐项对上我们的记法:**

| HeadQ | 我们 |
|---|---|
| null set $N_q$ | 零核 |
| behaviorally visible perturbation space | $\Phi$ 看得见的那部分 |
| 取商 $(\mathbb{R}^d)^n/N_q$ | 商映射,纤维内位置对下游不可见 |
| softmax-null / attention-inert | 落在零核里 |

**Corollary 1(Storage MSE does not order key behavior)** 就是「代价是 $\Phi$-相对的」那个论证,**而且比我们计划的强**:它证明存在存储 MSE 任意大而注意力完全不变的扰动,故存储 MSE 一般地无法排序行为损害。

**Proposition 2(Fixed-attention value surrogate)** 是 $\Phi$-对齐量化的值侧版本:从注意力读出推出 $A^2$ 加权 token 失真 $\sum_j p_j^2\lVert\Delta v_j\rVert^2/d$,并用它定值侧量化策略。

**我们计划的实验它已经做了:** null-space interventions、same-budget counterexamples、cache-only matched-MSE 面板,六个模型(GPT-2 124M 到 Mistral 1.1B),结论是「downstream damage follows visible burden rather than storage burden」。

**它没有的:** 只有 KV cache 一个界面;query 基由 **calibration-learned query-PCA** 得到,不是从权重取;无跨系统对照;无组装算子;无分词器 / VQ / AR 采样。

---

## 二、KV Cache VQ with Attention-Preserving Transforms(arXiv 2608.04074)—— FIB-4 的直接先例

把 KV cache 量化写成 transform coding,失真定为注意力乘积的误差而非重建误差:

$$\lVert QK^\top-Q\hat K^\top\rVert_F^2=\sum_j(k_j-\hat k_j)M_q(k_j-\hat k_j)^\top,\qquad M_q=Q^\top Q$$

他们自己称之为 "input-weighted quadratic distortion with constant sensitivity matrix"。**这就是 FIB-4 那个「按下游读出敏感度加权的量化误差」,而且他们给了闭式最优变换。**

**零增益条件他们也有:** "Orthogonal transforms are suboptimal for the high-resolution regime unless $M_q\propto I$"。**这正是我们准备预注册的那一条(各向同性时增益应为零)。**

**关键区别:变换全部来自 calibration statistics**——key 均值、key 协方差 $\tilde S_k$、query 二阶矩 $M_q=Q^\top Q$、score 二阶矩 $M_s=S^\top S$。投影权重 $W_Q,W_K,W_V$ 只用于定义 q/k/v,**不参与构造变换**。理论基础是经典 rate-distortion(Berger 1971)、transform coding(Goyal 2001)、Zador 1982、以及 **Linder 1999 的 non-difference distortion / companding**。

**它没有的:** 只有 KV cache;无权重级推导;无跨界面;自己声明 2 bit 下「use the theory to guide our design rather than as a guarantee」。

---

## 三、V-information(arXiv 2002.10689,ICLR 2020)

**Definition 1(Predictive Family)** 要求 optional ignorance;$H_\mathcal{V}(Y|X)=\inf_{f\in\mathcal V}\mathbb E[-\log f[x](y)]$;$I_\mathcal{V}(X\to Y)=H_\mathcal V(Y|\emptyset)-H_\mathcal V(Y|X)$。

**$\mathcal V$ 从哪来:明确是建模选择,且他们把这一点当卖点。** 原话:$\mathcal V$ 是 "a set of predictive models the agent is allowed to use, e.g., due to computational or statistical constraints",并写 "is explicit about the assumptions (as a feature instead of a bug)"。实践里逐应用挑(结构学习按边挑、基因网络用三阶多项式、视频用 PixelCNN++)。

**零值刻画是关于数据的,不是关于架构的:** Proposition 2.3 说 $X\perp Y$ 时 $I_\mathcal V=0$。**而零核定理说的是「$\Phi$ 看不见纤维」——那是架构的性质,与数据无关。** 这是一个实质区别。

**另两处方向相反:** V-information 可由计算创造(违反数据处理不等式);我们的串联次可加是反方向。**全文不含 quantization / discretization / tokenization。**

**故对审稿人那个问题的答案:** V-information 问「受限观察者能用多少信息」,我们问「一次商映射抹掉的东西下游读不读得到」;前者的族外生于建模选择,后者的族由架构强制。这个区别站得住。

---

## 四、IPM 族(arXiv 0901.2698)

$\mathrm{Cost}_\Phi$ 就是取 $\mathcal F=\Phi$、$P=\kappa\tau p$、$Q=p$ 的 IPM。标准特例:有界函数 → 全变差;1-Lipschitz → Wasserstein-1;有界 Lipschitz → Dudley;RKHS 单位球 → MMD。该文主结果:全变差是唯一既是 $\phi$-divergence 又是 IPM 的非平凡量。

**由上确界的单调性:$\mathcal F_1\subseteq\mathcal F_2\Rightarrow\mathrm{IPM}_{\mathcal F_1}\le\mathrm{IPM}_{\mathcal F_2}$。故我们结论第 58 条那句「含指标函数时上确界即全变差」是 IPM 的标准特例,须改为引用而非声称。**

**IPM 文献不讨论怎么选 $\mathcal F$,它把 $\mathcal F$ 当给定。故「$\Phi$ 从架构导出」在 IPM 这一支里是空白,不构成重复。**

---

## 五、两篇以为会冲突、读完是盟友

**Is Hierarchical Quantization Essential(2601.22244):** 单级 VQ 在匹配预算下追平层级 VQ 的重建。但**测的是多尺度拼接而非残差 VQ**,且**全文无逐级误差与总误差的理论**。其核心论证(上层是下层的确定性函数,故不提供额外重建信息)是零核推理的特例;其结论方向与我们从串联次可加推出的「塔不比部件更贵」一致。**盟友。** 规模:ImageNet 256²、1.1–1.7M 参数、4×A100。

**Task-Driven Semantic Quantization(2502.17842):** 用冻结 OneFormer 的分割图散度替代像素 MSE。**权重来自前向任务损失反传,不来自架构结构**;无零增益的理论刻画;读出是另一个任务头而非同一模型的下一层。**与 FIB-4 的区别清晰。**

---

## 六、对 FIBER 轮的直接后果

**1. FIB-2 的主结果须改写。** 「从架构推出可见量 → 证明朴素度量排不了序」这个论证在 KV cache 上已由 HeadQ Corollary 1 以定理形式完成。我们的增量只能是:**跨多个界面的同一推导 + 只用权重不用校准数据**。

**2. FIB-4 的零增益条件不能声称为新。** $M_q\propto I$ 时无增益已在 2608.04074 里。我们的版本要么是它的跨界面推广,要么是「用权重级替代校准统计」。

**3. 结论第 58 条须改为引用 IPM 标准结果。**

**4. 必须补 rate-distortion 那一支的定位**(Berger 1971、Zador 1982、Linder 1999 的 non-difference distortion / companding)。**那是 2608.04074 的整个理论基础,而我们的证据链里一篇都没有。**

**5. 真正独有的只剩三条:**

- 同一套账跨 AR 采样 / VQ / 分词器 / 检索 / 层级粒度(先例全部锁在单一界面内)
- 零核作为**与载体和数据无关**的一般结构刻画,带组装算子(串联次可加、并联未证)
- **$\Phi$ 从权重导出而非从校准数据导出**——这是与全部先例的唯一区别

**第三条同时是最大风险:若权重级推导预测不了实测次序,则我们只是用了一个更弱的信息源,那不是贡献。**

---

## 七、这一支扫全的结果(2026 年 KV cache 量化)

这支不是两篇,是一片。已定位:

| 论文 | 日期 | 失真定为什么 | 推导输入 |
|---|---|---|---|
| HeadQ(2605.03562) | 2026-05 | 分数可见的键误差(模去常数移位) | **calibration-learned query-PCA** |
| Runtime-Certified Bounded-Error Quantized Attention(2605.20868) | 2026-05 | 注意力近似误差的运行时证书 | — |
| Sound Runtime Risk Observability(2607.28699) | 2026-07 | 运行时风险门控 | — |
| Attention-Preserving Transforms(2608.04074) | 2026-08 | 注意力乘积误差,闭式最优变换 | **calibration statistics** |
| Through the Lens of Transform Coding(2608.14191) | 2026-08-14 | attention-aware distortion,可加分解 | **calibration set 分配比特** |

**模式是硬的:全部用校准数据或激活统计。** 2608.04074 明写投影权重 $W_Q,W_K,W_V$ 只用于定义 q/k/v、**不参与构造变换**;2608.14191 的比特分配在 calibration set 上做。

**另一支(纯权重,但读的不是下游):** SVD-Based Weight Preservation(2512.01343)从权重 SVD 定哪些权重该留高精度,自称 "data-free, structure-aware"。**但它测的是权重的重要性,不是下游读什么**;无任何界;**无零空间 / 不可见方向的声称**(残差权重只是「低优先级」,不是「按构造无害」);验证是三个 GLUE 任务上的准确率,结论是相关性而非推导。

**跨模态那一支(UniAR、MergeTok、UniTok、Kelix 等)是造系统,不是代价理论。不构成碰撞。**

**故缺口确认存在:「从权重导出下游读出结构 + 零核刻画 + 跨界面」这三者的合体,目前没有。**

---

## 八、扫全之后,存活的增量

| | 先例 | 我们 |
|---|---|---|
| 在可见坐标里度量 | **已有**(2026 多篇) | 不能声称为新 |
| 单界面的零核/商刻画 | **已有**(HeadQ Thm 1) | 不能声称为新 |
| 从读出算子推加权量化 | **已有**(2608.04074 闭式) | 不能声称为新 |
| 零增益条件 | **已有**($M_q\propto I$) | 不能声称为新 |
| **同一套账跨 AR / VQ / 分词器 / 检索 / 层级粒度** | 无(全部锁在单界面) | **存活** |
| **零核作为与载体、与数据无关的一般结构 + 组装算子** | 无(单界面实例) | **存活** |
| **$\Phi$ 从权重导出,不用校准数据** | 无(全部用校准) | **存活,但见下** |

**第三条的风险已经具体化:校准级推导现在是既有基线。** 若权重级推导的预测能力不如校准级,那我们不是「多了一条纪律」,是「用了一个更弱的信息源」。

**故 FIB-2 必须加一条对照臂:权重级推导 vs 校准级推导 vs 两个零模型。** 四者的预测能力一起报。这样即使权重级弱于校准级,结果仍是一个有内容的发现(「$\Phi$ 的可见性有多少能从架构单独读出」),而不是缺陷。**这条对照臂是本轮定位工作直接掉出来的,原设计里没有。**

---

## 九、还没读

- rate-distortion 原始结果(Berger 1971、Zador 1982、Linder 1999)
- 欠训练 token 检测那一支(P2 / FIB-6 尾需要,且要判是否与范数测量循环)
- HeadQ 与 2608.04074 之外的 2026 年 KV cache 量化文献(这一支在动,须扫全)
- 分词器理论四篇的复核(已有 deep-read 产物)
