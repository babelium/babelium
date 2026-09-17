# F 轮实验规格：可追溯问题审计与修正方案

> 审计时间：2026-08-24 16:53（Asia/Shanghai）
> 最后复核：2026-08-24 17:37（Asia/Shanghai）
> 唯一审计对象：远端现行规格 [`context/2026-08-23-19-18-f-round-spec.md`](../2026-08-23-19-18-f-round-spec.md)
> 固定版本：`6baa3e99fc0b50f383b9865ca4209fad801cdd6f`
> 审计边界：所有“发现的问题”都必须能回指上述原始规格；本地修订稿不作为问题来源。

## 0. 先给结论

本轮逐条回到原规格、官方配置和官方实现后，确认：

- **5 个开跑阻断项**：类型系统不闭合、理论量缺少经验估计器、P6 结构事实与因果预言混淆、P3 测量对象与“不可达”命题不一致、F5 模型类别与输入域错误；
- **11 个高优先级问题**：次可加推论、`Phi` 架构路径、DeepSeek 分支定义、P5 规模/绑定共变、`glb` 伪上界、`m=16` 截断、`p>0.05`、F2 内存协议、不可达 token 梯度、伪代码一致性、复现协议；
- **1 个精度风险**：RMSNorm 的“只留方向”需要写成带条件的近似；
- **4 个需收紧的问题/风险**：P4 长度暴露、KV cache 层次、成本与工期、开跑门定义。

因此当前判定仍是：**阶段零 `NO-GO`，不应开始正式测量。** 这不是说所有想法都错，而是当前规格还不能保证“跑出的数等于理论声称的量”，也不能保证第三方能复现同一次实验。

### 0.1 一页式问题登记表

| ID | 级别 | 原规格位置 | 一句话问题 | 首要修正 |
|---|---|---|---|---|
| [B01](#b01) | 阻断 | §1.2、§1.3、§2.2 | 单 token 商与整段 tokenizer 类型混用 | 分离序列层与局部选择层，逐式类型检查 |
| [B02](#b02) | 阻断 | §2.2、§6、§7.5 | 理论 Cost 没有经验测度和估计器 | 先冻结 $\mu$，再分线性/范数项估计 |
| [B03](#b03) | 阻断 | §3、§4.4、§6.3 | 低维不能推出跨模型 Cost 次序 | 结构事实、表示干预、架构因果分开 |
| [B04](#b04) | 阻断 | §2.4、§3、§7.4 | “不可达率”与有限语料未出现率混名 | matched BPE/Unigram 才作算法因果比较 |
| [B05](#b05) | 阻断 | §6.4、§7.6 | 一个候选不是 VQ，码字直送 encoder 类型错误 | 锁定真实离散 quantizer 与合法 cycle |
| [H01](#h01) | 高 | §2.5 | 次可加被误读成插层降代价 | 删除单调推论或另证充分条件 |
| [H02](#h02) | 高 | §6.1–§6.2 | `Phi_GQA` 不符合 decoder 执行路径 | attention/residual/block 分开定义 |
| [H03](#h03) | 高 | §6.3 | DeepSeek Q/KV/RoPE 分支不足以复现 | 按官方源码冻结四个分支 |
| [H04](#h04) | 高 | §3、§4.2 | P5 规模与 tied 状态共变 | untied 主梯与 tied 副梯分开 |
| [H05](#h05) | 高 | §7.5 | `glb` 不是上界却被用作分母 | 降为诊断参照，另定稳定 scale |
| [H06](#h06) | 高 | §6.2–§6.5 | `m=16` 未定义为下界/近似 | 主量与 `Phi_16` 敏感性分析分开 |
| [H07](#h07) | 高 | §3 | `p>0.05` 被当成反驳 | 支持/反驳/证据不足三态 |
| [H08](#h08) | 高 | §7.5 | F2 示例未实现自身的内存协议 | 流式统计并做等价性测试 |
| [H09](#h09) | 高 | §2.4 | untied 输入/输出行梯度被混写 | 分 tied/untied、E/W 四类陈述 |
| [H10](#h10) | 精度 | §1.3、§7.5 | RMSNorm 只近似尺度不变 | 写明 epsilon 条件并测近零输入 |
| [H11](#h11) | 高 | §7.4、§8 | tokenizer 参数、字节域、索引与对齐未闭合 | byte-level golden tests 与固定对齐算法 |
| [H12](#h12) | 高 | 全文 | 自包含、复现和治理字段不足 | manifests、锁文件、运行入口与 provenance |
| [R01](#r01) | 条件风险 | §3、§7.4–§7.5 | P4 可能只测到长度暴露机会 | hazard/survival 作主端点 |
| [R02](#r02) | 条件风险 | §2.7、§7.8 | tokenizer cache 与模型 KV cache 混层 | 分开定义，删除离散性的必要性 |
| [R03](#r03) | 资源风险 | §5、§7.7、§9 | 工期与显存缺 profiling 依据 | 冻结负载后分层 profiling |
| [R04](#r04) | 执行风险 | §9–§10 | 缺少机器可执行的开跑门 | G0/G1/G2 gate 与目录隔离 |

### 0.2 推荐阅读顺序

- **决定现在能否开跑**：B01–B05 → R04。
- **理解理论量为何尚不可测**：B01 → B02 → H01 → H06。
- **理解 P6/架构问题**：H02 → H03 → B03。
- **理解 P3/P4/P5 的实验构念**：B04 → H04 → H07 → R01。
- **理解实现与复现风险**：B05 → H08–H12 → R02–R04。

### 0.3 每张审计卡怎样读

每个问题都固定使用同一条证据链：

1. **发现的问题**：一句话说明缺陷；
2. **计划书位置**：给出章节、精确行号和固定版本链接；
3. **原文**：保留足够上下文的原句；
4. **为什么是问题**：给出逻辑、类型、架构或统计理由；
5. **证据**：区分规格内部证据、数学推导、官方配置/源码和尚未运行的实验；
6. **解决方案**：给出可落地改法；
7. **有效性验收**：说明怎样证明修正真的有效；
8. **原规格状态**：说明原规格是否已经意识到该限制，以及问题是否仍会阻断执行。

### 0.4 证据边界

- 已做：逐行审阅固定版本的原始规格；核对 Qwen2、Qwen3、DeepSeek-V2-Lite 的官方实现；核对相关模型官方配置；做代数反例和矩阵维度检查。
- 未做：没有下载权重，没有运行 tokenizer 全量扫描，没有运行 F2–F7，没有测显存和墙钟。
- 因而本文可以确认**定义、协议、架构和统计设计问题**，不能声称 P1–P10 已被数据支持或反驳。

---

## 1. 开跑阻断项

<a id="b01"></a>

### B01. 单 token 商映射与整段 tokenizer 被混成同一个 `tau`

**发现的问题**

同一个 `tau` 同时承担“整段字符串到 token 序列”和“局部对象到单个地址”的职责，导致 `A = Omega / ~`、`tau^{-1}(a)`、`kappa tau` 在类型上不能同时成立。

**计划书位置**

- §1.2，[原规格 L42–L46](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L42-L46)
- §1.3，[原规格 L64–L66](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L64-L66)
- §2.2，[原规格 L94–L100](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L94-L100)

**原文**

> “词表 / 码本。形式上是 $\Omega$ 的一个商集 $\Omega/\!\sim$——即「把 $\Omega$ 里彼此等价的元素合并成一个」的结果。”
>
> “$\tau:\Omega\to\mathcal{A}^*$，把内容切成 token 序列。注意值域是序列 $\mathcal{A}^*$ 而非单个 $\mathcal{A}$。”
>
> “纤维(fiber)：$\tau^{-1}(a)$，即所有被映到同一个地址 $a$ 的内容。”
>
> “$\mathrm{Cost}_\Phi(p)=\sup_{\varphi\in\Phi}\big|\mathbb{E}_{\kappa\tau p}[\varphi]-\mathbb{E}_p[\varphi]\big|$。”

**为什么是问题**

若 `tau: Omega -> A*`，则它的逆像自然接受的是序列 `w in A*`，而不是未嵌入序列空间的单个 `a in A`。同时 `kappa: A -> Omega` 不能直接与输出为 `A*` 的 `tau` 复合；此处至少需要序列延拓 `kappa*`。反过来，VQ 的局部量化确实是 `q: X -> A`，但它又不是文本整段 tokenizer。

这不是记号美观问题，而是会让“纤维”“截面”“无损往返”和两项 Cost 分别作用在哪一类对象上变得不确定。

**证据**

- 规格内部即可完成类型检查，不依赖实验。
- 一个复合映射只有在前一个映射的值域与后一个映射的定义域一致时才良定义；当前 `kappa o tau` 不满足这一条件。

**解决方案**

一种直接、可类型检查的修正是拆成两个系统，不再共用一个未经限定的 `tau`：

1. 文本序列层：`T: S -> A*`，`D: A* -> S`；规范切分、token healing、`R = T o D`、非规范质量都在这里定义。
2. 局部选择层：`q: X -> A`，`e: A -> X`；VQ、AR 的 soft/hard 选择、单地址 fiber 都在这里定义。
3. 若希望保留一个统一符号，也可以显式定义 singleton embedding、序列延拓和相应复合；关键验收标准是所有映射有合法类型，而不是必须采用本文唯一一套记号。
4. 若要把文本 token 与局部选择纳入同一个定理，新增桥接命题，明确何时一个序列位置可诱导局部 `q`。

**有效性验收**

- 为全文建立“映射签名表”，逐个复合检查定义域和值域；
- 全文搜索后，不再出现未定义的 `tau^{-1}(a)`、`kappa tau` 或把 `A`/`A*` 当同一对象的式子；
- 用文本序列、VQ 局部量化各给一个最小例子，所有公式均可代入。

**原规格状态：未修，阻断。**

---

<a id="b02"></a>

### B02. 理论 Cost 没有对应的经验估计器与聚合规则

**发现的问题**

原规格定义了总体期望差，却没有规定怎样从 position、prompt 和分支观测构造该期望的经验估计器，也没有冻结聚合顺序。不同实现者可以得到不同的“Cost”，而且都能声称遵守了三臂设计。

**计划书位置**

- 理论定义：§2.2，[原规格 L94–L100](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L94-L100)
- 原规格的三臂与比值：§7.5，[原规格 L500–L529](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L500-L529)
- 有限读出与统计纪律：§6.2–§6.5，[原规格 L353–L384](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L353-L384)

**原文**

> “$\mathrm{Cost}^{\mathrm{sel}}_\Phi=\sup_{\varphi\in\Phi}\Big|\mathbb{E}\big[\varphi(\textstyle\sum_a p_t(a)\,\mathrm{anc}(a))\big]-\mathbb{E}\big[\varphi(\mathrm{anc}(a_t))\big]\Big|$。”
>
> “取什么：每个位置的 logits（算出地址上的分布）、嵌入矩阵、以及下游第一层实际读到的量。”
>
> “不同模型的维数与嵌入尺度不同，绝对代价不可比。故 P5 一律用无量纲比 $\rho$。”

**为什么是问题**

对一个固定分支、且读出族确实包含该分支输出空间的全部单位线性函数时，理论量可化为：

`C_j = || E_i[f_j(u_avg_i) - f_j(u_cmt_i)] ||_2`。

规格没有说明期望对应什么抽样测度：是 token 等权、prompt 等权、序列等权，还是某个明确的数据生成分布；也没有说明先对有符号向量差取平均，还是先取逐位置范数再平均。两者一般不同：若两个等权位置的分支差分别为 `v` 和 `-v`，理论中的有符号期望差为 0；先取逐位置 L2 再聚合则大于 0。

上述 L2 化只覆盖完整单位线性子族。原规格的 $\Phi$ 还包含 $u\mapsto\lVert u\rVert_2$ 这一非线性项，并把多个投影分支并在同一函数族中；这些项必须按原定义分别估计，不能全部套进同一个向量 L2 公式。

**证据**

- 上述 `v/-v` 是精确代数反例，无需真实数据。
- 原始理论定义的期望位于绝对值/上确界内部，但 §7 没有给出保持该顺序的样本估计公式。

**解决方案**

1. **先冻结目标测度 $\mu$**：明确 prompt、序列、位置和温度的权重。是否 prompt-balanced 由作者在开跑前决定，审计报告不替代这一选择。
2. **线性分支主端点**：对每个固定分支计算 $\lVert\mathbb E_\mu[f_j(u^{\mathrm{avg}})-f_j(u^{\mathrm{cmt}})]\rVert_2$。
3. **范数项主端点**：单独计算 $|\mathbb E_\mu\lVert u^{\mathrm{avg}}\rVert_2-\mathbb E_\mu\lVert u^{\mathrm{cmt}}\rVert_2|$。
4. **函数族聚合**：若 $\Phi$ 是上述分支的并集，按原定义取各分支理论量的最大值；跨分支归一化若要引入，必须另行定义，不能继续称原始 Cost。
5. **异质性次端点**：逐位置相对 L2 的中位数、分位数和分箱；命名为 `local_shift`，不得沿用理论 Cost 的名称。

**有效性验收**

- 合成 `v/-v` 样例必须得到 `theory_cost = 0` 且 `local_shift > 0`；
- 交换样本顺序不改变结果；样本复制对结果的影响必须严格符合预先声明的 $\mu$；
- 纯尺度变化样例必须只影响预期的范数项，线性分支与范数分支分别出现在输出中；
- 论文理论表只引用 `theory_cost`，异质性图明确标为次分析。

**原规格状态：没有经验估计公式，阻断。**

---

<a id="b03"></a>

### B03. P6 的维数论证不能推出跨模型经验代价次序

**发现的问题**

P6 把 MLA 的低维瓶颈作为“代价更低”的理论来源，并将跨模型比较设为检验；但两个模型上的函数族没有集合包含关系，维数较小不能推出经验 Cost 较低。跨模型差异也无法单独归因于 MLA。

**计划书位置**

- P6 预言与来源：§3，[原规格 L202、L208–L210](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L202-L210)
- 模型对照：§4.4，[原规格 L263–L272](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L263-L272)
- 原规格自己承认的硬上限：§10.1，[原规格 L691–L695](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L691-L695)

**原文**

> “P6：MLA 架构的商坍缩代价低于非 MLA 对照。”
>
> “配上同为 MoE 的 Qwen3-30B-A3B 之后，稀疏性被锁住，只剩注意力机制在变。”
>
> “P6 的归因只到「这一对上成立」，不到「MLA 使代价降低」。”

**为什么是问题**

DeepSeek-V2-Lite 的直接 KV 压缩输出为 `512 + 64 = 576` 维，矩阵形状是 `576 x 2048`，所以秩至多 576、零空间至少 1472 维。Qwen3-30B-A3B 的 stacked K/V 为 `2 x 4 x 128 = 1024` 维，矩阵形状是 `1024 x 2048`，所以零空间至少 1024 维。

“1472 大于 1024”主要是矩阵行数的算术结果，不需要权重数据。更关键的是，`Phi_MLA` 与 `Phi_GQA` 是两个不同模型上的函数族，并没有被证明构成同一输入分布上的集合包含关系；“维数更小”本身不能推出两边经验 Cost 的次序。把两边都截成 `m=16` 也只统一了方向数量，没有统一具体函数、权重、输入分布和尺度。

此外，两个模型的专家数、top-k、层数、总参数、词表和训练数据均不同，“稀疏性被锁住”这句话与同节配置表自身矛盾。

**证据**

- [DeepSeek-V2-Lite 官方配置，revision `604d5664…`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json)
- [Qwen3-30B-A3B 官方配置，revision `ad44e777…`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)
- 线性代数：`rank(W) <= min(rows, columns)`，`nullity(W) = columns - rank(W)`。

**解决方案**

1. P6a 降级为“由架构配置派生的结构事实/实现 sanity check”，不计为理论获得经验支持。
2. P6b 若只在同一已训练模型中临时加入 rank-matched bottleneck、等维随机投影或低秩适配，只能称“表示干预”，不能称架构因果实验。
3. 若要做架构因果比较，需要配对训练/微调的有无 MLA 变体，尽量固定数据顺序、训练预算与多个随机种子；若资源做不到，就明确降级为探索性结果。
4. DeepSeek/Qwen 跨模型前向保留为探索性外部对照，只报告分支结果，不写 MLA 因果结论。
5. 删除“稀疏性被锁住，只剩注意力机制”这句；改成完整混淆项表。

**有效性验收**

- 先用随机满行秩矩阵复现形状决定的 nullity 基线；
- 经验权重结果必须同时报告“超出形状基线的谱特征”，不能只报 nullity；
- 只有同模型 rank-matched 干预才能进入因果主结论。

**原规格状态：虽在 §10 承认归因上限，但 P6 的预言与杀伤条件仍未降级，阻断。**

---

<a id="b04"></a>

### B04. P3 声称比较“不可达率”，实际只能测语料条件未出现率

**发现的问题**

规格已明确承认无法枚举所有字符串，只能在有限语料上收集出现过的 token；但 P3 仍把结果命名为“Unigram 族不可达率 < BPE 族”，并把不同语料、词表和预处理条件的现成 tokenizer 当作算法族样本。

**计划书位置**

- 理论推导：§2.4，[原规格 L135–L141](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L135-L141)
- P3：§3，[原规格 L195–L200](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L195-L200)
- 分词器族：§4.5，[原规格 L274–L279](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L274-L279)
- 实际测量边界：§7.4，[原规格 L474–L480](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L474-L480)；§10.1，[L693–L695](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L693-L695)

**原文**

> “故预言：Unigram 族的不可达率应低于 BPE 族。这是跨算法族的对照。”
>
> “$\mathcal{A}_{\mathrm{reach}}$ 的定义是「出现在某个规范切分里」。枚举全部字符串不可能，故只能给下界：在语料上跑分词器，收集出现过的 id 集合。”
>
> “报的是「在该语料下未出现」，不是「不可达」。”

**为什么是问题**

有限语料未出现只给出 reachability 的不完全证据；语料扩大后集合会继续缩小。不同现成 tokenizer 还同时改变训练语料、词表大小、normalizer、pre-tokenizer、特殊 token 规则和语言覆盖，无法把组间差异归因于 BPE/Unigram 算法。

此外，`r(a) != [a]` 只证明 token 单独往返不稳定；原规格也已承认“碎裂不等于不可达”。

**证据**

- 规格自身在 L476–L480 与 P3 的命名直接冲突。
- 同一 token 在当前语料未出现，不构成“任何字符串的规范切分都不会出现它”的证明。

**解决方案**

1. `r(a) != [a]` 命名为“单 token 往返不稳定率”。
2. 有限语料结果命名为“语料条件未出现率”，并随累计 token 数报告覆盖曲线。
3. 算法因果比较必须在同一训练语料、相同词表规模、normalization、pre-tokenization、特殊 token 规则和训练预算下重新训练 matched BPE/Unigram；同时冻结训练器版本和随机种子，并使用多个训练重复。
4. 现成 tokenizer 的比较只作描述性外部结果。

**有效性验收**

- 每张表标题都含数据集与语料规模；
- 覆盖曲线显示未出现率随语料增长的变化；
- matched 实验除算法外的配置散列完全相同；
- 论文不再把有限语料未出现直接简称为不可达。

**原规格状态：已意识到测量边界，但 P3 尚未同步改写，阻断。**

---

<a id="b05"></a>

### B05. F5 的一个具体候选不是 VQ，且“码字送回 encoder”输入域不成立

**发现的问题**

F5 把 `stabilityai/sd-vae-*` 列为 VQ 候选；其中可核查的具体仓库 `stabilityai/sd-vae-ft-mse` 是 `AutoencoderKL`，不满足离散码本要求。本文不据此判定所有名称匹配 `sd-vae-*` 的仓库。另一方面，“把码字当输入送回编码-量化”通常把 latent/codebook 向量直接传给接受图像/音频的 encoder，类型不成立。

**计划书位置**

- `Phi_VQ`：§6.4，[原规格 L376–L378](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L376-L378)
- F5 候选与测量：§7.6，[原规格 L563–L572](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L563-L572)

**原文**

> “候选（⬜ 待核实仓库与解码器首层）：图像 `CompVis/vqgan-*` 或 `stabilityai/sd-vae-*` 一类；音频 `facebook/encodec_24khz`；语音离散化模型。”
>
> “码字往返：把每个码字当输入送回编码-量化，看它是否量化回自己。”

**为什么是问题**

- `AutoencoderKL` 使用连续潜变量与 KL 正则，没有 F5 所需的离散 codebook/quantizer。
- 码字 `e_k` 位于 latent/codebook 空间；encoder 的定义域通常是原始信号空间。直接调用 encoder 是类型错误。
- 若只是对 `e_k` 再做最近码字查询，在码字互异且距离计算正常时返回自己是定义上的平凡结果，不能验证自编码器往返。

**证据**

- [`stabilityai/sd-vae-ft-mse` 官方配置，revision `31f26fde…`](https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/31f26fdeee1355a5c34592e401dd41e45d25a493/config.json) 明确写 `_class_name: AutoencoderKL`。
- 类型检查：`encoder: signal -> latent`，而 `codeword in latent`；必须先由 decoder 回到 signal。

**解决方案**

1. 仅选择明确暴露离散 codebook/quantizer 的 VQGAN、EnCodec 或其他离散 codec。
2. 若测 cycle consistency，定义为 `codeword/code-grid + 必需上下文 -> decoder -> signal -> encoder -> quantizer`；部分 decoder 不能接收孤立单码字，输入结构须逐模型冻结。
3. 若测 codebook 自最近邻，只把它称为实现 sanity check，不称为往返证据。
4. residual VQ 的逐级输入必须按真实 residual 条件定义，不能把每一级当独立同域映射。

**有效性验收**

- 模型 manifest 必须含 quantizer 类名、codebook 张量路径、encoder/decoder 输入输出 shape；
- 对每个模型跑 shape-only smoke test，并记录 code grid、带宽/层级上下文和 decoder 最小合法输入；
- identity/mock quantizer 应通过，故意打乱 decoder/encoder 的对照应使 cycle 指标恶化；
- AutoencoderKL 不进入 F5 主表。

**原规格状态：候选仓库仍未核实，且测量类型错误，阻断。**

---

## 2. 其他实质问题（H10 为精度修正，其余为高优先级）

<a id="h01"></a>

### H01. 次可加不等式不能推出“多插一层商可降低代价”

**发现的问题**

原文把一个上界不等式解释成单调下降结论。

**计划书位置**

§2.5，[原规格 L166–L168](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L166-L168)：

**原文**

> “已证的一条不等式（串联次可加）：$\mathrm{Cost}_\Phi(\tau_2\tau_1)\le\mathrm{Cost}_\Phi(\tau_1)+\mathrm{Cost}_\Phi(\tau_2)$。方向与直觉相反：多插一层商可以降低总代价。这是 P6 的来源。”

**为什么是问题**

由 `C12 <= C1 + C2` 只能知道 `C12` 不超过两项之和，不能推出 `C12 <= C1`。

**证据**

数值反例：`C1=1, C2=1, C12=1.5` 满足次可加不等式，但 `C12 > C1`，插入第二层后代价反而增加。

**解决方案**

删除单调下降推论。若理论需要“插入某类商降低代价”，必须另加充分条件并单独证明；P6 不能再以次可加性作为方向性来源。

**有效性验收**

用上述数值反例做文档单元测试；全文搜索不再把 subadditivity 写成 monotonicity。

**原规格状态：未修；该句仍直接作为 P6 来源。**

---

<a id="h02"></a>

### H02. 原 `Phi_GQA` 不符合 decoder layer 的真实执行路径

**发现的问题**

原规格把 Q/K/V 与 MLP 的 gate/up 投影写成原始提交向量经过同一次 RMSNorm 后的并列即时读出，并遗漏 identity residual；官方实现不是这条路径。

**计划书位置**

§6.1–§6.2，[原规格 L343–L364](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L343-L364)：

**原文**

> “提交后的量 $u=\mathrm{emb}(a_t)\in\mathbb{R}^d$ 依次经过……五个线性投影：$W_q\hat u$、$W_k\hat u$、$W_v\hat u$，以及前馈网络的 $W_{\mathrm{gate}}\hat u$、$W_{\mathrm{up}}\hat u$。”
>
> “$\Phi$ 的定义是「提交后紧接着读到的」。”

**为什么是问题**

Qwen2/Qwen3/DeepSeek decoder layer 都先保存 residual，RMSNorm 后进入 attention；attention 输出与 residual 相加后，再经过 post-attention norm 和 MLP。因此 MLP 读的是上下文依赖的新残差状态，不是原始 `u` 的即时线性读出；identity path 又完整携带了 `u` 的差异。

**证据**

- [Qwen2 decoder layer（Transformers v4.43.1）](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py#L545-L611)
- [Qwen3 decoder layer（Transformers v4.51.0）](https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py#L190-L306)
- [DeepSeek-V2-Lite 官方实现](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py)

**解决方案**

拆成三个明确对象：`Cost_Phi_attn`（即时 attention 分支）、`Cost_Phi_res`（identity path）、`block_effect`（完整第一层传播）。其中 residual 不能只有分支名称，还要明确对应的标量函数族或距离定义。三者分别报告，不合并成“全局 Cost”，也不用分支结果替代完整下游族。

**有效性验收**

- 为每个架构注册 forward hooks，与解析公式逐元素对齐；
- 同一输入下解析 Q/K/V 与真实模块输出在预注册容差内一致；
- MLP 输入必须等于 attention 后状态，而非原始 embedding；
- 三类输出使用不同字段名和表格。

**原规格状态：`Phi_GQA` 仍与真实执行路径不一致，高优先级。**

---

<a id="h03"></a>

### H03. DeepSeek 的 Q/KV 与 RoPE 分支需要按官方实现拆开

**发现的问题**

原规格只用一个抽象 “down projection + RoPE branch” 表示 MLA，没有明确 Q 的 non-RoPE/RoPE 拆分，也没有明确 compressed KV 与 RoPE key 的拆分，无法据此复现 DeepSeek-V2-Lite 的真实 Q/KV 路径。

**计划书位置**

§6.3，[原规格 L366–L374](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L366-L374)：

**原文**

> “下一层实际读到的不是 $W_k\hat u$，是 $W_k^{\uparrow}(W^{\downarrow}\hat u)$，中间过了 512 维瓶颈。”
>
> “$\Phi_{\mathrm{MLA}}=\{u\mapsto\langle W^{\downarrow}\hat u,\xi\rangle\}\cup\{\text{rope 分支的 }W^{\mathrm{rope}}\hat u\}\cup\{\lVert u\rVert_2\}$。”

**为什么是问题**

该模型 `q_lora_rank=null`，Q 是直接投影后拆为每头 128 维 non-RoPE 与 64 维 RoPE；KV 的 `kv_a_proj_with_mqa` 则拆为 512 维 compressed KV 与 64 维 RoPE key。不同分支的维度、旋转和后续用途不同，不能用一个未定义的 `W^rope` 或完整 Q 统一旋转代替。

**证据**

[DeepSeek-V2-Lite 固定 revision 官方源码](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py) 与 [同 revision 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json)。

**解决方案**

分别定义并报告 `q_nope`、`q_pe`、`compressed_kv`、`k_pe`；所有 RoPE 变换使用原 position id。不要把这些分支与 GQA 的不同维输出强合成一个标量。

**有效性验收**

解析计算的四个张量 shape 与 forward hook 完全一致，旋转前后只改变 RoPE 子维；在 position id 固定时与官方模块输出对齐。

**原规格状态：执行路径描述不足；在补齐前不能实现 F0 声称的架构冻结。**

---

<a id="h04"></a>

### H04. P5 的规模与 tied/untied 状态共变

**发现的问题**

Qwen2.5 小模型绑定输入/输出权重，大模型不绑定，绑定断点恰好与规模增长共变。原五点梯不能单独识别“规模效应”。

**计划书位置**

- P5：§3，[原规格 L201、L208](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L201-L208)
- 主梯与断点：§4.2，[原规格 L235–L251](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L235-L251)

**原文**

> “P5：选择坍缩代价随模型规模下降。”
>
> “断点一：绑定状态与规模共变。0.5B 与 1.5B 绑定，32B 不绑定……先核实 3B/7B 的绑定值以确定断点位置，把梯子切成两段各自看趋势。”

**为什么是问题**

原规格已写明 0.5B/1.5B 为 tied；固定 revision 又确认 3B 为 tied、7B/32B 为 untied。若直接跨 3B→7B 断点拟合，趋势可能来自参数量、绑定状态或二者交互。只把单个 checkpoint 的锚对齐率当协变量不能创造缺失的独立变化。

**证据**

[3B，revision `3aab1f19…`](https://huggingface.co/Qwen/Qwen2.5-3B/blob/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/config.json)、[7B，revision `d1497293…`](https://huggingface.co/Qwen/Qwen2.5-7B/blob/d149729398750b98c0af14eb82c78cfe92750796/config.json)、[32B，revision `1818d358…`](https://huggingface.co/Qwen/Qwen2.5-32B/blob/1818d35814b8319459f4bd55ed1ac8709630f003/config.json) 已足以确认绑定断点位于 3B 与 7B 之间。14B/72B 必须在 `model_manifest` 中另锁精确 revision，并由程序断言为 untied，才可进入建议的四点主梯。

**解决方案**

1. 确认性主梯只用 untied 的 7B/14B/32B/72B，结论限定为这四个 checkpoint 的条件趋势。
2. tied 的 0.5B/1.5B/3B 单独作描述性副梯。
3. entropy 是原 P5 所主张的可能中介机制，不应简单“控制掉”后才承认 P5；应把总趋势与 entropy 中介分解作为两个问题报告。
4. prompt bootstrap 只代表 prompt 抽样不确定性，不代表模型种子或模型家族。

**有效性验收**

- 主分析不跨 tied/untied 边界；
- 图注明确只有 4 个固定 checkpoint，不能写普遍 scaling law；
- 同时报总趋势与 entropy 分层/中介分析，不把二者混为同一杀伤条件。

**原规格状态：已经发现共变，但“五点主梯 + 协变量”仍不能识别纯规模效应，未闭合。**

---

<a id="h05"></a>

### H05. `glb` 不是数学上界，不能作归一化分母

**发现的问题**

全词表均值与概率加权均值之间没有距离上的序关系；把 `glb` 命名为“上界臂”并作为比值分母没有数学保证。

**计划书位置**

§7.5，[原规格 L508–L529](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L508-L529)：

**原文**

> “上界臂：$u^{\mathrm{glb}}=\frac1V\sum_a\mathrm{emb}(a)$，与 $t$ 无关。”
>
> “$\rho=\frac{\mathrm{Cost}^{\mathrm{sel}}(\text{不提交 vs 提交})}{\mathrm{Cost}^{\mathrm{sel}}(\text{上界 vs 提交})}$。”

**为什么是问题**

全词表均值与概率加权均值之间没有距离序关系；`glb` 距离还可能接近 0，使比值不稳定。

**证据**

一维反例：令 `cmt=0`、`avg=10`、`glb=1`，则 `|avg-cmt| > |glb-cmt|`，所以 `glb` 不是上界。

**解决方案**

将其改名为“全局均值诊断参照”，只用于离流形/实现诊断；跨模型无量纲化若使用待测分支的 pooled scale，必须在看结果前冻结 scale 定义、epsilon 和近零分母失败规则。

**有效性验收**

加入上述反例；报告不再出现 upper bound；分母接近零时有预注册失败规则。

**原规格状态：仍称“上界臂”并用作分母，未修。**

---

<a id="h06"></a>

### H06. `m=16` 只是截断子族，原规格没有定义近似误差

**发现的问题**

原规格先把 `xi` 定义为输出空间所有单位向量，随后明确改取 16 个 SVD 方向。取有限子族本身并非错误；问题是规格没有把所得量命名为完整 Cost 的下界/近似，也没有定义截断误差如何评估。

**计划书位置**

§6.2–§6.5，[原规格 L353–L382](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L353-L382)：

**原文**

> “$\xi$ 为该矩阵输出空间的单位向量。”
>
> “实践上取有限子族：$\xi$ 取该 $W$ 输出的前 $m$ 个主方向，由 $W$ 的奇异值分解确定。$m$ 在执行前写死为 $m=16$。”

**为什么是问题**

16 个固定方向只给一个下界/截断近似；除非 `Delta f` 恰落在其张成空间，否则不等于完整上确界。

**证据**

对完整单位线性族，`sup_{||xi||<=1} |<Delta f,xi>| = ||Delta f||_2`。取与前 16 个方向正交的非零 `Delta f`，截断结果为 0，而完整上确界大于 0。

**解决方案**

主分析直接使用完整输出向量 L2；`m=16` 只能作为另行命名、预注册的低成本敏感性分析，并报告捕获能量或与完整 L2 的误差。

**有效性验收**

构造与前 16 方向正交的 `Delta f`：截断值应为 0、完整 L2 应大于 0；主结论只引用后者。

**原规格状态：已承认它是有限子族，但未定义与完整 Cost 的关系，未闭合。**

---

<a id="h07"></a>

### H07. `p>0.05` 不能单独判定预言被反驳

**发现的问题**

P3、P4 把“不显著”直接列入杀伤条件，把“证据不足”误写成“反方向证据”。

**计划书位置**

§3，[原规格 L193–L204](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L193-L204)：

**原文**

> P3：“两族中位数之差 $\le 0$，或 Mann–Whitney 单尾 $p>0.05$。”
>
> P4：“在 {64,128,256,512,1024} token 五点上 Spearman 相关 $\le 0$ 或 $p>0.05$。”

**为什么是问题**

`p>0.05` 可能来自效应很小、样本不足、方差过大或检验功效低；它不区分零效应和未观察清楚。

**证据**

[ASA 关于 p 值的正式声明](https://doi.org/10.1080/00031305.2016.1154108)强调，科学结论不应只由某个阈值决定；此处还存在把不拒绝零假设当成接受反命题的逻辑错误。

**解决方案**

每条预言使用“支持 / 反驳 / 证据不足”三态；冻结效应方向、最小有意义效应、区间、实验单位、样本量和不足规则。P3/P4 分别按 B04/R01 的新端点重写。

**有效性验收**

模拟零效应、小样本和明确反向效应三种情形，输出必须分别落入证据不足/证据不足或等效/反驳，而不是全部由 `p` 阈值决定。

**原规格状态：P3/P4 仍以 `p>0.05` 作为杀伤条件，未修。**

---

<a id="h08"></a>

### H08. F2 伪代码没有实现正文要求的内存协议

**发现的问题**

F2 正文要求 `p` 分块相乘或及时释放，但示例先完整物化 logits、概率和全词表 cosine，并把 `p` 保留到函数返回。示例没有给出满足正文要求的生产实现。

**计划书位置**

- 完整张量伪代码：§7.5，[原规格 L531–L547](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L531-L547)
- 内存要求：§7.5，[原规格 L550–L557](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L550-L557)

**原文**

> “`logits = out.logits.float()`；`p = softmax(logits/T)`；`cos = normalize(u_avg) @ normalize(E).T`。”
>
> “故批量取 1，且 `p` 须分块乘或及时释放。”

**为什么是问题**

`logits`、`p` 和 `cos` 都可能达到 `(B,L,V)`；同时存在会放大峰值显存。五个温度若重复 decoder 前向，还会把算力估计放大。

**证据**

原规格 L557 自己计算：`L=1024, V=152064` 时，一个 fp32 `(1,L,V)` 张量约 623 MB。示例同时保留 logits 与 `p`，并另建相同词表维度的 cosine；但没有实现它随后要求的分块或释放策略。问题是规格不够可执行，而不是证明该计算在所有 80GB GPU 上必然 OOM。

**解决方案**

decoder hidden states 只算一次；lm head 沿词表行做两遍 fp32 流式扫描：第一遍求 max/logZ，第二遍累计 argmax、entropy 与 `p@E`；cosine 同样分块维护全局最大值。不同架构的 decoder/attention 入口由显式适配器调用。

**有效性验收**

- 小模型上与完整计算逐元素或在预注册容差内一致；
- 五个温度复用同一 hidden states；
- 记录峰值显存与墙钟；
- 人为中断后可从完整块边界恢复且结果相同。

**原规格状态：正文要求分块但示例仍完整物化，内部矛盾未修。**

---

<a id="h09"></a>

### H09. 不可达 token 的梯度描述对 untied 模型不成立

**发现的问题**

原文说不可达 token 的“嵌入与输出行向量只受负类梯度”，但 untied 输入 embedding 行在从未作为输入出现时通常没有数据梯度；负类 softmax 梯度作用于输出行。

**计划书位置**

§2.4，[原规格 L135–L141](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L135-L141)：

**原文**

> “它在训练中既不作为输入出现、也不作为目标出现，其嵌入与输出行向量只受负类梯度。”

**为什么是问题**

原文把输入 lookup 行和输出 softmax 行的梯度路径合并了；在 untied 模型中两者不同。

**证据**

- untied `E[a]`：token 不作为输入时，该行没有由 lookup 产生的数据梯度；
- untied `W[a]`：即使不是目标，softmax 分母仍产生负类梯度；
- tied：共享行可从 output side 收到负类梯度；
- special/control token 可能由模板直接注入，不能按普通文本 reachability 处理。

**解决方案**

P2/P7 按 tied/untied 分别陈述 `E` 与 `W` 的暴露和梯度路径；特殊 token、added token、padding 行单列。

**有效性验收**

在一个极小 tied/untied toy LM 上只用不含 token `a` 的 batch 反传：关闭 weight decay 和其他 optimizer 更新，只检查数据损失产生的梯度，验证 `E[a]`、`W[a]` 是否符合上述预测。

**原规格状态：未区分 tied/untied 梯度路径，高优先级。**

---

<a id="h10"></a>

### H10. RMSNorm 的“只留方向”是带条件的近似

**发现的问题**

原文把带 epsilon 的 RMSNorm 写成严格消除长度信息，并据此声称后续分支“天然只看方向”。这在通常激活尺度下可能是很好的近似，但不是无条件恒等式。

**计划书位置**

- 定义：§1.3，[原规格 L69–L72](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L69-L72)
- F2 混淆处理：§7.5，[原规格 L516–L521](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L516-L521)

**原文**

> “它会把输入向量的长度信息除掉，只留方向。”
>
> “天然只看方向、不受长度影响。”

**为什么是问题**

`RMSNorm(alpha u) = alpha u / sqrt(alpha^2 mean(u^2)+epsilon)`。当 `epsilon>0` 时，它不严格等于与 `alpha` 无关的常量；差异主要在近零范数区域。`gamma` 会改变方向几何，但在 `epsilon=0` 时并不会破坏对全局尺度 `alpha` 的不变性，因此不能把 `gamma` 当作残留尺度依赖的证据。

**证据**

[Qwen2 RMSNorm 官方实现](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py#L65-L79) 明确加入 epsilon。

**解决方案**

改成“当激活 RMS 远大于 `sqrt(epsilon)` 时近似尺度不变”；保存 norm 前后 RMS，并为近零向量设失败断言。不要用该近似证明 norm confound 已被完全消除。

**有效性验收**

对同方向不同尺度的向量扫 `alpha`，报告归一化输出差异；实际样本必须显示 `RMS/sqrt(epsilon)` 的分布远离 1，才可使用近似表述。

**原规格状态：需要改成条件性近似；这是精度风险，不单独阻断开跑。**

---

<a id="h11"></a>

### H11. tokenizer 伪代码与文字协议不一致，且存在索引风险

**发现的问题**

精确往返代码会删除 special token，文字要求关闭空格清理但代码未传参数；三臂索引把 `h_t` 产生的下一 token 与当前位置记号混用。

**计划书位置**

- 字级往返：§7.4，[原规格 L453–L468](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L453-L468)
- 序列往返：§7.4，[原规格 L482–L492](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L482-L492)
- 下一地址构造：§7.5，[原规格 L506–L514](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L506-L514)
- 空格规则：§8，[原规格 L632–L638](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L632-L638)

**原文**

> “`tok.decode(w, skip_special_tokens=True)`。”
>
> “`clean_up_tokenization_spaces` 须显式关掉。”
>
> “由隐状态 `h_t` 得地址分布 `p_t`……`a_t=argmax p_t`。”

**为什么是问题**

- 删除 special token 会改变序列内容，不能再称精确 `R = T o D`；正确做法是先按预注册规则排除不在普通字符串域的序列，剩余序列 decode 时不静默删除。
- `h_t` 通常预测 `a_{t+1}`，候选 embedding 也对应下一位置；当前记号易产生 off-by-one。

**证据**

- 规格 L488 的函数实际传入 `skip_special_tokens=True`，而 L638 又要求 decode 行为显式固定，二者不一致；
- 对 causal LM，位置 `t` 的 logits 定义下一 token 分布；因此原文的 `h_t -> p_t -> a_t` 记号至少需要单独声明索引约定，否则实现者会分别对齐到 `t` 或 `t+1`。

**解决方案**

显式固定 `skip_special_tokens=False`、`clean_up_tokenization_spaces=False`、`add_special_tokens=False`；特殊 token 预先分层。byte-level tokenizer 应优先通过字节接口核验，不能只凭字符串中出现 `�` 就判为不可解码，因为某个 token 也可能合法表示 U+FFFD。索引统一为 `h_t -> p(a_{t+1}) -> u_{t+1}`。对 `w` 与 `R(w)` 的插入、删除、替换，预先固定序列编辑对齐算法，才能定义“失败 token 位置数”。

**有效性验收**

建立 golden tests：ASCII、中文、多空格、合法 U+FFFD、byte fragment、special token、拼接边界和插入/删除/替换对齐；每个参数改变都必须使至少一个故意构造的测试失败，从而证明测试能抓错。

**原规格状态：文字要求与伪代码不一致，且下一位置索引未统一，高优先级。**

---

<a id="h12"></a>

### H12. 统计与复现协议未达到“自包含完整规格”的自我声明

**发现的问题**

文档自称所有必要信息都在一份文件中，但数据、模型 revision、分析端点、环境、输出 schema、恢复与验证测试仍未冻结；不同执行者会得到不同实验。

**计划书位置**

- 自我声明：开头，[原规格 L3–L7](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L3-L7)
- “预注册”的项目内定义：[原规格 L30–L35](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L30-L35)
- P2 文献空缺：[原规格 L212–L216](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L212-L216)
- F1 数字缺少可追溯引用：[原规格 L398–L403](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L398-L403)
- 未核实 model/tokenizer：§4，[原规格 L237–L245](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L237-L245)、[L274–L279](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L274-L279)
- 数据仍为空：§7.4，[原规格 L474–L480](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L474-L480)
- 尚未回填：§9–§10，[原规格 L658–L665](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L658-L665)、[L707–L709](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L707-L709)

**原文**

> “所有必要的定义、符号、模型清单、代码、成本、失败模式、以及尚未决定的事项，全部写在这一份里。”
>
> “语料定为多语混合、10B token 量级（具体来源空缺）。”

**为什么是问题**

按 BetterBench、Datasheets 与 Data Statements 的可操作字段复核，主要缺口如下：

| 类别 | 原规格仍缺少的内容 |
|---|---|
| 用途与边界 | 预期使用者、理论检验/工程诊断/排名的用途区分、不得跨模型家族/语言/任务/解码策略外推的范围 |
| 维护与治理 | 作者/维护者/联系渠道、变更日志、兼容与废弃政策、贡献/问题追踪方式、规格/代码/输出的许可证和引用方式 |
| 数据来源与抽样 | 数据集名称、revision、文件哈希、采集过程、原始用途、抽样机制、语言/地区/领域组成、划分和统计 |
| 数据处理与责任 | 版本化 preprocessing 顺序、normalization/过滤/截断/模板、纳入排除规则、去重/污染、偏差、许可、PII、再分发限制 |
| 模型与对照 | 模型/tokenizer/baseline 的精确 revision 与文件哈希、remote-code/gated 状态；凭证不得进入 manifest 或结果 |
| 主分析 | P1–P10 唯一 primary outcome、目标测度 $\mu$、prompt/sequence/token/语言加权、实验单位、最小有意义效应、功效或最小可检测效应、样本量、区间和多重比较 |
| 边界样本 | 提前 EOS、special token、缺失样本、序列编辑对齐、F7 argmax 并列时的确定性 tie-breaking |
| 生成与非确定性 | prompt/generation manifest、解码参数、随机种子、CUDA 确定性设置、允许的后端差异、停止条件 |
| F6 训练 | 训练数据、基线、优化器、学习率、batch、步数、评估频率、checkpoint、多个训练种子和停止规则 |
| 运行入口 | 依赖/容器锁、硬件与软件环境、完整运行命令、dry-run 与 confirmation-run 的独立入口 |
| QA 与恢复 | 分块 softmax/argmax/cosine、mask、索引、序列对齐和恢复的等价性测试；OOM、重试、损坏检测与幂等续跑 |
| 产物与溯源 | raw → derived → summary/figure 的 schema 和 provenance、缺失值、原子写入、发布位置、保留周期、重建与废弃策略 |
| 外部证据 | F1 文献题名、作者、版本/链接与数字所在表格；P2 文献读完前不能冻结“高度重合”的操作定义 |

**证据**

这些字段在全文没有固定值或被明确标为空缺；这是文档可执行性审计，不需要实际跑模型即可确认。

**解决方案**

在正式测量前生成 `data_manifest`、`model_manifest`、`generation_manifest`、`analysis_manifest`、机器可读 `phi_manifest` 和 `environment.lock`，另补一页 scope/governance 说明和完整运行入口；所有产物引用同一个 run id 与 git commit。将当前“看到数据前写死”的做法称为**内部预先规定**；只有提交到带时间戳、不可静默覆盖且有修改政策的 registry/commit 后，才称正式 preregistration。

**有效性验收**

在无网络的新环境中，仅凭锁文件和已缓存输入运行 0.5B 合成/小样本 smoke；不修改配置即可生成所有预定空表、日志和结果 schema。第二位执行者得到相同哈希与同容差结果。

**原规格状态：与“自包含完整规格”的声明不符，高优先级。**

---

## 3. 需要收紧的问题与条件性风险

<a id="r01"></a>

### R01. P4 的长度趋势可能只是暴露机会增加

**发现的问题**

若主端点是“一条序列是否至少出现一次非规范位置”，序列越长，即使每位置风险不变，累计失败概率也会机械上升。原规格同时提到序列级比例和 token 位置比例，但没有冻结哪一个是 P4 主端点，因此这是**条件性风险**，不是已证明的分析错误。

**计划书位置**

- P4：§3，[原规格 L199–L204](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L199-L204)
- 两种口径：§7.4，[原规格 L482–L498](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L482-L498)
- 五个长度：§7.5，[原规格 L557–L559](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L557-L559)

**原文**

> “非规范质量随长度单调增。”
>
> “须报两个口径：有多少条序列非规范；失败的 token 位置数/总 token 数。”

**为什么是问题**

若使用“至少一次失败”，长度增加会同时增加暴露机会，不能区分“每位置风险变大”与“只是机会更多”。

**证据**

固定每个边界独立失败概率为 `q` 时，长度 `L` 的“至少一次失败”概率仍为 `1-(1-q)^L`，会在机制完全不变时机械上升。

**解决方案**

主端点冻结为 per-token/per-boundary hazard 或首次失败 survival curve；累计“任一失败”只作次端点。prompt 跨长度配对，提前 EOS 有固定删失规则。

**有效性验收**

在固定 `q` 的模拟数据中，累计曲线应上升但 hazard 保持水平；分析必须正确判定“没有长度机制效应”。

**原规格状态：两种口径都要求报告，但 P4 主端点未冻结，需收紧。**

---

<a id="r02"></a>

### R02. tokenizer 前缀稳定与 Transformer KV cache 被写成同一层条件

**发现的问题**

标准 causal Transformer 对固定 token-id 前缀复用 KV，与“持续增长的原始字符串是否可增量 tokenize”是两个层次；原文把弱级成立直接称为 KV cache 合法的充要条件，表述过宽。

**计划书位置**

§2.7，[原规格 L178–L187](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L178-L187)：

以及 P10，[原规格 L592–L598](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L592-L598)。

**原文**

> “因果掩码保证弱级成立（这正是 KV cache 合法的充要条件），强级失败即提示词边界效应。”
>
> “故可缓存需要两个条件，不是一个：$\mathrm{sel}$ 值域离散（提交不可逆）加上 $\tau$ 保前缀（因果掩码）。”

**为什么是问题**

- 模型层：给定不变的 token-id 前缀，因果注意力使旧位置状态不依赖未来 token，故可复用 KV。
- tokenizer 层：若输入是不断增长的原始字符串，新增字符可能改变末端 tokenization，需要回退/re-tokenize。

第二个问题不能作为第一个问题一般形式下的必要条件；只有系统试图直接缓存“原始字符串到 token id”的增量流水线时才相关。

原文还把“`sel` 值域离散”写成缓存的一般必要条件，但缓存真正需要的是追加输入后已算状态可安全复用；连续状态系统也可能满足前缀不变性。离散性不是一般意义上的必要条件。

**证据**

固定 token-id 前缀时，causal attention 的旧位置计算图不包含未来位置；但在原始字符串末尾追加字符时，tokenizer 可以改变末端 token ids。两种现象可分别发生，说明它们不是同一个条件。

**解决方案**

分别定义 `model_kv_cache` 与 `incremental_tokenization_cache`。把模型 cache 条件写成“追加输入后已算状态保持不变且其位置/掩码语义兼容”，不要求状态空间必须离散。P10 仅核查双向 self-attention 下旧位置表示随新增 token 改变；token healing/边界问题放在 tokenizer cache 小节。

**有效性验收**

四格测试：固定/变化 token-id 前缀 × causal/bidirectional 模型；另外单测 raw-string 追加导致末端 retokenization 的案例。

**原规格状态：模型 cache 与 tokenizer cache 未分层，需收紧。**

---

<a id="r03"></a>

### R03. 成本和工期目前不可验证

**发现的问题**

权重容量量级估算可以作下界，但 F2/F4 没冻结 prompt 数、有效位置数、hidden-state 复用、温度扫描成本和实际硬件吞吐；墙钟承诺无法由规格复算。F6 的 512 GB 也不含 activation、allocator、通信和框架开销。

**计划书位置**

- 成本与时间：§5，[原规格 L293–L337](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L293-L337)
- F6：§7.7，[原规格 L574–L590](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L574-L590)
- 执行序：§9，[原规格 L667–L685](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L667-L685)

**原文**

> “F2/F4 实现 0.5 天、运行 1 天。”
>
> “32B 全参微调 + Adam……约 512 GB。”

**为什么是问题**

这些是粗略容量/日程估计，不是可核验 benchmark。尤其 F2 的成本取决于 `prompts x positions x temperatures x models`、是否缓存 hidden states 和分块实现；F6 还受 ZeRO stage、optimizer、activation checkpointing、序列长度和通信拓扑影响。

**证据**

原规格没有冻结 prompt 数、有效位置抽样数、硬件型号/吞吐、hidden-state 复用与 profiling 记录；因此无法从已给字段复算“一天”。512 GB 只相加了参数、梯度、Adam 状态和主权重，式中没有 activation、allocator 与通信 buffer。

**解决方案**

先在 0.5B/7B 各做固定 10 prompts profiling，记录阶段级 FLOPs、峰值显存、I/O、墙钟与温度复用收益，再按工作量外推区间。F6 使用所选训练栈的 memory estimator 加单步实测。

**有效性验收**

预测区间必须覆盖第二批 profiling 的实测；未 profiling 前所有工期写“待测”，512 GB 明确标为参数/梯度/优化器状态近似下界。

**原规格状态：仅有量级估计，没有 profiling 证据，需收紧。**

---

<a id="r04"></a>

### R04. 缺少机器可执行的 G0/G1/G2 开跑门

**发现的问题**

H12 已列出尚缺的字段；本卡只处理另一个问题：运行入口没有机器可执行的 gate，无法阻止文档、smoke 和正式测量三种状态混用。

**计划书位置**

- 阶段零：[L656–L665](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L656-L665)
- 可报告数字：[L667–L675](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L667-L675)
- 尚未回填：[L707–L709](https://github.com/Pthahnix/token-theory/blob/6baa3e99fc0b50f383b9865ca4209fad801cdd6f/context/2026-08-23-19-18-f-round-spec.md#L707-L709)

**原文**

> “上述任何一步都不需要等设备……产出第一批可报告的数。”
>
> “尚未回填：阶段零的六项；`Phi_VQ`；F5 具体模型仓库。”

**为什么是问题**

“能运行”不等于“可作为确认性证据”。若 H12 的字段未关闭就先看真实数字，会污染内部预先规定；后续改协议也难以区分源码纠错和结果驱动。纯文档核查、合成 smoke 与真实确认性测量需要不同权限和输出目录。

**证据**

原规格 L675 允许阶段一“产出第一批可报告的数”，L709 又明列阶段零六项、`Phi_VQ` 和 F5 仓库尚未回填，却没有一个程序条件把未关闭项与 confirmation run 关联起来。

**解决方案**

定义三个门：

- `G0 documentation`: B01–B05 裁决、所有 manifests 冻结；
- `G1 smoke`: 只允许合成/小样本 QA，结果不得进论文；
- `G2 measurement`: commit/tag、环境和主端点冻结后才运行真实确认性数据。

**有效性验收**

CI/运行入口读取 gate 文件；G0 未通过时拒绝 confirmation run。smoke 输出目录与正式结果目录物理分离。

**原规格状态：没有可执行的开跑门，需收紧。**

---

## 4. 修正优先级与最小可执行路径

### 第一步：先修理论对象，不碰 GPU

1. 修 B01：分离 `T/D` 与 `q/e`。
2. 修 B02：冻结目标测度 $\mu$；分别定义线性分支、范数项与 `local_shift`。
3. 重写 P3/P6/F5（B03–B05）。

这一步完成前，任何真实模型结果都不应进入论文证据。

### 第二步：修架构和统计协议

1. 固化 H02/H03 的架构 adapters 和 hooks。
2. 修 H04–H08：P5 分层、取消 `glb` 上界、完整 L2、三态判断、流式实现。
3. 修 H09–H11：梯度口径、RMSNorm 条件、tokenizer/index golden tests。

### 第三步：补齐复现与资源门

1. 完成 H12 的六份 manifest/lock。
2. 冻结 R01/R02 的端点与层次。
3. 只运行 profiling 与 smoke，验证 R03。
4. 所有 gate 通过后，再启动阶段一正式测量；阶段二 GPU 实验仍需独立 GO 决策。

---

## 5. 总验收清单

只有以下条件全部满足，阶段零才能从 `NO-GO` 改为 `GO`：

- [ ] 全文映射类型检查通过，`A`/`A*` 不再混用；
- [ ] 目标测度 $\mu$ 已冻结；线性分支、范数项与 `local_shift` 分离，并通过 `v/-v` 和纯尺度反例；
- [ ] P3 改为 matched tokenizer 因果比较，现成 tokenizer 只作描述性结果；
- [ ] P6 结构事实、表示干预与配对训练的架构因果检验分离；
- [ ] F5 仅含真实离散量化模型，code grid/上下文和所有输入输出 shape 可核查；
- [ ] Qwen2/Qwen3/DeepSeek adapter 与官方 forward hook 对齐；
- [ ] P5 不跨 tied/untied 断点作主趋势；
- [ ] `glb` 不再称上界、不再作分母；
- [ ] 完整单位线性子族的 L2 与 `Phi_16` 截断敏感性分析分开，并报告近似误差；
- [ ] P3/P4/P5 均采用支持/反驳/证据不足三态；
- [ ] 流式 softmax/argmax/entropy/`p@E`/cosine 与完整计算通过等价性测试；
- [ ] tokenizer 参数、下一位置索引、合法 U+FFFD、byte fragment、special token 和序列编辑对齐通过 golden tests；
- [ ] data/model/generation/analysis/phi manifests、environment lock、scope/governance 说明和完整运行入口齐全；
- [ ] profiling 给出可复核的显存与墙钟区间；
- [ ] G0/G1/G2 gate 被运行入口实际执行，而不只是写在文档里。

---

## 6. Documentation-audit 46 项完整性快照

这套框架原本用于 benchmark 文档；此处把它当作实验规格的严格复现性压力测试，而不是对研究价值打分。尚未开跑的项目若某项确实不适用，也应在规格中显式标为 `N/A` 并说明理由；原规格未说明时，本表按 `ABSENT` 处理。

| 类别 | PRESENT | PARTIAL | ABSENT | 总数 |
|---|---:|---:|---:|---:|
| Motivation & Scope | 3 | 2 | 3 | 8 |
| Data Documentation | 0 | 4 | 8 | 12 |
| Evaluation Protocol | 2 | 5 | 3 | 10 |
| Validity & Reliability | 1 | 3 | 4 | 8 |
| Maintenance & Governance | 0 | 0 | 8 | 8 |
| **总计** | **6** | **14** | **26** | **46** |

- `betterbench_score = 6 / 46 = 0.130`
- 对应等级：**F（文档尚不足以支持第三方负责任地复现）**
- 独立复现判断：**No**。主要缺少数据与处理链、确认性估计器、运行入口、版本锁、产物溯源和治理信息。
- 这里的 F 反映文档尚处于规划阶段，不表示理论本身已经被证伪。

<details>
<summary>展开 46 项逐条判定</summary>

| # | 类别 | 标准 | 判定 | 原因摘要 |
|---:|---|---|---|---|
| 1 | Motivation | 研究目的 | PRESENT | 开头明确检验十条预言 |
| 2 | Motivation | 目标构念定义 | PARTIAL | 有符号与理论量，但存在 B01/B02 |
| 3 | Motivation | 预期用途 | PARTIAL | 有理论检验目的，未区分工程诊断/排名用途 |
| 4 | Motivation | 已知限制 | PRESENT | §10 明列硬上限与未决项 |
| 5 | Motivation | 与现有 benchmark 关系 | ABSENT | 未给 benchmark 定位或替代关系 |
| 6 | Motivation | 目标用户 | ABSENT | 未说明作者、复现者或使用群体 |
| 7 | Motivation | 不测试什么 | PRESENT | §10.2 明列主动不做项 |
| 8 | Motivation | 版本史/changelog | ABSENT | 只有版本时间，没有变更记录和政策 |
| 9 | Data | 数据来源 | PARTIAL | 模型仓库已列，主语料仍为空 |
| 10 | Data | 采集方法 | ABSENT | 没有语料采集与抽样过程 |
| 11 | Data | 标注流程 | ABSENT | 未说明无需标注或具体标注流程 |
| 12 | Data | 标注者信息 | ABSENT | 未声明 N/A 或资格要求 |
| 13 | Data | 标注一致性 | ABSENT | 未声明 N/A 或一致性度量 |
| 14 | Data | 过滤/清洗 | PARTIAL | 有少量 tokenizer 陷阱，无完整 pipeline |
| 15 | Data | train/dev/test 划分 | ABSENT | 未定义数据划分或防泄漏规则 |
| 16 | Data | 规模与统计 | PARTIAL | 仅给“10B token 量级”等粗值 |
| 17 | Data | 数据格式/schema | ABSENT | 没有输入 observation schema |
| 18 | Data | 许可/使用条款 | PARTIAL | 知道需要许可，但未冻结数据许可 |
| 19 | Data | PII/伦理 | ABSENT | 无隐私和伦理说明 |
| 20 | Data | 已知数据偏差 | ABSENT | 未系统说明语言/领域偏差 |
| 21 | Evaluation | 主指标公式 | PARTIAL | 理论公式存在，经验估计器缺失 |
| 22 | Evaluation | 次指标 | PRESENT | entropy、delta、范数等已列 |
| 23 | Evaluation | 评估脚本 | PARTIAL | 只有不完整伪代码，无正式入口 |
| 24 | Evaluation | baseline 结果 | ABSENT | 尚未运行，也未定义随机/简单基线 |
| 25 | Evaluation | prompt/few-shot 格式 | ABSENT | prompt 模板与 manifest 未定 |
| 26 | Evaluation | 解码参数 | PARTIAL | 温度和长度部分明确，其余未闭合 |
| 27 | Evaluation | 后处理 | PARTIAL | 部分 tokenizer 参数存在但代码不一致 |
| 28 | Evaluation | 显著性方法 | PARTIAL | 有检验名，但 H07 指出判决错误 |
| 29 | Evaluation | 区间/方差 | ABSENT | 未统一规定 CI 与重复层级 |
| 30 | Evaluation | 硬件/算力 | PRESENT | GPU、存储和粗略成本已列 |
| 31 | Validity | 构念效度 | PARTIAL | 有理论映射，但 B02/B03/B04 未闭合 |
| 32 | Validity | 内容效度/采样 | PARTIAL | 模型网格有理由，数据项抽样未定 |
| 33 | Validity | 重测可靠性 | ABSENT | 无重复运行或容差协议 |
| 34 | Validity | 内部一致性 | ABSENT | 无跨样本/分支一致性定义 |
| 35 | Validity | 已知混淆 | PRESENT | 绑定、范数、离流形等已主动列出 |
| 36 | Validity | ceiling/floor | PARTIAL | 识别 $T\to0$ 平凡极限，未系统分析 |
| 37 | Validity | 难度校准 | ABSENT | 无 item/prompt 难度设计 |
| 38 | Validity | item 分析 | ABSENT | 无 discrimination/difficulty 分析 |
| 39 | Governance | 维护计划 | ABSENT | 未说明谁维护 |
| 40 | Governance | 更新频率 | ABSENT | 无更新周期 |
| 41 | Governance | 废弃政策 | ABSENT | 无兼容和 deprecation 规则 |
| 42 | Governance | 联系信息 | ABSENT | 无维护者联系方式 |
| 43 | Governance | 贡献指南 | ABSENT | 无贡献和审阅流程 |
| 44 | Governance | issue tracking | ABSENT | 文档未规定问题追踪入口 |
| 45 | Governance | leaderboard/提交 | ABSENT | 未说明无 leaderboard 或提交规则 |
| 46 | Governance | contamination 监控 | ABSENT | 无持续污染监控与更新策略 |

</details>

**CRITICAL gaps**：数据来源/处理/许可未冻结；主指标经验估计器和抽样测度未定义；prompt/生成/统计协议不完整；没有可复现运行入口与环境锁；没有输出 provenance 和机器可执行 gate。

---

## 7. 本地待审稿边界

本文件当前只是一份本地待审稿，内容限于“问题、原文、证据、修正方案与验收方法”。它：

- 不复制完整原规格；
- 不附阶段零清单或实验结果；
- 不修改远端现行规格；
- 不把源码/配置核对写成实验验证。

待上述阻断项获得裁决后，再决定是否提交；之后才另建时间戳目录准备完整的新规格版本。
