# Q1 — $\tau\kappa \neq \mathrm{id}$ 时的代价

**结论:代价推出来了,但推出来的过程把问题改掉了。截面性不是在文本输入上「失效」,是**我把它写在了错的层级上**——字级(单 token)上它确实假,序列级上它由无损性**自动成立**。修好之后:N1 的幂等与 N2 的零核定理**原样恢复**,而恢复的条件从「分词器的性质」变成「输入分布的性质」,该条件的失效点恰好是三个已知病灶。副产品是一条把 A 里的不可达 token 与已知的 glitch token 现象接上的推导。**

---

## 一、把 $r$ 写出来:失效的正确对象

记 $\Omega$ 为字符串集,$\mathcal{A}$ 为词表,$\tau:\Omega\to\mathcal{A}^*$(分词给出**序列**),$\kappa=\mathrm{anc}:\mathcal{A}\to\Omega$(单 token 的表面形),$\kappa^*:\mathcal{A}^*\to\Omega$ 为其拼接延拓。

**第一处此前没写准:$\tau$ 的值域是 $\mathcal{A}^*$ 而不是 $\mathcal{A}$。** 于是往返映射有两个,不是一个:

$$r:\mathcal{A}\to\mathcal{A}^+,\quad r(a)=\tau(\kappa(a))\qquad\text{(字级)}$$
$$R:\mathcal{A}^*\to\mathcal{A}^*,\quad R(w)=\tau(\kappa^*(w))\qquad\text{(序列级)}$$

$r$ 按字母作用可唯一延拓为自由幺半群的自同态 $r^*(w)=r(w_1)\cdots r(w_n)$。**$r^*$ 与 $R$ 一般不等**,而这个不等恰是 N4 的强级(保拼接)失效——先逐字往返再拼 vs 先拼再整体往返。

三件事随之分开,此前混作一件:

| 记法 | 等式 | 含义 | 实证 |
|---|---|---|---|
| 字级截面 | $r(a)=(a)\ \forall a$ | 每个 token 的表面形单独送去分词得回自己 | **假**(F3 要测的就是它) |
| 序列级截面 | $R|_{\mathrm{im}\,\tau}=\mathrm{id}$ | 规范切分往返不动 | **真**,见 §2 |
| 保拼接 | $r^*=R$ | 逐字往返 = 整体往返 | 假(prompt 边界效应) |

原第 51 条说「锚不是截面」,说的是第一行。N1/N2 需要的是第二行。**两行不是同一句话,而我此前当成了同一句。**

## 二、序列级截面由无损性自动成立(引理)

**引理 1。** 设 $\tau$ 确定且在字符串上无损($\kappa^*\tau=\mathrm{id}_\Omega$)。则 $R$ **幂等**,且

$$\mathrm{im}(R)=\mathrm{Fix}(R)=\tau(\Omega)=:\mathcal{A}^*_{\mathrm{can}}$$

**证明。** $R(R(w))=\tau\big(\kappa^*(\tau(\kappa^* w))\big)=\tau(\kappa^* w)=R(w)$,中间一步用 $\kappa^*\tau=\mathrm{id}$。幂等映射的像等于其不动点集;而 $\mathrm{im}(R)=\tau(\kappa^*(\mathcal{A}^*))\subseteq\tau(\Omega)$,反向包含由 $\tau(s)=R(\tau(s))$ 得。∎

$\mathcal{A}^*_{\mathrm{can}}$ = **规范切分集** = 该分词器真能产出的 token 序列。引理 1 说:**$\kappa^*$ 在 $\mathcal{A}^*_{\mathrm{can}}$ 上是 $\tau$ 的真截面。** 于是

- **N1 检验一恢复**:幂等要什么有什么,原推导逐行成立,只需把载体从 $\mathcal{A}$ 换成 $\mathcal{A}^*$、把「锚是截面」换成引理 1。
- **N2 零核定理恢复**:$\Leftarrow$ 方向那一步 $\bar\varphi(\tau\kappa\tau x)=\bar\varphi(\tau x)$ 现在成立,因为 $\tau x\in\mathcal{A}^*_{\mathrm{can}}$。

**代价:定理的假设换了性质。** 原来「锚是截面」是分词器的性质(查词表即可判);现在「$\pi$ 支撑在 $\mathcal{A}^*_{\mathrm{can}}$ 内」是**输入分布的性质**。这个交换不是把问题挪走了,它把问题挪到了对的地方——见 §4 的三个失效点。

## 三、代价:非规范质量

设 $\pi$ 为 $\mathcal{A}^*$ 上实际出现的 token 序列分布。$\Phi$ 里 $\sim$-不变的那一族记 $\bar\Phi$($\varphi=\bar\varphi\circ\tau$)。对这一族,

$$\varphi(\kappa^*w)-\varphi(w)\ \text{的类比量}=\bar\varphi(R w)-\bar\varphi(w)$$

于是**截面代价**:

$$\boxed{\ \mathrm{Cost}^{\mathrm{sec}}_{\bar\Phi}(\pi)\;=\;\sup_{\bar\varphi\in\bar\Phi}\Big|\mathbb{E}_{R_*\pi}[\bar\varphi]-\mathbb{E}_{\pi}[\bar\varphi]\Big|\ }$$

**它不是第三项,是第一项里此前被证明为零的那一块。** $\mathrm{Cost}_\Phi$ 限制到 $\sim$-不变子族上就是它;截面性成立时这一块恒零(N2 §2 的 $\Leftarrow$),失效时不零。所以判据的项数不变,是**第一项裂成两块**。

**上界(给 F3 的桥)。** 被积函数在 $\mathrm{Fix}(R)=\mathcal{A}^*_{\mathrm{can}}$ 上逐点为零,故

$$\mathrm{Cost}^{\mathrm{sec}}_{\bar\Phi}(\pi)\;\le\;2\lVert\bar\Phi\rVert_\infty\cdot\underbrace{\pi\big(\mathcal{A}^*\setminus\mathcal{A}^*_{\mathrm{can}}\big)}_{\text{非规范质量}}$$

且当 $\bar\Phi$ 含指标函数时上确界即 $\lVert R_*\pi-\pi\rVert_{\mathrm{TV}}$。**F3 要出的数是这个非规范质量,不是「失败 token 的比例」。** 两处差别都改实验设计:

1. **须按频率加权,不按 type 计数。** $\pi$ 是实际出现的分布;词表上均匀平均对应不到任何代价。
2. **松弛是真的松弛,不是不等号的客套。** 若 $R$ 把质量在两个 $\bar\varphi$ 值相同的序列之间搬,代价为零而非规范质量为正。**截面失效可以完全不付代价**——这与第 52 条(判据只吃推前对)是同一件事的第三次露头。

## 四、失效点:条件的否定恰好是三个已知病灶

$\pi$ 何时不支撑在 $\mathcal{A}^*_{\mathrm{can}}$ 内?$\pi$ 由 $\tau$ 作用于文本得到时,由构造支撑在里面(引理 1)。**所以违反只能来自不经 $\tau$ 而直接造出 token 序列的路径**,共三条:

| 违反路径 | 已知现象 |
|---|---|
| 两段各自分词后拼接 | prompt 边界效应;few-shot 模板拼接的不稳 |
| 模型自回归续写(逐位从 $\mathcal{A}$ 采样,从不过 $\tau$) | 生成文本重新分词后切法不同;续写与重编码不一致 |
| 在 token 边界处切割/截断/缓存复用 | token healing 要修的正是这个 |

**三条独立记录的病灶 = 一个推导出来的条件的三种违反方式。** 这是本节份量最重的一句:token healing 不是「锚不是截面」的补丁,它是**把 $\pi$ 拉回 $\mathcal{A}^*_{\mathrm{can}}$ 的投影算子**——正是 $R$ 本身。healing 的正确形式化就是「作用一次 $R$」。

## 五、字级失效的分类,与不可达 token

字级 $r$ 仍是可测的真东西,但它现在的意义变了:它不再是地基,是**词表自身的一个诊断**。按 $r$ 的形状分三档:

| 档 | 条件 | 含义 |
|---|---|---|
| 完好 | $r(a)=(a)$ | 该 token 是自身表面形的规范切分 |
| 碎裂 | $\lvert r(a)\rvert>1$ | 单独出现时该 token 不会被产出,只在更长上下文里被产出 |
| 改名 | $\lvert r(a)\rvert=1,\ r(a)\neq a$ | 另一个 token 编码同一串且优先 |

**引理 2(改名一步终止)。** 若 $r(a)=(b)$ 且 $b\neq a$,则 $\kappa(b)=\kappa^*(r(a))=\kappa(a)$(无损),故 $r(b)=\tau(\kappa(b))=\tau(\kappa(a))=(b)$,即 $b$ 完好。∎ 所以字级往返没有长于 1 的环,不存在「反复重分词震荡」这种病。

**推论(不可达 token)。** 记 $\mathcal{A}_{\mathrm{reach}}$ = 出现在某个规范切分里的 token 集。$a\notin\mathcal{A}_{\mathrm{reach}}$ 的 token **可以被模型输出,但没有任何文本输入能产出它**。这类 token 在训练中从不作为输入出现、也从不作为目标出现,因此其输入嵌入与输出行向量都只受负类梯度。**理论由此推出一个已被独立观测的现象:词表里存在一批「只写不读」的 token,其嵌入停在近初始化状态、行为异常。** 这正是 glitch / undertrained token(SolidGoldMagikarp 那一类)。

**预注册预言(F3 承担):$\mathcal{A}\setminus\mathcal{A}_{\mathrm{reach}}$ 与文献中用嵌入指标独立找出的 undertrained token 集应高度重合。** 杀伤条件:若两集合的重合度不显著高于随机(同频段对照下),则本推导错——不可达不是欠训练的原因。**须查证的既有工作:用嵌入范数/logit 指标检测 undertrained token 的那一支(Land & Bartolo 一类),我尚未读,不预设其内容。**

## 六、改掉的与新增的

**改掉:**

- 第 51 条的措辞错在层级。「锚不是商的截面」→ **「单 token 不是自身表面形的规范切分;而在序列级,锚由无损性自动是截面」**。N1/N2 的适用域**不缩小**,改的是载体($\mathcal{A}\to\mathcal{A}^*$)与假设的性质(分词器性质 → 输入分布性质)。
- P2 §1 的三条后果里,第 1、3 条作废(适用域不缩小、判据在文本输入上地位不待定);第 2 条保留但改述(可测的分类条件从「锚是截面」改为「$\pi$ 是否规范」)。
- 原第 11 条与 N1/N2 **不再冲突**:第 11 条讲的是字级,N1/N2 用的是序列级,两者可同真。S4 §1 那个「真冲突」的裁决因此升级——不是一方适用域缩小,是**双方层级不同**。

**新增可测量,全部零 GPU:**

1. 三档比例(完好/碎裂/改名),按 type 与按语料频率各一份。
2. $\lvert\mathcal{A}\setminus\mathcal{A}_{\mathrm{reach}}\rvert$,及其与 undertrained token 集的重合度。
3. 非规范质量 $\pi(\mathcal{A}^*\setminus\mathcal{A}^*_{\mathrm{can}})$,在三条违反路径上各测一次(拼接 / 自生成 / 边界切割)。第二条需要模型生成,与 F2 同跑。
