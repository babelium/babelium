# A4 — 换算签名

A1–A3 攒下六条硬要求:界面成对(A1 §5)、两方向代数身份不同且随安放方式互换(A2 §4 / A3 §5)、须含锚点(A3 §4)、须带「哪一面原始」参数(A3 §2)、地址是商(A2 §1)、缺席须外加(A2 §6)。A4 写出满足全部六条的签名。

## 1. 签名

$$\mathcal{I} = \big(\Omega^+,\ \mathcal{A},\ \mathcal{X},\ c,\ \mathrm{anc},\ \mathrm{sel},\ \omega\big)$$

| 件 | 是什么 | 来自 |
|---|---|---|
| $\Omega^+ = \Omega \sqcup \{*\}$ | 内容域,缺席已外加 | A2 §6 |
| $\mathcal{A} = \Omega^+/\sim$ | 地址集,商 | A2 §1 |
| $\mathcal{X}$ | 相似性承载,路径连通局部可微 | A1 §2 |
| $c$ | 分级比较 | A1 §2 / A3 §3 |
| $\mathrm{anc}: \mathcal{A} \to \mathcal{C}$ | 锚点,$\mathcal{C}$ 是锚点所居的承载 | A3 §4 |
| $\mathrm{sel}$ | **选择函数,定在纤维上** | 本件 §3 |
| $\omega \in \{\text{id-prim}, \text{sim-prim}\}$ | 哪一面原始 | A3 §2 |

$\omega$ 决定 $\mathcal{C}$,进而决定哪个映射是给定的、哪个是导出的:

| | $\mathcal{C}$ | 给定 | 导出 |
|---|---|---|---|
| id-prim | $\Omega$ | $\mathrm{anc}$ 是地址的表面形 | $\kappa = $ 表面形拼接(同态) |
| sim-prim | $\mathcal{X}$ | $\mathrm{anc}$ 是锚点向量 | $\kappa = \mathrm{anc}$(截面) |

两栏的 $\mathrm{anc}$ 是同一件事,只是落在不同承载里:文本里 $\mathrm{anc}(a)$ 是地址 $a$ 拼出的串,VQ 里是码字向量。**$\omega$ 不改签名的形,只改 $\mathrm{anc}$ 的值域。** 这是六条要求能装进一个签名而不是两个的原因。

$\tau$ 两栏同式:

$$\tau(x) = \mathrm{sel}\Big(\arg\max_{a \in \mathcal{A}} c\big(x, \mathrm{anc}(a)\big)\Big)$$

id-prim 时 $c$ 是「是否匹配」(二值),$\arg\max$ 给出全体可行切分;sim-prim 时 $c$ 分级,$\arg\max$ 一般给单点但纤维仍是连续集。两栏都需要 $\mathrm{sel}$,理由不同——这是 §3。

## 2. 一个校正:承诺不是「分级压成二值」

A1 §3 把承诺定位在分级↔二值的落差上。A4 写签名时这个定位不够用,须收紧。

id-prim 侧的 $c$ 本来就二值(串匹配),没有分级可压。但分词**确实**有承诺:同一个字符串有多种合法切分,选哪一种不由匹配决定。BPE 靠 merge 优先序定,greedy 靠最长匹配定——**承诺落在这个定序上,不落在匹配上。**

sim-prim 侧,$\arg\max$ 选出唯一锚点,但被映到该锚点的 $x$ 构成一整个纤维 $\tau^{-1}(a)$,回来时只能给出锚点这一个代表。**承诺落在「拿锚点代表整个纤维」上。**

两侧的共同结构不是分级压二值,是:

$$\boxed{\text{承诺} = \text{在欠定的选择上取定一个代表}}$$

id-prim 的欠定是**有限多个切分**,sim-prim 的欠定是**一个连续纤维**。二者只差纤维的基数,不差性质。

**这是 A1 §3 的推广而非否定。** 分级→二值是欠定的一种来源(纤维为连续),不是唯一来源。落差本身仍在(A1 §3 那张表不动),但承诺的定位从落差移到欠定上——移到更一般的位置,且两个安放方式因此被同一句话覆盖。承诺不再依赖「有连续面在场」这个条件,于是纯离散的界面(如 KG 检索、character→word)也在覆盖范围内。

## 3. $\mathrm{sel}$ 是签名里唯一有实质自由度的件

$\Omega^+, \mathcal{A}, \mathcal{X}, c, \mathrm{anc}, \omega$ 都是「界面长什么样」;$\mathrm{sel}$ 是「界面做什么」。它定在纤维上:

$$\mathrm{sel}: \{\text{纤维}\} \to \mathcal{A} \ \text{或} \ \Delta(\mathcal{A})$$

值域是 $\mathcal{A}$ 则承诺确定,是 $\Delta(\mathcal{A})$(分布)则承诺随机。已证那级允许随机分词而仍 exact,正是 $\mathrm{sel}$ 取分布值且各纤维支撑不相交的情形——**随机性住在 $\mathrm{sel}$ 里,不住在别处。**

代入五个安放方式:merge 优先序 / $\arg\max$ 或多项式采样 / 最近邻 / 沿 timestep 摊薄的一串 $\mathrm{sel}$ / 粒度层的选定 / 检索的 top-$k$ 截断。B 阶段逐一填。

A5 接手的正是 $\mathrm{sel}$ 被调用的**那一次事件**:签名说它存在,A5 说它何时发生、能否观测。

## 4. 界面成对:$\omega$ 相反的两个界面复合

A1 §5 的成对现在可以写出来。标准 LM:

$$\Omega \xrightarrow{\ \mathcal{I}_{\text{in}},\ \omega = \text{id-prim}\ } \mathcal{X} \xrightarrow{\ \text{计算}\ } \mathcal{X} \xrightarrow{\ \mathcal{I}_{\text{out}},\ \omega = \text{sim-prim}\ } \Omega$$

两个界面的 $\omega$ **必然相反**——输入侧从内容进、输出侧从表示出。于是:

- 两侧各恰好一次 $\mathrm{sel}$ 调用,位置相反(A1 §5 的成对由此得证而非观察)。
- 两侧的 $\mathcal{A}$ 是两个独立的商,无须相同(输入输出词表可独立调规模那条实证)。
- 两侧坏掉的复合相反(A3 §5),故误差不同源、不可合并计。

VQ 自编码器同构:编码侧 sim-prim、解码侧 id-prim,$\omega$ 同样相反。**「界面成对且 $\omega$ 相反」是一条可检验的结构预言**,B 阶段五个实例都要查。

## 5. 交给后续

- **A5** — 只剩 $\mathrm{sel}$ 的事件学:何时被调、能否观测、可否测量。承诺的可测量化(North Star 的交付物)整个落在 A5。
- **B** — 九格按签名重排为七件 + 两栏($\omega$、坏掉的复合)。B0 模板照 §1 那张表写。
- **C** — $\Phi$-弱条件(A3 §6)现在有了定义域:$\Phi$ 是定在 $\mathcal{X}$ 或 $\Omega$ 上的检验函数族,而 $\mathrm{sel}$ 的随机性使条件须在期望下陈述。C3 的正式陈述形如:界面 $\mathcal{I}$ 在 $\Phi$ 下成立 $\iff$ $\mathbb{E}_{\kappa\tau p}[\phi] = \mathbb{E}_p[\phi], \forall \phi \in \Phi$,其中 $\tau$ 含 $\mathrm{sel}$。
- **D1** — §4 的「$\omega$ 必然相反」是第三个候选:从签名推出、可对照、且能否掉东西(若某实例两侧 $\omega$ 相同,签名错)。
- **D3** — 签名七件里无一要求离散。$\Omega$ 可连续、$\mathcal{A}$ 可不可数、$\mathrm{anc}$ 可连续参数化。第四次。
