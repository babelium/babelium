# G4 行:层级粒度(蛋白:残基 ↔ 原子)

**取材方式**:论文原文。Jumper et al., *Highly accurate protein structure prediction with AlphaFold*, Nature 596 (2021),补充材料(SI)。以下页码为 SI 抽文本后的页序,Algorithm 号为原文所印。

---

## 六格

### 格 1 — 换算点

SI Algorithm 24 "Compute all atom coordinates"(SI p.31),前置说明在 1.8.4(SI p.30):

> "The Structure Module predicts backbone frames $T_i$ and torsion angles $\vec\alpha^f_i$ ... constructed by applying the torsion angles to the corresponding amino acid structure **with idealized bond angles and bond lengths**."

换算即此:残基级的(骨架框架 + 扭转角)→ 全原子坐标。**加粗处是商映射的所在**:真实分子中键长键角逐个不同,这里被取定成文献理想值。

上游的参数化在 SI p.24 交代:

> "To obtain all atom coordinates, we parameterize each residue by torsion angles."

原子按对扭转角的依赖分成 rigid group(SI p.24,Table 2):"We group the atoms according to their dependence on the torsion angles into 'rigid groups'","the position of atoms in the $\chi_2$-group depend on $\chi_1$ and $\chi_2$, but do not depend on $\chi_3$ or $\chi_4$"。

$\mathcal{A}$ 在这一行 = (残基类型, 框架, 扭转角) 三元组;$\mathrm{anc}$ = 该类型残基的文献理想几何;$\mathrm{sel}$ 无自由度(理想值查表,确定性)。

### 格 2 — 交给谁

两个接收方,与 G2 同构但通道不同。

**甲**:FAPE 损失,SI Algorithm 20 line 28 / Algorithm 28。
**乙**:结构模块下一层(8 层迭代),经 line 17 的中间损失。

### 格 3 — 交出什么

交出的是**全原子坐标** $\vec x^a_i$ 与全 rigid group 框架 $T^f_i$(SI p.25):

> "After the 8 layers, the final backbone frames and torsion angles are mapped to frames for all rigid groups (backbone and side chain) $T^f_i$ and all atom coordinates $\vec x^a_i$"

交出的是 $\mathbb{R}^3$ 中的点集,不是索引。但这些点**由地址完全决定**:给定(残基类型, 框架, 扭转角),Algorithm 24 是确定性函数。所以下游看到的原子坐标里,不含任何「真实键长键角偏离理想值」的信息。

### 格 4 — 接收方读什么

**甲(最终 FAPE)**,SI p.34:

> "final FAPE loss (Algorithm 20 line 28) scores all atoms in all backbone and side chain frames"

即 SI p.25 的 "assesses all atom coordinates (backbone and side chain)"。它把交出的原子坐标与**真实**原子坐标在各框架下对齐比较。真实坐标带有真实键长键角。

**乙(中间层)**,SI p.25:

> "The intermediate FAPE loss operates only on the backbone frames and $C\alpha$ atom positions to keep the computational costs low. For the same reason the side chain are here only supervised by their torsion angles."

即中间层读的是更粗的量:只有骨架框架与 $C\alpha$;侧链只经扭转角损失(Algorithm 27)监督。

总损失中 FAPE 权重 0.5(SI p.33):$0.5L_{\text{FAPE}} + 0.5L_{\text{aux}} + 0.3L_{\text{dist}} + 2.0L_{\text{msa}} + 0.01L_{\text{conf}}$。

### 格 5 — $\Phi$(由 3、4 推出)

由格 3,交出的是地址决定的原子坐标;由格 4,有两个不同粒度的读者。故:

$$\Phi^{\text{final}}_{\text{AF}} = \{\,\text{全部原子(骨架+侧链)在各 rigid group 框架下的对齐坐标}\,\}$$

$$\Phi^{\text{inter}}_{\text{AF}} = \{\,\text{骨架框架},\ \vec x_{C\alpha},\ \vec\alpha^f\,\}$$

$\Phi^{\text{inter}} \subsetneq \Phi^{\text{final}}$,且真包含关系由原文明说的省算理由造成("to keep the computational costs low")。

**这一行给出一件前三行都没有的东西:同一个界面,$\Phi$ 随深度变化。** 不是两个消费者读同一批量的不同子集(G2 是那样),而是同一条前向链上不同层的读者读不同粒度,且理由是算力预算。

### 格 6 — 纤维可见性(由 3 推出)

**部分可见,且可见的部分与不可见的部分能明确划线。**

纤维 = 与给定(类型, 框架, 扭转角)一致的全部真实原子构型,即键长键角在真实值域内变动所扫出的集合。

- **不可见的**:纤维内沿「键长键角偏离理想值」方向的变动。遮挡物是 1.8.4 那句 "idealized bond angles and bond lengths"——理想值是查表常量,任何真实偏离在 Algorithm 24 的输出里没有落点。
- **可见的**:FAPE 把输出原子与真实原子比较(SI p.34 "scores all atoms"),所以**理想化误差以损失值的形式被读到**,但只是其标量后果,不是纤维内的坐标。
- 另有一处结构性不可见:180° 旋转对称的 rigid group 造成命名歧义(SI p.31,Table 3),原文以「全局一致的重命名」处理。这是 $\sim$ 本身的歧义,不是纤维可见性问题,单列。

---

## 本行结论

- 换算 = 残基级地址 → 理想几何全原子,商的方向是键长键角。
- $\Phi$ **随深度变化**(中间层粗、最终层细),理由是算力,不是理论。前三行无此现象。
- 纤维部分可见:方向明确(理想化偏离不可见),标量后果可见。

## 待钉

- 页码为 SI 抽文本页序,Algorithm 号为原文印刷号。S6 引用宜以 Algorithm 号为准,页码复核。
