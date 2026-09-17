# K1 判决:杀「纤维对训练目标可见 ⟺ 提交代价被训练」

**判决:杀伤未中实质,但击中措辞。等价本身活下来并从六比零变成八比零;而它原来的写法必须废掉,因为「被训练」一词有两种读法,两种读法的真值不同。**

预注册的杀伤条件是「REINFORCE 候选成立则等价当场倒」。实测:成立与否取决于「被训练」怎么读——**预注册条件本身欠定**。这与 F1 判决 §2 的错误同类(那次是把「下沉应消失或减弱」写成了杀伤条件,实测是迁移守恒),即我写杀伤条件时反复在被测命题的关键词上留了歧义。第三次了,记在 §5。

---

## 一、两种读法,先分开写

设签名 $\mathcal{I}$ 的商 $\mathcal{A} = \Omega/\!\sim$,纤维即某地址 $a$ 的原像 $\{x : \tau(x) = a\}$,**纤维内位移**指载体 $x$ 与代表 $\mathrm{anc}(a)$ 之差所携带的那个量。

- **宽读**:提交这个操作进了训练图(有梯度经过它)。
- **严读**:纤维内位移这个量进了更新规则(目标函数或别的规则),即优化过程读得到载体在纤维里的位置。

G 表填的时候我用的是「可见/不可见」,默认了这两种读法同义。它们不同义。

## 二、REINFORCE:宽读下是反例,严读下不是

出处 `D:\anaconda3\Lib\site-packages\torch\distributions\__init__.py`:

- `:34` 更新式 $\Delta\theta = \alpha r \dfrac{\partial \log p(a|\pi^\theta(s))}{\partial\theta}$
- `:16-17` **「Whilst the score function only requires the value of samples $f(x)$, the pathwise derivative requires the derivative $f'(x)$.」**
- `:10-11` 「It is not possible to directly backpropagate through random samples.」
- `:46-49` 分类策略的实现:`m = Categorical(probs)` / `action = m.sample()`

`:16-17` 那句是本判决的枢轴。score function 只要 $f$ 在采样点上的**值**。把 $f$ 换成解码器损失、把 action 换成地址:回传的是一个标量 $r$ 乘一个只依赖 $\log p(a)$ 的梯度。$r$ 是在**已提交的那个点上**求出的数,不含载体在纤维里偏离代表多少的信息。

于是:

- 宽读:提交在训练图里(`:34` 的梯度正是穿过 `sample()` 那一步的替代),而纤维对解码器不可见 → **(不可见, 被训练)**,等价的 $\Leftarrow$ 向当场倒。
- 严读:纤维内位移从未进入任何更新规则 → **(不可见, 未被训练)**,等价成立。

**取严读。** 理由不是严读更方便,是宽读把等价讲成了废话的对立面:宽读的「被训练」只问梯度有没有经过某个位置,那它与「纤维」这个概念不发生关系,等价两端说的不是一回事。而 S1 建立纤维可见性这个区分量,要的正是「优化过程知不知道商掉了什么」。严读是这个意思的形式化,宽读不是。

代价须记:这一取舍使 REINFORCE 与 AR 在本等价下同类(都是「不可见, 未被训练」),而两者在别处显然不同(AR 的 $\mathrm{sel}$ 训练期不调用,REINFORCE 的每步都调用且付梯度)。**本等价因此不区分这两者;要区分得靠读者结构那一列的「提交时刻」。**

## 三、严读非平凡的证据:$\mathrm{sg}$ 的存在

严读若与「在目标函数里出现」同义,则等价近乎恒真,那就是 F-占位符那个陷阱升一级(S3 文件已登记该陷阱)。它不同义,证据是 VQ 原文的停梯度算子,arXiv:1711.00937 式 (3):

$$L=\log p(x|z_q(x))+\lVert \mathrm{sg}[z_e(x)]-e\rVert_2^2+\beta\lVert z_e(x)-\mathrm{sg}[e]\rVert_2^2$$

原文定义:$\mathrm{sg}$ 「is defined as identity at forward computation time and has zero partial derivatives」,作用是「effectively constraining its operand to be a non-updated constant」。

即:**中间项含有纤维内位移、且在目标函数里、且对 $z_e$ 的梯度恒为零。**「在目标函数里」与「梯度到得了载体」是两个独立取值的性质。严读不退化。

## 四、逐条修正

### 修正一:G0 表 G2 行那一格把两项当成了一项

我原来写「对目标可见,commitment loss 显式付费」,量记作 $\lVert z_e - e_k\rVert^2$。实际上**同一个纤维内位移在式 (3) 里出现两次,$\mathrm{sg}$ 位置相反**,原文对归属讲得很死:

> 「The decoder optimises the first loss term only, the encoder optimises the first and the last loss terms」,而「the embeddings are optimised by the middle loss term」。

中间项(原文另编为式 (4))把代表拉向载体——原文:「The VQ objective uses the l2 error to move the embedding vectors $e_i$ towards the encoder outputs $z_e(x)$」,且「Because this loss term is only used for updating the dictionary」。$\beta$ 项把载体拉向代表——原文:「To make sure the encoder commits to an embedding and its output does not grow, we add a commitment loss」。

**结论:纤维可见性不是纤维与目标之间的二元属性,它带方向——梯度给纤维的哪一侧。** 这是一个格子级的实错,不是措辞。

### 修正二:「被训练」不是二值,至少三档,因为存在非梯度通道

附录 A.1「VQ-VAE dictionary updates with Exponential Moving Averages」:EMA「to update the dictionary items instead of the loss term from Equation 3」,替掉的是式 (4) **中间项那一项**,式 (6)(7)(8):

$$N_i^{(t)} := N_i^{(t-1)}\gamma + n_i^{(t)}(1-\gamma),\quad m_i^{(t)} := m_i^{(t-1)}\gamma + \textstyle\sum_j z_{i,j}^{(t)}(1-\gamma),\quad e_i^{(t)} := m_i^{(t)}/N_i^{(t)}$$

$\gamma = 0.99$,原文说「This update is typically used in algorithms such as K-Means」,并注明该 EMA 变体 not used for the experiments in this work。$\beta$ 项在 EMA 下保留(编码器输出仍需被约束住不长大),重建项亦保留。

后果:**代表可以被纤维内统计量更新,而这个更新不在任何损失项里、也不是梯度。** 于是通道至少三档:

1. 梯度到载体($\beta$ 项)
2. 梯度到代表(中间项)
3. 非梯度更新代表(EMA 式 (6)(7)(8))

「可见性」若定义在目标函数上,第 3 档就成了隐形的——EMA 版 VQ 的码本明明每步都在读纤维内位置。**故可见性须定义在更新规则上,不是定义在目标函数上。**

### 修正三:计数从六比零改为八比零,且这不是「更多支持」

严读下逐行复核,新增两行:

| 行 | 纤维内位移进更新规则 | 提交代价被训练(严读) | 通道 |
|---|---|---|---|
| G1 AR | 否 | 否 | — |
| G2 VQ | 是 | 是 | 1 + 2 |
| G3′ 分箱 | 是(整箱积分对 $\mu_\theta$ 在箱内的位置敏感) | 是 | 1 |
| G4 层级 | 是(理想化偏离的标量后果) | 是 | 1 |
| G5a RAG | 是(top-$k$ 全读并边缘化,检索器受训) | 是 | 1 |
| G5b KG | 否(索引期,无梯度) | 否 | — |
| **K1 REINFORCE** | 否 | 否 | — |
| **K1 EMA-VQ** | 是 | 是 | 1 + 3 |

八比零。但**八比零的说服力不比六比零高多少**:新增两行是我为撞这条等价专门去找的,选样偏向能撞的方向,不是随机样本。真正的增益在通道那一列——它是这次撞出来的,原表没有。

## 五、覆盖边界(本判决不覆盖什么)

- **不覆盖 Gumbel-Softmax / 直通型松弛**。这一族在训练期让载体连续通过、推理期才硬提交,纤维内位移进不进更新规则取决于温度退火到哪一步,可能落在「部分」。未查。
- **不覆盖 RL 微调下的 AR**(PPO/GRPO 那一族)。按 §2,那里 $\mathrm{sel}$ 每步调用且付梯度,宽读为「被训练」严读为「否」,与 REINFORCE 同型;但它同时使 G1 行的「提交时刻=推理期」失效(RLHF 把提交搬进了训练期)。这一条直接威胁读者结构那张表,不威胁本等价。**登记为待测,优先级高于本条其余。**
- **不覆盖「纤维内位移」在无度量的商上怎么定义**。G1/G5b 判「否」用的是「没有任何规则读它」,不需要度量;但要判「是」得能说出位移是什么量,而 AR 的商($\mathcal{A}$ 上无自然度量)若哪天要判「是」,现在的措辞给不出量。这是本判决留给 S3 的坑。
- 预注册条件欠定这件事第三次发生(F1 一次,本次一次,加 F-占位符陷阱同源一次)。**从下一条杀伤起,杀伤条件必须写成「若 X 则倒」且 X 里的每个词已在别处定义过**,否则不算预注册。

## 六、新增待测

1. RL 微调后 AR 的提交时刻(见 §5),入口:是否存在训练期调用 `sel` 的路径。
2. Gumbel-Softmax 退火全程的纤维内位移可见性,可能给出本等价的第一个「部分」值。
3. EMA 版 VQ 与梯度版 VQ 的码本利用率差异——若差异显著,则通道 3 与通道 2 不只是实现选择,而是可测的不同后果。此项可查文献。

## 七、交出的规格(进产物三)

- **S-01**:判据不得把「提交代价被训练」当原语使用;必须指明梯度或更新给纤维的哪一侧。来源 §4 修正一(VQ 式 (3) 的两项 $\mathrm{sg}$ 反置)。**硬约束。** 来源倒的条件:若 VQ 原文对三项归属的陈述被读错,本条作废。
- **S-02**:判据里的「可见」必须定义在更新规则上,不得定义在目标函数上;否则漏掉非梯度通道。来源 §4 修正二(附录 A.1 式 (6)(7)(8))。**硬约束。**
- **S-03**:判据须能区分「提交操作在训练图里」与「纤维内位移进更新规则」;二者在 REINFORCE 上取相反值。来源 §2(`torch/distributions/__init__.py:16-17`)。**硬约束。**
- **S-04**:本等价不区分 AR 与 REINFORCE(§2 末),故判据若要区分二者须另取「提交时刻」为坐标。**偏好**,非硬约束。

## 八、状态

杀伤一:**未杀掉,理由已给**(严读下无反例,且严读是本项目要的那个意思,非贪图方便)。按 S2 的门,「杀不动并说明理由」是产出。
副产品三项:通道三档、纤维可见性带方向、预注册措辞的第三次同类失误。
杀伤清单剩:杀伤二(AR 是唯一跨空间安放)。
