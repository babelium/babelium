# K2 判决:杀「AR 是唯一跨空间安放」

**判决:杀成了,以第二种许可方式——AR 其实同空间。跨空间那一类是空集,唯一性因此空真而无内容。G0 §5「一个结构上的异类被当成了典型」作废。**

杀伤不需要外部证据。它从我自己的签名里就地倒掉:我把 $c$ 的值表当成了载体所居的空间。这是本项目至今最省的一次反驳,也是最该早发现的一次。

---

## 一、原断言与它错在哪

G0 §5 原文:AR「换算前活在 $\mathbb{R}^{|A|}$(logits),换算后活在 $\mathbb{R}^d$(embedding)」,故须外部构造一座桥。

签名是 $\tau(x) = \mathrm{sel}(\arg\max_a c(x, \mathrm{anc}(a)))$。**载体是 $x$,不是 $c(x,\mathrm{anc}(\cdot))$。** logits 是 $c$ 在全部地址上取值排成的表,不是 $x$ 住的地方。我把值表误当成了空间。

对照可知这个错有多硬:VQ 也要算 $\lVert z_e - e_k\rVert^2$ 对全部 $k$,那也是一个 $\mathbb{R}^{|A|}$ 的表,原文式 (1) 正是对它取 $\arg\min$。若 logits 使 AR 跨空间,则距离表同样使 VQ 跨空间。**两者要么都跨要么都不跨,而 VQ 原文明说同空间(§3.2 直通估计的依据)。** 结论只能是都不跨。

## 二、AR 同空间,逐行钉死

`D:\anaconda3\Lib\site-packages\transformers\models\llama\modeling_llama.py`:

- `:421` `hidden_states = self.norm(hidden_states)`(`:365` `self.norm = LlamaRMSNorm(config.hidden_size, ...)`)
- `:438` `self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)` —— **无 bias**
- `:487` `logits = self.lm_head(hidden_states[:, slice_indices, :])`
- `:430` `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`
- `:361` `nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)`

绑定生效时,把 $h_t$ 记作 $\mathrm{norm}(\text{hidden}_t) \in \mathbb{R}^d$:

$$\text{logit}_a = \langle h_t,\ \mathrm{emb}(a)\rangle,\qquad \tau = \arg\max_a \langle h_t, \mathrm{emb}(a)\rangle$$

$c$ 就是载体与锚之间的内积,$\mathrm{anc} = \mathrm{emb}$。换算前 $h_t \in \mathbb{R}^d$,换算后 $\mathrm{emb}(a_t) \in \mathbb{R}^d$。**同一个 $\mathbb{R}^d$,可直接相减。** 与 VQ 同型:AR 是残差流在嵌入码本上的**内积量化**。

**同空间这个结论与是否绑定无关**,须与上式区分开。不绑定时 $\text{logit}_a = \langle h_t, w_a\rangle$,上式失效;但 $w_a$ 与 $\mathrm{emb}(a)$ 都落在 $\mathbb{R}^d$,$h_t$ 也在 $\mathbb{R}^d$,故换算前后仍同空间。同空间只依赖两件事:载体是残差流、交出的是嵌入表的一行——二者皆由构造保证。绑定只决定「打分用的表与交出的表是否同一张」,即 §3 修正四那件事,不决定空间。

## 三、逐条修正

### 修正一:F0 的桥是我自己造出来的问题,不是架构给的

G0 §5 说 F0 那个式子「是构造,不是架构自带的运算」,并把这归因于跨空间。现在:$h_t - \mathrm{emb}(a_t)$ 直接可减,提交落差与 VQ 的 $\lVert z_e - e_k\rVert$ 同型。**桥不需要造。** 需要绕道 $\Phi(\sum_a p_t(a)\mathrm{emb}(a))$ 的唯一原因,是我把切口开在了 logits 与 embedding 之间;切口开在 $h_t$ 与 $\mathrm{emb}(a_t)$ 之间就没有这个问题。切口位置由签名指定($x$ 是载体),所以原切口开错了。

### 修正二:换算点的坐标要改

G1 行原记「换算点在 `generation/utils.py:2789-2793`」。那是 $\mathrm{sel}$ 被执行的**代码位置**,不是换算界面的**空间位置**。换算界面是 $h_t \mapsto \mathrm{emb}(a_t)$ 这个 $\mathbb{R}^d \to \mathbb{R}^d$ 的映射,其像恰为码本 $\{\mathrm{emb}(a)\}$。代码位置与界面位置须分列,原表混作一格。

### 修正三:唯一残留的不对称不是空间,是范数

$\arg\max_a \langle x, e_a\rangle$ 与 $\arg\min_a \lVert x - e_a\rVert^2$ 等价当且仅当 $\lVert e_a\rVert$ 对 $a$ 为常数(因 $\lVert x-e\rVert^2 = \lVert x\rVert^2 - 2\langle x,e\rangle + \lVert e\rVert^2$,只有 $\lVert e\rVert^2$ 项妨碍)。

**故 AR 是内积量化,VQ 是欧氏量化;二者重合当且仅当嵌入范数齐一。** 这是同空间之后剩下的全部差别,且它是个数值事实,可测(查 `embed_tokens.weight` 的行范数散布),不是理论选择。

与此配对的第二个不对称有出处支撑:VQ 有 $\beta$ 项「To make sure the encoder commits to an embedding and its output does not grow」(1711.00937 §3.2),**AR 没有任何项约束 $h_t$ 待在码本附近**。`:421` 的 RMSNorm 只管 $h_t$ 自身的尺度,不管它与 $\{\mathrm{emb}(a)\}$ 的相对尺度。于是可测预言:AR 的 $\lVert h_t - \mathrm{emb}(a_t)\rVert$ 无界压制,VQ 的有。这条把 K1 §4 的「梯度给纤维的哪一侧」接上了 AR 行——AR 两侧都不给。

### 修正四(新发现,比杀伤本身重):不绑定权重时,签名的单一 anc 不够用

`:430` 的绑定受 `config.tie_word_embeddings` 控制,不是所有模型都绑。不绑时 $\text{logit}_a = \langle h_t, w_a\rangle$ 而交出的是 $\mathrm{emb}(a_t)$,且 $w_a \neq \mathrm{emb}(a)$。即:

$$c \text{ 用的锚} = w_a,\qquad \kappa \text{ 交出的锚} = \mathrm{emb}(a)$$

**两个锚映射。签名 $\mathcal{I} = (\Omega^+, \mathcal{A}, \mathcal{X}, c, \mathrm{anc}, \mathrm{sel}, \omega)$ 只有一个 $\mathrm{anc}$,表达不了这件事。** 而 VQ 是单锚:$e_k$ 既进式 (1) 的距离又进式 (2) 交给解码器(原文「The input to the decoder is the corresponding embedding vector $e_k$ as given in equation 2」)。

这不是措辞问题,是签名的表达力缺口。且它给出一个新的二值区分:**选谁用的锚与交出的锚是否同一个。** 绑定的 Llama、VQ 皆同一;不绑的 LM 不同一。此列与纤维可见性、与提交时刻均不重合。

**按 S3 的轴门追问「若无此轴,哪个已知现象无法解释」:它答得出——绑定与不绑定在小模型上的实测差异(绑定省参数且常不掉分)在单锚签名下无从表述,因为单锚签名根本写不出「不绑定」这个配置。** 但门要在 S3 统一过,此处不放行。

## 四、覆盖边界

- 未覆盖 softmax 之后:$\arg\max$ 对 softmax 单调不变,故 §2 的推导与是否过 softmax 无关;但**采样**(`utils.py:2791` 的 `multinomial`)不是 $\arg\max$,采样下「载体到最近锚」这个几何说法失效,交出的可能是任意锚。同空间结论不受影响(仍是 $\mathbb{R}^d\to\mathbb{R}^d$),几何解释受影响。
- 未覆盖 MoE 路由、未覆盖带 bias 的 lm_head(有 bias 时 $c$ 是仿射而非内积,锚的说法要改写)。
- 未查:`embed_tokens.weight` 的行范数散布(修正三的数值前提)。本机无本地权重,须实测或查文献。
- G0 §5 的表格其余各行(VQ/分箱/层级/RAG 同空间)不受影响,倒的只是「AR 在异空间那一列」以及由此得出的唯一性与「异类被当成典型」。

## 五、新增待测

1. `embed_tokens.weight` 行范数散布:决定内积量化与欧氏量化差多远。
2. AR 的 $\lVert h_t - \mathrm{emb}(a_t)\rVert$ 是否随训练发散(修正三的预言),与 VQ 有 $\beta$ 项者对比。
3. 双锚是否可折叠:是否存在把 $w_a$ 与 $\mathrm{emb}(a)$ 统一处理的写法(如让 $\mathrm{anc}$ 取值在 $\mathbb{R}^d\times\mathbb{R}^d$),抑或必须承认签名要两个锚。此项决定签名要不要改,优先。

## 六、交出的规格(进产物三)

- **S-05**:判据不得预设两面不可比而引入外部桥;AR 的两面同在 $\mathbb{R}^d$,提交落差可直接相减。来源 §2(`modeling_llama.py:421/430/438/487`)。**硬约束。** 来源倒的条件:仅当载体被证明不是残差流,或交出的被证明不是嵌入表的一行。**与 `tie_word_embeddings` 无关**——绑定只管打分的表与交出的表是否同一张(S-06/S-07 管那件事),不管空间。此处初稿曾把本条的存亡挂在绑定上,是错的,已改。
- **S-06**:判据须能表述「$c$ 用的锚」与「$\kappa$ 交出的锚」不是同一个映射的情形,否则不覆盖不绑定权重的 LM。来源 §3 修正四。**硬约束,且它同时要求签名扩容。**
- **S-07**:判据涉及「最近」时须指明度量;内积极大与欧氏极小等价仅当锚范数齐一。来源 §3 修正三。**硬约束。**
- **S-08**:S3 的适用域声明中「排除跨空间安放」这个许可答案**现已无用**——跨空间类为空。S3 文件里那一条须划掉。来源本判决。**硬约束(删除类)。**

## 七、状态

杀伤二:**杀成了**。跨空间类空集;唯一性空真;G0 §5 结论段作废,§5 表格中 AR 移入同空间列。
副产品:双锚缺口(签名表达力,最重)、内积量化 vs 欧氏量化的精确条件、AR 无提交约束项这一可测不对称。
S3 连带受影响:S-08 要求删掉它的一个许可答案;F0/F2 的桥式简化。
杀伤清单**清空**。按停止条件,S2 的杀伤阶段结束,转产物一(面相矩阵)与产物三(规格清单)。
