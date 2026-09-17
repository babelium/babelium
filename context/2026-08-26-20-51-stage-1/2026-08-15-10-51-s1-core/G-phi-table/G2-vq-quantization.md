# G2 行:VQ 量化

**取材方式**:论文原文式子。van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*, arXiv:1711.00937(v2, 2018-05-30)。式号为原文所印。

---

## 六格

### 格 1 — 换算点

原文式 (1),后验被定义成 one-hot:

$$q(z=k\mid x) = \begin{cases} 1 & k = \arg\min_j \lVert z_e(x) - e_j \rVert_2 \\ 0 & \text{otherwise}\end{cases}$$

原文措辞:"The posterior categorical distribution $q(z|x)$ probabilities are defined as one-hot as follows"。

$\mathrm{sel}$ 在这里是确定性的 $\arg\min$,无随机性。$\mathrm{anc}$ 的像是码本 $\{e_j\}$。

### 格 2 — 交给谁

解码器 $p(x \mid z_q(x))$,即原文式 (3) 第一项里的那个条件分布。

### 格 3 — 交出什么(最要紧的一格)

原文式 (2):

$$z_q(x) = e_k, \quad k = \arg\min_j \lVert z_e(x) - e_j \rVert_2$$

原文明确写出接收方拿到的是什么:"The input to the decoder is the corresponding embedding vector $e_k$ as given in equation 2." 另一处呼应:categorical 样本 "index an embedding table","These embeddings are then used as input into the decoder network."

**交出的是码字向量,不是索引。** 且原文第 3.2 节给出一句决定性的结构事实:直通估计之所以成立,是因为编码器输出与解码器输入**处在同一个 $D$ 维空间**,梯度可以从 $z_q(x)$ 原样抄回 $z_e(x)$。

这一点把我填表前的预判判错了。我原以为 VQ 与 AR 同属「只交出地址」,实则不同:AR 换算前活在 $\mathbb{R}^{|A|}$、换算后活在 $\mathbb{R}^d$,两个空间;VQ 换算前后**同一个空间**,$z_e$ 与 $z_q$ 可以直接相减。G1 格 5 里那个「桥是我构造的」的说法在 VQ 这里不成立——这里的桥是架构自带的。

### 格 4 — 接收方读什么

两个接收方,不是一个。

**接收方甲:解码器。** 读 $e_k$,算重建对数似然,即式 (3) 第一项 $\log p(x \mid z_q(x))$。

**接收方乙:目标函数本身。** 式 (3):

$$L = \log p(x\mid z_q(x)) + \lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2 + \beta \lVert z_e(x) - \mathrm{sg}[e]\rVert_2^2$$

后两项读的是 $z_e(x) - e$,即**换算前的对象与换算后的对象之差**。原文 $\beta = 0.25$,报告在 $0.1$ 到 $2.0$ 之间稳定。

第三项在原文里就叫 commitment loss。也就是说:这个界面的提交落差不是我要去测的东西,是架构**自己已经在算并且在惩罚**的一项。

### 格 5 — $\Phi$(由 3、4 推出)

由格 3,交出的对象是 $\mathbb{R}^D$ 中的向量;由格 4,它被两个不同的消费者读。因此这一行的 $\Phi$ 必须分两层写,合并会丢掉信息:

$$\Phi_{\text{VQ}}^{\text{dec}} = \{\,\phi:\ \phi \text{ 是 } e_k \text{ 的函数}\,\}$$

$$\Phi_{\text{VQ}}^{\text{obj}} = \Phi_{\text{VQ}}^{\text{dec}} \ \cup\ \{\,\lVert z_e - e_k\rVert_2^2\,\}$$

第二个量族严格更大,且大出来的那一项**恰好是纤维内的位置**。

### 格 6 — 纤维可见性(由 3 推出)

**分裂:对解码器不可见,对目标函数可见。**

纤维在这里是 $\{z : \arg\min_j \lVert z - e_j\rVert = k\}$,即 Voronoi 胞。

- 对解码器不可见,遮挡物是式 (2) 的赋值 $z_q(x) = e_k$:整个胞内的 $z_e$ 塌成同一个 $e_k$,解码器无法区分胞内任何两点。
- 对目标函数可见,通道是式 (3) 第三项:$\lVert z_e - \mathrm{sg}[e]\rVert^2$ 直接读出 $z_e$ 在胞内离中心多远。另一条通道是直通估计,凭的是格 3 那句「同一个 $D$ 维空间」。

---

## 本行结论

- 交出的是向量不是索引,换算前后同空间。
- 纤维可见性**分裂**成两个消费者两个答案,这是 G1 没有的结构。
- 提交落差在 VQ 里是架构自带的一项(commitment loss),在 AR 里要外部构造。

## 与 G1 对照产生的一条新东西

AR 与 VQ 还差另一件事,是填这两行时才对上的:

**AR 的 $\mathrm{sel}$ 在训练期从不被调用。** teacher forcing 下,`generation/utils.py:2789-2793` 那段根本不在训练图里,提交只发生在推理。而 VQ 的式 (1) 在训练期每一步都执行,且式 (3) 第三项为它付代价。

于是同一个签名 $\mathcal{I}$ 的两次安放,在「提交是否被训练目标看见」这一项上取了相反值。这不是我事先列出的差异维度,是从格 4 掉出来的。它落在何处待汇总判断,先登记。

## 待钉

- 式 (3) 三项与 $\beta$ 数值取自原文,已引原句。若 S6 要引用,需回到 PDF 正文再核一次印刷式号(当前经 ar5iv 渲染版读取)。
