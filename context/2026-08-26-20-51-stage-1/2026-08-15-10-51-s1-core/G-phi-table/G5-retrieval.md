# G5 行:检索(RAG ↔ KG 两极)

**取材方式**:论文原文。Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv:2005.11401;Edge et al., *From Local to Global: A Graph RAG Approach*, arXiv:2404.16130。

**填表前的预判**:两极「一个交出内容、一个交出身份」,故分开填。**判定结果:预判错。两极最终都交出文本。** 下面是理由。

---

## 5a 极:RAG

### 格 1 — 换算点

原文 §2.1(此渲染版未印式号,登记为未编号展示式):

$$p_{\text{RAG-Seq}}(y|x) \approx \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y|x,z)$$

$$p_{\text{RAG-Tok}}(y|x) \approx \prod_i^N \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i|x,z,y_{1:i-1})$$

**换算点不在「选中哪篇」,而在 top-$k$ 的截断。** 原文对 $z$ 的处理是边缘化而非取定:"treat $z$ as a latent variable and marginalize over seq2seq predictions";RAG-Sequence "marginalize[s] it via a top-K approximation";RAG-Token "can draw a different latent document for each target token and marginalize accordingly"。

$k$:训练用 $k\in\{5,10\}$,两档无显著差异;测试期(Appendix A)开放域 QA 的 RAG-Token 用 15,RAG-Sequence 配 Thorough Decoding 用 50,MS-MARCO 与 Jeopardy 用 10。RAG-Sequence 在 NQ 上取更多单调变好,RAG-Token 约在 10 处见顶。

### 格 2 — 交给谁

生成器(BART)。

### 格 3 — 交出什么

原文 §2.3:

> "To combine the input $x$ with the retrieved content $z$ when generating from BART, we simply concatenate them."

交出的是**段落原文**,不是文档 ID。原文强调记忆 "is comprised of raw text rather [than] distributed representations"。

### 格 4 — 接收方读什么

$p_\theta(y_i \mid x, z, y_{1:i-1})$,即把 $z$ 的文本当普通上下文读。RAG-Token 每步可换 $z$,RAG-Sequence 整序列固定 $z$ 再在 $k$ 篇间加权。

### 格 5 — $\Phi$

$$\Phi_{\text{RAG}} = \{\,\text{生成器条件在 } (x,z) \text{ 上的下一 token 分布}\,\}$$

即 $\Phi$ 就是 G1 那一族,因为 $z$ 被 concat 进上下文后走的是同一条路。

### 格 6 — 纤维可见性

**top-$k$ 内完全可见,top-$k$ 外完全不可见。**

- 内:$k$ 篇全部被读,权重 $p_\eta(z|x)$ 显式出现在边缘化式里。这里**没有提交**——纤维内每个代表都参与了计算。
- 外:第 $k+1$ 篇及以后不可见,遮挡物是 top-$k$ 截断本身。

---

## 5b 极:KG(GraphRAG)

### 格 1 — 换算点

两处,且**分处两个不同时刻**。

**索引期**:图 → 文本。原文:"Entity descriptions are aggregated and summarized for each node and edge";社区摘要由 "adding various element summaries (for nodes, edges, and related claims) to a community summary template" 构成;高层递归自低层:"Community summaries from lower-level communities are used to generate summaries for higher-level communities"。

**查询期**:摘要 → 分块。"Community summaries are randomly shuffled and divided into chunks of pre-specified token size"。

### 格 2 — 交给谁

查询期的 LLM。map 步 "the summaries are used to provide partial answers to the query independently and in parallel";reduce 步 "Intermediate community answers are sorted in descending order of helpfulness score"。

### 格 3 — 交出什么(判定所在)

**自然语言文本,不是实体 ID,不是图结构。**

> "Given a user query, the community summaries generated in the previous step can be used to generate a final answer"

提示词侧证:社区答案提示把输入当报告("responding to questions about a dataset by synthesizing perspectives from multiple analysts"),全局答案提示指 "questions about data in the tables provided",而那些 table 是社区报告。

ID 只以引用记号残留于文本中("Points supported by data should list the relevant reports as references")。实体与关系的数字 ID 确实出现在索引期的报告生成输入里(形如 `id,source,target,description`),但那一步在查询之前,其产物是文本报告。

### 格 4 — 接收方读什么

把摘要文本当普通上下文读,与 5a 同。

### 格 5 — $\Phi$

$$\Phi_{\text{KG}} = \Phi_{\text{RAG}}$$

同一族。差别不在 $\Phi$,在换算发生的**时刻**:KG 把图→文本的换算前置到索引期,查询期的 LLM 面对的已是文本。

### 格 6 — 纤维可见性

**不可见,且遮挡物有两层。**

- 第一层:图→文本摘要。同一子图的不同拓扑细节被摘进同一段文本后不可区分,遮挡物是摘要模型本身(有损且不可逆)。
- 第二层:社区检测决定哪段文本进哪个块——图结构**间接塑造了检索**却对查询期 LLM 不可见。原文的这一点值得单记:图在起作用,但不在 $\Phi$ 里。

---

## 本行结论

- 两极的格 5 **相同**,预判的「内容 vs 身份」对立在 $\Phi$ 层面不成立。
- 真差别在两处别的地方:**换算时刻**(RAG 查询期 / KG 索引期),与**是否提交**(RAG 在 top-$k$ 内不提交 / KG 的摘要是一次不可逆提交)。
- RAG 给出全表唯一一个「纤维内不提交」的实例:边缘化而非取定。这是 $\mathrm{sel}$ 可以退化到「不调用」的存在性证据。

## 待钉

- 2005.11401 的两式在此渲染版**无印刷式号**,已如实登记为 §2.1 未编号展示式。S6 引用需回 PDF 确认是否有编号。
- 2404.16130 的引文均为散文句,无式号,正常。
