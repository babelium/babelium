# Why do LLMs attend to the first token?

## 论文身份

- arXiv:2504.02732；COLM 2025
- 作者：Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu
- 主题：首 token sink 与 Transformer 的 over-mixing / over-squashing。

## 第一遍速读

核心判断：论文将首 token sink 解释为控制信息混合、降低 token 扰动敏感性的结构性手段，并预测深度、上下文长度和 data packing 会系统改变 sink 强度；值得全文阅读，`read_deeper = true`。

摘要的主结果原文：“In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers.”（PDF p.1，Abstract）

负重图表：Figure 1（理论核心示意）；Figure 2（有/无 BOS 的扰动实验）；Figure 3–4（sink head 与 apostrophe head）；Figure 5–6（训练、规模、上下文规律）；Table 2–3（data packing 与移除 BOS）。结论把 sink 强度随规模和上下文增长归因于模型对扰动更脆弱。

## 第二遍理解

论文把 attention 看作跨层反复施加的 mixing operator：深度和上下文增长会放大一个 token 扰动对其他 token 表示的影响，并推动 rank / representational collapse。首 token sink 把一部分 attention 质量导向低范数 BOS value，相当于让某些 heads 默认近似 no-op，从而降低有效 mixing rate；在需要局部模式时 head 才转向有内容的 token。理论 Jacobian 上界给出深度、head 数、context length 与敏感度的联系；Gemma 7B 扰动实验、从头训练的 context-length 与 packing 消融、Llama 3.1 尺度对照均符合预测。论文解释的是“为什么 sink 有用”，不是 sink key/value 的完整生成机制。

## 关键实验与限制

- Gemma 7B：把 “greatest” 替换为 token 数不变的 “best”，移除 BOS 后扰动在各层和位置传播更强；若干 heads 的 attention 也更平滑（PDF pp.5–6，Figures 2–3）。
- 近似 no-op 个案：apostrophe head 在条件满足时关注 apostrophe，否则关注 BOS；BOS value norm 最小，故默认更新残差较少（PDF pp.6–7，Figure 4）。
- context-length 从头训练：约 120M 参数、每个模型同为 5B tokens；128 长度几乎无 sink，长度越长 sink 越强，验证 loss 曲线相近（PDF pp.7–8，Figure 5；p.13，Appendix A.1–A.2）。
- 尺度对照：Llama 3.1 8B / 70B / 405B 在 `ε=0.8` 下 sink metric 为 45.97 / 73.49 / 78.29（PDF p.8，Table 1）。这是同家族相关性，不是控制训练数据和训练预算的严格因果实验。
- packing 干预：无固定 BOS 时仍在第一个实际 token 形成 sink；训练时固定 BOS 后，推理移除它使 sink metric 近零且 loss 从约 2.69 升至 7.56/7.78（PDF p.9，Table 2）。但移除训练时始终存在的 BOS 同时造成分布偏移，无法把性能损失完全归因于 sink。
- downstream 移除 BOS：Gemma 7B 多任务显著下降，RULER-4096 从 82.57 到 0（PDF p.10，Table 3）；同样受上述 OOD/位置变化混杂。
- 范围限于 causal decoder LMs；没有 encoder、ViT、diffusion 对照，也未直接测预测熵或语义边界。

## 统一问题与精确证据

### 1. sink 由什么触发，承担什么计算功能

触发所需的关键属性是“序列最早、所有后续 token 都可见”的稳定位置，而非 BOS 的语义身份。其计算功能是吸收 attention mass、降低 head 的实际残差更新，控制跨 token mixing；具体 head 可借此实现“条件不满足则 no-op”的 if/else。

> “The presence of attention sinks slows down the mixing of information between tokens and hence makes Transformers more robust to perturbations of prompts.”（PDF p.2，Figure 1 caption）

> “The attention sink in the ⟨bos⟩ token seems to provide a direct mechanism to construct this ‘approximate no-op’.”（PDF p.6，§3.2）

> “Otherwise, LMs employ the first token (which need not be ⟨bos⟩) to avoid over-mixing.”（PDF p.9，§5 summary）

### 2. 支持语义、不确定性还是纯结构解释

证据支持结构性 mixing-control 解释，不支持由语义或预测不确定性决定首边界。任意 first token 可承担该角色；BOS 是否固定只改变模型构造 sink 的方式。论文没有检查 sink value 是否携带语义，亦未使用熵作为触发变量。

> “If there is no ⟨bos⟩ during training, the sink forms at the first token regardless, but is slightly weaker.”（PDF p.9，§5）

> “The only important property the sink should have is that it exists at the first position in the sequence to help prevent mixing of the subsequent tokens.”（PDF p.9，§5）

### 3. 消融或干预因果证据

最有力的是从头训练时操纵 context length 与 BOS/data packing：前者在相同总 tokens 和相近 loss 下系统改变 sink 强度，后者说明 BOS 身份非必要。Gemma 移除 BOS 的扰动和下游实验是直接干预，但会同时改变输入分布与位置，因果解释应收窄为“依赖该首位槽位”，不能纯化为“只因 sink”。

> “We vary the pre-training context length, making sure that each training step processes the same amount of tokens such that the total tokens processed by each model is 5B.”（PDF p.7，§4.1）

> “Initially, there are no attention sinks; the rate at which sinks develop is generally increasing with the context length (until saturating).”（PDF p.7，§4.1）

> “Removing the attention sink (i.e., removing the ⟨bos⟩ token) in Gemma 7B at inference time consistently lowers the performance, and this drop is particularly pronounced on the long-context ruler benchmark.”（PDF p.9，§5）

### 4. 对 AR 依赖与跨模态对照的意义

本文机制高度依赖 causal mask 产生的非对称信息传播：首 token 是天然全局可见且自身不混入过去内容的槽位；长 context 的 over-squashing 也是 decoder-only 分析。因此它是“AR 条件下首 sink 为什么形成”的强证据，却不是 sink/register 的普遍理论。BERT 的 null/no-op 仅在背景中被提及，没有实证对照。

> “In this work, we focus on decoder-only Transformer models, that apply a causal mask to the attention mechanism.”（PDF p.2，§2）

> “The sum ranging over j such that j ≤ i is due to the causal mask.”（PDF p.3，§2）

> “A similar ‘no-op’ phenomenon [was observed] in the encoder-Transformer model BERT.”（PDF p.3，§2；作者对既有工作的概述，并非本文实验）
