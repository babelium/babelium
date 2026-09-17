# R4 — GPU 组:F2/F4(前向)、F5(VQ)、F6(微调)

**四条预言吃卡:P4(非规范质量随长度)、P5(第二项随规模降)、P6(MLA)、P8(范数走势)。本篇给两臂的构造、三个混淆项的对照、以及 F6 的真实成本。**

---

## F2/F4 · 两臂前向

### 两臂的确切构造

位置 $t$ 上,由 $h_t$ 得地址分布 $p_t=\mathrm{softmax}(Wh_t/T)$。

- **提交臂**:下一位置输入 $u^{\mathrm{cmt}}_t=\mathrm{emb}(a_t)$,$a_t=\arg\max p_t$。
- **不提交臂**:$u^{\mathrm{avg}}_t=\sum_a p_t(a)\,\mathrm{emb}(a)$。
- **上界臂(本轮新增,理由见下)**:$u^{\mathrm{glb}}=\frac{1}{V}\sum_a\mathrm{emb}(a)$,与 $t$ 无关。

第二项代价 $=\sup_{\varphi\in\Phi}|\mathbb{E}\varphi(u^{\mathrm{avg}})-\mathbb{E}\varphi(u^{\mathrm{cmt}})|$,$\Phi$ 按 R2 §1 取。

### 三个混淆项与对照(不缩水,给对照)

**混淆一:范数。** $u^{\mathrm{avg}}$ 是凸组合,方向互相抵消,故 $\lVert u^{\mathrm{avg}}\rVert\ll\lVert u^{\mathrm{cmt}}\rVert$ 是几乎必然的。**若不处理,第二项代价几乎全部是范数差,与「承诺」无关。**

处置:**RMSNorm 在第一步就把范数除掉了**($\hat u=u/\mathrm{RMS}(u)\cdot\gamma$),故 $\Phi$ 里除 $\varphi_0=\lVert u\rVert$ 之外的成员**天然不受范数影响,只看方向**。因此:$\varphi_0$ 那一项**单独报**,不与其余合并取上确界。**这是 R2 §1 把 $\lVert u\rVert$ 单列的实质理由,不是补全。**

**混淆二:离流形程度。** $u^{\mathrm{avg}}$ 是模型没见过的输入。CC-06 里我把这个当成了「只测单步」的理由,那是错的。正确做法是**把它当协变量测掉**:

- 定义离流形量 $\delta_t=1-\max_a\cos(u^{\mathrm{avg}}_t,\mathrm{emb}(a))$,与代价一同记录;
- 报代价对 $\delta$ 分箱后的曲线,而非单一均值;
- **上界臂给出 $\delta$ 的极端参照**:$u^{\mathrm{glb}}$ 是最离流形的合法输入。**若第二项代价接近上界臂,则测到的是「离流形」而非「承诺」。**

**混淆三:温度。** $T\to0$ 时 $u^{\mathrm{avg}}\to u^{\mathrm{cmt}}$,代价平凡趋零。故须扫 $T\in\{0.5,0.7,1.0,1.3,2.0\}$(**事先写死**),报代价-温度曲线。**同时以 $p_t$ 的熵作第二协变量分箱**——第 18 条说代价是「熵 × $\Phi$-敏感度」二维量,故**若代价是熵的单值函数,则第 18 条那半错**;这是顺带的一次检验。

### 跨规模可比性(P5 的前提)

不同模型 $d$ 不同、嵌入尺度不同,**绝对代价不可比**。故 P5 一律用**无量纲比**:

$$\rho=\frac{\mathrm{Cost}^{\mathrm{sel}}(\text{avg vs cmt})}{\mathrm{Cost}^{\mathrm{sel}}(\text{glb vs cmt})}$$

分母是同一模型、同一 $\Phi$ 下的上界臂。**这是上界臂存在的主要理由**,离流形参照是副产品。

### 代码骨架

```python
@torch.no_grad()                       # 纯推理:无梯度、无 detach 问题
def two_arms(model, ids, T, phi):      # phi: 由 R2 §1 从权重 SVD 预先算好并冻结
    E   = model.get_input_embeddings().weight          # (V, d)
    out = model(input_ids=ids, output_hidden_states=False)
    logits = out.logits.float()                        # fp32:softmax 前升精度
    p   = torch.softmax(logits / T, dim=-1)            # (B, L, V)
    u_cmt = E[p.argmax(-1)]                            # (B, L, d)
    u_avg = p @ E                                      # (B, L, d) 加权平均锚
    u_glb = E.mean(0).expand_as(u_cmt)                 # 上界臂
    ent   = -(p * p.clamp_min(1e-12).log()).sum(-1)    # 熵,协变量
    cos   = torch.nn.functional.normalize(u_avg, dim=-1) @ \
            torch.nn.functional.normalize(E, dim=-1).T
    delta = 1 - cos.max(-1).values                     # 离流形量
    return {k: phi(v) for k, v in
            [("cmt", u_cmt), ("avg", u_avg), ("glb", u_glb)]} | \
           {"ent": ent, "delta": delta}
```

**四处 dtype/统计口径,写死:**

1. `logits.float()` 再 softmax:bf16 softmax 在长尾上下溢,会改熵与 $u^{\mathrm{avg}}$。
2. `p @ E` 在 fp32 下做($V=152064$ 的加权和,bf16 累加误差不可接受)。
3. **padding 位置一律不进统计**:按 attention mask 掉,且**须报被掉的比例**。
4. **序列首位不进统计**:首位无前文,$p_1$ 由 BOS 决定,与其余位置分布不同族。

**`p @ E` 的显存**:$B\times L\times V$ 的 $p$ 在 fp32 下,$L=1024$、$V=152064$ 时单条序列 = 623 MB。**故 $B=1$、且 $p$ 须分块乘或就地释放。** 这是本组唯一的显存瓶颈,不在权重上。

### P4(自生成非规范质量)与 P8(范数走势)同跑

P4 需要模型自己续写的 token 序列:五个长度点 {64,128,256,512,1024},每点 ≥ 500 条,交给 R3 的 `noncanonical_mass`。P8 记录逐步的 $\lVert h_t-\mathrm{emb}(a_t)\rVert$,**须同时记 $\lVert h_t\rVert$ 以区分「距离增大」与「整体尺度增大」**——不分则 P8 测到的可能只是残差流范数随深度增长这件已知事实。

## F5 · VQ 侧

**候选模型(跨 domain,⬜ 待核实 repo 与解码器首层):** 图像 `CompVis/vqgan-*` / `stabilityai/sd-vae-*` 一类;音频 `facebook/encodec_24khz`(RVQ,**且它是 Q2 §3 的塔的实例**);语音 `openai/whisper` 系的离散变体。**至少覆盖三个 domain,不局限单卡小模型。**

测两件:
1. **码字往返**:把码字 $e_k$ 当输入送回编码-量化,看是否量化回自己。这是 VQ 侧的 AAR,**且因为 VQ 单锚,它与 P1 的绑定情形同型**。
2. **RVQ 是塔**:EnCodec 的 residual VQ 是多级量化,**正好检验 Q2 §4 的次可加性**——逐级代价与总代价的关系。这是本轮 Q2 掉出来的新用途,原 F5 没有。

## F6 · 微调,成本要报实数

32B 全参 + Adam:参数 64 GB(bf16)+ 梯度 64 GB + Adam 两态 fp32 256 GB + 主权重 fp32 128 GB ≈ **512 GB**,须 8×80GB + ZeRO-3,或 offload 换时间。

**三条路,代价明说:**

| 路 | 显存 | 代价 |
|---|---|---|
| 全参 32B | 8×80GB | 贵;但唯一能对上「从头训 vs 事后改」这个对照 |
| LoRA 32B | 1–2×80GB | 便宜;**但结论弱化**:LoRA 不改注意力的 K-bias 结构,而 F1 那件事正是结构改动 |
| 全参 8B + LoRA 32B | 1×80GB + 2×80GB | 折中:8B 给机制、32B 给规模,**两者结论不可合并陈述** |

**我的判断:F6 排最后,且若预算只够一条,选「全参 8B + LoRA 32B」并明确写两者结论不合并。** 这不是缩水成设计——**是预算决定,如实标注**(CC-06 的规矩)。

**梯度流须写死的一处**:Zero-Sink 那类改动若引入可训练的 K-bias,则该 bias 与 $W_k$ 同时受训,二者共变。**要分离,须做冻结 $W_k$ 只训 bias 的第三臂。** 不做则「bias 起作用」与「$W_k$ 适应了 bias」分不开。

## 状态

- 两臂变三臂,**上界臂是 P5 可比性的前提**,不是装饰。
- 三个混淆项各配对照:范数由 RMSNorm 吃掉 + $\varphi_0$ 单列;离流形当协变量分箱 + 上界臂给参照;温度扫五点 + 熵作第二协变量(顺带检验第 18 条)。
- 显存瓶颈在 $p\in\mathbb{R}^{B\times L\times V}$ 而非权重,故 $B=1$。
- F5 多一个新用途:RVQ 是塔,检验 Q2 §4 的次可加性。
- F6 三条路的成本报实数,推荐路的弱化如实标为预算决定。
