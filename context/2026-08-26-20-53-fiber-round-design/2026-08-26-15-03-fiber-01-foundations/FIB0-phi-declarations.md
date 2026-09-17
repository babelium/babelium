# FIB-0 · 三份 $\Phi$ 声明(按官方源码)

> 2026-08-26 · 消审计 H02 / H03 · 全部行号来自 `huggingface/transformers` main 分支实测抽取
>
> **结论:原声明有一处是错的、两处不完整,而改正之后掉出一条新的可预注册预言。错的那处是把 MLP 的两个投影算作读提交向量——源码显示 MLP 读的是 attention 之后的残差。不完整的两处是 MLA 的 RoPE 分支绕过瓶颈,以及 VQ 解码器有两条输入路径。新预言来自 Qwen3 的 per-head q_norm/k_norm:它是一次额外的商,故同规模下 Qwen3 的 q/k 位点代价应低于 Qwen2。**

---

## 一、$\Phi$ 的定义,精确到「紧接着」

$\Phi$ = 提交交出 $u=e(a)$ 之后,**紧接着读到它的那些标量函数**。「紧接着」必须精确,否则往深处走会把后续层的非线性混进来,那时测的不再是提交的代价。

**判定规则(执行前冻结):$\varphi\in\Phi$ 当且仅当 $\varphi$ 是 $u$ 的函数且在计算图上 $u$ 到 $\varphi$ 的路径不经过任何其他位置的信息。** 经过了别的位置就不是这一次提交的读出。

B02 已定:**范数不在 $\Phi$ 里,它落在零核里**(归一化第一步除掉它)。故下面每份声明里 $\Phi$ 都是线性或仿射族。

---

## 二、$\Phi_{\mathrm{GQA}}$ — Qwen2 系

`modeling_qwen2.py` 的 `Qwen2DecoderLayer.forward` 逐行:

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)      # ← 提交向量在这里被读
hidden_states, _ = self.self_attn(...)                   # ← 读它的是 q/k/v 三个投影
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                  # ← 读的不是提交向量
```

### 错的那处(审计 H02)

**原声明把 $W_{\mathrm{gate}}$ 与 $W_{\mathrm{up}}$ 列为 $\Phi$ 的成员。源码否掉这一条:** MLP 的输入是 `post_attention_layernorm(residual + attn_out)`,而 `attn_out` 混入了**所有位置**的信息(注意力是跨位置的)。按 §1 的判定规则,MLP 的投影不是这一次提交的读出。

$$\boxed{\ W_{\mathrm{gate}},W_{\mathrm{up}}\notin\Phi\ }$$

**这不是细节。** 原声明的五个位点里两个是错的,而 FIB-2 的主结果就是这些位点之间的次序——**位点划错,整个主结果是错的**。

### 三处源码级事实,各有后果

**一、q/k/v 有 bias,故读出是仿射不是线性。**

```python
self.q_proj = nn.Linear(config.hidden_size, ..., bias=True)   # Qwen2 硬编码 True
```

$\varphi(u)=\langle W\hat u+b,\xi\rangle$。**bias 在两臂之差里相消**,故 B02 的估计器不受影响($\Delta_j$ 里 $b$ 消掉)。但声明必须写明是仿射,否则「$\Phi$ 是线性族」这句话不严格。

**二、RoPE 在投影之后作用于 q/k,不作用于 v。**

```python
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

故 q/k 位点的读出带位置依赖:$\varphi^{(t)}(u)=\langle \mathrm{RoPE}_t(W_q\hat u),\xi\rangle$。RoPE 是正交变换,**故它不改 $\lVert\cdot\rVert$、不改零核,但改「哪个 $\xi$ 方向」**。处置:q/k 位点的主方向须逐位置定,或用 RoPE 不变的量(奇异值谱)。**v 位点无此问题,是干净的仿射读出。**

**三、位点数比原设想少,但可以按头恢复。**

去掉 MLP 两个之后,每层只剩 q/k/v 三个位点。**三个点上做次序检验太少。** 恢复办法:**按头拆**。GQA 下 query 头各自独立、kv 头被共享,故

| 模型 | query 头 | kv 头 | 可用位点数 |
|---|---|---|---|
| Qwen2.5-32B | 40 | 8 | 40 + 8 + 8 = 56 |
| Qwen2.5-72B | 64 | 8 | 64 + 8 + 8 = 80 |

**按头拆是合法的,而且比原设想更强:** 各头读的是 $W$ 的不同行空间切片,**函数形式完全相同、只有行空间不同**——这正是次序检验想要的对照(变的只有「读哪个子空间」)。原设想里五个位点的函数形式各不相同(注意力投影 vs 前馈投影),那反而是个混淆项。

$$\Phi_{\mathrm{GQA}}=\Big\{u\mapsto\langle W^{(h)}\hat u+b^{(h)},\xi\rangle\ :\ W^{(h)}\in\{W_q^{(h)},W_k^{(h)},W_v^{(h)}\},\ \lVert\xi\rVert\le1\Big\}$$

$\hat u=\mathrm{RMSNorm}_{\text{input}}(u)$,$h$ 遍历各头,q/k 分支额外带 $\mathrm{RoPE}_t$。

---

## 三、$\Phi_{\mathrm{Qwen3}}$ — Qwen3 系(与 Qwen2 是两份,不是一份)

原规格把 Qwen2 与 Qwen3 合并为一份 $\Phi_{\mathrm{GQA}}$。**源码否掉这个合并:**

```python
self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # per-head
self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
```

**Qwen3 在投影之后、RoPE 之前,对每个头各做一次 RMSNorm。Qwen2 没有这一步。**

### 这是一次额外的商,故给出一条新预言

按 B01,商的复合是串联(塔)。per-head norm 把**每个头内的长度信息**除掉,故它是提交之后的第二次商:

$$u\ \xrightarrow{\ \hat u=\mathrm{RMSNorm}\ }\ \xrightarrow{\ W_q^{(h)}\ }\ \xrightarrow{\ \text{per-head norm}\ }\ \xrightarrow{\ \mathrm{RoPE}_t\ }\ \text{读出}$$

**按零核定理,多一次商 ⇒ 零核更大 ⇒ 代价更低。** 而 Q2 §4 已证串联次可加(且否掉了「沿塔单调不减」)。

$$\boxed{\ \textbf{预注册预言:同规模下,Qwen3 的 q/k 位点代价低于 Qwen2 的 q/k 位点;v 位点无此差异(v 不过 per-head norm)}\ }$$

**这条预言的价值在于它的对照结构:**

- 网格里已有 **Qwen2.5-32B 与 Qwen3-32B**,hidden 5120、64 层**完全相同**,词表差 128 行
- **v 位点是内部对照**:两个模型的 v 都不过 per-head norm,故 v 上不应有差异。**若 v 上也出现同向差异,则该差异不来自 per-head norm,来自训练或别的东西**
- 杀伤条件:q/k 上无差异,或 v 上出现同量级同向差异

**这条是本次按源码核对直接掉出来的,原规格里没有。** 它比原 P6(MLA vs GQA 跨厂商)干净得多——同厂、同规模、同层数、架构差异只有这一处。

**一处必须记的混淆项:** Qwen2.5-32B 的 `rms_norm_eps`=1e-05,Qwen3-32B 是 1e-06,差一个数量级。按 B02 §4,$\epsilon$ 大者的 $\epsilon$ 残余更强。**故这一对上必须同时报 $\lVert u\rVert^2/(\epsilon d)$ 的分布,否则「Qwen3 代价更低」可能有一部分来自 $\epsilon$ 不同。**

$$\Phi_{\mathrm{Qwen3}}=\Big\{u\mapsto\langle \mathrm{RoPE}_t\big(N_h(W^{(h)}\hat u)\big),\xi\rangle\ :\ W^{(h)}\in\{W_q^{(h)},W_k^{(h)}\}\Big\}\cup\Big\{u\mapsto\langle W_v^{(h)}\hat u,\xi\rangle\Big\}$$

Qwen3-32B 的 `attention_bias=False`(实测),故无 bias 项。

---

## 四、$\Phi_{\mathrm{MLA}}$ — DeepSeek-V2 系(审计 H03)

`modeling_deepseek_v2.py` 的实际路径:

```python
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)              # → 512 + 64
kv_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_nope = self.kv_a_layernorm(kv_nope)                               # norm 只作用于 512 那半
k_pe   = k_pe.view(...)                                             # ← 64 维,不过 norm、不过瓶颈
# V2-Lite 的 q_lora_rank is None,故 query 走这一支:
query_states = self.q_proj(hidden_states)                           # 直接读,满秩
```

### 原规格漏掉的:RoPE 分支绕过瓶颈

原规格写「MLA 在 q/k/v 之前多插一次低秩压缩,故 $\Phi_{\mathrm{MLA}}$ 维数 = 512 + 64 低于 $\Phi_{\mathrm{GQA}}$」。**源码显示这个描述把三条路混成一条:**

| 分支 | 实际路径 | 有瓶颈吗 | 有 norm 吗 |
|---|---|---|---|
| KV 主路 | $h\to W^{\downarrow}\to$ 512 $\to$ `kv_a_layernorm` | **有**(512) | 有 |
| **RoPE 键路** | $h\to$ 直接切出 64 维 | **没有** | **没有** |
| **Query 路(V2-Lite)** | $h\to W_q\to$ 16 头 × 192 | **没有**(`q_lora_rank=None`) | 无 |

**故「MLA 的 $\Phi$ 维数更低」只对 KV 主路成立。** Query 路是满秩读出(16 头 × (128+64) = 3072 维,而 hidden 只有 2048),RoPE 键路是 64 维的直接读出、不过任何压缩。

**这一处改了 P6 的推导来源。** 原来那句「从两个 config 数值读出来」($\mathrm{kv\_lora\_rank}=512$ 对 $\mathrm{hidden}=2048$)只覆盖三条路里的一条。P6 现在已并入 FIB-3(配对预训练),但 $\Phi$ 声明必须写对,否则 FIB-2 在 MLA 界面上的位点是错的。

**可用位点(V2-Lite,16 头):**

$$\Phi_{\mathrm{MLA}}=\underbrace{\{u\mapsto\langle N(W^{\downarrow}\hat u),\xi\rangle\}}_{\text{512 维瓶颈,1 个位点}}\ \cup\ \underbrace{\{u\mapsto\langle\mathrm{RoPE}_t(W^{\mathrm{pe}}\hat u),\xi\rangle\}}_{\text{64 维,1 个位点}}\ \cup\ \underbrace{\{u\mapsto\langle W_q^{(h)}\hat u,\xi\rangle\}}_{\text{16 个头}}$$

**这个结构反而对 FIB-2 更有用:同一个模型内同时有「过瓶颈」与「不过瓶颈」的位点。** 于是「$\Phi$ 越小代价越低」可以在**同模型内**检验,不需要跨模型——这正是改形状要的东西,而且它比原来跨厂商比 MLA/GQA 干净。

---

## 五、$\Phi_{\mathrm{VQ}}$ — Emu3(审计 B05 的类型问题一并处理)

`modeling_emu3.py` 的解码路径:

```python
quant = self.quantize.embedding(hidden_states.flatten())   # 码字
post_quant = self.post_quant_conv(quant)                   # Conv3d, kernel (3,1,1)
video = self.decoder(post_quant, quant)                    # ← 两个参数,不是一个
```

### 原规格漏掉的:提交向量走两条路进解码器

原规格写 $\Phi_{\mathrm{VQ}}=\{e\mapsto\langle\mathrm{Conv}_1(e),\xi\rangle\}$,只有一条。**源码显示 `decoder` 接两个参数:**

| 路 | 入口 | 作用 |
|---|---|---|
| 主路 | `post_quant_conv(quant)`,Conv3d kernel (3,1,1) | 常规解码输入 |
| **条件路** | 原始 `quant` → `Emu3VQVAESpatialNorm.conv_y`,Conv2d kernel 1 | 逐层作条件归一化 |

**故 $\Phi_{\mathrm{VQ}}$ 有两个分支。** 而这两条路的性质不同:主路的 kernel 是 (3,1,1) 故**跨时间维混了邻居**,条件路的 kernel 是 1 故**逐位置纯线性**。

**按 §1 的判定规则,主路严格说不合格**——kernel (3,1,1) 意味着它读的是三个时间位置的码字,不只这一次提交。**处置:主路只取 kernel 中心那一片的权重作为「这一次提交的读出」,并把这个限制写明;条件路无此问题。**

$$\Phi_{\mathrm{VQ}}=\underbrace{\{e\mapsto\langle W^{\mathrm{pq}}_{\text{center}}e,\xi\rangle\}}_{\text{主路,取中心片}}\ \cup\ \underbrace{\{e\mapsto\langle W^{\mathrm{cy}}e,\xi\rangle\}}_{\text{条件路,逐层各一个位点}}$$

**顺带记两件:** 该 VQ 模型在 `__init__` 里 `self.eval()`,官方注明「Emu3's VQ model is frozen」,故它只能作诊断不能作 FIB-4 的干预臂(干预要从头训,已在设计里);量化器是标准最近邻(`distances = 2*matmul(...)` 那一段),`codebook_size=32768`、`embed_dim=4`。

---

## 六、四份不是三份

原规格说「三份声明」。按源码,**是四份**:

| 份 | 覆盖 | 与原规格的差别 |
|---|---|---|
| $\Phi_{\mathrm{GQA}}$ | Qwen2 系、Llama 系 | **删掉 MLP 两个位点**;按头拆;q/k 带 RoPE |
| $\Phi_{\mathrm{Qwen3}}$ | Qwen3 系 | **新增一份**(per-head q_norm/k_norm) |
| $\Phi_{\mathrm{MLA}}$ | DeepSeek-V2 系 | **三条路而非一条**,只有 KV 主路过瓶颈 |
| $\Phi_{\mathrm{VQ}}$ | Emu3 | **两条路而非一条**;主路须取中心片 |

---

## 七、三条纪律(冻结)

1. **主方向由权重矩阵的 SVD 定,不用任何激活统计。** 用了激活就是看了数据。RoPE 位点用 RoPE 不变的量(奇异值谱)或逐位置定主方向。
2. **主分析用完整向量 L2**(B02 §6),$m=16$ 截断降为廉价对照 `cost_16`。
3. **四份声明的差异随结果一起报**,因为跨架构比较的公平性完全取决于它。

---

## 八、须回填到别处

1. **FIB-2 的位点表:** 从「五个位点」改为「按头拆,Qwen2.5-32B 有 56 个」;MLA 界面上同时有过瓶颈与不过瓶颈的位点,故「$\Phi$ 越小代价越低」可在同模型内检验。
2. **新预言进预言表:** 同规模下 Qwen3 的 q/k 位点代价 < Qwen2 的 q/k,v 位点作内部对照。混淆项:两者 `rms_norm_eps` 差一个数量级,须同时报 $\epsilon$ 残余。
3. **结论表新增一条:** 按源码核对 $\Phi$ 时发现原声明把 MLP 投影算作提交读出,而 MLP 读的是 attention 之后的残差。**这是「实现细节反过来改理论」的第五例**,且它是唯一一处**如果不核对就会让主结果整体错**的。

