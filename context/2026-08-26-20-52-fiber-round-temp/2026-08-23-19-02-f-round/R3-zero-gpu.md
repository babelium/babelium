# R3 — 零 GPU 组:F7(权重级)与 F3(分词器级)

**六条预言不吃卡:P1/P2/P7 只要两个张量,P3 只要分词器,P9/P10 是结构核查。本篇给到可执行的代码级,并逐条标出会静默出错的地方。**

**编号调整:原 F3 混了两件不同的事(分词器往返、权重性质),现拆开——F3 = 纯分词器,F7 = 纯权重两张表。理由是二者的输入、成本、失败模式全不同,混编会让「F3 零成本」这句话再犯一次旧账。**

---

## F7 · 权重级:锚对齐率、行范数、不可达交叉

### 输入

每个模型只取两个张量:`model.embed_tokens.weight`($E\in\mathbb{R}^{V\times d}$)与 `lm_head.weight`($W\in\mathbb{R}^{V\times d}$)。绑定时二者同一。

按分片取,不下全权重(R1 §4):先读 `model.safetensors.index.json` 的 `weight_map`,取这两个 key 指向的分片文件名,只下那两个。

### 会静默出错的四处

**一、matmul 的 dtype。** 权重是 bf16(8 位尾数)。$d=5120$ 维内积在 bf16 下累加误差量级约 $\sqrt{d}\cdot2^{-8}\approx0.28$ 相对量,而 argmax 的前两名常常差得比这小。**bf16 下算 argmax 会给出与 fp32 不同的答案,而这个错不会报错,只会让 AAR 偏低。** 故:**matmul 与 argmax 一律在 fp32 下做**,输入用 `.float()` 升精度。**这不是保险,是必须**——AAR 的整个内容就是 argmax 的身份。

**二、152064² 装不下。** 输出 23.1e9 个 fp32 元素 = 92 GB。须按 $b$ 分块,且**分块后要维护全局最大值与全局索引**,不能各块取局部 argmax 再拼。

**三、pad 行。** Qwen2.5-32B 的 `vocab_size=152064` 而 `len(tokenizer)` 更小(R1 §2:差 128)。**这 128 行从未被训练,其行为随机,会污染 AAR 的分子与分母。** 故 AAR 报两次:全 $V$ 行、以及只取 `id < len(tokenizer)` 且非 added-token 的行。

**四、绑定模型上 AAR 不是恒等于 1 的平凡结果。** $\arg\max_b\langle e_a,e_b\rangle=a$ 要求 $\lVert e_a\rVert^2\ge\langle e_a,e_b\rangle$ 对所有 $b$,由 Cauchy–Schwarz 得 $\langle e_a,e_b\rangle\le\lVert e_a\rVert\lVert e_b\rVert$,故**只在 $\lVert e_b\rVert\le\lVert e_a\rVert$ 时才保证**。范数散布大时,长向量会抢短向量的 argmax。**所以 P1 与 P7 不是两条独立预言,P7 是 P1 的机制。** 这一点是写代码时才看清的,须回填进结论表。

### 代码

```python
import json, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

def anchor_stats(emb_path, head_path, emb_key, head_key, model_id, block=4096):
    E = load_file(emb_path)[emb_key].float()          # fp32,理由见「一」
    W = E if head_path is None else load_file(head_path)[head_key].float()
    V, d = E.shape
    tok = AutoTokenizer.from_pretrained(model_id)
    n_real = len(tok)                                  # 含 added token 的真实上界

    row_norm = E.norm(dim=1)                           # P7
    best_val = torch.full((V,), -float("inf"))
    best_idx = torch.zeros(V, dtype=torch.long)
    for s in range(0, V, block):                       # 按 b 分块,理由见「二」
        logits = E @ W[s:s+block].T                    # (V, block) fp32
        v, i = logits.max(dim=1)
        upd = v > best_val                             # 维护全局 argmax
        best_idx[upd] = i[upd] + s
        best_val[upd] = v[upd]
        del logits
    hit = (best_idx == torch.arange(V))
    return {
        "AAR_full":  hit.float().mean().item(),                 # 分母 = vocab_size
        "AAR_real":  hit[:n_real].float().mean().item(),        # 分母 = len(tokenizer),理由见「三」
        "n_pad":     V - n_real,
        "norm_cv":   (row_norm.std() / row_norm.mean()).item(), # P7 的杀伤量
        "norm_q":    torch.quantile(row_norm, torch.tensor([0.,.01,.5,.99,1.])).tolist(),
        "miss_ids":  torch.nonzero(~hit).flatten().tolist(),    # 交给 P2 求交
    }
```

**`block=4096` 的理由**:每块 $152064\times4096$ fp32 = 2.5 GB,峰值可控。CPU 上 32B 全程约 240 TFLOPs(R1 §4),分钟到小时级,**不是零算力**。

## F3 · 分词器级:三档、不可达、非规范质量

### 三档分类(Q1 §5)

```python
def roundtrip_class(tok):
    out = {"intact": [], "fragment": [], "rename": [], "undecodable": []}
    V = len(tok)
    for a in range(V):
        if a in tok.all_special_ids: continue          # 特殊 token 无表面形,单列
        s = tok.decode([a], skip_special_tokens=False)
        if "�" in s:                              # 见「坑一」
            out["undecodable"].append(a); continue
        back = tok.encode(s, add_special_tokens=False) # 见「坑二」
        if back == [a]:            out["intact"].append(a)
        elif len(back) > 1:        out["fragment"].append(a)
        else:                      out["rename"].append(a)
    return out
```

**坑一(会静默毁掉整个数):byte-level BPE 的 token 未必是合法 UTF-8。** GPT-2 系与 Llama/Qwen 的 byte-level BPE 里,一个 token 可以是多字节字符的**一半**。`decode()` 对这种 token 返回 U+FFFD 替换符,再 encode 得到的是替换符的编码——**必然不等于原 token,于是被误记为「碎裂」**。中日韩词表里这类 token 数以千计,**不隔离则「碎裂率」这个数几乎全部是这个 bug**。故须单列 `undecodable`,并在报告里明确它既不是完好也不是失效,而是**该 token 本身不构成一个字符串**——它是字节级载体上的对象,不在 Q1 §1 的 $\Omega$(字符串集)里。**这是理论边界的一次真实收窄,不只是工程细节。**

**坑二:`add_special_tokens=False` 必须显式给。** 默认为真时每次 encode 都加 BOS,`back` 永不等于 `[a]`,**得到的失败率会是 ~100%**。

**坑三:`decode` 的 `clean_up_tokenization_spaces`** 在部分 tokenizer 上默认改空格。须显式关掉并记录该参数值。

### 不可达 token(P2)

$\mathcal{A}_{\mathrm{reach}}$ 的定义是「出现在某个规范切分里」。**枚举全部字符串不可能,故只能给下界**:在语料上跑 $\tau$,收集出现过的 id 集合。**报的是「在该语料下未出现」,不是「不可达」。** 两者差别须写明:前者是可测的,后者是前者在语料 → ∞ 时的极限。**故 P2 的正确陈述是「语料未出现集 ∩ undertrained 集」,而语料规模是协变量。** 语料定为多语混合、≥ 10B token 量级(具体来源在 R4 定)。

**碎裂档给出的是真正的不可达证据**:$|r(a)|>1$ 意味着该 token 单独出现时不会被产出,但它仍可能在更长上下文里被产出。**故「碎裂」不等于「不可达」,不许混报。** 真不可达的判据是语料未出现,碎裂只是嫌疑。

### 非规范质量(第 58 条,三条违反路径各一次)

```python
def noncanonical_mass(tok, seqs):                 # seqs: List[List[int]],真实出现的 token 序列
    bad = tot = 0
    for w in seqs:
        s = tok.decode(w, skip_special_tokens=True)
        R_w = tok.encode(s, add_special_tokens=False)   # R = τ∘κ*
        tot += 1; bad += (R_w != w)                     # 不动点判定
    return bad / tot
```

三条路径的 `seqs` 来源:**(a) 拼接** — 两段文本各自 encode 后 concat;**(b) 自生成** — 模型续写的 token 序列(须模型,与 F2 同跑);**(c) 边界切割** — 在随机 token 边界处截断再拼。**(a) 与 (c) 零 GPU,(b) 不是。**

**须报序列级与 token 级两个口径**:上式是「有多少序列非规范」;第 58 条的上界要的是**按频率加权的质量**,故还须报 $\sum$(不动点失败的 token 位置数)/(总 token 数)。**两个数会差很多**(一个长序列只错一处也算整条失败),混用会把上界算错。

## P9 / P10 · 结构核查

**P9**:逐实例查 $\mathrm{anc}$ 的值域是 $\Omega$ 还是 $\mathcal{X}$。标准 LM 输入侧 = 字符串(id-prim)、输出侧 = 向量(sim-prim),相反 ✓。VQ-VAE 编码侧 sim-prim、解码侧 id-prim ✓。**待查的两个:BLT(byte→patch 无固定词表)、H-Net(动态分块)——这两个是最可能出反例的,因为它们的「词表」是动态的。**

**P10**:判据是「追加 token 后已有前缀的 K/V 是否逐元素不变」。取一个双向模型(如 `FacebookAI/xlm-roberta-base`),前向两次(长度 $n$ 与 $n{+}1$),比较前 $n$ 位的 K/V。**须用同一随机种子、同一 dtype、且关掉 dropout(`model.eval()`)**,否则差异来自随机性而非结构。**允许的数值容差:bf16 下 1e-2、fp32 下 1e-5;超出即判为「不相等」。** 阈值事先定。

## 状态

- F7 与 F3 拆开,理由是输入与失败模式不同。
- **四处会静默出错的地方已定死处置**:fp32 argmax、分块维护全局索引、pad 行两个分母、byte-level 非法 UTF-8 隔离。
- **一条理论收窄**:byte-level 的 undecodable token 不在 $\Omega$(字符串集)里,Q1 的载体对这类词表须改为字节串。**回填结论表。**
- **一条结论合并**:P7 是 P1 的机制(Cauchy–Schwarz 只在 $\lVert e_b\rVert\le\lVert e_a\rVert$ 时保证),二者不独立。**回填结论表。**
- **一条预言弱化**:P2 只能测「语料未出现」,不能测「不可达」;语料规模是协变量。
