# G1 行:自回归采样(输出界面)

**取材方式**:本地源码。`transformers` 5.5.4,`D:\anaconda3\Lib\site-packages\transformers`。以下行号均可直接跳转核对。

---

## 六格

### 格 1 — 换算点

`generation/utils.py:2789-2793`

```python
if do_sample:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

换算的输入是 `next_token_scores`(shape `[B, |A|]`,由 `utils.py:2762` 取 `outputs.logits[:, -1, :]`,再经 `utils.py:2765` 的 `logits_processor` 变形),输出是 `next_tokens`(shape `[B]`,整数)。

`sel` 的位置在这里得到确认:第 2791 行的 `multinomial` 与第 2793 行的 `argmax` 是同一个换算的两种 `sel`,除此之外这个循环里没有第二处随机性。温度、top-k、top-p 全在 2765 行的 `logits_processor` 里,即**在换算之前改的是 c,不是 sel**。

### 格 2 — 交给谁

`generation/utils.py:2800` 把结果并入序列:

```python
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

下一轮 `utils.py:2746-2750` 用它调 `prepare_inputs_for_generation` 再进 `model_forward`。接收方即模型本体的 forward,以 llama 为例是 `models/llama/modeling_llama.py:384` 的 `LlamaModel.forward`。

### 格 3 — 交出什么(最要紧的一格)

`modeling_llama.py:385-389`

```python
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

if inputs_embeds is None:
    inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
```

交出的是**整数张量**。`embed_tokens` 是 `nn.Embedding(config.vocab_size, config.hidden_size, ...)`(`modeling_llama.py:361`),对整数索引做查表。

因此:换算之后送进下一阶段的对象是 `emb(a_t)`,它是**地址的函数,且只是地址的函数**。`next_token_scores` 里除 $a_t$ 以外的 $|A|-1$ 个分量在第 2800 行之后没有任何路径进入下一轮 forward——第 2809 行 `del outputs` 把它们连同 logits 一起丢弃。

这一格另有一个不在预期内的事实。第 385 行的互斥检查说明 forward 的签名**本来就接受连续向量**:传 `inputs_embeds` 时整条查表被跳过。也就是说「这次不提交、把 $\sum_a p_t(a)\,\mathrm{emb}(a)$ 直接送进去」是这个架构的合法调用,不是改造。F2 的对照臂因此不需要动模型代码,两条臂共用同一个 forward。

### 格 4 — 接收方读什么

接收方拿 `inputs_embeds` 做的事,顺着 `modeling_llama.py` 往下:加位置信息(`:394-397` 的 `position_ids`)、进 decoder layer 栈、层内经 `eager_attention_forward`(`:205-221`)。其中

`modeling_llama.py:212-218`

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
attn_weights = nn.functional.dropout(attn_weights, ...)
attn_output = torch.matmul(attn_weights, value_states)
```

读的是 $Q,K,V$ 三个线性像,即 `emb` 的线性函数;它们进的是内积、softmax、再对 $V$ 加权求和。整条链条对 `emb(a_t)` 的依赖全部经由「与其他位置的向量做内积」和「作为被加权的 value 出现」。

顺带确认 F1 的前提:第 216 行的 softmax 沿 `dim=-1` 归一,被归一的那一维只由 `key_states` 的位置数供给(第 209 行 `repeat_kv(key, ...)`),没有任何附加常数项或空 key。「一个都不该选」在这份实现里没有合法归宿——F1 的推导前提由此钉在 `modeling_llama.py:216`。

### 格 5 — $\Phi$(由 3、4 推出)

由格 3:下一阶段能看到的全部信息是 `emb(a_t)`。由格 4:它对这个向量的用法是线性像的内积与加权和。

$$\Phi_{\text{AR}} = \{\,\phi : \phi \text{ 是 } \mathrm{emb}(a)\text{ 的函数}\,\}$$

且格 4 进一步把它收窄:实际被读取的是 $W_Q\,\mathrm{emb}$、$W_K\,\mathrm{emb}$、$W_V\,\mathrm{emb}$ 三个线性像及其内积。

这里必须记一笔严格的话:$\Phi_{\text{AR}}$ 的定义域是 $\mathcal{A}$(地址集),不是 $\Delta(\mathcal{A})$(地址上的分布)。换算前的对象 `next_token_scores` 活在 $\mathbb{R}^{|A|}$,换算后的对象活在 $\mathbb{R}^{d}$,**两者不在同一个空间**。所以「提交前 vs 提交后」不能由架构自己相减——A5 §5 与 F0 里那个 $\Phi(\sum_a p_t(a)\,\mathrm{emb}(a)) - \Phi(\mathrm{emb}(a_t))$ 的桥是**我构造的**,靠的是 `emb` 的线性性把分布推到同一个 $\mathbb{R}^d$ 里,不是这个架构本身执行的运算。格 3 里发现的 `inputs_embeds` 入口让这个构造可以被执行,但不改变它是构造这一点。这个区分要带进汇总。

### 格 6 — 纤维可见性(由 3 推出)

**看不见。挡住它的是 `nn.Embedding` 的查表本身。**

纤维在这里是「$\arg\max$ 或采样所放弃的那些地址」。查表的入参是整数,一个整数只对应一行,被放弃的地址对应的行不会被取出。这不是被概率压低到可忽略,而是根本没有进入下一轮计算图:第 2800 行只拼接了 `next_tokens`,第 2809 行删掉了 `outputs`。

所以 AR 输出界面的纤维内 $\Phi$-敏感度**恒等于零**,是定义级的零而非经验上的小。这正是 F0 推翻 A5 §5 原步骤 3 的那条理由,现在有了行号。

---

## 本行结论

- $\Phi_{\text{AR}}$ = `emb(地址)` 的函数族,实际被读的是三个线性像。
- 纤维完全不可见,遮挡物是查表。
- 换算前后不同空间,提交代价必须靠外部构造的桥来测。
- 附带钉死两处:`sel` 只在 2791/2793 出现一次,温度与 top-k 改的是 $c$ 不是 $\mathrm{sel}$;attention 的 softmax 无弃权项(`:216`)。

## 待钉

无。本行六格全部出自可跳转的本地源码。
