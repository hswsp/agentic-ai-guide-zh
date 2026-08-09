---
layout: home
title: DPO——直接偏好优化
permalink: /part2/ch06-dpo.html
---

## 动机

PPO 需要在显存中维持 4 个模型（policy、reference、reward 模型、value head）、复杂的 RL 基础设施，并且以不稳定著称。DPO [[159]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rafailov2023direct)] 提出的问题是：*我们能否跳过 RL，直接从偏好中学习？*

**关键洞察**：在 RLHF 目标（reward 最大化 + KL 惩罚）下的最优 policy 拥有**解析解**。由此我们可以推导出一个有监督 loss，隐式优化同一个目标。

## 数学推导

**第 1 步**：RLHF 目标：$\max_\pi \mathbb{E}_{x,y\sim\pi}[r(x,y)] - \beta D_\text{KL}[\pi\|\pi_\text{ref}]$

**第 2 步**：最优解为：$\pi^*(y\mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y\mid x) \exp\left(\frac{r(x,y)}{\beta}\right)$

**第 3 步**：重排式子，用 policy 表达 reward：$r(x,y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta \log Z(x)$

**第 4 步**：代入 Bradley-Terry 偏好模型 $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$。$Z(x)$ 项相互抵消！

$$
\boxed{\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right]}
$$

> **DPO 实际在做什么**
>
> 将**隐式 reward** 定义为 $\hat{r}(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$。
>
> DPO 在最小化一个交叉熵 loss，其中“标签”是：被选回答的隐式 reward 应高于被拒回答。Margin 由 $\beta$ 控制：
>
> - $\beta$ 较大：需要较大的 margin $\rightarrow$ policy 移动得更激进 $\rightarrow$ 有遗忘风险
> - $\beta$ 较小：较小的 margin 就够了 $\rightarrow$ policy 更靠近 reference $\rightarrow$ 更保守
>
> Reference 模型起到正则化作用：policy 必须用偏好对齐来“证明”自己偏离 reference 的合理性。

## Gradient 分析

DPO 的 gradient 可分解为：
$$
\nabla_\theta \mathcal{L} = -\beta \cdot \underbrace{\sigma(-\hat{r}_w + \hat{r}_l)}_{\text{weight: higher when model is wrong}} \cdot \left[\nabla_\theta \log\pi_\theta(y_w\mid x) - \nabla_\theta \log\pi_\theta(y_l\mid x)\right]
$$
**解读**：Gradient 会提升“被选”项的概率，降低“被拒”项的概率。当模型当前更偏好错误答案时，权重最大——也就是说，它把学习重点放在“易混淆”的成对样本上。

> **一个具体的 DPO 例子**
>
> **Prompt**：“向一个 10 岁的小孩解释量子纠缠。”
>
> **被选**（$y_w$）：“想象你有两枚魔法硬币。你抛其中一枚，如果它是正面，另一枚不管离多远都会立刻变成反面！”\
>
> $\log\pi_\theta(y_w\mid x) = -15.3$，$\log\pi_\text{ref}(y_w\mid x) = -16.1$
>
> **被拒**（$y_l$）：“量子纠缠是一种现象，其中两个粒子相互关联，以致于一个粒子的量子态无法被独立描述。”\
>
> $\log\pi_\theta(y_l\mid x) = -12.8$，$\log\pi_\text{ref}(y_l\mid x) = -12.5$
>
> **隐式 reward**：$\hat{r}_w = 0.1 \times ((-15.3) - (-16.1)) = 0.08$，$\hat{r}_l = 0.1 \times ((-12.8) - (-12.5)) = -0.03$
>
> **Loss 输入**：$\sigma(0.08 - (-0.03)) = \sigma(0.11) = 0.527$
>
> **Loss**：$-\log(0.527) = 0.64$——模型仅略微偏好被选回答。Gradient 会用力推动。
>
> 训练之后：被选回答的概率提升、被拒回答的概率下降，直到 margin 稳定在约 $1/(2\beta)$。

## TRL 实现

下面给出一个使用 HuggingFace TRL 的最小可运行示例。

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 数据集格式：{"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences")

lora_config = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # KL 正则强度
    learning_rate=5e-7,          # 学习率非常低，利于稳定
    loss_type="sigmoid",         # 标准 DPO loss
    max_length=2048,             # 最大序列长度
    max_prompt_length=1024,      # prompt 截断长度
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # 有效 batch = 16
    gradient_checkpointing=True,
    bf16=True,
    num_train_epochs=1,          # DPO 易过拟合——只跑 1 个 epoch！
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,             # 配合 LoRA 时，ref = base 模型（无需另外拷贝！）
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
# 需要监控的关键指标：train/rewards/chosen、train/rewards/rejected、train/rewards/margins
```

## DPO 的完整机制

本节给出 DPO 的完整计算细节——训练时在 token 级别究竟发生了什么。

### 序列级对数概率

DPO 中的关键量是：在给定 prompt $x$ 下，**整段序列** $y = (y_1, y_2, \ldots, y_T)$ 的对数概率。它是**逐 token 对数概率之和**：

$$
\boxed{\log \pi_\theta(y\mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})}
$$

其中每一项 $\log \pi_\theta(y_t \mid x, y_{<t})$ 都是位置 $t$ 处的 log-softmax 输出，对应序列中*实际*出现的 token $y_t$。这与标准语言建模中的交叉熵 loss 完全一致——只是这里我们**求和**而非求均值。

**关键细节**：Gradient 会流过 $y_w$ 和 $y_l$ 中的**每一个 token 位置**。中间 token 不做掩码——每个 token 都对序列级对数概率有贡献。

### DPO Loss 的分解

从 loss 开始：
$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \cdot h_\theta(x, y_w, y_l)\right)\right]
$$
其中“隐式 reward margin” $h_\theta$ 为：
$$
h_\theta(x, y_w, y_l) = \underbrace{\log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{chosen reward proxy}} - \underbrace{\log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{rejected reward proxy}}
$$

展开为 token 级各项：
$$
\boxed{h_\theta = \sum_{t=1}^{\lvert y_w \rvert}\!\left[\log\pi_\theta(y_w^t \mid x, y_w^{<t}) - \log\pi_{\text{ref}}(y_w^t \mid x, y_w^{<t})\right] - \sum_{t=1}^{\lvert y_l \rvert}\!\left[\log\pi_\theta(y_l^t \mid x, y_l^{<t}) - \log\pi_{\text{ref}}(y_l^t \mid x, y_l^{<t})\right]}
$$

### Forward Pass：逐步详解

对一个训练样本 $(x, y_w, y_l)$：

1. **拼接**：组成两条序列：$[x; y_w]$ 与 $[x; y_l]$。在 batch 内 pad 到相同长度。
2. **Forward pass（policy $\pi_\theta$）**：把两条序列都喂进模型；在每个回答位置收集 logits。
3. **提取对数概率**：在回答中每个位置 $t$ 处，取 $\log\text{softmax}(\text{logits}_t)[y_t]$——即实际 token 的对数概率。
4. **对 token 求和**：
$$
\text{logp\_chosen} &= \sum_{t \in \text{response positions}} \log\pi_\theta(y_w^t \mid x, y_w^{<t}) \\
\text{logp\_rejected} &= \sum_{t \in \text{response positions}} \log\pi_\theta(y_l^t \mid x, y_l^{<t})
$$
5. **减去参考**（预计算或来自第二次 forward pass）：
$$
\text{ratio\_w} &= \text{logp\_chosen} - \text{ref\_logp\_chosen} \\
\text{ratio\_l} &= \text{logp\_rejected} - \text{ref\_logp\_rejected}
$$
6. **计算 loss**：$\mathcal{L} = -\log\sigma(\beta \cdot (\text{ratio\_w} - \text{ratio\_l}))$
7. **Backward pass**：Gradient 沿步骤 5 $\rightarrow$ 4 $\rightarrow$ 3 $\rightarrow$ 2 回传，更新 $\theta$。

### Token 级 Gradient 分析

**是不是每个 token 都收到 gradient？** 是。在被选序列中，位置 $t$ 处 logits 上的 gradient 为：

$$
\frac{\partial \mathcal{L}}{\partial \text{logits}_t^{(w)}} = -\underbrace{\sigma(-\beta \cdot h_\theta)}_{\text{scaling factor}} \cdot \beta \cdot \frac{\partial \log\pi_\theta(y_w^t \mid \cdot)}{\partial \text{logits}_t^{(w)}}
$$

**关键洞察**：缩放因子 $\sigma(-\beta \cdot h_\theta)$ 在两条序列的**所有 token 间共享**，相当于一个自适应学习率：

- 当 $h_\theta$ 较小（模型分不清被选与被拒）：缩放 $\approx 0.5$——gradient 强，激进地学。
- 当 $h_\theta$ 较大（模型已经偏好被选）：缩放 $\approx 0$——gradient 可忽略，避免过拟合。

**对被选 token 的影响**：概率被*提升*（对数概率被推高）。\

**对被拒 token 的影响**：概率被*降低*（对数概率被压低）。\

**相对 reference**：只有相对 $\pi_{\text{ref}}$ 的*差值*才重要。如果模型本就给被选回答高概率（与 reference 一致），那么 gradient 就几乎为零。

### 逐 Token vs. 序列级：长度归一化

一个微妙问题：越长的序列对数概率天然越低（求和的项更多，且每项 $\leq 0$）。若 $\lvert y_w \rvert \gg \lvert y_l \rvert$，loss 会偏向于偏好更短的回答。

**解决方案**：

- **长度归一化的 DPO**：用 $\frac{1}{\lvert y \rvert}\sum_t \log\pi_\theta(y_t\mid\cdot)$ 替换 $\log\pi_\theta(y\mid x)$。一些实现采用此做法（SimPO 即如此）。
- **标准 DPO**：使用原始求和（不归一化）。这会*隐式*惩罚冗长——模型必须对被选回答中的每个 token 都赋予高概率。
- **实际影响**：在 benchmark 上，长度归一化的 DPO 能减少长度博弈，但可能损害指令遵循质量。生产中更常用未归一化的标准版本。

### 标签掩码：哪些 token 会收到 Gradient

> **DPO 中哪些 token 收到 Gradient**
>
> - **Prompt token**（$x$）：**无 gradient**。Loss 仅在回答位置上计算。Prompt token 提供上下文，但其 logits 不参与 $\log\pi(y\mid x)$。
> - **被选回答 token**（$y_w$）：**所有 token 都收到 gradient**。每个 $y_w^t$ 都贡献到求和；gradient 推高它们的概率。
> - **被拒回答 token**（$y_l$）：**所有 token 都收到 gradient**。每个 $y_l^t$ 都贡献到求和；gradient 压低它们的概率。
> - **Padding token**：**无 gradient**。通过 attention mask 屏蔽掉。

### 伪代码：DPO 训练步

> **DPO Forward + Backward（PyTorch 风格）**
>
> ```python
> def dpo_loss(model, ref_model, batch, beta=0.1):
>     """一个 DPO 训练步。"""
>     # batch 包含：input_ids_chosen、input_ids_rejected、
>     #            labels_chosen、labels_rejected（prompt 被掩为 -100）
>
>     # 1. Forward pass：得到逐 token 对数概率
>     logps_chosen = get_sequence_logprob(model, batch["chosen"])
>     logps_rejected = get_sequence_logprob(model, batch["rejected"])
>
>     # 2. 参考模型对数概率（预计算或在此处计算）
>     with torch.no_grad():
>         ref_logps_chosen = get_sequence_logprob(ref_model, batch["chosen"])
>         ref_logps_rejected = get_sequence_logprob(ref_model, batch["rejected"])
>
>     # 3. 计算隐式 reward margin
>     chosen_rewards = beta * (logps_chosen - ref_logps_chosen)
>     rejected_rewards = beta * (logps_rejected - ref_logps_rejected)
>
>     # 4. DPO loss = -log(sigmoid(被选 reward - 被拒 reward))
>     loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
>     return loss
>
> def get_sequence_logprob(model, sequences):
>     """仅在回答 token 上求和的对数概率。"""
>     outputs = model(sequences["input_ids"], attention_mask=sequences["mask"])
>     logits = outputs.logits[:, :-1, :]  # 为 next-token 预测做位移
>
>     # 收集实际 token 的对数概率
>     labels = sequences["labels"][:, 1:]  # 位移后的标签
>     log_probs = F.log_softmax(logits, dim=-1)
>     token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
>
>     # 掩码：只对回答 token 求和（labels != -100）
>     mask = (labels != -100).float()
>     return (token_logps * mask).sum(dim=-1)  # 形状：[batch_size]
> ```

### 常见陷阱

> **DPO 实现中的陷阱**
>
> - **忘记屏蔽 prompt**：如果 prompt token 被纳入对数概率求和，模型就会优化 prompt 的似然（无意义），且有效 $\beta$ 也会出错。
> - **用均值替代求和**：$\frac{1}{T}\sum_t \log\pi$ 与 $\sum_t \log\pi$ 会带来不同的隐式长度惩罚。$\pi_\theta$ 与 $\pi_{\text{ref}}$ 之间必须保持一致。
> - **过期的参考模型**：若 $\pi_{\text{ref}}$ 与 $\pi_\theta$ 相距过远（如 base 模型 vs. 微调后的模型），KL 项会占主导，gradient 消失。解决办法：使用 SFT checkpoint（而非 base）作为 reference。
> - **$\beta$ 过大**：放大对数概率差值 $\rightarrow$ sigmoid 饱和 $\rightarrow$ gradient 归零。从 $\beta = 0.1$ 起步，在 $[0.05, 0.5]$ 区间内调参。
> - **$\beta$ 过小**：理论上允许 policy 更大程度地偏离 reference（KL 约束更弱），但 gradient $\propto \beta \cdot \sigma(-\beta h)$ 会变得极小 $\rightarrow$ loss 地形平坦 $\rightarrow$ 收敛极慢。模型“被允许”走得很远，却几乎收不到任何告诉它*往哪走*的信号。

## DPO 变体及其各自的失败场景

> **DPO 何时会失败**
>
> **1. 分布漂移**：偏好数据来自旧模型。当前 policy 生成的文本截然不同 $\rightarrow$ loss 在无关样本上做优化。
>
> **2. 无探索**：无法发现数据集之外的行为。陷入局部最优。
>
> **3. Reference 崩塌**：Reference 太强，policy 动弹不得；太弱则失去正则化作用。
>
> **4. 数据质量**：噪声标签会毒化训练。与对大量样本平均的 PPO 不同，DPO 会“记住”单个的样本对。
>
> **5. 偏好数据多样性**：要确保被选/被拒成对样本覆盖质量差异的完整谱系（而不仅是“好 vs 极差”）。在*解法思路*上有差异的样本对（而不仅是质量差异）能教出更丰富的 policy 区分能力。

## $\beta$ 选择指南

| $\beta$ | **风格** | **使用场景** |
| --- | --- | --- |
| 0.01 | 非常激进 | 仅当数据极其干净、且你需要大幅度分布迁移时使用 |
| 0.05 | 激进 | 数据较好，希望相对 SFT 取得明显改进 |
| 0.1 | 标准 | 默认起点。质量与稳定性的良好平衡 |
| 0.2 | 保守 | 噪声较多，或模型已接近期望行为 |
| 0.5 | 非常保守 | 安全微调场景，必须保住已有能力 |

## DPO 的 Batch Size 配置与扩展

与在单序列 token 预测上工作的标准 SFT 不同，DPO 使用一个比较“被偏好序列与被拒序列”的**成对 loss**。这从根本上改变了显存利用与优化稳定性。

### 全局 Batch Size 目标

跨多种 DPO 实现的经验证据给出了一个最佳全局 batch size 范围：
$$
\boxed{B_{\text{global}} \in [32, 128]}
$$

- $B_{\text{global}} < 32$：隐式 reward 估计中的 gradient 噪声严重 $\rightarrow$ policy 在多个对齐目标之间（如有用 vs 安全）破坏性振荡。
- $B_{\text{global}} > 128$：收敛速度边际收益递减；分布式算力间通信开销巨大。

### 数学分解

由于 DPO 同时加载**两份**模型副本（活动 policy $\pi_\theta$ + 冻结 reference $\pi_{\text{ref}}$），每条序列的显存翻倍。全局 batch size 可分解为：
$$
\boxed{B_{\text{global}} = B_{\text{micro}} \times N_{\text{GPUs}} \times K_{\text{accum}}}
$$

- $B_{\text{micro}}$：每设备 micro-batch 大小（每次 forward pass 的偏好对数）。
- $N_{\text{GPUs}}$：并行处理数据的设备数量。
- $K_{\text{accum}}$：在权重更新前累计 gradient 的步数。

**成对倍数因子**：单个 DPO 数据样本包含 prompt（$x$）、被选（$y_w$）和被拒（$y_l$）。每个 micro-batch 的实际 tensor 负载为：
$$
T_{\text{sequences}} = 2 \times B_{\text{micro}}
$$

对于在 80GB GPU 上、上下文长度 4096--8192 token 的 $>$7B 参数模型，物理上限被刚性约束在 $B_{\text{micro}} \in [1, 2]$。

### 分布式扩展配置


**DPO 训练的分布式扩展配置（目标 $B_{\text{global}} = 64$）。**
| **配置** | **单 GPU** | **8 卡节点** |
| --- | --- | --- |
| $B_{\text{global}}$ | 64 | 64 |
| $B_{\text{micro}}$ | 2（4 条序列） | 2（4 条序列） |
| $N_{\text{GPUs}}$ | 1 | 8 |
| $K_{\text{accum}}$ | 32 步 | 4 步 |
| 吞吐 | 串行/慢 | 高并行吞吐 |

### 显存优化：预计算 Reference 对数概率

DPO loss：
$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

由于 $\pi_{\text{ref}}$ 在整个训练过程中**完全静态**，其输出可被预先计算：

> **Reference 模型驱逐策略**
>
> 1. 训练开始前，仅用 $\pi_{\text{ref}}$ 对数据集 $\mathcal{D}$ 执行一次 forward pass。
> 2. 将标量 $\log \pi_{\text{ref}}(y_w\mid x)$ 和 $\log \pi_{\text{ref}}(y_l\mid x)$ 缓存到磁盘。
> 3. **把 $\pi_{\text{ref}}$ 完全从 GPU 显存中驱逐。**
>
> **效果**：可用 GPU 显存翻倍 $\rightarrow$ $B_{\text{micro}}$ 可从 1--2 提升到 4--8，从而最大化硬件利用率与训练吞吐。
>
> *实现*：在 TRL 中，于 `DPOConfig` 设置 `precompute_ref_log_probs=True`。对 70B 模型，这能在整个集群上节省约 140GB 的 GPU 显存。

## DPO 扩展与变体

直接偏好优化（Direct Preference Optimization, DPO）通过推导 reward 函数与最优 policy 之间的解析映射，把 RLHF 重新表述为一个有监督学习问题。标准 DPO loss 为：

$$
\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(q,y_w,y_l)}\!\left[
\log \sigma\!\left(
\beta \log \frac{\pi_\theta(y_w\mid q)}{\pi_{ref}(y_w\mid q)}
- \beta \log \frac{\pi_\theta(y_l\mid q)}{\pi_{ref}(y_l\mid q)}
\right)
\right],
$$

其中 $y_w$ 为被偏好（获胜）的回答，$y_l$ 为被拒（失败）的回答，$\beta$ 控制 KL 惩罚强度。下面的小节涵盖最重要的扩展与变体。

### f-DPO——广义 f-Divergence DPO

> **超越反向 KL**
>
> 标准 DPO 使用反向 KL 散度作为 policy 与 reference 之间的正则项。反向 KL 是*寻峰*的：它倾向于把概率质量集中在少数高 reward 回答上。前向 KL 则是*覆盖*的：它把概率质量分散去覆盖所有合理回答。f-DPO [[161]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wang2023fdpo)] 推广到任意 f-散度，允许实践者在这些行为间做权衡。

f-DPO loss 用 f-散度生成器的导数替换对数比：

$$
\mathcal{L}_{f-DPO} = -\mathbb{E}\!\left[
f'\!\left(\frac{\pi_\theta(y_w\mid q)}{\pi_{ref}(y_w\mid q)}\right)
- f'\!\left(\frac{\pi_\theta(y_l\mid q)}{\pi_{ref}(y_l\mid q)}\right)
\right],
$$

其中 $f'$ 是 f-散度生成器函数的导数。

> **TRL 中的 f-Divergence 选项**
>
> - **reverse_kl**：$f'(u) = \log u$。即标准 DPO。寻峰。
> - **forward_kl**：$f'(u) = -1/u$。覆盖式。多样性更好。
> - **js_divergence**：$f'(u) = \log(2u/(u+1))$。寻峰/覆盖之间的平衡。
> - **alpha_divergence**：$f'(u) = u^{\alpha-1}$。在前向 KL 与反向 KL 之间插值。

> **TRL 中的 f-DPO**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> # Jensen-Shannon 散度（平衡）
> config = DPOConfig(
>     f_divergence_type="js_divergence",
>     beta=0.1,
> )
>
> # Alpha 散度（alpha=0：前向 KL；alpha=1：反向 KL）
> config_alpha = DPOConfig(
>     f_divergence_type="alpha_divergence",
>     f_alpha_divergence_coef=0.5,   # alpha 参数
>     beta=0.1,
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **何时使用 f-DPO**
>
> - 当你想在多样性与质量之间取得平衡时，使用 **JS 散度**。
> - 对多样性至关重要的创意任务，使用**前向 KL**。
> - 对有唯一正确答案的任务，使用**反向 KL**（即标准 DPO）。
> - 使用**alpha 散度**可连续插值并精调这一权衡。

### Robust DPO

> **偏好数据中的噪声标签**
>
> 人类偏好标注是有噪声的。标注者会意见分歧、会出错，有时会把“被偏好/被拒”的标签搞反。标准 DPO 把所有标签视为 ground truth，可能导致模型对噪声过拟合。Robust DPO [[162]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chowdhury2024robustdpo)] 在已知噪声模型下解析性地对 loss 去偏。

假设每个标签以概率 $\epsilon$（噪声率）被翻转。去偏后的 loss 为：

$$
\boxed{
\mathcal{L}_{robust} =
\frac{(1-\epsilon)\,\mathcal{L}_{DPO}(y_w, y_l)
- \epsilon\,\mathcal{L}_{DPO}(y_l, y_w)}{1 - 2\epsilon},
$$

其中 $\mathcal{L}_{\text{DPO}}(y_w, y_l)$ 是把 $y_w$ 视为被偏好的标准 DPO loss，而 $\mathcal{L}_{\text{DPO}}(y_l, y_w)$ 是把标签翻转后的 loss。该修正去除了标签噪声引入的偏置。

> **Robust DPO 的直觉**
>
> 该公式是一个线性组合，把翻转标签的贡献“减掉”。当 $\epsilon=0$ 时，它退化为标准 DPO；当 $\epsilon=0.5$ 时，分母趋于零——标签纯属噪声，无法学习。实际中，$\epsilon \in [0.05, 0.2]$ 覆盖了大多数真实标注噪声水平。

> **TRL 中的 Robust DPO**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="robust",
>     label_smoothing=0.1,   # 对应 epsilon = 0.1
>     beta=0.1,
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

### TR-DPO——信任域 DPO

> **过期 Reference 模型问题**
>
> 标准 DPO 在整个训练过程中使用固定的 reference 模型 $\pi_{\text{ref}}$。随着 policy $\pi_\theta$ 改进，KL 惩罚 $\beta \log(\pi_\theta/\pi_{\text{ref}})$ 不断增长，最终会主导 loss 并阻止进一步改进。TR-DPO [[163]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-gorbatenko2024trdpo)] 周期性地更新 reference 模型，使其跟随当前 policy。

TR-DPO 使用指数滑动平均（EMA）更新 reference 模型：

$$
\pi_{ref}^{(t+1)} \leftarrow
\alpha \cdot \pi_\theta^{(t)} + (1-\alpha) \cdot \pi_{ref}^{(t)},
$$

其中 $\alpha \in (0,1)$ 是混合系数。每 $T_{\text{sync}}$ 个 gradient 步执行一次。

> **TRL 中的 TR-DPO**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="sigmoid",        # 标准 DPO loss
>     sync_ref_model=True,        # 启用 TR-DPO
>     ref_model_mixup_alpha=0.6,  # alpha：混入当前 policy 的比例
>     ref_model_sync_steps=512,   # T_sync：每 512 步同步一次
>     beta=0.1,
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **何时使用 TR-DPO**
>
> - Policy 已偏离初始 reference 较远的长训练任务。
> - 当观察到 DPO loss 因 KL 惩罚主导而过早平台化时。
> - 从当前 policy 收集新偏好数据的迭代 DPO 流水线。
> - $\alpha$ 接近 1 表示快更新 reference；接近 0 表示慢更新。

### EXO——精确优化

> **DPO 的 KL 方向问题**
>
> DPO 是在反向 KL 约束下求解最优 policy 而推导出的。然而，由此得到的 loss 实际上在 reward 空间中优化的是一个*前向* KL 目标，这个方向是错的。EXO [[164]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-ji2024exo)] 通过使用反向 KL 概率匹配修正了这一点，这正是对齐在理论上正确的目标。

EXO 最小化模型分布与目标（reward 最优）分布之间的反向 KL：

$$
\mathcal{L}_{EXO} = \mathbb{E}_{y \sim \pi_\theta}\!\left[
\log \frac{\pi_\theta(y\mid q)}{p^*(y\mid q)}
\right],
$$

其中 $p^*(y\mid q) \propto \pi_{\text{ref}}(y\mid q) \exp(r(y,q)/\beta)$ 即最优 policy。实践中，EXO 利用可用的偏好对来近似该量：

$$
\mathcal{L}_{EXO} \approx -\mathbb{E}\!\left[
\log \sigma\!\left(
\beta \log \frac{\pi_{ref}(y_w\mid q)}{\pi_\theta(y_w\mid q)}
- \beta \log \frac{\pi_{ref}(y_l\mid q)}{\pi_\theta(y_l\mid q)}
\right)
\right].
$$

注意：相对于 DPO，$\pi_\theta$ 与 $\pi_{\text{ref}}$ 的角色被*对调*了。

> **TRL 中的 EXO**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="exo_pair",
>     beta=0.1,
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

### NCA——噪声对比对齐

> **DPO 的似然崩塌**
>
> DPO 已知的失败模式之一是*似然崩塌*：模型学到了降低被拒回答的概率，但同时也降低了被选回答的概率（因为 loss 只在意它们的*差值*）。NCA [[165]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chen2024nca)] 增加了一个绝对似然项以防止这一点。

NCA 把对齐重新表述为噪声对比估计。Loss 包含三项：

$$
\boxed{
\mathcal{L}_{NCA} =
-\log \sigma(r_w)
- \tfrac{1}{2}\log \sigma(-r_w)
- \tfrac{1}{2}\log \sigma(-r_l),
$$

其中 $r_y = \beta \log(\pi_\theta(y\mid q)/\pi_{\text{ref}}(y\mid q))$ 是隐式 reward。第一项鼓励 $y_w$ 获得高 reward；第二、三项则同时惩罚 $y_w$ 与 $y_l$ 上的高 reward（防止崩塌）。

> **TRL 中的 NCA**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="nca_pair",
>     beta=0.01,   # beta 较小：绝对似然项占主导
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **何时使用 NCA**
>
> - 当观察到 DPO 训练过程中被选回答的概率在下降时。
> - 对于看重绝对回答质量、而不仅仅是相对排序的任务。
> - 使用较小的 $\beta$（如 0.01）让绝对似然项获得更高权重。

### SLiC-HF——序列似然校准

> **作为简化替代的 Hinge Loss**
>
> DPO 中的 log-sigmoid loss 虽然平滑，但当 margin 较大时收敛缓慢。SLiC-HF [[166]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-zhao2023slichf)] 使用 hinge loss：当 margin 超过阈值时为零，否则为线性。它更简单、更快，且效果出乎意料地有竞争力。

SLiC-HF loss 为：

$$
\mathcal{L}_{SLiC} = \max\!\left(0,\;
\delta - \beta\log\frac{\pi_\theta(y_w\mid q)}{\pi_{ref}(y_w\mid q)}
+ \beta\log\frac{\pi_\theta(y_l\mid q)}{\pi_{ref}(y_l\mid q)}
\right),
$$

其中 $\delta$ 是 margin 阈值。当模型在被选与被拒回答之间已经赋予 $\delta$ 大小的 margin 时，loss 为零。

> **TRL 中的 SLiC-HF**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="hinge",
>     beta=0.1,
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

### Iterative RPO——推理偏好优化

> **DPO 会“忘记”如何生成**
>
> 标准 DPO 训练模型去*判别*被选与被拒回答的优劣。但对于推理任务，模型还需要*生成*正确的推理轨迹。一个能判别却不能生成的模型在推理时毫无用处。RPO 在被选回答上加入一个 NLL（负对数似然）项，确保模型学会生成它。

RPO loss 将 DPO 与 SFT 结合：

$$
\mathcal{L}_{RPO} =
\lambda_1 \mathcal{L}_{DPO}(y_w, y_l)
+ \lambda_2 \mathcal{L}_{NLL}(y_w),
$$

其中 $\mathcal{L}_{\text{NLL}}(y_w) = -\log \pi_\theta(y_w\mid q)$ 是作用于被选回答的标准语言建模 loss。

> **TRL 中的 Iterative RPO**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="sigmoid",            # 标准 DPO loss
>     rpo_alpha=1.0,                  # NLL 正则权重（RPO）
>     beta=0.1,
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=ref_model,
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **何时使用 RPO**
>
> - 模型必须生成逐步解答的推理任务（数学、代码、逻辑）。
> - 当 DPO 训练导致模型流畅度或生成质量下降时。
> - 迭代式流水线：生成 rollout、打标签、用 RPO 训练、再重复。
> - NLL 项起到正则化作用，防止 policy 崩塌。

### SimPO——简化版偏好优化

> **无参考模型的偏好学习**
>
> DPO 需要一个 reference 模型来计算隐式 reward，这会让显存翻倍并增加复杂度。SimPO [[167]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-meng2024simpo)] 取消了 reference 模型，转而用回答的*平均对数概率*作为隐式 reward，并通过长度归一化项防止模型偏好短回答。

SimPO 将隐式 reward 定义为：

$$
r_{SimPO}(y\mid q) = \frac{\beta}{\lvert y \rvert} \log \pi_\theta(y\mid q),
$$

而 loss 为：

$$
\boxed{
\mathcal{L}_{SimPO} = -\mathbb{E}\!\left[
\log \sigma\!\left(
\frac{\beta}{\lvert y_w \rvert}\log\pi_\theta(y_w\mid q)
- \frac{\beta}{\lvert y_l \rvert}\log\pi_\theta(y_l\mid q)
- \gamma
\right)
\right],
$$

其中 $\gamma > 0$ 是目标 reward margin，确保被选回答的 reward 至少比被拒回答高出 $\gamma$。

> **SimPO vs DPO vs ORPO**
>
> - **DPO**：使用 reference 模型；基于比率的隐式 reward。
> - **ORPO**：无 reference；在 SFT loss 上加入 odds-ratio 项。
> - **SimPO**：无 reference；长度归一化的对数概率 reward + margin。
> - SimPO 比 DPO 简单（无 reference 模型），且比 ORPO 更具理论性。
> - SimPO 中的长度归一化至关重要：若没有它，模型会偏好长回答。

> **TRL 中的 SimPO**
>
> ```python
> from trl import DPOConfig, DPOTrainer
>
> config = DPOConfig(
>     loss_type="simpo",
>     simpo_gamma=0.5,   # 目标 reward margin gamma
>     beta=2.5,          # 长度归一化系数
>     # 无需 ref_model！
> )
>
> trainer = DPOTrainer(
>     model=model,
>     ref_model=None,    # SimPO 是无 reference 的
>     args=config,
>     train_dataset=dataset,
> )
> ```
