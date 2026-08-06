---
layout: home
title: 偏好优化变体
permalink: /part2/ch08-preference-optimization-variants.html
---

# 偏好优化变体


本章涵盖一类用不同目标、数据假设或架构权衡来扩展或替换 DPO 的方法。每种方法都解决标准 offline DPO 的某个具体局限：分布偏移（Online DPO）、需要配对数据（KTO）、对噪声标签过拟合（IPO）、参考模型显存开销（ORPO）或训练复杂度（Best-of-N）。

## Online DPO

### 动机

标准 DPO 的主要局限：偏好数据由一个 *不同的* 模型生成（通常是较旧的 checkpoint，甚至是不同的模型族）。随着训练推进，policy 生成的文本与训练 pair 已完全不同 $\rightarrow$ loss 在一个无关分布上进行优化。

**Online DPO 方案** [guo2024direct]：每一步都从 *当前* policy 生成新鲜的偏好 pair，用 reward model 判定，然后应用 DPO loss。

### 算法

1. 从当前 $\pi_\theta$ 为每个 prompt 生成 $K$ 条响应
2. 用 reward model $r_\phi$ 给所有响应打分
3. 构造 pair：最高分 = chosen，最低分 = rejected
4. 在这些新鲜 pair 上应用 DPO loss
5. 重复（每步重新生成）

> **Online DPO = 两全其美**
>
> - 来自 DPO：简单的监督式 loss，无 value function，无 GAE，优化稳定
> - 来自 PPO：on-policy 数据，超越数据集的自我提升，无分布偏移
> - 与 GRPO 的关键差异：使用 DPO loss（基于 pair）而非 PPO loss（逐样本 advantage）
>
> **权衡**：需要 reward model（DPO 不需要），但不需要 value head（PPO 需要）。复杂度居中。


### TRL 实现

下面是基于 HuggingFace TRL 的最小可运行示例。

```python
from trl import OnlineDPOConfig, OnlineDPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "RLHFlow/ArmoRM-Llama3-8B-v0.1", torch_dtype=torch.bfloat16)

online_dpo_config = OnlineDPOConfig(
    output_dir="./online_dpo_output",
    learning_rate=5e-7,
    beta=0.1,                    # DPO beta（与标准 DPO 同义）
    num_generations=4,           # 每个 prompt 的 K 条响应
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_new_tokens=512,
    temperature=0.7,
    bf16=True,
    num_train_epochs=1,
    logging_steps=10,
)

trainer = OnlineDPOTrainer(
    model=model,
    reward_model=reward_model,
    args=online_dpo_config,
    train_dataset=prompt_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### Online DPO vs Offline DPO vs PPO

|  | **数据** | **模型** | **Loss** | **最适合** |
| --- | --- | --- | --- | --- |
| Offline DPO | 静态 pair | 2 (policy + reference) | DPO | 快速对齐、算力有限 |
| Online DPO | 从 $\pi_\theta$ 新鲜采样 | 3 (policy + reference + reward model) | DPO | 当 DPO 停滞、需要探索时 |
| PPO | 从 $\pi_\theta$ 新鲜采样 | 4 (policy + reference + reward model + value head) | PPO clip | 极致质量、复杂推理 |

## KTO —— Kahneman-Tversky Optimization

### 动机

DPO 需要 *成对* 偏好：对同一 prompt，你既需要好的也需要坏的响应。实际上大多数反馈是 *未配对* 的：用户对单个响应给出点赞/点踩，没有匹配的 pair。

**KTO 的洞见** [ethayarajh2024kto]：使用前景理论（来自行为经济学）。人类对损失的感受比对收益更强烈。“点踩”应当产生比“点赞”更强的 gradient。

### Loss 函数

$$
\boxed{\mathcal{L}_\text{KTO} = \mathbb{E}_{y_w}\left[\lambda_w (1 - v(x, y_w))\right] + \mathbb{E}_{y_l}\left[\lambda_l \cdot v(x, y_l)\right]}
$$
其中 $v(x,y) = \sigma\left(\beta \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} - z_\text{ref}\right)$，$z_\text{ref}$ 是期望 KL 散度（一个滑动 baseline）。

> **基于前景理论的 KTO 直觉**
>
> **好响应**（$y_w$）：模型通过提高它们的概率获得“效用”。但收益递减——一旦它已相当可能，就不再用力推。
>
> **坏响应**（$y_l$）：损失厌恶意味着生成坏文本的惩罚被加权得比生成好文本的奖励更强。默认 $\lambda_l = 1.0$，$\lambda_w = 1.0$，但你可以设 $\lambda_l > \lambda_w$。
>
> **关键优势**：每个训练样本都是独立的！无需匹配 pair。可直接使用点赞/点踩数据。

> **KTO 数据格式**
>
> 与 DPO 需要 \verb|{"prompt": ..., "chosen": ..., "rejected": ...}| 不同
>
> KTO 只需要：\verb|{"prompt": ..., "completion": ..., "label": true/false}|
>
> 这意味着你可以使用：
>
> - 生产流量中的点赞/点踩
> - 论坛上的赞/踩
> - 二值化的人类评分（4--5 星 = 好，1--2 星 = 坏）
> - 任何单条响应的质量信号

### TRL 实现

下面是基于 HuggingFace TRL 的最小可运行示例。

```python
from trl import KTOConfig, KTOTrainer

# 数据格式：{"prompt": str, "completion": str, "label": bool}
# label=True 表示好的，label=False 表示坏的
kto_dataset = [
    {"prompt": "What's 2+2?", "completion": "The answer is 4.", "label": True},
    {"prompt": "What's 2+2?", "completion": "It might be 5.", "label": False},
]

kto_config = KTOConfig(
    output_dir="./kto_output",
    beta=0.1,
    desirable_weight=1.0,        # 好样本的权重
    undesirable_weight=1.0,      # 坏样本的权重（损失厌恶时调高）
    learning_rate=5e-7,
    max_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
)

trainer = KTOTrainer(
    model=model,
    ref_model=ref_model,  # LoRA 时可为 None
    args=kto_config,
    train_dataset=kto_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 何时选择 KTO

- 你有二值反馈但 *没有* 匹配 pair
- 大规模的生产点赞/点踩数据
- 一类占主导（例如 90% 好、10% 坏）—— KTO 对不平衡更友好
- 在噪声标签下快速迭代（对噪声比 DPO 更鲁棒）

## IPO —— Identity Preference Optimization

### 动机

DPO 有一个退化解：通过让 chosen 与 rejected 之间的 margin *无限大* 来达到零 loss。实际上这意味着 DPO 会过拟合——把 chosen 概率推到 1、rejected 概率推到 0，记住训练数据。

**IPO 的修复** [azar2024general]：与其使用会饱和的 log-sigmoid，不如使用一个针对 *特定* margin 的平方 loss。loss 在有限差距处最小，而非无穷远。

### Loss 函数

$$
\boxed{\mathcal{L}_\text{IPO} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]}
$$

> **IPO vs DPO：通过目标 Margin 实现正则化**
>
> DPO：$\sigma(\text{margin}) \to 1$ 为最优。Margin $\to \infty$，没有自然停止点。
>
> IPO：Margin $\to \frac{1}{2\beta}$ 为最优。平方 loss 同时惩罚 **过小** 与 **过大** 的 margin。
>
> 结果：IPO 对噪声标签更鲁棒（错标的 pair 影响有界），且因为不死记硬背而泛化更好。

### TRL 实现

下面是基于 HuggingFace TRL 的最小可运行示例。

```python
from trl import DPOConfig, DPOTrainer

# IPO 在 TRL 中作为 DPO 的 loss_type 变体实现
ipo_config = DPOConfig(
    output_dir="./ipo_output",
    beta=0.1,
    loss_type="ipo",             # 关键差异！
    learning_rate=5e-7,
    max_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    num_train_epochs=1,
)

trainer = DPOTrainer(
    model=model, ref_model=None, args=ipo_config,
    train_dataset=pref_dataset, tokenizer=tokenizer, peft_config=lora_config,
)
trainer.train()
```

### 何时选择 IPO 而非 DPO

- 噪声偏好数据（众包、AI 判定带错误）
- 观察到 DPO 过拟合（训练 loss $\to$ 0 但评估退化）
- 想要更保守、更鲁棒的对齐
- 需要多个 epoch（DPO 在第一个 epoch 后退化；IPO 更稳定）

## ORPO —— Odds Ratio Preference Optimization

### 动机

迄今所有方法都需要参考模型——要么是独立副本（显存翻倍），要么通过 LoRA 隐式存在。ORPO [hong2024orpo] 通过将监督微调（Supervised Fine-Tuning, SFT）与偏好对齐合并到单一 loss 中，彻底消除了参考模型。

**关键洞见**：用生成 chosen 与生成 rejected 的 *odds ratio*（几率比）作为偏好信号。SFT 部分自然防止坍塌（无需 KL 正则）。

### Loss 函数

$$
\boxed{\mathcal{L}_\text{ORPO} = \underbrace{\mathcal{L}_\text{SFT}(y_w)}_{\text{standard NLL on chosen}} - \lambda \cdot \underbrace{\log\sigma\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)}_{\text{preference alignment via odds ratio}}}
$$
其中 $\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$。

> **ORPO：一次完成 SFT + 对齐**
>
> **SFT 项**：训练模型把 chosen 响应生成好（标准语言建模）。
>
> **Odds ratio 项**：额外推动模型偏好 chosen 而非 rejected。Odds ratio 是天然的对比，不需要参考模型。
>
> **为什么不需要参考？**SFT loss 已将模型锚定在合理文本上，扮演了其他方法中 KL-to-reference 的角色。一个模型、一次前向、一个 loss。显存减少 50%！

### TRL 实现

下面是基于 HuggingFace TRL 的最小可运行示例。

```python
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    output_dir="./orpo_output",
    beta=0.1,                    # Odds ratio 权重（lambda）
    learning_rate=5e-7,
    max_length=2048,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
    num_train_epochs=1,
    gradient_checkpointing=True,
)

trainer = ORPOTrainer(
    model=model,                 # 无需 ref_model！
    args=orpo_config,
    train_dataset=pref_dataset,  # 与 DPO 同格式：prompt/chosen/rejected
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
```

### 何时选择 ORPO

- 显存受限：负担不起参考模型副本（70B 可节省 70--140GB）
- 从 base 模型起步（尚未 SFT）—— ORPO 同时做 SFT
- 想要最简单的 pipeline：一个模型、一个 loss、一次训练
- 一开始就有良好的偏好数据

> **ORPO 的局限**
>
> - 研究不如 DPO/PPO 充分——70B 及以上规模的成熟配方较少
> - SFT 部分意味着它需要高质量的 chosen 响应（不仅是相对偏好）
> - 调试更难：两个 loss 分量可能冲突

> **另见：SimPO**
>
> **SimPO** [meng2024simpo] 是另一种无参考的偏好方法，使用长度归一化的 log-probability 作为隐式 reward，彻底消除参考模型。它在 Section 「DPO 扩展与变体」 与其他 DPO 扩展一同介绍，因为它们都共享无参考的理念。

## Best-of-N 采样（拒绝采样）

### 动机

有时候最简单的方法获胜。Best-of-N 采样（Rejection Sampling） [nakano2021webgpt] 在 RL 阶段 *完全不需要训练*——只需生成多个候选并挑选最佳。

### 算法

1. 对每个 prompt，从 policy 生成 $N$ 条响应（通常 $N = 4$--$64$）
2. 用 reward model 给所有响应打分
3. 选择最高分响应
4. （可选）将选中的响应作为下一轮的 SFT 数据

$$
\boxed{\text{Best-of-N response}: \quad y^* = \arg\max_{y_i \sim \pi_\theta(\cdot|x)} r_\phi(x, y_i)}
$$

> **为什么 Best-of-N 是合法的"RL"方法**
>
> **推理时**：Best-of-N 在不改动模型权重的情况下提升输出质量。$N=64$ 时胜率比 greedy 提升 10--20%——有时匹配甚至超过 PPO。
>
> **作为训练方法**（拒绝采样微调 / RFT）：
>
> 1. 生成大量响应，选出最佳
> 2. 在选出的响应上做 SFT
> 3. 重复（迭代精炼）
>
> 许多生产模型就是这样训练的：比 PPO 简单、几乎同样有效、完全稳定。
>
> **理论联系** [gao2023scaling]：Best-of-N 实现了隐式的 KL 约束 policy：$\pi_\text{BoN}(y|x) \propto \pi_\theta(y|x)^{1-1/N} \cdot r(x,y)^{1/N}$。

### TRL 实现

下面是基于 HuggingFace TRL 的最小可运行示例。

```python
from transformers import pipeline
import numpy as np

# 推理时 Best-of-N（手工实现）
gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def best_of_n(prompt, n=16, temperature=0.8):
    """生成 N 个候选并返回最高 reward 者。"""
    candidates = gen_pipeline(
        prompt, num_return_sequences=n,
        temperature=temperature, do_sample=True, max_new_tokens=512,
    )
    scores = [reward_model.score(prompt, c["generated_text"]) for c in candidates]
    return candidates[np.argmax(scores)]["generated_text"]

best_response = best_of_n(prompt, n=16)

# 训练：拒绝采样微调（RFT）
from trl import SFTConfig, SFTTrainer

# 步骤 1：生成并筛选
all_responses = []
for prompt in prompts:
    candidates = [generate(prompt, temp=0.9) for _ in range(16)]
    scores = [reward_model.score(prompt, c) for c in candidates]
    best_idx = np.argmax(scores)
    if scores[best_idx] > threshold:  # 质量门槛
        all_responses.append({"prompt": prompt, "completion": candidates[best_idx]})

# 步骤 2：在最佳响应上做 SFT
sft_config = SFTConfig(output_dir="./rft_output", learning_rate=2e-5, num_train_epochs=2, max_seq_length=2048)
trainer = SFTTrainer(model=model, args=sft_config, train_dataset=all_responses, tokenizer=tokenizer)
trainer.train()
# 步骤 3：用更新后的模型从步骤 1 重复（迭代 RFT）
```

### Best-of-N 的扩展律

| $N$ | **质量增益** | **成本** | **说明** |
| --- | --- | --- | --- |
| 1 | 基线 | $1\times$ | 标准采样 |
| 4 | +5--8% 胜率 | $4\times$ | 最低有用值。良好的成本/质量比 |
| 16 | +10--15% 胜率 | $16\times$ | 强力。常匹配 PPO 质量 |
| 64 | +15--20% 胜率 | $64\times$ | 开始边际递减 |
| 256 | +18--22% 胜率 | $256\times$ | 仅用于关键应用 |

> **Best-of-N 作为基线**
>
> 始终在相同算力预算下将你的 RL 方法与 Best-of-N 比较。若 64 GPU-小时的 PPO 打不过 64 GPU-小时生成的 Best-of-N，你的 PPO 有 bug。

## 总结：如何选择对齐方法

我们已经纵览了偏好优化与基于 RL 的对齐方法的完整全景。本节将关键权衡整合为一份参考，帮助实践者根据自身约束选择合适方案。


**各对齐方法的横向对比。**
| **方法** | **模型** | **数据** | **算力** | **稳定性** | **最适合** |
| --- | --- | --- | --- | --- | --- |
| PPO | 4 | 在线（生成） | 非常高 | 低 | 极致质量、复杂推理 |
| GRPO | 2（无 critic） | 在线（生成） | 高 | 中 | 数学/代码（可验证 reward） |
| DPO | 2 | 离线 pair | 低 | 高 | 风格/安全、算力有限 |
| Online DPO | 3 | 在线（生成） | 中 | 中-高 | 无分布偏移的 DPO |
| KTO | 2 | 未配对二值 | 低 | 高 | 生产反馈、点赞/点踩 |
| IPO | 2 | 离线 pair | 低 | 非常高 | 噪声标签、抗过拟合 |
| ORPO | 1 | 离线 pair | 非常低 | 高 | 显存受限、SFT+对齐合并 |
| Best-of-N | 1+RM | 在线（生成） | 中 | 完美 | 强基线、数据生成 |


![近似的质量 vs. 算力前沿。位于 SFT 上限线之上的方法超越了单纯监督微调所能达到的水平。位置仅为示意，与模型有关。](/figures/fig_029_fig29.png)

> **决策树：该用哪种方法？**
>
> 1. **有可验证 reward 吗？**（数学/代码）$\rightarrow$ **GRPO**
> 2. **在复杂任务上需要极致质量吗？** $\rightarrow$ **PPO**
> 3. **有成对偏好吗？** $\rightarrow$ **DPO**（若噪声大则用 IPO）
> 4. **仅有未配对的二值反馈？** $\rightarrow$ **KTO**
> 5. **显存受限、从 base 模型起步？** $\rightarrow$ **ORPO**
> 6. **DPO 停滞、想要 on-policy？** $\rightarrow$ **Online DPO**
> 7. **需要快速建立强基线？** $\rightarrow$ **Best-of-N / RFT**

