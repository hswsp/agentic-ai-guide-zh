---
layout: home
title: GRPO —— 组相对策略优化
permalink: /part2/ch07-grpo.html
---

# GRPO —— 组相对策略优化


组相对策略优化（Group Relative Policy Optimization, GRPO） [shao2024deepseekmath] 是一种专为语言模型设计的强化学习算法，它消除了对独立 value network（critic）的需求。GRPO 由 DeepSeek 在 DeepSeekMath 工作中提出，随后在 DeepSeek-R1 [deepseek2025r1] 中被扩展到更大规模，已经迅速成为 LLM 训练中的主流 RL 方法——被大多数开源对齐框架（TRL、OpenRLHF、veRL）作为默认算法采用。

其核心思想看似简单却出人意料地有效：与其训练一个神经网络去预测期望 reward（即 PPO 中的 critic），GRPO 通过对同一 prompt 生成多条响应、利用该组的 reward 统计量作为 baseline，从经验上 *估计* 这一基线。这从显存中去掉了一整个模型，将工程复杂度减半，而且——令人意外的是——往往优于 PPO，因为经验基线比训练不充分的 value function 更准确。

GRPO 在下列场景中尤其有效：

- **推理任务**：具有可验证 reward（数学、代码），二值正确性提供干净信号。
- **大模型**（70B 及以上）：去掉 critic 带来的显存节省至关重要。
- **多轮对话（Multi-Turn）与 Agent 场景**：跨工具调用（Tool Calling）的价值估计本就难以处理。

本章涵盖 GRPO 的动机、算法、关键变体（Dr. GRPO、DAPO、2-GRPO、GDPO）以及基于 TRL 的实战实现。

## 动机

PPO 的 value model（critic）在语言任务上存在三大问题：

1. **显存**：value head 与 policy 共享主干（70B 模型占 140GB）。若分离则显存翻倍。
2. **精度**：对部分序列预测期望 reward 极其困难。value function 经常出错 $\rightarrow$ advantage 出错 $\rightarrow$ gradient 方向出错。
3. **训练**：value head 需要大量样本才能收敛。RL 早期它给出嘈杂的预测，破坏 policy 学习的稳定性。

**GRPO 的关键洞见** [shao2024deepseekmath]：与其学习 $V(s)$，不如从一组样本中经验地 *估计* 它。对同一 prompt 生成 $G$ 条响应，计算它们的 reward，并用组统计量作为 baseline。

## 算法

1. 对每个 prompt $x$，采样 $G$ 条 completion：$\{y_1, \ldots, y_G\} \sim \pi_\theta(\cdot|x)$
2. 对每条打分：$r_i = R(x, y_i)$
3. 组内归一化：$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$，其中 $\mu_G = \frac{1}{G}\sum_j r_j$，$\sigma_G = \text{std}(\{r_j\})$
4. 用这些 advantage 应用 PPO 风格的 clipped 更新

$$
\boxed{\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \qquad L = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_i,\; \text{clip}(r_t(\theta), 1{\pm}\epsilon)\hat{A}_i\right)\right] - \beta D_\text{KL}[\pi_\theta\|\pi_\text{ref}]}
$$

> **为什么组内归一化有效**
>
> **组均值近似 $V(s)$**：若对同一 prompt 采样足够多的响应，其平均 reward 就是期望 reward 的蒙特卡洛估计，即 value function。
>
> **高于均值 = 好动作**：$\hat{A}_i > 0$ 表示该响应对该 prompt 比平均更好，应强化。
>
> **低于均值 = 坏动作**：$\hat{A}_i < 0$ 表示比平均更差，应抑制。
>
> **归一化**：除以 $\sigma_G$ 确保 advantage 在不同 reward 量级的 prompt 之间具有尺度不变性。
>
> **DeepSeek-R1 突破** [deepseek2025r1]：纯 GRPO 配合二值正确性 reward（答对 $r=1$，答错 $r=0$）在数学/代码上训练时，模型自发涌现出思维链（Chain-of-Thought，CoT）推理、自我验证和错误纠正——完全没有被显式指示这么做。


![GRPO 实战：对一个数学 prompt 采样 $G{=}5$ 条响应。三条正确（$r{=}1$），两条错误（$r{=}0$）。组均值 $\mu_G{=}0.6$ 充当 baseline；正确响应获得正 advantage（强化），错误响应获得负 advantage（抑制）。](/figures/fig_029_fig29.png)

## TRL 实现

下面是基于 HuggingFace TRL 的最小可运行示例。

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,           # G = 组大小
    temperature=1.0,             # 高温度以保证组内多样性
    max_completion_length=2048,  # 最大响应长度
    beta=0.04,                   # KL 惩罚系数
    learning_rate=1e-6,
    per_device_train_batch_size=2,  # 每设备 prompt 数（×8 生成 = 16 响应）
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    logging_steps=10,
    # vLLM 生成以加速（GRPO 由于 8 倍生成，速度至关重要）
    use_vllm=True,
    vllm_gpu_memory_utilization=0.7,
)

# Reward 函数：数学任务的二值正确性
def reward_fn(completions, prompts, **kwargs):
    """返回 float 列表：正确为 1.0，错误为 0.0。"""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        answer = extract_answer(completion)
        expected = get_ground_truth(prompt)
        rewards.append(1.0 if answer == expected else 0.0)
    return rewards

# 可以组合多个 reward 函数！
def format_reward_fn(completions, **kwargs):
    """对使用正确 LaTeX 格式的额外奖励。"""
    return [0.5 if "\\boxed{" in c else 0.0 for c in completions]

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=[reward_fn, format_reward_fn],  # 多目标！
    train_dataset=math_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## 组大小分析

| $G$ | **信号质量** | **算力** | **使用场景** |
| --- | --- | --- | --- |
| 2 | 极嘈杂（掷硬币） | 低 | 不推荐 —— 太嘈杂，无法稳定学习 |
| 4 | 中等 | 中等 | 快速实验、简单任务（通过率 $>$ 50%） |
| 8 | 良好（标准） | 高 | 默认。适合大多数任务的良好平衡 |
| 16 | 极佳 | 非常高 | 困难任务（通过率 $<$ 20%），需要多次尝试才能拿到正样本 |
| 32 | 近乎完美 | 极高 | 仅在算力充裕且任务非常困难时使用 |

> **关键：组内必须同时包含成功与失败**
>
> 若 $G$ 条响应全对（$r_i = 1 \;\forall i$）：所有 advantage = 0，无学习信号！\
>
> 若全错：同样的问题。Prompt 难度必须与模型能力匹配。\
>
> **Goldilocks 原则**：将 prompt 过滤至当前模型 20--80% 通过率。随着模型改进，每 500 步重新筛选。

## GRPO 变体与扩展

### GRPO 组中的多样性

> **RL 训练中的模式坍塌**
>
> 若没有多样性压力，RL 训练的 LLM 会坍塌到狭窄的高 reward 响应集合上：
>
> - 模型对每种题型学到一个“模板”答案
> - 熵急剧下降；模型变得确定性
> - Reward hacking 变得更容易（狭窄输出更容易被利用）
> - 泛化能力受损：模型记住的是 reward 模式而非推理
>
> KL 惩罚 $\beta D_\text{KL}[\pi_\theta \| \pi_\text{ref}]$ 是主要的多样性机制，但单靠它并不够。

> **GRPO 组多样性**
>
> GRPO 对每个 prompt 生成 $N$ 条响应并使用组内排序。组内多样性至关重要：
>
> - **高温度**（$\tau=0.8$--$1.0$）：确保响应多样，便于有意义的比较
> - **较大的 $N$**（8--16）：样本越多，越可能同时包含好的和差的方案
> - **DAPO 的“不重复”惩罚**：拒绝组内重复响应，强制探索
> - 若 $N$ 条响应完全相同：advantage 为零，无学习信号
> - 若响应过于多样（随机）：reward 信号嘈杂，学习缓慢
>
> **甜点**：温度选取应让方案有差异，同时仍然切题。


**RL 训练中促进多样性的方法。**
| **方法** | **如何促进多样性** |
| --- | --- |
| 熵奖励（Entropy bonus） | 在 reward 中加入 $\alpha H(\pi_\theta)$。直接惩罚低熵（确定性）policy。 |
| KL 惩罚 | $-\beta D_\text{KL}[\pi_\theta \| \pi_\text{ref}]$ 防止坍塌到单一模式。 |
| 拒绝采样（Rejection Sampling） | 生成大量候选，按 reward 保留 top-$k$。自然筛选出多样的高质量响应。 |
| Best-of-N | 推理时：生成 $N$ 条响应，全部评分，返回最佳。多样性来自采样。 |
| 带多样性 pair 的 DPO | 在 chosen/rejected 不仅质量不同、*方法*也不同的 pair 上训练。 |
| 多 reward | 使用多个 reward model（安全、有用性、代码质量）。防止坍塌到单一维度。 |

> **多样性 vs. 质量的权衡**
>
> 更多样性并非总是更好：
>
> - 过度多样（高熵）= 随机、无用的响应
> - 多样性不足（低熵）= 重复的、被 reward hack 的响应
> - **监控**：训练中跟踪响应熵、唯一 n-gram 比例和 reward 分布宽度。若三者同时下降，你就遇到了坍塌问题。

#### 用于 RL 数据收集的 Verbalized Sampling

后训练对齐（基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）、直接偏好优化（Direct Preference Optimization, DPO））经常因为 *典型性偏差*（typicality bias）而降低输出多样性：人类标注者系统性地偏好熟悉的、“典型的”文本而非新颖的替代方案。这种模式坍塌是数据层面的现象，并非纯算法问题。

Verbalized Sampling（VS） [zhang2025verbalized] 是一种无需训练的 prompt 策略，通过要求模型在单次生成中 **显式言语化多条响应上的概率分布** 来规避这种坍塌。

> **Verbalized Sampling —— 核心思想**
>
> 与其采样单一响应（这会坍塌到众数），不如 prompt 模型输出 *多个候选响应及其概率*：
>
> `‘‘Generate 5 jokes about coffee and their corresponding probabilities.’’`
>
> 模型会产出如下列表：
>
> 1. 笑话 A（概率：0.35）
> 2. 笑话 B（概率：0.25）
> 3. 笑话 C（概率：0.20）
> 4. 笑话 D（概率：0.12)
> 5. 笑话 E（概率：0.08）
>
> 然后从这个言语化分布中采样。由于模型显式表征了整个分布（而不仅是 argmax），那些概率较低但富有创意/多样的响应就变得可达。

```python
# Verbalized Sampling：prompt 模型输出分布
def verbalized_sample(model, tokenizer, task, n=5):
    prompt = (
        f"{task}\n\n"
        f"Generate {n} different responses and assign a probability "
        f"to each (probabilities should sum to 1.0). "
        f"Format: [response] (probability: X.XX)"
    )
    output = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    # 从输出中解析响应及概率
    responses, probs = parse_verbalized_distribution(
        tokenizer.decode(output[0])
    )
    # 从言语化分布中采样
    import random
    chosen = random.choices(responses, weights=probs, k=1)[0]
    return chosen
```

> **为什么 Verbalized Sampling 有效**
>
> - **绕过模式坍塌**：从对齐模型的标准采样高度集中在一两个“安全”响应。VS 强迫模型说出那些它 *知道* 但通常不会浮现出来的替代方案。
> - **多样性是语义的**：与温度缩放（词法噪声）不同，VS 产出真正不同的方案——模型会就不同选项进行推理。
> - **随能力扩展**：能力更强的模型产出校准更好的言语化分布——它们从 VS 中受益 *更多*（创意写作上获得 1.6--2.1$\times$ 多样性增益）。
> - **无需训练**：不需要微调或修改解码；在推理时与任何遵循指令的模型协同工作。
> - **用于 GRPO**：用 VS 为每个 prompt 生成 $G$ 个响应候选——确保组内包含语义多样的方案，而非仅表层变化。

在深入各种扩展之前，让我们简要回顾前几节建立的基础 GRPO 算法。其核心机制——采样一组 completion、归一化它们的 reward、并应用 clipped policy gradient——简洁优雅。然而，实践者很快发现了一些具体的失败模式：预训练偏差稀释 gradient（Dr. GRPO）、对称 clipping 限制探索（DAPO）、过大的组大小造成浪费（2-GRPO）、以及多目标场景下 reward 尺度失衡（GDPO）。下面各节依次解决这些问题。

> **GRPO 基线回顾**
>
> 给定 prompt $q$，从当前 policy $\pi_\theta$ 采样 $G$ 条 completion $\{o_1,\dots,o_G\}$。计算 reward $\{r_1,\dots,r_G\}$ 并归一化：
> $$
> \hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \qquad
> \mu_r = \frac{1}{G}\sum_{i=1}^G r_i, \quad
> \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i-\mu_r)^2}.
> $$
> 逐 token 的 clipped surrogate loss 为：
> $$
> \mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}
> \sum_{t=1}^{|o_i|}
> \min\!\Bigl(
> \rho_{i,t}\,\hat{A}_i,\;
> clip(\rho_{i,t},1{-}\epsilon,1{+}\epsilon)\,\hat{A}_i
> \Bigr),
> $$
> 其中 $\rho_{i,t} = \pi_\theta(o_{i,t}|q,o_{i,<t})\,/\,\pi_{\text{old}}(o_{i,t}|q,o_{i,<t})$。

### DAPO —— Dynamic Adaptive Policy Optimization

> **为什么需要 DAPO？**
>
> 基础 GRPO 使用 *对称* clipping：无论 policy 想增加还是减小某 token 的概率，约束都是相同的。但探索与利用具有不同的风险特征。增大一个好 token 的概率通常是安全的；而抑制一个恰好出现在坏 completion 中的 token，若该 token 本身是中性的，则可能是灾难性的错误。DAPO [yu2025dapo] 引入五项针对性修复，共同显著改进训练稳定性与最终性能。

#### 组件 1 —— 非对称 Clipping（Clip-Higher）

标准 PPO/GRPO 在 $[1-\epsilon, 1+\epsilon]$ 处对称地 clip importance ratio。DAPO 将其替换为非对称区间：

$$
\boxed{
clip_{DAPO}(\rho, A) =
\begin{cases}
clip(\rho,\, 1-\epsilon,\, 1+\epsilon_{high}) & if  A > 0 \
clip(\rho,\, 1-\epsilon,\, 1+\epsilon) & if  A \le 0
\end{cases}
$$

其中 $\epsilon_{\text{high}} > \epsilon$（典型取值：$\epsilon=0.2$，$\epsilon_{\text{high}}=0.28$）。当 advantage 为正时，允许 policy 更进一步偏向好 token；当 advantage 为负时，应用常规的保守 clipping 以避免过度抑制。

#### 组件 2 —— Token 级 Loss 聚合

基础 GRPO 将 loss 除以 *序列数* $G$。DAPO 则除以所有序列的 *token 总数*：

$$
\mathcal{L}_{token} =
-\frac{1}{\sum_{i=1}^G |o_i|}
\sum_{i=1}^G \sum_{t=1}^{|o_i|}
\min\!\bigl(\rho_{i,t}\hat{A}_i,\;
clip_{DAPO}(\rho_{i,t},\hat{A}_i)\,\hat{A}_i\bigr).
$$

这样可以防止长 completion 仅因为 token 数更多而主导 gradient 信号。

#### 组件 3 —— 过长过滤（Overlong Filtering）

当一条 completion 被截断（在最大长度预算内没有 EOS token），它会提供 *误导性* 信号：模型因为那些被正确生成、只是恰好出现在截断边界之前的 token 而被惩罚。DAPO 将这些 completion 整条 mask 掉：

$$
m_i = \mathbf{1}[EOS \in o_i], \qquad
\mathcal{L}_{filtered} =
-\frac{\sum_{i=1}^G m_i \sum_t (\cdots)}{\sum_{i=1}^G m_i |o_i|}.
$$

#### 组件 4 —— 软过长惩罚（Soft Overlong Punishment）

比起硬 mask，更软的变体应用一个长度惩罚，随着 completion 接近最大长度 $L_{\max}$ 平滑增长：

$$
r_i \leftarrow r_i - \lambda \cdot \max\!\left(0,\, \frac{|o_i| - L_{cache}}{L_{\max} - L_{cache}}\right),
$$

其中 $L_{\text{cache}}$ 是一个“安全”长度阈值。

#### 组件 5 —— 动态采样（Dynamic Sampling）

DAPO 会重新采样那些整组 completion 获得相同 reward（全对或全错）的 prompt，因为此类组在归一化后贡献零 gradient。这能在整个训练过程中保持有效 batch size 稳定。

> **TRL 中的 DAPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     # 非对称 clipping
>     epsilon=0.2,
>     epsilon_high=0.28,          # Clip-Higher
>     # Token 级 loss
>     loss_type="dapo",           # 启用 token 级聚合
>     # 过长过滤
>     mask_truncated_completions=True,
>     # 生成预算
>     max_completion_length=1024,
>     num_generations=8,
>     # 注意：DAPO loss 内部已处理零方差组过滤
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> trainer.train()
> ```

> **何时使用 DAPO**
>
> - completion 经常触及长度上限的长篇推理任务。
> - 任何观察到 reward 方差在训练中途坍塌的场景。
> - 当基础 GRPO 表现不稳定（loss 尖峰、熵坍塌）时。
> - 在大多数任务上推荐作为基础 GRPO 的 *即插即用改进*。

### GSPO —— Group Sequence Policy Optimization

> **Off-Policy 问题**
>
> GRPO *逐 token* 地 clip importance ratio。但一条 500 token 的序列，即使每个单独的 ratio 都在 $[1-\epsilon, 1+\epsilon]$ 内，逐 token ratio 的乘积可能大或小到天文数字。当在同一 batch 上进行多次 gradient 步（off-policy）时，这种不匹配迅速放大，clipping 界限在序列级别上变得毫无意义。

GSPO [chen2025gspo] 将 *序列级* importance weight 定义为逐 token ratio 的几何平均，等价于完整序列概率比的 $|o_i|$ 次方根：

$$
\boxed{
s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{old}(o_i \mid q)}\right)^{1/|o_i|}
= \exp\!\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{old}(o_{i,t}|q,o_{i,<t})}\right).
$$

这是 *长度归一化* 的序列概率比。GSPO loss 对每条序列 clip 这个单一标量：

$$
\mathcal{L}_{GSPO} = -\frac{1}{G}\sum_{i=1}^G
\min\!\Bigl(s_i(\theta)\,\hat{A}_i,\;
clip(s_i(\theta),1{-}\epsilon,1{+}\epsilon)\,\hat{A}_i\Bigr).
$$

> **GSPO 与 GRPO 的 Clipping 对比**
>
> - **GRPO**：独立 clip $|o_i|$ 个逐 token ratio。一条序列可以所有 ratio 都在界内、却拥有 $10^{50}$ 的乘积 ratio。
> - **GSPO**：对每条序列 clip 一次几何平均。保证 *序列级* policy 变化有界。
> - GSPO 在 off-policy 重要性采样上理论正确；GRPO 只是近似。

> **TRL 中的 GSPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     # 序列级重要性采样
>     importance_sampling_level="sequence",   # GSPO 模式
>     # Off-policy：每个 batch 复用多次 gradient 步
>     steps_per_generation=4,
>     num_generations=8,
>     epsilon=0.2,
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **何时使用 GSPO**
>
> GSPO 在 `steps_per_generation > 1`（off-policy 训练）时最有用。对于纯 on-policy 训练（$\text{steps\_per\_generation}=1$），它与 GRPO 的差异可忽略。Off-policy 训练能极大降低生成成本（最昂贵的步骤），使得 GSPO + off-policy 成为强力的效率选择。

### Dr. GRPO —— Debiased Reward GRPO

> **预训练偏差问题**
>
> 标准 GRPO 在组内归一化 advantage，但 *预训练分布* 会引入系统性偏差：在预训练数据中常见的 token 即便不携带任务相关信息也会获得较大 gradient。Dr. GRPO [liu2024drgrpo] 识别并纠正这种偏差，将 gradient 信号聚焦于 *有信息量的* token。

Dr. GRPO 修改逐 token gradient 权重，以考虑该 token 对 reward 信号的边际贡献。模型本就赋予高概率的 token（无论 reward 如何）会被降权：

$$
w_{i,t} = \hat{A}_i \cdot \bigl(1 - \pi_{ref}(o_{i,t}|q,o_{i,<t})\bigr),
$$

其中 $\pi_{\text{ref}}$ 是参考（预训练）模型。这是一种 *token 效率* 形式：gradient 被集中到 policy 真正需要改变的 token 上。

> **TRL 中的 Dr. GRPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     loss_type="dr_grpo",
>     num_generations=8,
>     beta=0.04,   # KL 惩罚系数
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     ref_model=ref_model,   # token 加权需要参考模型
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **何时使用 Dr. GRPO**
>
> - 当训练任务在预训练与 RL 之间词表分布严重不匹配时。
> - 当观察到常见的填充 token 主导 gradient 时。
> - 与接近初始 policy 的参考模型配合良好。

### 2-GRPO —— 最简的两次 Rollout GRPO

> **"It Takes Two" 洞见**
>
> "It Takes Two" 论文 [xu2025twograpo] 从经验和理论上证明：$G=2$ 的 GRPO（每个 prompt 仅两条 completion）在大多数推理基准上能匹配甚至超过 $G=16$ 的 GRPO。这令人意外——为什么更少样本反而足够？

关键洞见是：GRPO 的有效性并 *不* 主要来自准确的 advantage 估计（那需要较大的 $G$），而是来自一个结构上类似 DPO 的隐式 *对比目标*：

$$
\mathcal{L}_{2-GRPO} \approx
-\mathbb{E}_{(o^+, o^-) \sim \pi_\theta}\!\left[
\log \sigma\!\left(
\beta \log \frac{\pi_\theta(o^+|q)}{\pi_{old}(o^+|q)}
- \beta \log \frac{\pi_\theta(o^-|q)}{\pi_{old}(o^-|q)}
\right)
\right],
$$

其中 $o^+$ 是较高 reward 的 completion，$o^-$ 是较低 reward 的。$G=2$ 时这一对比结构是显式的；$G=16$ 时同样的信号存在，但被冗余对稀释。

> **2-GRPO 的算力节省**
>
> - $G=2$ vs $G=16$：**生成算力减少 8$\times$**。
> - 生成通常是瓶颈（占 60--80% 的 wall-clock 时间）。
> - 端到端总训练加速：约 4--6$\times$。
> - 在 GSM8K、MATH 和代码基准上无精度损失。

> **TRL 中的 2-GRPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     num_generations=2,      # 关键改动 —— 只做两次 rollout
>     loss_type="grpo",       # 标准 GRPO loss 即可
>     epsilon=0.2,
>     # G=2 时 batch size 至少为 2 * num_prompts_per_step
>     per_device_train_batch_size=2,
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **2-GRPO 的注意事项**
>
> $G=2$ 时 advantage 归一化只对两个值进行，因此归一化后的 advantage 总是 $\{+1, -1\}$（若两 reward 相等则为 $\{0, 0\}$）。这意味着 gradient 幅度与 reward 差距无关。对于 reward 差异的 *幅度* 很重要的任务（例如部分得分），更大的 $G$ 仍然有利。

### SAPO —— Soft Adaptive Policy Optimization

> **硬 Clipping 的脆弱性**
>
> PPO 风格的 clipping 产生不连续 gradient：clip 带外 gradient 为零，带内非零。这种“悬崖效应”会在边界附近造成不稳定，并使信赖域对 $\epsilon$ 的选择敏感。SAPO [han2025sapo] 用一个平滑、温度可控的 gate 函数替换硬 clip。

SAPO 用一个平滑替代品替换 $\min(\rho A, \mathrm{clip}(\rho,\cdot)\,A)$ 目标：

$$
\boxed{
\mathcal{L}_{SAPO}(\rho, A) =
\begin{cases}
-A \cdot \sigma\!\left(\dfrac{\rho - 1}{\tau_+}\right) \cdot \rho
& if  A > 0 \[8pt]
-A \cdot \sigma\!\left(\dfrac{1 - \rho}{\tau_-}\right) \cdot \rho
& if  A \le 0
\end{cases}
$$

其中 $\sigma$ 是 sigmoid 函数，$\tau_+, \tau_-$ 是非对称温度参数。温度越高 gate 越软（更多探索）；温度越低越接近硬 clipping。

> **SAPO 温度直觉**
>
> - $\tau_+ = 1.0$：正 advantage 的中等 gate（允许探索）。
> - $\tau_- = 1.05$：负 advantage 略软的 gate（避免过度抑制）。
> - $\tau \to 0$：恢复硬 PPO clipping。
> - $\tau \to \infty$：恢复未 clip 的 policy gradient。

> **TRL 中的 SAPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     loss_type="sapo",
>     sapo_temperature_pos=1.0,    # 正 advantage 的 tau_+
>     sapo_temperature_neg=1.05,   # 负 advantage 的 tau_-
>     num_generations=8,
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> ```

### TIS 与 MIS —— Truncated 与 Masked Importance Sampling

> **隐蔽的 vLLM 概率不匹配**
>
> 当使用 vLLM 进行快速生成时，vLLM 返回的 log-probability 与训练前向传播中计算的不同 [zhong2025tismis]。这 *不是* bug —— 它源于不同的 CUDA 内核、不同的浮点精度以及不同的 Attention 实现（如 FlashAttention vs PagedAttention）。这种不匹配悄悄破坏了 on-policy 假设：用于计算 importance ratio 的“旧 policy”概率是错的，导致 gradient 估计出现偏差。

#### Truncated Importance Sampling（TIS）

TIS 通过将 gradient 乘以一个截断的修正因子来纠正偏差：

$$
\boxed{
w_{TIS}(o_i) = \min\!\left(C,\; \frac{\pi_{train}(o_i|q)}{\pi_{vllm}(o_i|q)}\right),
$$

其中 $\pi_{\text{train}}$ 是训练前向传播给出的概率，$\pi_{\text{vllm}}$ 是 vLLM 报告的概率。在 $C$ 处截断可防止极端修正破坏训练稳定性。

#### Masked Importance Sampling（MIS）

MIS 采取更激进的策略：对任何修正 ratio 超过阈值 $C$ 的序列，将其 gradient 置零：

$$
w_{MIS}(o_i) = \mathbf{1}\!\left[\frac{\pi_{train}(o_i|q)}{\pi_{vllm}(o_i|q)} \le C\right].
$$

这更保守，但避免了大（即使被截断）修正权重的风险。

#### 序列级 vs Token 级 IS

TIS 和 MIS 都既可在 token 级、也可在序列级应用：

- **序列级**：将 ratio 计算为所有 token 的几何平均（如 GSPO 那样）。理论正确但方差更高。
- **Token 级**：为每个 token 单独计算 ratio。有偏（逐 token 修正的乘积不等于序列修正），但方差更低。

> **TRL 中的 TIS 和 MIS**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> # 针对 vLLM 概率不匹配的 Truncated IS 修正
> config_tis = GRPOConfig(
>     use_vllm=True,
>     vllm_importance_sampling_correction=True,
>     vllm_importance_sampling_mode="sequence_truncate",  # TIS
>     vllm_importance_sampling_cap=5.0,                   # C 阈值
> )
>
> # Masked IS 修正
> config_mis = GRPOConfig(
>     use_vllm=True,
>     vllm_importance_sampling_correction=True,
>     vllm_importance_sampling_mode="sequence_mask",      # MIS
>     vllm_importance_sampling_cap=3.0,
> )
> ```

> **何时使用 TIS/MIS**
>
> - 使用 vLLM 生成时 **始终** 考虑开启。
> - 当不匹配较小时（同模型、不同精度）首选 TIS。
> - 当不匹配较大或不可预测时首选 MIS。
> - 序列级 IS 在理论上更受推荐；token 级是实用上的折中。

### VESPO —— Variational Sequence-Level Soft Policy Optimization

> **原理性的 Reward 重塑**
>
> 大多数 GRPO 变体只是启发式地修改 clipping 机制。VESPO 从变分推断框架推导出一个原理性的 reward 重塑核，将 policy optimization 视为近似后验推断。VESPO [luo2025vespo] 推导出的核是平滑、非对称的，并天然地处理异步或 off-policy 训练中的陈旧性问题。

VESPO 从变分目标推导出每条 trajectory $\tau$ 的加权函数 $W(\tau)$。最终的 gradient 权重形式为：

$$
\boxed{
g(\tau) = W(\tau)^k \cdot \exp\!\bigl(\lambda(1 - W(\tau))\bigr),
$$

其中 $W(\tau) = \pi_\theta(\tau)/\pi_{\text{old}}(\tau)$ 是序列级 importance weight，$k$ 控制加权的锐度，$\lambda$ 控制对陈旧（低权重）trajectory 的指数衰减。该核：

- 处处平滑（在 clip 边界没有不连续 gradient）。
- 通过指数项天然地降权陈旧 trajectory（$W \ll 1$）。
- 是非对称的：高权重 trajectory（$W > 1$）与低权重 trajectory 区别对待。

> **TRL 中的 VESPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     loss_type="vespo",
>     vespo_k_pos=2.0,         # 锐度指数（正 advantage）
>     vespo_lambda_pos=3.0,    # 陈旧性衰减（正 advantage）
>     num_generations=8,
>     steps_per_generation=2,  # off-policy；VESPO 处理陈旧性
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> ```

### DPPO —— Direct Policy Divergence Policy Optimization

> **Ratio Clipping 的问题**
>
> PPO 的 ratio clipping 是约束新旧 policy 间 KL 散度的代理。但这个代理并不完美：clipping 对低概率 token 过度惩罚（这里概率的小绝对变化对应较大的 ratio 变化），而对高概率 token 惩罚不足（这里大的绝对变化对应较小的 ratio 变化）。DPPO [an2025dppo] 用 *直接散度估计* 替换 ratio clipping。

DPPO 直接使用新旧 policy 分布间的 Total Variation（TV）或 KL 散度来计算信赖域约束：

$$
\mathcal{L}_{DPPO} = -\mathbb{E}\!\left[
\hat{A} \cdot \pi_\theta(o|q) \cdot \mathbf{1}[D(\pi_\theta \| \pi_{old}) \le \delta]
\right],
$$

其中 $D$ 是所选的散度度量。在实践中，DPPO 用 token 级二值或 top-$k$ mask 来近似：

- **binary_tv**：mask 掉 $|\pi_\theta - \pi_{\text{old}}| > \delta$ 的 token。
- **binary_kl**：mask 掉 $\pi_\theta \log(\pi_\theta/\pi_{\text{old}}) > \delta$ 的 token。
- **topk_tv**：仅保留按 TV 贡献排序的 top-$k$ token。
- **topk_kl**：仅保留按 KL 贡献排序的 top-$k$ token。

> **DPPO —— 概念性实现**
>
> DPPO 目前尚未作为内置 TRL trainer 提供。一种自定义实现可基于 GRPOTrainer，修改 loss 以基于分布散度（TV 或 KL）而非标准概率 ratio 进行 clip：
>
> ```python
> # 伪代码：DPPO 需要自定义 trainer 子类
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     num_generations=8,
>     beta=0.04,
> )
> # 重写 loss 计算以使用分布式 clipping：
> # 当 TV(pi_new || pi_old) > delta 时 clip，而非
> # 当 pi_new/pi_old 超过 [1-eps, 1+eps] 时
> ```

> **DPPO 处于研究阶段**
>
> DPPO 是较新的研究贡献，尚未集成到主流 RL 库中。它在你观察到标准 ratio clipping 失效时（例如在 token 概率分布高度偏斜的任务上）最有用。

### ScaleRL 和 CISPO

> **RL 的扩展律**
>
> ScaleRL 论文 [luo2025scalerl] 系统研究了什么因素能让 LLM 的 RL 训练有效扩展。关键发现是：两项修改——batch 级 reward 缩放与 DAPO 风格的 token 级 loss——共同解锁了规模化下的强性能，单独任一项都不够。CISPO（Clipped IS Policy Optimization）就是由此得到的算法。

#### Batch 级 Reward 缩放

标准 GRPO 在单个 prompt 的 $G$ 条 completion 组内归一化 reward。CISPO 则在 *整个 batch* 上归一化 reward：

$$
\hat{A}_i = \frac{r_i - \mu_{batch}}{\sigma_{batch} + \epsilon},
$$

其中 $\mu_{\text{batch}}$ 和 $\sigma_{\text{batch}}$ 在当前训练 batch 的所有 reward 上计算。这提供更稳定的 baseline，并防止任何单一 prompt 主导 gradient。

#### CISPO Loss

CISPO 结合了 batch 级缩放、DAPO 的 token 级 loss 聚合以及非对称 clipping：

$$
\mathcal{L}_{CISPO} =
-\frac{1}{\sum_{i,t} m_{i,t}}
\sum_{i=1}^G \sum_{t=1}^{|o_i|} m_{i,t} \cdot
\min\!\bigl(\rho_{i,t}\hat{A}_i,\;
clip_{DAPO}(\rho_{i,t},\hat{A}_i)\,\hat{A}_i\bigr),
$$

其中 $m_{i,t}$ 是过长过滤 mask。

> **TRL 中的 CISPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     loss_type="cispo",
>     scale_rewards="batch",          # batch 级 reward 归一化
>     mask_truncated_completions=True,
>     epsilon=0.2,
>     epsilon_high=5.0,               # CISPO 的 epsilon_max（ScaleRL 论文）
>     num_generations=8,
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[reward_fn],
>     args=config,
>     train_dataset=dataset,
> )
> ```

> **ScaleRL 关键发现**
>
> 1. 单独的 batch 级 reward 缩放：改进有限。
> 2. 单独的 token 级 loss：改进有限。
> 3. 两者结合：**协同效应** —— 显著优于单独任一项。
> 4. batch 越大，从 batch 级缩放中受益越多。
> 5. CISPO 是大规模 RL 训练的推荐默认方案。

### GDPO —— Group Reward-Decoupled Policy Optimization

> **多 Reward 坍塌问题**
>
> 在多目标 RL 中（例如同时优化正确性与格式），标准 GRPO 归一化 *组合后* 的 reward。若某一项 reward 方差远高于另一项，它会主导归一化 advantage，相当于忽略其他 reward。这就是 *advantage 坍塌*：低方差 reward 贡献近乎为零的 gradient。GDPO [zhong2025gdpo] 在聚合前 *独立* 归一化每项 reward。

其核心机制在聚合前 *独立* 归一化每项 reward：

$$
\boxed{
\hat{A}_n^{(i)} = \frac{r_n^{(i)} - \mu_n}{\sigma_n + \epsilon}, \qquad
\hat{A}^{(i)} = \sum_{n=1}^N w_n \hat{A}_n^{(i)},
$$

其中 $r_n^{(i)}$ 是 completion $i$ 的第 $n$ 项 reward，$\mu_n$ 和 $\sigma_n$ 是组内第 $n$ 项 reward 的均值与标准差，$w_n$ 是用户指定权重。

> **GDPO 与标准多 reward GRPO 的对比**
>
> - **标准**：$\hat{A}^{(i)} = \frac{\sum_n w_n r_n^{(i)} - \mu_{\text{combined}}}{\sigma_{\text{combined}}}$。高方差 reward 占主导。
> - **GDPO**：单独归一化每项 reward 后再组合。每项 reward 按其权重 $w_n$ 按比例贡献。
> - 当 reward 量级或方差差异极大时，GDPO 不可或缺。

> **TRL 中的 GDPO**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     multi_objective_aggregation="normalize_then_sum",
>     reward_weights=[1.0, 0.5],   # [正确性, 格式] 的权重
>     num_generations=8,
> )
>
> def correctness_reward(completions, **kwargs):
>     return [1.0 if is_correct(c) else 0.0 for c in completions]
>
> def format_reward(completions, **kwargs):
>     return [0.1 if has_good_format(c) else 0.0 for c in completions]
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[correctness_reward, format_reward],
>     args=config,
>     train_dataset=dataset,
> )
> ```

### GOPO —— Group Ordinal Policy Optimization

GOPO [choi2025gopo] 始于一个简单观察：reward model 是用成对比较（“A 是否优于 B？”）训练的，因此只有其输出的 **排序** 是可信的——原始数值分数本身没有内在意义。然而 GRPO 直接把这些原始幅度喂入 advantage 计算。对于不可验证 reward 的任务——摘要、开放式聊天、指令跟随——这种不匹配引入了噪声，因为 0.6 reward 分的差距可能在输出空间某个区域反映真实质量，而在另一个区域毫无意义。

**关键洞见**：完全丢弃 reward 幅度。仅使用组内 reward 的 **序数排名**。

**算法**：给定一组 $N$ 条响应 $\{o_1, \ldots, o_N\}$ 及其 reward $\{r_1, \ldots, r_N\}$：

1. 按 reward 对响应排名：赋予 rank $\text{rank}(o_i) \in \{1, \ldots, N\}$（1 = 最差，$N$ = 最佳）。
2. 用基于 rank 的分数替换原始 advantage：
$$
\boxed{\hat{A}_i^{\text{GOPO}} = f\!\left(\frac{\text{rank}(o_i)}{N}\right)}
$$
其中 $f$ 是单调变换（例如线性映射到 $[-1, 1]$ 或分位数归一化）。
3. 用基于 rank 的 advantage 应用 PPO 风格的 clipped 目标。

**与 GRPO 的对比**：

| **方面** | **GRPO** | **GOPO** |
| --- | --- | --- |
| Advantage 信号 | $\hat{A}_i = (r_i - \mu)/\sigma$（使用幅度） | $\hat{A}_i = f(\text{rank}_i / N)$（仅使用序数 rank） |
| 对 reward 尺度的敏感性 | 高 —— 校准不良的 RM 分数会扭曲 advantage | 无 —— 对单调 reward 变换不变 |
| 最适合 | 可验证 reward（二值、校准良好） | 不可验证 reward（基于 RM、幅度嘈杂） |

**经验性增益**（在不可验证任务上相对 GRPO）：

- 整个优化过程中，reward 曲线（训练和留出集）都位于 GRPO 之上
- 由独立 LLM 评估器判定的胜率在大多数训练 checkpoint 上有所改进
- 收敛显著更快——以更少的 gradient 步数达到 GRPO 的最终质量
- 优势随 reward model 变得更嘈杂或校准更差而扩大

> **GOPO vs. GRPO 的使用选择**
>
> - **使用 GRPO**：当 reward 可验证且精确时（数学正确性、代码测试通过/失败、二值信号）。幅度携带有意义的信息。
> - **使用 GOPO**：当 reward 来自学习的 reward model、面向主观任务（有用性、风格、安全）时。RM 的相对排序可信，但其绝对分数是任意的。

