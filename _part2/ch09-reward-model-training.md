---
layout: home
title: 奖励模型训练
permalink: /part2/ch09-reward-model-training.html
---

奖励模型是连接人类偏好与 RL 训练信号的桥梁。训练良好的奖励模型对成功的 RLHF 至关重要;训练不佳的奖励模型会导致奖励黑客攻击(reward hacking)和行为失准。本章涵盖奖励模型的理论基础、实用训练技术以及架构选择。

## Bradley-Terry 模型 —— 完整推导

Bradley-Terry 模型 [bradley1952rank] 是成对偏好学习(pairwise preference learning)的标准概率框架。给定对 prompt $q$ 的两个回答 $y_1$ 和 $y_2$,该模型假设:

$$
P(y_1 \succ y_2 \mid q) = \sigma(r(y_1, q) - r(y_2, q))
= \frac{e^{r(y_1,q)}}{e^{r(y_1,q)} + e^{r(y_2,q)}},
$$

其中 $r: \mathcal{Y} \times \mathcal{Q} \to \mathbb{R}$ 是标量 Reward 函数,$\sigma$ 是 sigmoid 函数。

#### 最大似然估计

给定偏好对数据集 $\mathcal{D} = \{(q^{(k)}, y_w^{(k)}, y_l^{(k)})\}_{k=1}^N$,最大似然估计(MLE)目标为:

$$
\mathcal{L}_{BT}(\phi) =
-\frac{1}{N}\sum_{k=1}^N \log \sigma\!\bigl(r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)})\bigr),
$$

其中 $r_\phi$ 是由 $\phi$ 参数化的神经网络。这是一个二元交叉熵 Loss,其中“正”类别对应偏好的回答。

> **Bradley-Terry 假设**
>
> 1. 偏好具有*传递性*:若 $y_1 \succ y_2$ 且 $y_2 \succ y_3$,则 $y_1 \succ y_3$。
> 2. 偏好由*标量*Reward 决定(不存在多维偏好)。
> 3. 偏好概率仅依赖于 Reward 的*差值*。
> 4. 各偏好对之间相互*独立*(不存在标注者效应)。
>
> 这些假设在实践中常被违反,这催生了一些扩展模型,例如用于排序的 Plackett-Luce 模型和多维奖励模型。

#### Margin Loss 扩展

一种常见的扩展是引入 margin $m$,以确保胜出与落败 Reward 之间存在最小间隔:

$$
\mathcal{L}_{margin} =
-\frac{1}{N}\sum_{k=1}^N \log \sigma\!\bigl(r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)}) - m\bigr).
$$

## 奖励模型架构

> **LLM 上的分类头**
>
> 标准的奖励模型架构是:取一个预训练 LLM,将其语言建模头(将隐藏状态映射到词表 logits)替换为一个标量回归头(将最终隐藏状态映射到单一 Reward 值)。

该架构由以下部分组成:

1. **骨干网络(Backbone)**:预训练 LLM(例如 Llama、Mistral),将 prompt-response 对编码为一系列隐藏状态。
2. **池化(Pooling)**:提取最后一个 Token 位置(decoder-only 模型)或 `[CLS]` Token(encoder 模型)处的隐藏状态。
3. **回归头(Regression head)**:线性层 $W \in \mathbb{R}^{d \times 1}$,将池化后的隐藏状态映射为标量 Reward。

> **TRL 中的奖励模型训练**
>
> ```python
> from trl import RewardConfig, RewardTrainer
> from transformers import AutoModelForSequenceClassification
>
> # 加载带标量头(num_labels=1)的模型
> model = AutoModelForSequenceClassification.from_pretrained(
>     "meta-llama/Llama-3.1-8B-Instruct",
>     num_labels=1,
> )
>
> config = RewardConfig(
>     output_dir="reward_model",
>     per_device_train_batch_size=4,
>     gradient_accumulation_steps=4,
>     learning_rate=1e-5,
>     num_train_epochs=1,
>     # Margin loss
>     center_rewards_coefficient=0.01,
> )
>
> trainer = RewardTrainer(
>     model=model,
>     args=config,
>     train_dataset=dataset,  # 必须包含 chosen/rejected 列
> )
> trainer.train()
> ```

## 奖励模型训练技巧

#### Reward 中心化

原始的奖励模型输出可能存在任意尺度和偏移。对 Reward 进行中心化(减去均值)可以稳定 RL 训练:

$$
r_{centered}(y, q) = r_\phi(y, q) - \mathbb{E}_{y' \sim \pi_\theta}[r_\phi(y', q)].
$$

在 TRL 中,这通过 `center_rewards_coefficient` 参数实现,它会向奖励模型 Loss 中添加一个正则项,惩罚均值非零的 Reward。

#### 长度偏差校正

奖励模型往往存在*长度偏差*:无论质量如何,都倾向于给较长的回答分配更高的 Reward。可通过以下方式校正:

1. **长度归一化(Length normalisation)**:将 Reward 除以回答长度。
2. **长度受控训练(Length-controlled training)**:将长度作为特征,训练模型使其对长度不变。
3. **校准(Calibration)**:事后回归以剔除长度效应。

#### Margin Loss

在 Bradley-Terry Loss 中加入 margin $m$,可确保奖励模型对偏好回答与非偏好回答打出具有显著区分度的分数:

$$
\mathcal{L}_{margin} = \max\!\bigl(0,\; m - (r_w - r_l)\bigr).
$$

## 过程奖励模型 vs 结果奖励模型

> **PRM 与 ORM 对比**
>
> | **属性** | **ORM** | **PRM** |
> | --- | --- | --- |
> | Reward 信号 | 仅最终答案 | 每一步推理 |
> | 训练数据 | (prompt, answer, correct?) | (prompt, steps, step labels) |
> | 标注成本 | 低 | 高 |
> | 信用分配 | 稀疏 | 稠密 |
> | Reward 黑客 | 较易被攻击 | 较难被攻击 |
> | 最适合 | 简单任务 | 多步推理 |
> | 推理成本 | 低 | 高(需对每步打分) |

> **何时使用 PRM**
>
> 过程奖励模型(PRM)在以下场景中最有价值:
>
> - 任务需要多步推理(数学、代码、逻辑)。
> - 最终答案是二元的(正确/错误),但中间步骤质量参差不齐。
> - 你希望将奖励模型用于*搜索*(例如带步骤分数的 beam search)。
> - 你能获得步骤级标注(或可自动生成)。
>
> 对于简单任务(情感、毒性、事实性),ORM 已经足够,且成本低得多。

> **LLM RLHF 中的 PBRS**
>
> **原始 Reward**:二元正确性(最终答案正确为 1,否则为 0)—— 对多步推理而言极其稀疏。
>
> **势函数(Potential function)**:$\Phi(s) =$ 来自验证器的部分得分(例如,逻辑上有效的中间推理步骤的比例)。
>
> **塑形后的 Reward**:Agent 对每个有效推理步骤都获得增量信号,同时保留“最优 Policy 仍然最大化最终答案正确率”的理论保证。
>
> **实际实现**:
>
> - 对思维链(Chain-of-Thought,CoT)中每一步打分的过程奖励模型(PRM)
> - 代码生成中的中间编译检查
> - 多部分答案的部分匹配分数
>
> 这是基于势函数的 Reward 塑形(Potential-Based Reward Shaping, PBRS) [ng1999policy] 在 LLM 场景下的直接应用 —— “塑形 Reward 保留最优 Policy”这一理论保证使得 PRM 成为在推理任务中获取稠密 Reward 的有原则方法。

#### 自动 PRM 标注

步骤级标注可通过以下方法自动生成:

1. **蒙特卡洛 rollout**:对每个中间步骤,采样多个补全(completion),将到达正确答案的比例作为该步的 Reward。
2. **LLM-as-judge**:使用一个强大的 LLM 来评估每一步。
3. **形式化验证**:对数学/代码任务,使用验证器逐步检查。

## RLVR 中基于规则的 Reward

可验证奖励的强化学习(Reinforcement Learning from Verifiable Rewards, RLVR)使用确定性、基于规则的 Reward 函数,而非习得的奖励模型。这能显著减少奖励黑客行为(尽管模型仍可能利用格式技巧、边界情况或测试记忆来作弊),DeepSeek-R1 [deepseek2025r1] 即采用此方法。

> **TRL 中基于规则的 Reward 函数**
>
> ```python
> import re
>
> def format_reward(completions, **kwargs):
>     """对使用 <think>...</think><answer>...</answer> 格式给予 Reward。"""
>     rewards = []
>     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
>     for completion in completions:
>         text = completion[0]["content"]
>         rewards.append(1.0 if re.fullmatch(pattern, text, re.DOTALL) else 0.0)
>     return rewards
>
> def correctness_reward(completions, ground_truth, **kwargs):
>     """对最终答案正确给予 Reward。"""
>     rewards = []
>     for completion, gt in zip(completions, ground_truth):
>         text = completion[0]["content"]
>         match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
>         if match:
>             answer = match.group(1).strip()
>             rewards.append(1.0 if answer == gt else 0.0)
>         else:
>             rewards.append(0.0)
>     return rewards
>
> def code_execution_reward(completions, test_cases, **kwargs):
>     """对通过测试用例的代码给予 Reward。"""
>     import subprocess, tempfile, os
>     rewards = []
>     for completion, tests in zip(completions, test_cases):
>         code = completion[0]["content"]
>         passed = 0
>         for test in tests:
>             with tempfile.NamedTemporaryFile(
>                 mode="w", suffix=".py", delete=False
>             ) as f:
>                 f.write(code + "\n" + test)
>                 fname = f.name
>             try:
>                 result = subprocess.run(
>                     ["python", fname], capture_output=True,
>                     timeout=5, text=True
>                 )
>                 passed += int(result.returncode == 0)
>             except Exception:
>                 pass
>             finally:
>                 os.unlink(fname)
>         rewards.append(passed / len(tests))
>     return rewards
> ```

> **基于规则 Reward 的陷阱**
>
> - **格式作弊(Format gaming)**:模型学会生成正确格式但内容错误。务必将格式 Reward 与正确性 Reward 结合使用。
> - **测试用例泄漏**:如果测试用例存在于训练数据中,模型会将其记住。
> - **超时利用**:模型可能生成会超时的代码(以回避失败)。请使用严格的超时并显式惩罚超时。
> - **Reward 稀疏性**:二元 Reward(0/1)对复杂任务而言可能过于稀疏。考虑使用部分得分或中间 Reward。

## 多目标 Reward —— 组合策略

当使用多个 Reward 信号进行训练时,组合策略会显著影响最终 Policy。

> **多 Reward 组合策略**
>
> 1. **加权求和(Weighted sum)**:$r = \sum_n w_n r_n$。简单但对尺度敏感。
> 2. **先归一化再求和(GDPO)**:在组内将每个 Reward 归一化为零均值、单位方差,再加权求和。尺度无关。
> 3. **字典序(Lexicographic)**:按优先级顺序优化 Reward;仅当高优先级 Reward 平局时才考虑低优先级的 Reward。
> 4. **约束法(Constrained)**:在次要 Reward 的约束下最大化主 Reward。
> 5. **Pareto**:维护一组 Policy 的 Pareto 前沿,并根据偏好进行选择。

> **TRL 中的多 Reward 训练**
>
> ```python
> from trl import GRPOConfig, GRPOTrainer
>
> config = GRPOConfig(
>     # GDPO:对每个 Reward 独立归一化
>     multi_objective_aggregation="normalize_then_sum",
>     reward_weights=[1.0, 0.3, 0.1],  # 正确性、格式、长度
>     num_generations=8,
> )
>
> trainer = GRPOTrainer(
>     model=model,
>     reward_funcs=[
>         correctness_reward,
>         format_reward,
>         length_penalty_reward,
>     ],
>     args=config,
>     train_dataset=dataset,
> )
> ```

## 基于列表排序的 Reward

虽然 Bradley-Terry 模型处理的是*成对*偏好($y_w \succ y_l$),但许多实际场景需要同时对多个回答进行排序。列表式(listwise)奖励模型从完整排序中学习,提供更丰富的训练信号,并实现更好的校准。

#### 动机:超越成对比较

> **为何要用 listwise?**
>
> - **更丰富的信号**:对 $K$ 个回答的排序包含 $\binom{K}{2}$ 个隐式成对比较,同时还能捕捉*相对 margin*(例如排名第 1 比第 3 好多少)。
> - **更好的校准**:成对 BT 模型只学习 Reward 的*差值*;listwise 模型学习的是绝对 Reward 尺度。
> - **天然契合 GRPO**:GRPO 对每个 prompt 生成 $N$ 个回答并排序 —— listwise Reward 与该工作流直接对齐。
> - **标注效率**:对 5 个回答排序比独立标注所有 10 个可能的成对组合更快。

#### Plackett-Luce 模型

Plackett-Luce(PL)模型 [plackett1975analysis] 是将 Bradley-Terry 推广至完整排序的标准扩展。给定 $K$ 个回答 $y_1, \ldots, y_K$ 及其排序 $\pi$(其中 $\pi(1)$ 为最佳):

> **Plackett-Luce 似然**
>
> $$
> P(\pi \mid q) = \prod_{i=1}^{K} \frac{e^{r_\phi(y_{\pi(i)}, q)}}{\sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)}}
> $$
> **直觉**:顺序地从剩余项中选出最佳。每一步选中 $\pi(i)$ 的概率是对剩余项做 Softmax。
>
> **Loss 函数**:
> $$
> \mathcal{L}_{PL}(\phi) = -\frac{1}{\lvert \mathcal{D} \rvert} \sum_{(q, \pi) \in \mathcal{D}} \sum_{i=1}^{K-1} \left[ r_\phi(y_{\pi(i)}, q) - \log \sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)} \right]
> $$

> **Plackett-Luce 退化为 Bradley-Terry**
>
> 当 $K=2$ 时,PL 模型给出:$P(y_1 \succ y_2) = \frac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))$ —— 这正是 Bradley-Terry 模型。PL 是其严格的推广。

#### ListMLE 与基于排序的 Loss

> **列表式 Loss 函数**
>
> - **ListMLE** [xia2008listwise]:直接最大化真实排序的 PL 似然。简洁有效。
> - **ListNet** [cao2007listnet]:最小化模型的 top-1 概率分布与真实分布之间的 KL 散度:
> $$
> \mathcal{L}_{ListNet} = -\sum_{i=1}^{K} P_{true}(y_i  is best) \cdot \log P_{model}(y_i  is best)
> $$
> 其中 $P_{\text{model}}(y_i \text{ is best}) = \frac{e^{r_\phi(y_i)}}{\sum_j e^{r_\phi(y_j)}}$。
> - **LambdaRank** [burges2006lambdarank]:用排序指标(例如 NDCG)的变化对成对 Gradient 加权。当排名靠前的质量更重要时尤为有用。
> - **RankNet** [burges2005ranknet]:对所有成对组合求和的成对交叉熵 —— 等价于在从排序中抽取的所有 $\binom{K}{2}$ 个对上应用 BT。

#### GRPO 与拒绝采样中的 listwise Reward

> **与 GRPO 的集成**
>
> GRPO 天然产生有序的分组:对每个 prompt,$N$ 个回答被打分并排序。可以直接在这些排序上训练 listwise 奖励模型:
>
> 1. **生成**:从 Policy 中对每个 prompt 采样 $N=8$ 个回答。
> 2. **排序**:使用已有的奖励模型(或人工标注者)产生完整排序 $\pi$。
> 3. **训练 listwise RM**:在 $(q, \pi)$ 元组上优化 PL Loss。
> 4. **在 GRPO 中使用**:listwise RM 对每个回答分配标量 Reward $r(y_i, q)$;GRPO 将优势计算为 $\hat{A}_i = (r_i - \mu) / \sigma$。
>
> **相对成对方法的优势**:listwise RM 同时看到全部 $N$ 个回答,可学到排名第 1 应当比排名第 $N$ 拥有高得多的 Reward(而非仅仅“比另一个回答略好”)。

#### 实用考量

> **Listwise 训练挑战**
>
> - **标注成本**:完整排序昂贵。部分排序(8 个中取 top-3)能在质量损失很小的前提下降低成本。
> - **并列**:真实排序常有并列。使用 PL 的并列扩展:对并列项分配相等的概率质量。
> - **位置偏差**:标注者倾向偏好排在前面的项。随机化展示顺序并训练去偏。
> - **列表长度**:训练时 $K=4$--8 较为常见。更长的列表($K>16$)只会增加噪声,收益有限。
> - **一致性**:不同标注者的排序可能不一致。使用标注者间一致性($\kappa > 0.6$)作为质量过滤。

> **Plackett-Luce 训练代码**
>
> ```python
> import torch
> import torch.nn.functional as F
>
> def plackett_luce_loss(rewards, rankings):
>     """
>     Args:
>         rewards: (batch, K) - 对 K 个回答预测的标量 Reward
>         rankings: (batch, K) - 真实排序索引(0 = 最佳)
>     Returns:
>         标量 Loss
>     """
>     batch_size, K = rewards.shape
>     # 按真实排序顺序对 Reward 排序
>     sorted_rewards = torch.gather(rewards, 1, rankings)  # (batch, K)
>
>     # PL 对数似然:对各位置求和
>     loss = 0.0
>     for i in range(K - 1):
>         # 对剩余项(位置 i 到 K)做 log-softmax
>         remaining = sorted_rewards[:, i:]           # (batch, K-i)
>         log_probs = remaining[:, 0] - torch.logsumexp(remaining, dim=1)
>         loss -= log_probs.mean()
>
>     return loss / (K - 1)
>
> # 示例:每个 prompt 8 个回答,由标注者排序
> rewards = reward_model(responses)          # (batch, 8)
> rankings = torch.argsort(human_scores, descending=True)  # 最佳在前
> loss = plackett_luce_loss(rewards, rankings)
> loss.backward()
> ```

