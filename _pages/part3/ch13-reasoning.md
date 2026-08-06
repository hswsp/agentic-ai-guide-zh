---
layout: home
title: 面向大型推理模型的强化学习
permalink: /part3/ch13-reasoning.html
---

# 面向大型推理模型的强化学习

大型推理模型的出现是现代 AI 最重要的进展之一。与优化下一个 Token 预测的标准语言模型训练不同，面向推理的强化学习教会模型在*回答之前先思考*——在推理时分配额外算力以探索、验证并精炼中间步骤。本节系统地介绍支撑这一范式的方法、架构与扩展律。

> **关键洞见：推理即搜索问题**
>
> 多步推理可以被刻画为在部分解构成的树上的**搜索问题**。树中每个节点是一个推理状态（思维链的前缀），每条边是一个推理步骤（一个 Token 或一个句子），叶子节点是最终答案。面向推理的强化学习教会模型高效地在这棵树中导航——探索有前途的分支、从死胡同回溯，并把算力分配到最重要的地方。

## 动机与背景

### 为什么推理需要不同的强化学习方法

标准的基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）针对一段完整回答优化一个标量奖励。对于需要多步推理的任务——数学、形式化验证、竞赛编程、科学推导——这种表述在以下几个方面不够充分：

- **稀疏奖励**：一道数学题可能需要 20 个中间步骤；单一的结果奖励无法为导致错误的中间步骤提供梯度信号。
- **长时序**：推理链可能跨越数百到数千个 Token，造成严重的信用分配问题。
- **组合搜索**：有效推理路径的空间呈指数级膨胀；模型必须学会高效搜索这一空间。
- **可验证性**：与主观的文本质量不同，数学与逻辑的正确性是可客观验证的，从而无需人工标注即可自动计算奖励。

### 思维链：涌现行为 vs 训练得到的能力

思维链（Chain-of-Thought，CoT）推理最初是在足够大的语言模型中被观察到的一种*涌现*能力[wei2022chain]：当用逐步示例进行 Prompt 时，大型模型（通常 $\geq$ 100B 参数）会自发地产生中间推理步骤，并由此提升准确率。这引出了一个基本问题：CoT 是规模带来的涌现性质，还是可以被显式训练？

正如 DeepSeek-R1 及相关工作所表明的，答案是**两者皆是**——但有若干重要的微妙之处：

- **涌现的 CoT** 源于上下文学习并需要大型基础模型。它脆弱、对 Prompt 敏感，且泛化不稳定。
- 通过 RL **训练得到的 CoT** 让模型*内在地*把生成推理链作为生成过程的一部分，与提示风格无关。这些链条更长、更具探索性，并表现出质上不同的行为（自我纠正、回溯、验证）。

> **「顿悟时刻」现象（DeepSeek-AI 等，2025）**
>
> 在对推理模型进行 RL 训练时，DeepSeek 的研究者观察到一个引人注目的涌现行为：训练到某一阶段，模型开始在推理链中途自发地*重新审视*自己最初的思路，使用诸如「Wait, let me reconsider…」或「Actually, I think I made an error…」这样的措辞。这种自我纠正行为——*并未*被显式训练——纯粹由「最大化最终答案准确率」的 RL 目标涌现而出。它表明 RL 能够发现对求解难题工具性有用的元认知策略。

### 测试时计算的扩展律

推动推理模型研究的一个核心经验发现是：**测试时算力与性能之间存在可预测的扩展关系**。记 $C_{\text{train}}$ 为训练算力（FLOPs），$C_{\text{test}}$ 为推理算力（生成的 Token 数）。关键观察是：

$$
\text{Accuracy}(C_{\text{train}}, C_{\text{test}}) \approx f\!\left(\alpha \log C_{\text{train}} + \beta \log C_{\text{test}}\right)
$$

其中 $f$ 是某个单调函数，$\alpha, \beta > 0$ 为常数。这意味着：使用更多推理算力的较小模型可以匹敌使用较少推理算力的较大模型——这是算力与性能权衡的根本性转变。

![测试时算力扩展曲线示意图。在各种模型规模下，性能随推理 Token 数对数线性提升；较小模型在更多算力下可以逼近较大模型在更少算力下的表现。](/figures/fig_044_test_time_scaling.png)

其实践含义深远：**推理模型用推理算力换取训练算力**。不必总是部署尽可能大的模型，而是可以部署一个具备推理能力的较小模型，并在难题上分配更多 Token 用于「思考」。

## 测试时扩展方法

上述扩展律表明，在推理阶段投入更多算力可以显著提升推理性能。本节系统介绍将测试时扩展（Test-Time Scaling）落地的各类**方法**——从简单的思维链到复杂的树与图搜索算法。这些方法构成了一个用推理成本换取准确率的连续谱，理解其结构对于设计现代推理系统至关重要。

![测试时扩展方法的连续谱。每种方法都以额外的推理算力换取更高的推理准确率。各方法在概念上层层递进：CoT 引入显式推理，Self-Consistency 加入采样，ToT 引入结构化搜索，GoT 加入合并操作，而 MCTS 加入学习到的价值引导。](/figures/fig_045_test_time_spectrum.png)

### 思维链（CoT）

思维链 Prompt[wei2022chain] 是所有测试时扩展方法的基础。模型不再直接输出答案，而是生成中间推理步骤，将复杂问题分解为可处理的子问题。

**零样本 CoT。**

Kojima 等[kojima2022large] 证明，仅在 Prompt 末尾追加「Let's think step by step」就能在没有任何示例的情况下激发推理行为。这一简单触发器可以激活足够大模型（$\geq$ 100B 参数）中的潜在推理能力。

**少样本 CoT。**

Wei 等[wei2022chain] 表明，提供少量带有显式推理轨迹的示例能够让较小的模型也有效地进行推理：

$$
\text{Prompt} = [(x_1, z_1, y_1), (x_2, z_2, y_2), \ldots, (x_k, z_k, y_k), (x_{\text{test}}, \texttt{?})]
$$

其中 $z_i$ 是为示例 $(x_i, y_i)$ 手工编写的推理轨迹。

**形式化刻画。**

CoT 将单步预测 $p(y|x)$ 转化为多步序列生成：

$$
p(y|x) = \sum_{z} p(y|x, z) \cdot p(z|x) \approx p(y|x, z^*) \cdot p(z^*|x)
$$

其中 $z^* = (z_1, z_2, \ldots, z_T)$ 是贪心推理链。对所有可能链条求和是不可计算的；标准 CoT 仅采用单一样本（贪心或温度采样）。

**局限性。**

单链 CoT 是脆弱的：一旦早期推理步骤出错，后续所有步骤都会建立在错误基础上，且没有任何恢复机制。

### 自洽性（多数投票）

自洽性（Self-Consistency）[wang2023selfconsistency] 通过采样**多条独立的推理链**并对最终答案取多数投票，来缓解 CoT 单链脆弱的问题：

$$
\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbf{1}[y_i = y], \quad \text{where } (z_i, y_i) \sim p(\cdot | x), \; T > 0
$$

**关键性质**：

- 使用温度 $T > 0$ 的采样以产生多样的链条（通常 $T = 0.7$--$1.0$）
- 链条之间无交互——完全可并行化
- 准确率随 $N$ 单调提升（$N \approx 40$ 之后收益递减）
- 在 GSM8K 上：CoT = 56.5\%，Self-Consistency（$N$=40）= 74.4\%（使用 PaLM-540B[chowdhery2022palm]）
- 等价于**使用结果奖励的 Best-of-N**（多数投票充当隐式的 ORM）

> **多数投票为何有效**
>
> 若模型生成正确推理链的概率为 $p > 0.5$，则由大数定律可知：当 $N \to \infty$ 时，对 $N$ 个独立样本做多数投票的准确率趋近 100\%。即便 $p = 0.3$（模型通常是错的），只要正确答案集中在某一个值上、而错误答案彼此分散，多数投票仍能恢复出正确答案。这是测试时扩展的统计基础。

### 思维树（ToT）

思维树（Tree-of-Thoughts，ToT）[yao2024tree] 将 CoT 从**线性链**推广为**树形结构**，使模型能够探索多条推理路径、评估中间状态，并从前景不佳的分支回溯。这为推理过程引入了刻意的规划。

**核心抽象。**

推理问题被分解为对一棵树的搜索，其中：

- **根节点**：初始问题陈述 $x$
- **节点**：部分推理状态 $s = (x, z_1, \ldots, z_k)$
- **边**：单个推理步骤（「想法」）$z_{k+1}$
- **叶子**：包含最终答案的完整解答
- **价值函数**：$V(s)$ 估计某个部分解的前景如何

**形式化定义。**

$$
\text{ToT} = (\mathcal{G}, \mathcal{E}, V, \pi_\theta, \text{Search})
$$

其中：

- $\mathcal{G}$：**想法生成器**——产生 $b$ 个候选的下一想法：$\{z^{(1)}, \ldots, z^{(b)}\} \sim \pi_\theta(\cdot | s)$
- $\mathcal{E}$：**状态评估器**——为部分解打分：$V(s) \in \{$*sure*, *maybe*, *impossible*$\}$ 或 $V(s) \in [0, 1]$
- $\pi_\theta$：生成想法的语言模型
- $\text{Search}$：搜索算法（BFS 或 DFS）

![思维树在「24 点」任务上的运行：对 {4, 9, 10, 13} 进行四则运算得到 24。在每一层，模型生成 $b=3$ 个候选想法，分别评估（sure/maybe/impossible），剪掉前景不佳的分支，并扩展最有希望的分支。绿色路径通向解答；红色路径在早期被剪枝。](/figures/fig_046_tot_example.png)

**搜索算法。**

**BFS（广度优先搜索）：**

1. 为当前深度的每个节点生成 $b$ 个候选想法
2. 用 $V(\cdot)$ 评估所有候选
3. 保留前 $k$ 个最有希望的状态（束搜索）
4. 将这 $k$ 个状态全部推进到下一层
5. 重复，直到找到解答或达到深度上限

**DFS（深度优先搜索）：**

1. 为当前状态生成 $b$ 个候选想法
2. 评估：若 $V(s) =$ *impossible*，立即回溯
3. 若 $V(s) =$ *sure/maybe*，向更深处递归（选择最有希望的一个）
4. 若达到深度上限仍未得到解答，则回溯
5. 继续直到找到解答或所有分支都被探索完

> **示例：ToT——价值评估 Prompt**
>
> `EVAL_PROMPT` 用于 LLM 评估部分推理状态；`GEN_PROMPT` 用于想法生成。

```python
# LLM 用于评估部分推理状态：
EVAL_PROMPT = """Evaluate if this partial solution can reach 24.

Numbers remaining: [4, 4, 10]
Steps so far: 13 - 9 = 4

Can these remaining numbers (4, 4, 10) be combined using +,-,*,/
to make 24?

Analysis: 4 * (10 - 4) = 4 * 6 = 24. Yes!

Judge: sure/maybe/impossible
Answer: sure"""

# 想法生成 Prompt：
GEN_PROMPT = """Input: 4 9 10 13
Possible next steps:
1. 13 - 9 = 4 (left: 4 4 10)
2. 10 + 13 = 23 (left: 4 9 23)
3. 9 - 4 = 5 (left: 5 10 13)
..."""
```

**计算成本。**

对于分支因子为 $b$、深度为 $d$、束宽为 $k$ 的 ToT：

$$
\text{LLM calls (BFS)} = \underbrace{k \cdot b}_{\text{generation}} + \underbrace{k \cdot b}_{\text{evaluation}} = 2kb \text{ per level} \implies \text{Total} = 2kbd
$$

对于 24 点游戏：$b=3, k=2, d=3 \implies 36$ 次 LLM 调用，而标准 CoT 仅需 1 次。

**结果。**

在 24 点游戏（一个具挑战性的算术推理任务）上，ToT 达到 74\% 的成功率，而 CoT 仅 4\%——在相同基础模型（GPT-4）之上，结构化搜索带来了巨大的提升。

### 思维图（GoT）

思维图（Graph-of-Thoughts，GoT）[besta2024graph] 将 ToT 从树推广到**有向无环图（DAG）**，引入了一项关键能力：**合并**来自不同分支的部分解。这让模型可以把多条推理路径中的洞见综合到一个精炼的解答中。

**关键操作。**

GoT 在 ToT 之上引入了三种操作：

- **Generate**：从一个状态产生新的想法（与 ToT 相同）
- **Aggregate/Merge**：把多个想法合并为一个更精炼的想法——这是树结构中不可能做到的
- **Refine**：基于反馈对一个想法进行迭代改进
- **Score**：评估想法的质量（与 ToT 的价值函数相同）

![CoT（线性链）、ToT（树——只能分支不能合并）与 GoT（DAG——分支可以合并）的对比。对于一个排序任务，GoT 可以把数组拆为子问题、独立（并行）求解，然后**合并**结果——在纯树结构中不可能做到。这使得分治式推理成为可能。](/figures/fig_047_got_comparison.png)

**图操作（形式化）。**

设 $\mathcal{V} = \{v_1, \ldots, v_n\}$ 为想法节点，$\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ 为有向边。GoT 支持：

$$
\textbf{Generate}(v) : v \to \{v_{c_1}, \ldots, v_{c_b}\} \quad \text{(创建子节点)}
$$

$$
\textbf{Aggregate}(v_1, \ldots, v_k) \to v_{\text{merged}} \quad \text{(将 $k$ 个想法合并为一个)}
$$

$$
\textbf{Refine}(v, n) \to v' \quad \text{(通过 $n$ 次迭代改进 $v$)}
$$

$$
\textbf{Score}(v) \to s \in [0, 1] \quad \text{(评估想法质量)}
$$

**Aggregate** 操作是关键差异：它从多个父节点向单个子节点连边，从而形成 DAG 而非树。这使得以下能力成为可能：

- **分治**：拆分问题 $\to$ 并行求解子问题 $\to$ 合并解答
- **集成推理**：生成多个视角，然后综合出最佳想法
- **迭代精炼**：把评估结果反馈回去以改进早期想法

**结果。**

在排序任务（一项需要合并的任务）上，GoT 在质量相当时相比 ToT 节省 62\% 成本。在集合求交与关键词计数任务上，由于合并操作支持更高效的分解，GoT 在相同质量下减少了 30--40\% 的 LLM 调用。

### 结合奖励模型的 Best-of-N

Best-of-N 采样（Rejection Sampling）（BoN）[nakano2021webgpt, stiennon2020learning] 是最简单的扩展方法，它利用一个**学习得到的奖励模型**在候选中进行选择：

$$
y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} R_\phi(x, y), \quad y_i \sim \pi_\theta(\cdot | x)
$$

**按奖励模型类型的变体**：

- **结合 ORM 的 BoN**：对完整解答打分，选择得分最高者。当 ORM $\approx$ 正确性检查时等价于 Self-Consistency。
- **结合 PRM 的 BoN**：对每一推理步打分，选择最小步得分最高的解答（即在任何步骤上出错的可能性最小）。
- **加权 BoN**：按奖励对候选加权：$y^* \sim \text{softmax}(R(y_1)/\tau, \ldots, R(y_N)/\tau)$。

> **BoN 扩展律**
>
> 对于单样本准确率为 $p$ 的模型，$N$ 次采样中至少有一个正确样本的概率为：
>
> $$
> P(\text{success with BoN}) = 1 - (1-p)^N
> $$
>
> 在完美奖励模型（总能正确选择的预言机）下：
>
> - $p = 0.3, N = 10$：成功率 = $97\%$
> - $p = 0.1, N = 50$：成功率 = $99.5\%$
>
> 在实践中，不完美的奖励模型会限制有效的 $N$——超过 $N \approx 64$--$256$ 后，奖励模型的误差开始主导，准确率趋于停滞甚至下降（**奖励黑客**）。

### 面向推理的蒙特卡洛树搜索（MCTS）

蒙特卡洛树搜索（MCTS）[kocsis2006bandit, silver2016mastering] 将 ToT 的结构化探索与**学习到的价值估计**及**访问计数统计**结合起来，从而最优地分配推理算力。MCTS 最初为博弈而设计（AlphaGo[silver2016mastering]），后被 AlphaProof[alphaproof2024]、rStar[qi2024mutual] 等系统改造用于 LLM 推理。

**算法（针对 LLM 推理改造）。**

每次 MCTS 迭代包含四个阶段：

![面向推理的 MCTS 的四个阶段：(1) **选择**：用 UCB 遍历树以找到一个有前途的叶子节点；(2) **扩展**：从该叶子节点生成新的推理步骤；(3) **模拟**：将推理补全到终止状态并进行评估；(4) **反向传播**：沿路径更新价值估计。](/figures/fig_048_mcts_phases.png)

**推理中的 UCB。**

节点选择使用 PUCT（Predictor + UCB applied to Trees）：

$$
a^* = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s, a)} \right]
$$

其中 $P(s,a) = \pi_\theta(a|s)$ 是 LLM 从状态 $s$ 生成步骤 $a$ 的先验概率。这使探索偏向 LLM 本就认为可能的步骤，而 UCB 项鼓励尝试探索不足的备选方案。

> **面向数学推理的 MCTS：运行示例**
>
> **问题**：证明 $\sqrt{2}$ 是无理数。
>
> **第 1 次迭代**（Selection $\to$ 根节点，Expansion）：
>
> - 生成 3 个候选的首步：
>   1. "Assume for contradiction that $\sqrt{2} = p/q$ in lowest terms."（$P = 0.7$）
>   2. "Consider the decimal expansion of $\sqrt{2}$ = 1.414..."（$P = 0.15$）
>   3. "Use the fundamental theorem of arithmetic."（$P = 0.10$）
> - 从 $z_1$ 做 Rollout：4 步内得到正确证明 $\to$ $r = 1.0$
> - 从 $z_2$ 做 Rollout：失败（小数展开不能证明无理性）$\to$ $r = 0.0$
> - 反向传播：$Q(s_0, z_1) = 1.0$，$N(s_0, z_1) = 1$
>
> **第 2 次迭代**（Selection：按 UCB 选中 $z_1$）：
>
> - 从状态"Assume $\sqrt{2} = p/q$..."进行扩展：
>   1. "Then $2 = p^2/q^2$, so $p^2 = 2q^2$."（$P = 0.8$）
>   2. "Then $p$ and $q$ share no common factors."（$P = 0.15$）
> - 从 $z_4$ 做 Rollout：正确续推 $\to r = 1.0$
> - 反向传播：$Q(s_0, z_1) = 1.0$，$Q(s_1, z_4) = 1.0$
>
> **20 次迭代之后**：树已探索 8 条不同的推理路径。访问次数最多的路径被选为最终证明：$z_1 \to z_4 \to z_6 \to z_8$（基于奇偶性论证的经典反证法）。

**对比：ToT vs MCTS。**

| 维度 | ToT | MCTS |
| --- | --- | --- |
| 价值估计 | LLM Prompt（"sure/maybe/impossible"） | 学习得到的价值网络 + Rollout 统计 |
| 探索方式 | 固定束宽；不回访 | UCB 自适应地把预算分配给有前途的节点 |
| 算力分配 | 各深度层均匀 | 聚焦：在更难的子问题上做更多模拟 |
| 训练集成 | 无训练；纯 Prompt | 可将 MCTS 策略蒸馏回基础模型[silver2016mastering] |
| 最适用于 | 分支简单的问题（24 点） | 需要深度探索的复杂问题（证明、代码） |

### 推理步级的束搜索

束搜索——在 NMT 与文本生成中长期作为标准方法——也可以在*推理步*层面而非 Token 层面应用。我们不再跟踪前 $k$ 个 Token 序列，而是跟踪前 $k$ 个**推理前缀**：

$$
\mathcal{B}_d = \text{top-}k\left\{ (s_1, \ldots, s_d) : \sum_{i=1}^d \log \pi_\theta(s_i | s_{<i}) + \lambda \cdot V_\phi(s_1, \ldots, s_d) \right\}
$$

其中评分将 LLM 的对数概率（流畅性）与价值模型估计（正确性）相结合。这本质上是用学习到的价值函数取代提示式价值函数的 ToT-BFS。

### 迭代精炼与自我纠正

与探索*广度*（多条并行链）不同，迭代精炼把算力投入到*深度*——反复改进同一个解答：

$$
y^{(t+1)} = \text{LLM}\!\left(\text{"Improve this solution:"}, y^{(t)}, \text{"Errors found:"}, e^{(t)}\right)
$$

其中 $e^{(t)}$ 可来自：

- **自我验证**：让模型检查自己的答案
- **外部验证**：运行代码、符号化地校验数学
- **评判模型**：由一个独立模型识别错误

代表性方法：**Self-Refine**[madaan2023selfrefine]（迭代自我反馈）、**Reflexion**[shinn2023reflexion]（通过存入记忆的反思实现的语言化 RL）以及 **LATS**[zhou2024lats]（树搜索 + 基于反思的剪枝）。

### 方法对比与选型指南

| 方法 | 结构 | LLM 调用 | 可并行 | 需要 RM？ | 最适用于 |
| --- | --- | --- | --- | --- | --- |
| CoT[wei2022chain] | 链 | 1 | N/A | 否 | 简单—中等难度问题 |
| Self-Consistency[wang2023selfconsistency] | 并行链 | $N$ | ✓ 完全 | 否（多数投票） | 答案离散的数学题 |
| Best-of-N + ORM | 并行链 | $N$ + 1 | ✓ 完全 | 是（ORM） | 有良好 RM 的通用任务 |
| Best-of-N + PRM | 并行链 | $N$ + $N{\cdot}K$ | ✓ 完全 | 是（PRM） | 复杂多步推理 |
| ToT[yao2024tree] | 树（BFS/DFS） | $O(kbd)$ | 部分 | LLM 作判官 | 结构化搜索问题 |
| GoT[besta2024graph] | DAG | $O(kbd)$ | 部分 | LLM 作判官 | 可分解的问题 |
| MCTS[kocsis2006bandit] | 树 + 价值 | $O(N_{\text{sim}} \cdot d)$ | 部分 | 是（价值网络） | 困难证明、编程 |
| Self-Refine[madaan2023selfrefine] | 线性（迭代） | $2T$ | 否 | 自我评判 | 开放式生成 |
| LATS[zhou2024lats] | 树 + 反思 | $O(N \cdot d)$ | 部分 | LLM 作判官 | Agent 任务 |

> **何时选用哪种方法**
>
> - **预算 $<$ 5$\times$ 基础成本**：使用 CoT 或 Self-Consistency。性价比最高。
> - **预算 5--50$\times$**：使用结合 PRM 的 Best-of-N（若你有好的奖励模型）或 $b=3, k=2$ 的 ToT-BFS。
> - **预算 50--500$\times$**：使用配备已训练价值函数的 MCTS。这正是 DeepSeek-R1 与 OpenAI o1 所处的区间——以长推理链实现的隐式树搜索。
> - **需要并行**：Self-Consistency 与 Best-of-N 完全可并行；ToT/MCTS 需要顺序的深度扩展。
> - **无可用奖励模型**：使用 Self-Consistency（多数投票）或采用 LLM 作判官评估的 ToT。
> - **可分解的问题**：当问题具有天然子问题时（排序、多文档综合、带模块的代码），GoT 表现出色。

> **推理模型中的隐式测试时扩展**
>
> 现代推理模型（DeepSeek-R1[deepseek2025r1]、OpenAI o1/o3[openai2024o1, openai2025o3]）通过生成长思维链来执行**隐式测试时扩展**。它们的「思考」Token 起到了类似 MCTS Rollout 的作用：模型探索多种思路、回溯（"Wait, let me reconsider..."）、验证中间步骤，并在更难的子问题上分配更多 Token。R1/o1 训练的关键洞见是：GRPO/RL 让模型在*一次生成内*就执行这种隐式搜索，从而无需外部编排（ToT Prompt、MCTS 基础设施）。模型自身即成为搜索算法。

## DeepSeek-R1

DeepSeek-R1[deepseek2025r1] 是首个在主要基准上达到或超越 OpenAI o1 的完全开源大型推理模型。其训练流程在技术上完全透明，已成为基于 RL 的推理的事实参考实现。

### 两阶段训练流程

**阶段 1：冷启动监督微调**

基础模型（DeepSeek-V3）首先在一个经过精心整理的小规模长思维链示例数据集上做微调。这个「冷启动」阶段服务于两个目的：

1. **格式初始化**：模型学会在给出最终答案之前，以 `<thinking>...</thinking>` 的格式输出推理。
2. **稳定性**：若没有冷启动 SFT，直接在基础模型上从零进行纯 RL 会导致训练动力学不稳定与退化输出（例如混语、重复循环）。

冷启动数据集仅包含 $\sim$ 数千条样本，刻意保持较小规模，以避免过度约束 RL 后续将发现的推理风格。

**阶段 2：基于 GRPO 的强化学习**

冷启动 SFT 之后，模型进入使用组相对策略优化（Group Relative Policy Optimization, GRPO）的大规模 RL 阶段。R1 中使用的完整 GRPO 目标见「R1 中的 GRPO 公式」一节。

> **R1 训练流程总结**
>
> 1. **基础模型**：DeepSeek-V3（671B MoE，37B 激活参数）
> 2. **冷启动 SFT**：$\sim$ 数千条长 CoT 样本，格式：`<thinking>...</thinking><answer>...</answer>`
> 3. **RL 阶段**：在数学与代码问题上使用带可验证奖励的 GRPO
> 4. **拒绝采样（Rejection Sampling）**：生成多个解答，保留正确的
> 5. **在 RL 输出上做 SFT**：在高质量的 RL 生成链上微调
> 6. **最终 RL**：用于对齐与有用性的第二轮 RL

### 奖励设计：准确率奖励与格式奖励

R1 的一个关键设计选择是**不使用过程奖励模型（PRM）**。取而代之，R1 使用两种简单且可自动计算的奖励：

**准确率奖励**

对答案可验证的数学问题：

$$
r_{\text{acc}}(y, y^*) = \begin{cases} 1 & \text{if } \texttt{verify}(y, y^*) = \texttt{True} \\ 0 & \text{otherwise} \end{cases}
$$

其中 $y$ 是模型的最终答案（从 `<answer>` 标签中提取），$y^*$ 是真实答案。`verify` 函数使用符号数学比较（如 SymPy）来处理等价形式。

对于代码问题，准确率奖励由通过测试用例决定：

$$
r_{\text{acc}}^{\text{code}}(y, \mathcal{T}) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \mathbf{1}[\texttt{execute}(y, t) = \texttt{expected}(t)]
$$

**格式奖励**

用于强制 `<thinking>...</thinking>` 结构：

$$
r_{\text{fmt}}(y) = \begin{cases} 1 & y \text{ 含有有效的 thinking 与 <answer> 标签} \\ 0 & \text{否则} \end{cases}
$$

**组合奖励**

$$
r(y, y^*) = r_{\text{acc}}(y, y^*) + \lambda_{\text{fmt}} \cdot r_{\text{fmt}}(y)
$$

在原始实现中 $\lambda_{\text{fmt}} = 0.1$（足够小以不至于主导优化，足够大以防止格式崩塌）。

> **没有过程奖励模型**
>
> R1 一个引人注目且令人意外的发现是：**不需要过程奖励模型（PRM）**。尽管推理链很长，仅靠结果奖励就足以让 RL 发现高质量的推理策略。作者推测，数学/代码奖励的可验证性提供了足够的信号，而 PRM 自身又会引入新的失效模式（步级别的奖励黑客）。这与 OpenAI 所采取的方法形成对比（见「OpenAI o1/o3 系列」一节）。

### R1 中的 GRPO 公式

GRPO[shao2024deepseekmath] 是一种策略梯度方法，它通过对一*组*采样回答估计优势，从而避免训练独立的价值网络。对于一个问题 $q$，GRPO 从当前 Policy $\pi_\theta$ 采样 $G$ 个回答 $\{y_1, y_2, \ldots, y_G\}$，并相对组均值计算优势。

**分组采样与优势归一化**

给定问题 $q$，采样 $G$ 个输出：

$$
\{y_i\}_{i=1}^G \sim \pi_\theta(\cdot \mid q)
$$

使用组合奖励公式的奖励函数计算 Reward $\{r_i\}_{i=1}^G$。第 $i$ 个回答的归一化优势为：

$$
\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}
$$

其中 $\mu_r = \frac{1}{G}\sum_{i=1}^G r_i$，$\sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_r)^2}$，$\epsilon = 10^{-8}$ 用于数值稳定。

**GRPO 目标**

GRPO 目标对概率比做截断（与近端策略优化（Proximal Policy Optimization, PPO）相同），并对参考 Policy $\pi_{\text{ref}}$ 加上 KL 惩罚：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q \sim \mathcal{D},\, \{y_i\} \sim \pi_\theta(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\left( \rho_{i,t}\, \hat{A}_i,\; \text{clip}(\rho_{i,t}, 1{-}\varepsilon, 1{+}\varepsilon)\, \hat{A}_i \right) - \beta\, \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] \right]
$$

其中：

- $\rho_{i,t} = \dfrac{\pi_\theta(y_{i,t} \mid q, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid q, y_{i,<t})}$ 是逐 Token 的概率比
- $\varepsilon \in \{0.1, 0.2\}$ 是 PPO 的截断参数
- $\beta > 0$ 控制 KL 惩罚的强度
- $|y_i|$ 是第 $i$ 个回答的长度（长度归一化可防止偏向较短回答）

**KL 惩罚的具体形式**

KL 散度项逐 Token 计算：

$$
\mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] = \mathbb{E}_{y \sim \pi_\theta(\cdot|q)} \left[ \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid q, y_{<t})}{\pi_{\text{ref}}(y_t \mid q, y_{<t})} \right]
$$

在实践中，R1 使用一种 KL 的无偏估计器，通过下列近似避免在每一步都计算 $\pi_{\text{ref}}$：

$$
\mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] \approx \frac{\pi_{\text{ref}}(y_t \mid q, y_{<t})}{\pi_\theta(y_t \mid q, y_{<t})} - \log \frac{\pi_{\text{ref}}(y_t \mid q, y_{<t})}{\pi_\theta(y_t \mid q, y_{<t})} - 1
$$

该估计器恒为非负，且当 $\pi_\theta = \pi_{\text{ref}}$ 时为零。

> **GRPO 实战：组大小与稳定性**
>
> 在 R1 的训练中，每个问题采样 $G = 8$ 个回答。这是一个关键超参数：
>
> - 过小（$G=2$）：优势估计方差高，训练噪声大。
> - 过大（$G=32$）：计算成本线性增长，收益递减。
> - $G=8$：经验上在方差缩减与计算成本之间取得平衡。
>
> 分组采样还提供了天然的**课程学习信号**：随着训练推进，模型的平均奖励 $\mu_r$ 上升，方差 $\sigma_r$ 下降。所有 $G$ 个回答全对（或全错）的问题贡献零梯度，从而自然地将学习聚焦于模型能力前沿的问题上。

### 蒸馏：R1-Distill 系列

R1 的一项重要实践贡献在于证明：**推理能力可以通过在 R1 生成链上进行监督微调蒸馏到小得多的模型中**。R1-Distill 系列（1.5B、7B、8B、14B、32B、70B 参数）的训练方式为：

1. 使用 R1（671B）为一个大规模问题集生成长 CoT 解答
2. 过滤，仅保留正确的解答
3. 在这些解答上对较小的基础模型（Qwen2.5、Llama-3）进行微调

> **小模型：蒸馏 vs RL**
>
> 一个引人注目的发现：**在小模型上，蒸馏胜过从零开始的 RL 训练**。DeepSeek-R1-Distill-Qwen-7B 在 MATH 基准上的得分高于直接用 GRPO 训练的 7B 模型。这表明：
>
> - 小模型缺乏通过 RL 探索去发现推理策略的能力
> - 但它们*可以*学会模仿大模型发现的推理策略
> - 小模型的瓶颈是*探索*，而非*表示*

蒸馏路径引出了一个关于推理本质的重要问题：小模型究竟是真的在「推理」，还是只是在对推理链的表面形式进行模式匹配？经验上，蒸馏模型对新颖问题类型表现出一定的泛化能力，提示其确实在某种程度上内化了推理策略，而非纯粹的记忆。

## OpenAI o1/o3 系列

OpenAI 的 o1[openai2024o1]（2024 年 9 月发布）以及后续的 o3/o4-mini[openai2025o3] 模型代表了推理模型开发的商业前沿。尽管完整的技术细节仍属专有，但已公开的系统卡（system card）、技术报告和实证观察为理解其方法论提供了充足的线索。

### 带隐藏推理 Token 的思维链 RL

o1 最具标志性的架构选择是使用**隐藏推理 Token（hidden reasoning tokens）**：模型生成一段不向用户展示的内部思维链（称为「推理轨迹（reasoning trace）」或「思考 Token（thinking tokens）」）。最终只返回答案本身。这一设计具有若干意涵：

- **无格式约束**：隐藏推理可以使用任意格式，包括草稿纸式记法、伪代码，甚至非英文推理。
- **风格上无奖励作弊（reward hacking）**：由于用户从未看到推理过程，模型没有为了「好看」而牺牲「有用」的压力。
- **专有保护**：推理过程不被暴露，防止直接模仿。

训练流程被描述为「使用 RL 训练模型进行推理」，其 RL 目标作用于完整的（隐藏推理 + 最终答案）序列，奖励仅基于最终答案的质量。

### 过程奖励模型与结果奖励模型

OpenAI 的方法据信在结果奖励之外还使用了**过程奖励模型（Process Reward Model, PRM）**[lightman2023lets]，这与 DeepSeek-R1 仅采用结果奖励的方式形成对比。这一推断基于 OpenAI 已发表的 PRM 研究（PRM800K 数据集，"Let's Verify Step by Step"）以及 o1 系统卡中对推理链 RL 训练的描述，尽管 o1/o3 的精确训练配方并未公开披露。

**结果奖励模型（ORM）**

结果奖励模型（Outcome Reward Model, ORM）对完整回复 $(q, y)$ 打分：

$$
R_{\text{ORM}}(q, y) \in [0, 1]
$$

对于可验证任务（数学、代码），它简化为精确匹配验证；对于开放式任务，则使用一个学习得到的奖励模型。

**过程奖励模型（PRM）**

PRM 为推理链 $y = (s_1, s_2, \ldots, s_K)$ 中的每一个推理步骤 $s_k$ 分配奖励：

$$
R_{\text{PRM}}(q, y) = \sum_{k=1}^{K} \gamma^{K-k} \cdot r_k(q, s_1, \ldots, s_k)
$$

其中 $r_k \in [0,1]$ 是步骤级奖励，$\gamma \in (0,1]$ 是折扣因子。步骤级奖励 $r_k$ 估计部分解 $(s_1, \ldots, s_k)$ 能够导向正确最终答案的概率：

$$
r_k(q, s_1, \ldots, s_k) = P(\text{correct final answer} \mid q, s_1, \ldots, s_k)
$$

> **PRM 与 ORM：信用分配的权衡**
>
> **ORM** 提供干净、明确的奖励，但存在严重的信用分配问题：在一条 50 步推理链早期的单个错误步骤，所获得的零奖励与完全随机的回复并无区别。
>
> **PRM** 提供稠密奖励以直接处理信用分配，但也引入了新的挑战：
>
> - **训练数据**：步骤级标签需要人工标注或自动化生成（Math-Shepherd，见「过程奖励模型」一节）。
> - **奖励作弊**：模型可能学会产生在 PRM 眼中*看起来*正确、但实际上并不正确的步骤。
> - **分布漂移**：在某一推理链分布上训练的 PRM 可能无法泛化到 RL 产生的新颖推理链。
>
> 经验证据表明，PRM 对*搜索*（在候选解中选择）是有益的，但对*训练*的收益则不那么明确。

### 推理时计算扩展

o1 技术报告展示了一条清晰的扩展律：在困难推理任务上，**更多的思考 Token 单调地提升性能**。这通过一个控制隐藏推理 Token 数量上限的「思考预算（thinking budget）」参数得以实现。

设 $T$ 为思考 Token 预算。经验上观察到的扩展律近似为：

$$
\text{Pass@1}(T) \approx a - b \cdot T^{-c}
$$

其中 $a, b, c > 0$ 为常数，$a$ 表示渐近准确率上限，$c$ 刻画提升速率。在 AIME 2024 上，配备完整思考预算的 o1 达到约 83\% 的准确率，而未使用扩展思考的 GPT-4o 仅约 13\%。

### 训练计算量与测试时计算量

o1/o3 系列带来的一个根本性洞见是**计算等价原理（compute equivalence principle）**：训练计算量 $C_{\text{train}}$ 与测试时计算量 $C_{\text{test}}$ 之间存在一条权衡曲线，曲线上的各点能够达到相近的性能：

$$
\text{Performance}(C_{\text{train}}, C_{\text{test}}) = g\!\left(\alpha C_{\text{train}}^{p} + \beta C_{\text{test}}^{q}\right)
$$

经验上，对于推理任务 $p \approx q$，这意味着训练计算量与测试时计算量大致可互相替代。这对部署有深远影响：一个更小、更便宜但拥有扩展思考能力的模型，可以在困难问题上匹敌更大的模型，代价是更高的延迟。

### o3 与 o4-mini 的架构洞察

尽管 o3 和 o4-mini 的细节大多仍属专有，但已经浮现出一些观察：

- **o3**：思考预算显著大于 o1；在 ARC-AGI 上达到接近人类的表现（高算力下 87.5\%）。据信在推理时使用了更复杂的搜索策略。
- **o4-mini**：证明了经过 RL 训练推理能力的*更小*模型同样极具竞争力。在 AIME 2025 上配合扩展思考达到 93\%，表明在数学任务上模型规模的重要性不及推理能力。
- **工具调用（Tool Calling）**：o3/o4-mini 将工具调用（代码执行、网络搜索）集成到推理过程中，使模型能够以程序化方式验证中间步骤。

## QwQ 与 Qwen 推理模型

阿里巴巴 Qwen 团队开发了一系列推理模型（QwQ-32B[qwen2024qwq]、Qwen3[qwen2025qwen3]），与 DeepSeek-R1 一同代表了开源前沿。其方法在若干关键方面有所不同。

### 多阶段 RL 流水线

Qwen 的推理流水线采用了更精细的多阶段方法：

1. **基础预训练**：具备强大数学与编码能力的 Qwen2.5 基础模型
2. **多样推理任务上的 SFT**：在涵盖广泛推理任务（数学、代码、科学、逻辑）的混合数据上进行监督微调（Supervised Fine-Tuning, SFT）
3. **拒绝采样微调（Rejection Sampling Fine-Tuning, RFT）**：对每个问题生成 $N$ 个解，保留正确的解，再进行微调
4. **RL 第一阶段**：在数学和代码上使用可验证奖励的组相对策略优化（Group Relative Policy Optimization, GRPO）
5. **RL 第二阶段**：更广泛的 RL，包括指令遵循与安全

### 拒绝采样与 RL 的结合

Qwen 方法的一项关键创新是**拒绝采样与 RL 的迭代结合**：

1. **初始化**：以 SFT 模型作为 Policy $\pi_0$。
2. **拒绝采样（Rejection Sampling）**：采样 $N$ 个解 $\{y_i\}_{i=1}^N \sim \pi_{k-1}(\cdot \mid q)$，保留正确的解 $\mathcal{Y}^+(q) = \{y_i : r(y_i, y^*) = 1\}$。
3. **SFT 更新**：$\pi_k^{\text{SFT}} \leftarrow \text{SFT}(\pi_{k-1}, \bigcup_q \mathcal{Y}^+(q))$
4. **RL 更新**：$\pi_k \leftarrow \text{GRPO}(\pi_k^{\text{SFT}}, \mathcal{D})$
5. **重复**步骤 2--4 共 $K$ 次迭代，得到最终 Policy $\pi_K$。

拒绝采样步骤提供高质量的正样本来锚定 Policy，而 RL 则在当前分布之外进行探索。这种组合比纯 RL 更稳定，又比纯 SFT 更有能力。

### 工具集成推理

QwQ-32B 与 Qwen3 模型支持**工具集成推理（tool-integrated reasoning）**：模型可以在其推理链中调用外部工具（Python 解释器、搜索引擎、计算器）。这通过特殊 Token 实现：

```text
<thinking>
Let me solve this step by step.
First, I'll compute the eigenvalues of the matrix.

<tool_call>
{"name": "python", "arguments": {"code": "import numpy as np\nA = np.array([[2,1],[1,3]])\neigenvalues = np.linalg.eigvals(A)\nprint(eigenvalues)"}}
</tool_call>

<tool_response>
[1.38196601 3.61803399]
</tool_response>

The eigenvalues are approximately 1.382 and 3.618.
These are (5 +/- sqrt5)/2, which are the golden ratio and its conjugate...
</thinking>
<response>
<answer>The eigenvalues are (5 +/- sqrt5)/2</answer>
</response>
```

RL 训练奖励基于最终答案计算，但模型会学会策略性地使用工具，因为工具调用能提升得到正确答案的概率。

## 关键方法及其数学基础

### 用于推理的蒙特卡洛树搜索

蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）为将推理视作树搜索提供了一个有原则的框架。在 AlphaProof[alphaproof2024] 等相关系统中，MCTS 作用于推理步骤而非棋类走子。

**状态与动作空间**

- **状态** $s_k$：部分推理链 $(q, r_1, r_2, \ldots, r_k)$，其中 $r_i$ 为推理步骤
- **动作** $a$：下一个推理步骤（一句话或一段话）
- **终止状态**：包含最终答案的状态
- **奖励**：$R(s_{\text{terminal}}) = r_{\text{acc}}$（见准确率奖励公式）

**部分解的价值函数**

价值函数 $V(s_k)$ 估计从部分状态 $s_k$ 出发达到正确答案的概率：

$$
V(s_k) = P(\text{correct answer} \mid s_k) \approx \frac{1}{M} \sum_{m=1}^{M} R(\text{rollout}_m(s_k))
$$

其中 $\text{rollout}_m(s_k)$ 是从 $s_k$ 出发、使用当前 Policy 走到终止状态的一次蒙特卡洛 rollout。

**UCB 探索**

节点选择使用为推理场景改编的置信上界（Upper Confidence Bound, UCB）公式：

$$
\text{UCB}(s_k, a) = Q(s_k, a) + c_{\text{puct}} \cdot \pi_\theta(a \mid s_k) \cdot \frac{\sqrt{N(s_k)}}{1 + N(s_k, a)}
$$

其中：

- $Q(s_k, a) = \frac{1}{N(s_k,a)} \sum_{\text{visits}} V(s_{k+1})$ 为子状态的平均价值
- $\pi_\theta(a \mid s_k)$ 为 Policy 先验（语言模型对步骤 $a$ 的概率）
- $N(s_k)$ 为状态 $s_k$ 的访问次数
- $N(s_k, a)$ 为边 $(s_k, a)$ 的访问次数
- $c_{\text{puct}}$ 为探索常数

**MCTS 引导的训练**

MCTS 可被用于生成高质量训练数据：

$$
\mathcal{L}_{\text{MCTS}}(\theta) = -\sum_{k} \sum_{a} \pi_{\text{MCTS}}(a \mid s_k) \log \pi_\theta(a \mid s_k)
$$

其中 $\pi_{\text{MCTS}}(a \mid s_k) \propto N(s_k, a)^{1/\tau}$ 是 MCTS Policy（带温度 $\tau$ 的访问计数分布）。

### 过程奖励模型

**Math-Shepherd：自动化 PRM 训练**

Math-Shepherd[wang2024mathshepherd] 提出了一种无需人工步骤级标注、自动训练 PRM 的方法。其核心洞见是使用**基于结果的估计（outcome-based estimation）**：如果存在从 $s_k$ 出发的某一补全能够到达正确答案，则将步骤 $s_k$ 标注为正确。

形式化地，对于部分解 $(s_1, \ldots, s_k)$：

$$
\hat{r}_k = \mathbf{1}\!\left[\exists\, (s_{k+1}, \ldots, s_K) : \text{verify}(s_K, y^*) = 1\right]
$$

在实践中，这通过从 $s_k$ 采样 $M$ 个补全并检查是否有任何一个正确来估计：

$$
\hat{r}_k \approx \mathbf{1}\!\left[\sum_{m=1}^{M} \text{verify}(\text{complete}_m(s_k), y^*) > 0\right]
$$

随后用二元交叉熵训练 PRM：

$$
\mathcal{L}_{\text{PRM}}(\phi) = -\sum_{k=1}^{K} \left[ \hat{r}_k \log r_\phi(s_k) + (1-\hat{r}_k) \log(1 - r_\phi(s_k)) \right]
$$

**用 PRM 进行 Best-of-N 选择**

PRM 的一个主要应用是 **Best-of-N 选择**：生成 $N$ 个候选解，选择 PRM 分数最高的那一个：

$$
y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} R_{\text{PRM}}(q, y)
$$

这比（使用 ORM 的）多数投票更有效，因为 PRM 能够区分通过不同质量推理路径达到相同答案的解。

### 结果奖励模型与多数投票

**多数投票（自洽性，Self-Consistency）**

测试时扩展（Test-Time Scaling）最简单的形式是多数投票[wang2023selfconsistency]：生成 $N$ 个解并返回最常见的答案：

$$
y^* = \arg\max_{a} \sum_{i=1}^{N} \mathbf{1}[y_i = a]
$$

在每个解独立正确概率为 $p > 0.5$ 的假设下，多数投票正确的概率为：

$$
P(\text{majority correct}) = \sum_{k=\lceil N/2 \rceil}^{N} \binom{N}{k} p^k (1-p)^{N-k} \xrightarrow{N \to \infty} 1
$$

**用 ORM 加权的多数投票**

ORM 可以通过按置信度加权来改进多数投票：

$$
y^* = \arg\max_{a} \sum_{i=1}^{N} R_{\text{ORM}}(q, y_i) \cdot \mathbf{1}[y_i = a]
$$

### 用于推理的自博弈

自博弈（self-play）方法让模型同时扮演*生成器*和*验证器*两种角色，从而生成训练数据。

**STaR：自学推理者（Self-Taught Reasoner）**

STaR[zelikman2022star] 通过迭代方式自举推理能力：

1. 为问题集生成推理链
2. 保留导向正确答案的推理链（拒绝采样）
3. 在保留的推理链上微调
4. 用改进后的模型重复上述过程

其核心洞见是模型可以为正确答案进行*合理化（rationalize）*：即便它无法从零开始解决一个问题，给定答案后它仍能生成一条合理的推理链，这条推理链可作为训练数据。

**自博弈 RL**

在用于推理的自博弈 RL 中，模型同时生成问题与解答：

$$
\mathcal{L}_{\text{self-play}}(\theta) = \mathbb{E}_{q \sim \pi_\theta^{\text{gen}}} \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|q)} \left[ r(y, y^*) \right]
$$

其中 $\pi_\theta^{\text{gen}}$ 生成问题，$\pi_\theta^{\text{solve}}$ 解答它们。生成器因产生具有挑战性但可解的问题而获得奖励。

### 可验证奖励的强化学习（RLVR）

可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）[lambert2024tulu3] 是一个使用**真值验证（ground-truth verification）**作为奖励信号的框架，适用于任何可以自动检查正确性的领域。

**可验证领域**

- **数学**：通过 SymPy、Lean 或 Isabelle 进行符号验证
- **代码**：单元测试执行
- **形式逻辑**：证明检查
- **事实问答**：数据库查找
- **游戏**：胜负结果

**RLVR 目标**

$$
\mathcal{L}_{\text{RLVR}}(\theta) = -\mathbb{E}_{(q, y^*) \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_\theta(\cdot|q)} \left[ \text{verify}(y, y^*) \right] + \beta \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right]
$$

RLVR 相对于基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）的关键优势在于**不存在奖励模型误差**：由于奖励由确定性验证器而非学习得到的模型计算，因此不存在针对有缺陷奖励模型的奖励作弊。唯一的失败模式是模型找到了能通过验证、但并非真正正确的解（例如，利用代码评估中测试用例的弱点）。

> **用于代码的 RLVR：奖励作弊挑战**
>
> 在代码生成中，验证器是一个测试套件。用 RLVR 训练的模型可能学会：
>
> - **硬编码测试输出**：对每个测试输入返回预期输出，而不实现真正的算法
> - **利用薄弱测试**：通过所有提供的测试，但在边界情形下失败
>
> 缓解措施包括：使用大规模、多样化的测试套件；引入对抗性测试用例；使用基于执行的奖励来惩罚硬编码（例如，检查解是否在 $O(n \log n)$ 时间内运行）。

### 旅程学习（Journey Learning）

旅程学习（Journey Learning）[qin2024o1journey] 主张在**完整推理轨迹**上训练，包括失败的尝试和纠正，而不仅仅是成功的最终解。

**动机**

标准的拒绝采样会丢弃失败尝试。但失败尝试包含有价值的信息：

- 哪些方法行不通（负样本）
- 如何识别错误并从中恢复（纠正模式）
- 问题空间的结构（探索数据）

**Journey Learning 目标**

给定一条可能包含回溯的轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$：

$$
\mathcal{L}_{\text{journey}}(\theta) = -\sum_{t=0}^{T} w_t \log \pi_\theta(a_t \mid s_t)
$$

其中权重 $w_t$ 被设计为强调：

- 最终导向成功的步骤（$w_t > 1$）
- 错误之后的纠正步骤（$w_t > 1$）
- 失败分支中的步骤（$w_t < 1$，但 $> 0$）

### Quiet-STaR：在每个 Token 上推理

Quiet-STaR[zelikman2024quietstar] 将推理范式扩展到*每一个 Token 位置*：模型不仅在最终答案之前生成一条推理链，而是在每个 Token 位置都生成一段「思考（thought）」。

**形式化**

对于每个 Token 位置 $t$，模型在预测下一个 Token $x_{t+1}$ 之前先生成一段隐藏思考 $z_t$：

$$
P(x_{t+1} \mid x_{\leq t}) = \mathbb{E}_{z_t \sim \pi_\theta(\cdot | x_{\leq t})} \left[ \pi_\theta(x_{t+1} \mid x_{\leq t}, z_t) \right]
$$

在实践中，这通过混合带思考与不带思考的预测来近似：

$$
P(x_{t+1} \mid x_{\leq t}) = \alpha \cdot \pi_\theta(x_{t+1} \mid x_{\leq t}, z_t) + (1-\alpha) \cdot \pi_\theta(x_{t+1} \mid x_{\leq t})
$$

**使用 REINFORCE 训练**

由于思考 $z_t$ 是离散潜变量，梯度使用 REINFORCE 估计：

$$
\nabla_\theta \mathcal{L}_{\text{QS}} = \mathbb{E}_{z_t} \left[ \nabla_\theta \log \pi_\theta(z_t \mid x_{\leq t}) \cdot \left( \log P(x_{t+1} \mid x_{\leq t}, z_t) - b_t \right) \right]
$$

其中 $b_t$ 为基线（例如无思考预测 $\log \pi_\theta(x_{t+1} \mid x_{\leq t})$）。

> **Quiet-STaR 的计算代价**
>
> Quiet-STaR 将推理成本提升为 $L_z + 1$ 倍（$L_z$ 为思考长度），且作用于*每一个* Token 位置。对于长度为 $T$ 的序列，若思考长度 $L_z = 8$，计算量增加 $9\times$。这使得 Quiet-STaR 在长序列上若无重大工程优化（如对思考的投机解码、缓存等）便不可行。

## 推理的扩展律

近期工作[snell2024scaling, wu2024empirical] 已经证实测试时计算量与推理性能之间存在可预测的扩展关系，将经典扩展律[kaplan2020scaling] 延伸到了推理阶段。

### 训练计算量与测试时计算量的权衡

推理模型的根本性扩展问题是：**给定固定的总计算预算 $C_{\text{total}} = C_{\text{train}} + N \cdot C_{\text{test}}$（其中 $N$ 为查询数量），应如何分配计算？**

设 $\mathcal{A}(C_{\text{train}}, C_{\text{test}})$ 表示用 $C_{\text{train}}$ FLOPs 训练、每个查询给定 $C_{\text{test}}$ 推理 FLOPs 的模型准确率。经验上：

$$
\mathcal{A}(C_{\text{train}}, C_{\text{test}}) \approx 1 - \exp\!\left(-a \cdot C_{\text{train}}^{\alpha} \cdot C_{\text{test}}^{\beta}\right)
$$

其中 $a, \alpha, \beta > 0$ 为常数。对于固定总预算 $C_{\text{total}}$，最优分配满足训练与推理之间每 FLOP 的边际回报相等的条件：

$$
\frac{\partial \mathcal{A}}{\partial C_{\text{train}}} = \frac{1}{N} \cdot \frac{\partial \mathcal{A}}{\partial C_{\text{test}}}
$$

直觉上：一个 FLOP 的训练计算惠及全部 $N$ 个查询，而一个 FLOP 的测试时计算只惠及单个查询。在最优点，测试时计算的单查询边际价值是训练计算的 $N$ 倍（因为训练成本被摊销）。将其应用于推理扩展律公式可得到最优训练计算占比：

$$
\frac{C_{\text{train}}^*}{C_{\text{total}}} = \frac{\alpha}{\alpha + \beta}
$$

对于特定预算结构 $C_{\text{total}} = C_{\text{train}} + N \cdot C_{\text{test}}$，在乘性准确率模型下该占比与 $N$ 无关。然而在实际中 $\alpha$ 和 $\beta$ 是问题相关的：对于高流量部署（大 $N$），即便基础模型的微小提升也会占主导，倾向于训练投入；对于低流量、高价值的查询（小 $N$），测试时计算则更具成本效益。

### 何时投资于更长推理链 vs 更好的基础模型

> **推理链长度与模型容量**
>
> 对于容量为 $C$ 的模型在难度为 $D$ 的问题上，最优推理链长度 $L^*$ 满足：
>
> $$
> L^* \propto \frac{D}{C^{\gamma}}
> $$
>
> 其中 $\gamma > 0$。这意味着：
>
> - **困难问题**无论模型规模如何都需要更长的推理链
> - 在相同问题难度下，**更大的模型**需要更短的推理链
> - **收益递减**：超过 $L^*$ 后，额外的 Token 不再带来收益，反而可能有害（过度思考）

「过度思考（overthinking）」现象——拥有极长推理链的模型反而比拥有适中推理链的模型表现*更差*——已被经验观察到，归因于：

- 长推理链中错误的累积（误差传播）
- 偏离主解路径的干扰
- 对错误中间结论的过度自信

### 最优 Token 预算分配

对于给定 Token 预算 $B$ 的模型，「思考」Token $T_{\text{think}}$ 与「作答」Token $T_{\text{answer}}$ 之间的分配应满足：

$$
T_{\text{think}}^* = \arg\max_{T} \mathcal{A}(T, B - T)
$$

经验上，最优划分是问题相关的：

- **简单问题**：$T_{\text{think}}^* / B \approx 0.3$（30\% 思考）
- **困难问题**：$T_{\text{think}}^* / B \approx 0.8$（80\% 思考）
- **极困难问题**：$T_{\text{think}}^* / B \approx 0.95$（95\% 思考，极简作答）

这促成了**自适应思考预算（adaptive thinking budgets）**：为更难的问题分配更多 Token，难度可由模型在初次解题尝试中的不确定度来估计。

## 推理模型比较

| 方法 | PRM | ORM | MCTS | 蒸馏 | 工具 | 开源 |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI o1/o3 | ✓ | ✓ | 未知 | -- | ✓ | $\times$ |
| DeepSeek-R1 | $\times$ | ✓ | $\times$ | ✓ | $\times$ | ✓ |
| QwQ / Qwen3 | 部分 | ✓ | $\times$ | $\times$ | ✓ | ✓ |
| AlphaProof | ✓ | ✓ | ✓ | -- | ✓ | $\times$ |
| Math-Shepherd | ✓ | ✓ | $\times$ | -- | $\times$ | ✓ |
| STaR / Quiet-STaR | $\times$ | ✓ | $\times$ | -- | $\times$ | ✓ |

## 总结与开放问题

RL 用于推理模型的领域进展极为迅速。已经浮现出若干关键经验：

1. **可验证奖励已经足够**：对于具备真值验证的领域（数学、代码），仅靠结果奖励就足以让 RL 发现复杂的推理策略，无需过程奖励模型。
2. **测试时计算是一个新维度**：推理模型引入了新的扩展维度——推理计算——在困难推理任务上它与训练计算大致可互相替代。
3. **蒸馏极为有效**：大型推理模型可以通过对生成推理链进行监督微调，将能力迁移到小得多的模型上，往往优于对小模型直接进行 RL 训练。
4. **涌现的元认知**：在推理任务上的 RL 训练会涌现出未被显式训练的自我纠正与验证行为。

> **RL 用于推理中的开放问题**
>
> 若干根本性问题仍待回答：
>
> - **泛化**：在数学/代码上训练得到的推理能力，能否迁移到其他领域（科学推理、规划、社会推理）？
> - **忠实性（Faithfulness）**：生成的推理链对最终答案是因果性负责的，还是事后合理化？
> - **最优搜索**：推理时的最优搜索策略是什么——束搜索（beam search）、MCTS，还是其他？
> - **奖励设计**：对于没有真值验证器的领域，如何为推理设计可靠的奖励信号？
> - **过度思考**：模型如何学会分配*恰当*的思考量——既不过少也不过多？
> - **组合式推理**：经过 RL 训练的推理模型能否解决需要组合多种不同推理技能的问题？

推理模型的发展代表了一次范式转变：从*知道*事情的语言模型，转向能够*推演*事情的语言模型。本节所描述的 RL 方法是驱动这一转变的主要引擎，它们的持续发展很可能成为未来数年 AI 研究的中心焦点。
