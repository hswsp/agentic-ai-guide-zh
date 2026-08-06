---
layout: home
title: 强化学习导论
permalink: /part1/ch03-rl-intro.html
---

# 强化学习导论

强化学习（Reinforcement Learning，RL）是一种范式，其中**智能体（agent）**通过与**环境（environment）**交互、接收**奖励（rewards）**作为反馈，并优化其**策略（policy）**以最大化长期累积奖励来学习做出序贯决策[sutton2018reinforcement]。与监督学习（需要带标签的输入-输出对）不同，RL 通过*试错（trial and error）*来发现最优行为。

![强化学习概览：智能体与环境交互，接收奖励作为反馈，并通过试错更新其策略。与从带标签数据对学习的监督学习不同，RL 通过经验最大化奖励来学习应当做什么。](/figures/fig_020_fig20.png)

## 马尔可夫决策过程（Markov Decision Process，MDP）

MDP 是一个五元组 $(S, A, P, R, \gamma)$：

- $S$：状态空间——环境所有可能的配置
- $A$：动作空间——智能体可用的所有动作
- $P(s'|s, a)$：转移函数——在状态 $s$ 下采取动作 $a$ 后到达状态 $s'$ 的概率
- $R(s, a, s')$：奖励函数——一次状态转移获得的即时标量反馈
- $\gamma \in [0, 1]$：折扣因子——未来奖励相对于即时奖励的权重

**马尔可夫性质（Markov Property）**：未来只依赖于当前状态，而不依赖于历史：

$$
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)
$$

这一性质使问题变得可处理。

> **直觉：智能体-环境交互循环**
>
> 在每个时间步 $t$：
>
> 1. 智能体观察状态 $s_t$
> 2. 智能体根据策略 $\pi(a|s)$ 选择动作 $a_t$
> 3. 环境转移到 $s_{t+1} \sim P(\cdot|s_t, a_t)$
> 4. 智能体接收奖励 $r_t = R(s_t, a_t, s_{t+1})$
> 5. 重复直到终止状态或时间步上限 $T$

## 核心概念与定义

**策略（Policy）** $\pi(a|s)$：从状态到动作概率的映射。确定性策略：$a = \pi(s)$；随机性策略：$a \sim \pi(\cdot|s)$。

**回报（Return）**（累积折扣奖励）：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

**价值函数（Value Function）**（在策略 $\pi$ 下从状态 $s$ 出发的期望回报）：

$$
V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s\right]
$$

**动作价值函数（Action-Value Function）**（从状态 $s$ 出发、采取动作 $a$、其后遵循 $\pi$ 的期望回报）：

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]
$$

**优势函数（Advantage Function）**（动作 $a$ 相对于平均水平好多少）：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**贝尔曼方程（Bellman Equations）**（递归关系）：

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]
$$

$$
Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')\right]
$$

> **关键：最优策略与贝尔曼最优性**
>
> 最优策略 $\pi^*$ 满足：
>
> $$
> V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^*(s')\right]
> $$
>
> $$
> Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right]
> $$
>
> 一旦得到 $Q^*$，最优策略就直接是：$\pi^*(s) = \arg\max_a Q^*(s,a)$。

## 强化学习方法的分类

强化学习算法可以从多个维度进行分类。理解这一分类体系有助于针对特定问题选择合适的方法。

![强化学习方法的分类。](/figures/fig_021_fig21.png)

> **关键：关键分类区分**
>
> **无模型（Model-Free） vs 基于模型（Model-Based）**：
>
> - **Model-Free**：直接从交互经验中学习策略或价值函数；无需建模环境的转移机制。对 LLM 最实用（语言的转移机制难以建模）。
> - **Model-Based**：学习或使用环境转移模型 $P(s'|s,a)$。可以提前规划。样本效率更高，但需要准确的模型。
>
> **基于价值（Value-Based） vs 基于策略（Policy-Based）**：
>
> - **Value-Based**：学习 $Q(s,a)$ 或 $V(s)$，并通过 $\arg\max_a Q(s,a)$ 得到策略。适合离散、小规模动作空间（如 Atari），在连续或大规模动作空间上表现不佳。
> - **Policy-Based**：直接参数化并优化 $\pi_\theta(a|s)$。天然适合连续或高维动作空间，对 LLM 至关重要（词表 = 32K--128K 个动作）。
> - **Actor-Critic**：两者结合——策略（actor）提议动作，价值函数（critic）评估动作。用于 LLM 的 PPO 即属于 actor-critic 方法。
>
> **同策略（On-Policy） vs 异策略（Off-Policy）**：
>
> - **On-Policy**：从*当前*策略生成的数据中学习。每次更新后必须重新生成数据。例如：REINFORCE、PPO、A2C。更稳定但样本效率较低。
> - **Off-Policy**：从*任意*策略生成的数据中学习（包括旧版本或其他智能体）。可以复用过往经验。例如：Q-Learning、DQN、SAC。样本效率高但更难稳定。

## 时序差分（Temporal Difference，TD）学习

TD 学习[sutton1988learning]采用自举（bootstrap）思想——它使用其他价值估计来更新价值估计，无需等待完整 episode 结束。

### 理解 TD 误差：以「惊讶」作为学习信号

**TD 误差（TD Error）**衡量智能体对未来奖励的**当前估计**与采取一步后**新估计**之间的差异。简单来说，就是智能体*原本以为*会发生的事，与*实际*发生的事加上对未来的新预期之间的差。它代表了智能体的「惊讶」。

> **示例：直觉——开车类比**
>
> 设想你正在开车回家，预计需要 30 分钟。
>
> - **预测**：总共 30 分钟。
> - **现实变化**：10 分钟后，你遇到意外的道路施工。GPS 更新提示，你还需要 35 分钟。
> - **TD 误差**：总预期时间现在是 45 分钟（已过 10 分钟 + 剩余 35 分钟）。新估计（45 分钟）与旧估计（30 分钟）之差就是 **+15 分钟的 TD 误差**。你下次会用这个「惊讶」来调整路线。
>
> **正的 TD 误差**意味着结果好于预期 $\rightarrow$ 提升该状态的价值。
>
> **负的 TD 误差**意味着结果差于预期 $\rightarrow$ 降低该状态的价值。

### TD 误差公式

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

- $R_{t+1}$：采取动作后获得的**即时奖励**。
- $\gamma V(S_{t+1})$：下一个状态的估计**折扣价值**（智能体从下一个状态起预期获得的回报，按折扣因子 $\gamma$ 缩放）。
- $V(S_t)$：当前状态价值的**原始估计**。

组合项 $(R_{t+1} + \gamma V(S_{t+1}))$ 被称为 **TD 目标（TD Target）**。因此：

$$
\text{TD Error} = \text{TD Target} - \text{Old Estimate}
$$

### 智能体如何使用 TD 误差

智能体调整其价值函数以将 TD 误差驱向零：

$$
V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t
$$

- 若 $\delta_t > 0$：结果好于预测 $\rightarrow$ 增加 $V(S_t)$，使智能体倾向追求该状态。
- 若 $\delta_t < 0$：结果差于预测 $\rightarrow$ 降低 $V(S_t)$，使智能体避开该状态。
- 若 $\delta_t = 0$：预测完美 $\rightarrow$ 无需更新（已收敛）。

> **直觉：TD 与 Monte Carlo 的比较**
>
> **Monte Carlo**：等到 episode 结束，使用实际回报 $G_t$。无偏但方差高（单条完整轨迹可能不具代表性）。
>
> **TD**：每一步使用估计的未来价值 $\gamma V(s_{t+1})$ 进行更新。有偏（依赖 $V$ 的准确性），但方差低得多（单步更新，不会复合噪声）。
>
> **TD($\lambda$)**：在 TD(0) 与 Monte Carlo 之间插值。$\lambda=0$：纯 TD；$\lambda=1$：纯 MC。这正是 GAE 在 PPO 中所做的事情（取 $\lambda=0.95$）。

**TD 目标**：$y_t = r_t + \gamma V(s_{t+1})$——我们要靠近的「更好的估计」。

**多步 TD**（n 步回报）：

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

## Q-Learning

Q-Learning[watkins1989learning]是基础性的**异策略、基于价值**的算法。它直接学习最优 $Q^*$，与所遵循的策略无关。

**更新规则**：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]
$$

> **直觉：为什么 Q-Learning 是异策略的**
>
> 更新使用 $\max_{a'} Q(s_{t+1}, a')$——下一状态下*最优*动作的价值，与智能体实际采取的动作无关。这意味着目标始终在最优策略下计算，即便行为策略以随机方式探索（$\epsilon$-贪心）。
>
> 这就是 Q-Learning 可以从重放缓冲区（Replay Buffer）、演示数据或任何经验来源中学习的原因。数据无需来自当前策略。

**SARSA**[rummery1994online]（同策略替代方案）：使用*实际采取的*动作而非最大值：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]
$$

**深度 Q 网络（Deep Q-Networks，DQN）**[mnih2015human]：用神经网络 $Q_\theta(s,a)$ 取代表格化的 $Q(s,a)$。关键创新包括：经验重放缓冲区（异策略数据复用）、目标网络（稳定性）、$\epsilon$-贪心探索。

**DQN 损失函数**：网络通过最小化从重放缓冲区采样的 mini-batch 上的均方 TD 误差来训练：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\!\left[\left(r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_\theta(s, a)\right)^2\right]
$$

其中 $Q_{\bar{\theta}}$ 是**目标网络（target network）**——$Q_\theta$ 的冻结副本，每 $C$ 步才更新一次（如 $C = 10{,}000$）。这避免了「移动目标」问题：没有它的话，预测与目标会同时变化，导致发散。

**梯度更新**：对损失关于 $\theta$ 求梯度（注意：目标 $y$ 被视为常量——梯度不流经 $\bar{\theta}$）：

$$
\nabla_\theta \mathcal{L} = -\mathbb{E}\!\left[\underbrace{\left(r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_\theta(s, a)\right)}_{\text{TD error } \delta}\; \nabla_\theta Q_\theta(s, a)\right]
$$

$$
\theta \leftarrow \theta - \alpha \cdot \delta \cdot \nabla_\theta Q_\theta(s, a)
$$

**学习流程**（每个训练步）：

1. **行动**：通过 $\epsilon$-贪心选择动作：以概率 $\epsilon$ 选择随机动作，否则 $a = \arg\max_a Q_\theta(s, a)$。在前 100 万步内将 $\epsilon$ 从 1.0 衰减到 0.01。
2. **存储**：将转移 $(s, a, r, s', d)$ 存入重放缓冲区 $\mathcal{B}$（容量约 100 万）。
3. **采样**：从 $\mathcal{B}$ 中均匀采样 32 条转移作为 mini-batch。
4. **计算目标**：$y = r + \gamma(1 - d)\max_{a'} Q_{\bar{\theta}}(s', a')$（若为终止状态则未来价值为零）。
5. **更新**：对 $(y - Q_\theta(s,a))^2$ 进行梯度下降。将梯度裁剪到 $[-1, 1]$（Huber 损失变体）。
6. **同步目标网络**：每 $C$ 步执行 $\bar{\theta} \leftarrow \theta$。

### 理解重放缓冲区

**重放缓冲区（Replay Buffer）**[lin1992self]（经验回放）是一种数据存储机制，保存过往经验以便智能体后续再学习。智能体不会在执行动作后立刻丢弃数据，而是将转移存入记忆库，并随机采样 mini-batch 用于训练。

**存储内容**：每条转移是一个元组：

$$
e_t = (s_t, a_t, r_t, s_{t+1}, d_t)
$$

其中 $d_t$ 是布尔标志，表示 episode 是否结束。

> **关键：为何重放缓冲区不可或缺**
>
> - **打破数据相关性**：连续步之间高度相关。神经网络在序列数据上泛化较差。从缓冲区随机采样可使训练样本近似独立同分布（i.i.d.）。
> - **防止灾难性遗忘**：没有缓冲区，智能体在通过某个困难关卡后，若在后续 10K 步中持续在更后面的关卡上失败，可能会忘记如何通过先前那关。缓冲区可确保它持续练习旧的场景。
> - **提升样本效率**：运行环境可能很慢。重放缓冲区允许同一条转移用于多次权重更新，从每一步中榨取更多价值。

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 有界队列

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 通过随机选取经验打破相关性
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```

> **直觉：优先级经验回放（Prioritized Experience Replay，PER）**
>
> 在标准缓冲区中，所有经验的采样概率相等。但有些经验信息量大得多。**PER**[schaul2016prioritized]按 **TD 误差幅度**对采样概率进行加权——如果某条转移引发了巨大「惊讶」（$|\delta_t|$ 高），智能体就更频繁地采样它，从而更快修正模型。在 Atari 基准上可将学习速度加快 2--3 倍。

> **警告：为什么 Q-Learning 对 LLM 不适用**
>
> 语言生成中的动作空间是整个词表（$|A| = 32\text{K}$--$128\text{K}$），状态空间是所有可能的 token 序列（无限）。在每个 token 位置对 128K 个动作计算 $\max_a Q(s,a)$ 是不可行的。这就是 LLM RL 使用**基于策略**方法（PPO、GRPO）的原因。

## 策略梯度方法——REINFORCE

与其学习价值函数再推导策略，不如直接优化策略参数 $\theta$ 以最大化期望回报[williams1992simple]。

**目标**：$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T r_t\right]$

**策略梯度定理（Policy Gradient Theorem）**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

> **关键：策略梯度定理——形式化推导（5 步）**
>
> **第 1 步**：定义目标。我们希望最大化期望回报：
>
> $$
> J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T r_t\right] = \sum_\tau P(\tau|\theta) R(\tau)
> $$
>
> 其中 $P(\tau|\theta) = p(s_0)\prod_{t=0}^T \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t, a_t)$ 是轨迹概率。
>
> **第 2 步**：求梯度。只有 $\pi_\theta$ 项依赖于 $\theta$（状态转移 $p$ 与 $\theta$ 无关）：
>
> $$
> \nabla_\theta J = \sum_\tau \nabla_\theta P(\tau|\theta)\, R(\tau)
> $$
>
> **第 3 步**：应用**对数求导技巧（log-derivative trick）**：$\nabla_\theta P(\tau|\theta) = P(\tau|\theta)\, \nabla_\theta \log P(\tau|\theta)$：
>
> $$
> \nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta \log P(\tau|\theta)\, R(\tau)\right]
> $$
>
> **第 4 步**：展开 $\log P(\tau|\theta)$。$\log p(s_0)$ 与 $\log p(s_{t+1}|s_t,a_t)$ 在 $\nabla_\theta$ 下消失：
>
> $$
> \nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)
> $$
>
> **第 5 步**：合并。未来奖励不依赖于过去动作（因果性），因此每个 $\nabla\log\pi$ 仅与未来回报 $G_t = \sum_{t'=t}^T r_{t'}$ 配对：
>
> $$
> \nabla_\theta J = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
> $$

> **直觉：这一结果为何优美**
>
> 该梯度**不需要对环境的状态转移** $p(s'|s,a)$ **求导**。对数求导技巧将其转化为一个期望，只需*运行策略并观察奖励*就能估计。把 $G_t$ 替换为优势 $\hat{A}_t = G_t - V(s_t)$ 可以在不引入偏差的前提下降低方差（因为对任何仅依赖状态的基线 $b(s)$，都有 $\mathbb{E}[\nabla\log\pi \cdot b(s)] = 0$）。

**REINFORCE 算法（REINFORCE）**[williams1992simple]（Williams, 1992）：

1. 在 $\pi_\theta$ 下采样完整轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
2. 为每个时间步计算回报 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$
3. 更新：$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$

> **直觉：REINFORCE 直觉——「奖励加权的最大似然」**
>
> $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是提升动作 $a_t$ 概率的方向。乘以 $G_t$ 意味着：
>
> - 高奖励轨迹：提升所有所采取动作的概率（$G_t$ 为正）
> - 低奖励轨迹：降低所采取动作的概率（扣除基线后 $G_t$ 为负）
>
> 这是一种监督学习，「标签」是你采取的动作，并按结果好坏加权。

**基线带来的方差缩减**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]
$$

任何与 $a_t$ 无关的基线 $b(s_t)$ 都能在保持梯度无偏的同时降低方差。最佳选择：$b(s_t) = V^\pi(s_t)$。此时 $G_t - V(s_t) \approx A^\pi(s_t, a_t)$ = 优势。

> **警告：REINFORCE 的局限**
>
> - **方差高**：每次梯度仅使用一条轨迹，需要数千样本才能保证更新稳定。
> - **无自举**：必须等待完整 episode（没有部分回报信号）。
> - **样本效率低**：数据使用一次即丢弃（同策略）。
> - **缺乏步长控制**：策略更新步可能灾难性地过大。
>
> 这些局限催生了如下演进路径：REINFORCE $\to$ Actor-Critic $\to$ TRPO $\to$ **PPO**。

## Actor-Critic 方法

将策略梯度（actor）与学习得到的价值函数（critic）结合，以在保持策略优化灵活性的同时降低方差。

**结构**：

- **Actor** $\pi_\theta(a|s)$：策略，提议动作。
- **Critic** $V_\phi(s)$ 或 $Q_\phi(s,a)$：评估状态/动作的好坏，提供低方差基线。

**Actor 更新**（使用 critic 提供的优势）：

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right], \quad \hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

**Critic 更新**（最小化 TD 误差）：

$$
\mathcal{L}_\text{critic} = \mathbb{E}\left[(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2\right]
$$

> **关键：面向 LLM 的 PPO 演进路径**
>
> 1. **REINFORCE**[williams1992simple]：高方差、无自举 $\rightarrow$ 对 LLM 不可行
> 2. **A2C/A3C**[mnih2016asynchronous]（Advantage Actor-Critic）：使用基于 TD 的优势，方差更低，但步长无界。
> 3. **TRPO**[schulman2015trust]：约束策略更新前后的 KL 散度。稳定但代价高（二阶方法）。
> 4. **PPO**[schulman2017proximal]：通过裁剪策略比率，仅用一阶优化就达到与 TRPO 类似的稳定性。是 LLM RL 训练的标准方法。
> 5. **GRPO**：完全去掉 critic，使用组内统计量作为基线。更简单，对可验证奖励有效。

## 广义优势估计（Generalized Advantage Estimation，GAE）

**动机**：Actor-Critic 框架需要对优势 $A(s,a) = Q(s,a) - V(s)$ 进行良好估计——这个动作比平均水平好多少？但这里存在根本性的张力：

- **1 步 TD 优势**（$r_t + \gamma V(s_{t+1}) - V(s_t)$）：方差低（仅含一步随机性），但**有偏**——若价值函数 $V$ 不准确，优势估计就会系统性偏离。
- **Monte Carlo 优势**（$G_t - V(s_t)$）：无偏（使用实际回报），但**方差高**——许多随机奖励之和在不同 episode 间波动剧烈。

GAE[schulman2016high]（Schulman 等，2016）通过单一参数 $\lambda \in [0, 1]$ 在两种极端之间提供**平滑插值**。它对所有 $n$ 的 $n$ 步优势估计进行指数加权平均，给出一种在偏差与方差之间权衡的原则化方式。

**核心思想**：在每个时间步计算 1 步 TD 误差 $\delta_t$，然后用指数衰减权重 $(\gamma\lambda)^l$ 进行混合——近期 TD 误差权重完整，远期的被降权：

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

![GAE 数据流：每个 TD 残差 $\delta_{t+l}^V$ 在求和前被乘以权重 $(\gamma\lambda)^l$。$\lambda$ 越大，纳入的未来残差越多（偏差更低，方差更高）。](/figures/fig_022_fig22.png)

> **直觉：$\lambda$ 控制什么——偏差-方差权衡**
>
> - $\lambda = 0$：$\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。完全信任价值函数。方差低，但若 $V$ 不准确则有偏。
> - $\lambda = 1$：$\hat{A}_t = \sum_l \gamma^l r_{t+l} - V(s_t)$。完整 Monte Carlo 回报减去基线。无偏但方差极高。
> - $\lambda = 0.95$（标准取值）：黄金平衡点。主要信任 $V$，但在远期效应上用实际回报修正。之所以可行，是因为价值头在初始训练后会变得准确。
>
> 针对 LLM 的常用取值：$\gamma = 1.0$（无时间折扣——单轮对话中所有 token 权重相等），$\lambda = 0.95$。

### GAE 中偏差与方差的直观映射

在监督学习中，偏差与方差源于模型的结构性假设。而在通过 GAE 进行的强化学习中，它们源于**你信任有缺陷的模型有多深 vs 你信任混乱的环境有多深**：

- **偏差（系统性偏离）**：当估计器依赖于价值网络 $V_\theta$ 的结构性假设与不完美预测时产生。如果 $\theta$ 训练不足或容量不够，基线猜测就会系统性出错。
- **方差（样本抖动）**：当估计器依赖于长且无约束的环境轨迹时产生。随机转移、随机种子与策略执行噪声在长时间步上累积，导致不同 rollout 之间的实际样本奖励剧烈波动。

### 架构光谱：边界情形分析

![GAE 中的偏差 vs 方差：$\lambda$ 控制这一权衡。较小的 $\lambda$（左）通过自举得到高偏差/低方差；较大的 $\lambda$（右）使用完整 Monte Carlo 回报得到低偏差/高方差。最优选择（$\lambda \in [0.9, 0.95]$）在稳定训练与准确的长程信用分配之间取得平衡。](/figures/fig_023_fig23.png)

超参数 $\lambda$ 充当了两种基本估计范式之间的「滑尺」。

> **关键：高偏差/低方差的限制（$\lambda = 0$）**
>
> $$
> \hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
> $$
>
> - **行为**：优势在很大程度上由参数 $\theta$ 的当前状态决定。
> - **直觉**：高度**有偏**，因为网络在 1 步窗口内为自己打分；若 $V_\theta$ 不准确，梯度步就会被破坏。**方差低**，因为它忽略 $t+1$ 步之后的未来随机事件，从而产生平滑、稳定的参数更新。
> - **风险**：策略陷入次优局部极小——永远发现不了复杂的延迟奖励序列。

> **关键：低偏差/高方差的限制（$\lambda = 1$）**
>
> 当 $\lambda = 1$ 时，中间的价值项相互裂项相消，GAE 退化为 Monte Carlo 回报减去基线：
>
> $$
> \hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V_\theta(s_t)
> $$
>
> - **行为**：舍弃自举式前瞻，将整个 episode 的实际现实进行累加。
> - **直觉**：对真实的环境转移完全**无偏**——衡量的是实际奖励而非神经网络近似。然而表现出极端的**高方差**：episode 早期的微小扰动可能导致完全不同的总回报，使策略更新变得不稳定。
> - **风险**：破坏性梯度更新；训练爆炸。

### 权衡矩阵

通过选择 $\lambda \in [0.95, 0.99]$，GAE 可以最小化优势估计的总均方误差：

**GAE 参数选择的操作性对比。**

| 配置 | 统计特性 | 核心依赖 | 实际风险 |
| --- | --- | --- | --- |
| $\lambda = 0$ | 高偏差、低方差 | 模型参数（$\theta$） | 策略陷入次优局部极小 |
| $\lambda \in [0.95, 0.99]$ | 均衡（最优 MSE） | 混合融合 | 需要根据环境随机性进行调参 |
| $\lambda = 1$ | 低偏差、高方差 | 经验性环境 rollout | 破坏性梯度更新；训练爆炸 |

### $\lambda$ 调参诊断

监控训练曲线能直接揭示当前主导因素是偏差还是方差：

1. **高方差指标**：策略熵骤降，同时价值函数的解释方差变得高度为负或剧烈波动 $\rightarrow$ 策略更新噪声大。**对策**：降低 $\lambda$ 以平滑目标更新。
2. **高偏差指标**：智能体在早期获得稳定训练，但完全无法发现复杂的延迟奖励序列 $\rightarrow$ 由于自举导致对长程依赖的低估。**对策**：将 $\lambda$ 提高至接近 $1.0$，让策略接触真实的下游轨迹信号。

## 同策略 vs 异策略——详细对比

|  | On-Policy | Off-Policy |
| --- | --- | --- |
| **数据来源** | 仅当前策略 $\pi_\theta$ | 任意策略（重放缓冲区） |
| **更新之后** | 旧数据失效，必须重新生成 | 旧数据仍可使用 |
| **样本效率** | 低（数据仅用一次） | 高（数据可多次复用） |
| **稳定性** | 更稳定（分布一致） | 可能发散（分布不匹配） |
| **示例** | REINFORCE、PPO、A2C、GRPO | Q-Learning、DQN、SAC、DPO |
| **用于 LLM** | PPO、GRPO（每步重新生成） | DPO（静态偏好数据集） |

> **直觉：RLHF 方法的同/异策略属性**
>
> **PPO/GRPO 是同策略**：用当前策略生成回复，计算优势，更新，丢弃数据，再次生成。这就是为何生成占据 60% 算力——每一步都要重新生成。
>
> **DPO 是异策略**：在固定的偏好数据集上训练。训练过程中无需生成。便宜得多，但会遭受分布偏移问题（随策略变化数据逐渐过时）。
>
> **在线 DPO 是混合体**：生成新鲜数据（同策略生成），但使用 DPO 的监督损失（异策略式优化）。兼具两者优势。
>
> **PPO 的巧妙之处**：使用裁剪比率 $r = \pi_\text{new}/\pi_\text{old}$，从同一批同策略数据中榨取多步梯度更新（通常 4 个 epoch），以可控方式使其变得「略带异策略」。

## 基于模型 vs 无模型

|  | Model-Free | Model-Based |
| --- | --- | --- |
| **所学内容** | 直接学习策略 $\pi$ 和/或价值 $V$/$Q$ | 环境模型 $\hat{P}(s'|s,a)$ |
| **规划能力** | 无规划，反应式决策 | 可以模拟未来轨迹 |
| **样本效率** | 低（必须经历所有情况） | 高（可在想象中规划） |
| **准确性** | 无模型偏差 | 模型误差会累积 |
| **何时使用** | 复杂/未知的转移 | 简单转移，追求效率 |
| **示例** | PPO、DQN、SAC[haarnoja2018soft] | MuZero[schrittwieser2020mastering]、Dreamer[hafner2020dream]、AlphaGo[silver2016mastering] |

> **直觉：为什么 LLM RL 是无模型的**
>
> 语言生成的状态转移是平凡的（将 token 追加到序列——确定性转移）。环境「模型」并非瓶颈。难的是**奖励**——预测人类的偏好。这使得 model-based 方法对 LLM RL 来说没有必要。
>
> RLHF 中的奖励模型在某种意义上也可以被看作一种「模型」（它预测人类偏好），但它被用作奖励信号，而非用于规划/模拟。LLM RL 本质上是无模型的策略优化。

## 奖励塑形（Reward Shaping）

**奖励塑形（Reward Shaping）**[ng1999policy]是一种由开发者修改或补充环境原始奖励函数的技术。其主要目的是将**稀疏奖励**场景（智能体仅在任务最终完成时才获得反馈）转换为带有中间反馈信号的**稠密奖励**场景，以加速收敛。

### 数学框架

设时间步 $t$ 的原始奖励为 $R_t(s, a, s')$。重塑后的奖励添加了一个辅助塑形函数 $F$：

$$
R'_t(s, a, s') = R_t(s, a, s') + F(s, a, s')
$$

> **警告：朴素塑形的风险：奖励黑客（Reward Hacking）**
>
> 如果 $F(s, a, s')$ 被随意设计，智能体会找到结构性漏洞来最大化辅助信号，同时忽略全局目标。
>
> **示例**：一个因到达中间地标而获得奖励的导航智能体可能学会无限绕着某个检查点循环以累积无限奖励——而永远不会到达目的地。
>
> 在 LLM 中：若模型因「听起来自信」而获得奖励，它可能学会无论准确与否都以「Absolutely!」开头。

### 基于势函数的奖励塑形（Potential-Based Reward Shaping，PBRS）

为了从数学上保证塑形**不会**改变最优策略，可使用**基于势函数的奖励塑形（PBRS）**。塑形函数 $F$ 被约束为跨状态的标量势函数 $\Phi$ 之差：

$$
F(s, a, s') = \gamma\, \Phi(s') - \Phi(s)
$$

其中 $\Phi: \mathcal{S} \to \mathbb{R}$ 是一个实值势函数，衡量某状态相对目标的接近程度，$\gamma$ 是折扣因子。

完整的 PBRS 奖励：

$$
R'(s, a, s') = R(s, a, s') + \gamma\, \Phi(s') - \Phi(s)
$$

### 理论保证

> **关键：PBRS 策略不变性定理**
>
> - **策略不变性**：在重塑奖励 $R'$ 下的最优策略 $\pi^*$ 与原始奖励 $R$ 下的最优策略**完全相同**。塑形不会引入次优行为。
> - **回路免疫**：任何从某状态出发又回到同一状态的循环轨迹，其净势能变化恰好为零（$\Phi(s) - \Phi(s) = 0$）。智能体无法利用回路来作弊式刷奖励。
> - **收敛加速**：尽管最优策略不变，塑形后的奖励提供了更稠密的梯度信号，使智能体在稀疏奖励环境下收敛速度提升 5--50 倍。
