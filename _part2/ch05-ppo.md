---
layout: home
title: PPO——近端策略优化
permalink: /part2/ch05-ppo.html
---

## 动机与历史

**问题**：朴素的 policy gradient 更新对步长没有任何约束。仅仅一次倒霉的 batch 就可能把 policy 推入一个生成垃圾文本的区域 $\rightarrow$ 垃圾文本获得低 reward $\rightarrow$ 下一次 gradient 让情况更糟 $\rightarrow$ 不可挽回地崩溃。

**解决方案的演进**：

1. **TRPO** [[148]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schulman2015trust)]（2015）：对新旧 policy 之间的 KL 散度施加约束。效果完美，但需要昂贵的二阶优化（Fisher 信息矩阵、共轭梯度）。
2. **PPO**（2017） [[149]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schulman2017proximal)]：用一个简单的一阶 clipped 目标实现类似的稳定性。实现复杂度低了 10$\times$，效果几乎一样，且能轻松扩展到分布式训练。

## Clipped 目标

PPO 的核心创新是一个 clipped surrogate 目标：它能阻止破坏性的大幅 policy 更新，同时保持实现简单。

$$
\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]}
$$
其中 $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$ 是概率比。

> **Clipping 直觉——关键洞察**
>
> $\min$ 算子构造了一个**悲观下界**：
>
> - **好的 action（$\hat{A} > 0$）**：我们希望提升其概率。Surrogate $r\hat{A}$ 会随 $r$ 增大而增大，但 clip 将收益上限锁在 $r = 1 + \epsilon$。*“别因为一个好例子就贪心。”*
> - **坏的 action（$\hat{A} < 0$）**：我们希望降低其概率。$r\hat{A}$ 会随 $r$ 减小而改善，但 clip 将收益上限锁在 $r = 1 - \epsilon$。*“别因为一个坏例子就遗忘得太激进。”*
>
> 净效应：每次更新步中 policy 的变化最多在 $\pm$20% 之内，既能防止灾难性崩溃，也能防止过度自信地走向特化。

## 完整的 PPO Loss

$$
L = L^{\text{CLIP}} - c_1 \underbrace{(V_\theta(s_t) - V^{\text{target}}_t)^2}_{\text{value loss}} + c_2 \underbrace{H[\pi_\theta(\cdot\mid s_t)]}_{\text{entropy bonus}}
$$

- **Value loss**（$c_1 = 0.1$）：训练 critic 去预测 return；同样被 clip 以保持稳定。
- **熵奖励**（$c_2 = 0.01$）：防止过早收敛到确定性 policy，对探索至关重要。

## PPO Gradient 与更新规则的推导

本节追溯从 RL 目标到 PPO 更新规则的数学路径，说明 clipped surrogate *为何*有效。

### 第 1 步：RL 目标

目标是在 policy 下最大化期望累计 reward：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T r_t\right]
$$

### 第 2 步：Policy Gradient 定理

$J(\theta)$ 关于 policy 参数的 gradient：
$$
\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t\mid s_t) \cdot \hat{A}_t\right]}
$$

其中 $\hat{A}_t$ 是优势函数（即在 state $s_t$ 下 action $a_t$ 相对平均 action 的好坏程度）。用 advantage 替代完整 return 是为了降低方差。

### 第 3 步：离策略数据的重要性采样

PPO 使用 $\pi_{\theta_{\text{old}}}$ 收集数据，却更新 $\pi_\theta$。为修正这种分布失配，需应用重要性采样：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} \nabla_\theta \log \pi_\theta(a_t\mid s_t) \cdot \hat{A}_t\right]
$$

定义概率比 $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$。利用恒等式 $\nabla_\theta \log f = \frac{\nabla_\theta f}{f}$，可得：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\nabla_\theta\, r_t(\theta) \cdot \hat{A}_t\right]
$$

这意味着最大化** surrogate 目标**：
$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]
$$

### 第 4 步：无约束 Surrogate 的问题

$L^{\text{CPI}}$ 是一个合法的目标，但若无约束，单次 gradient 步就可能让 $r_t(\theta)$ 远离 1.0，导致：

- 重要性权重变得极端 $\rightarrow$ 方差升高
- Policy 进入未经检验的区域 $\rightarrow$ reward 模型给出不可靠的分数
- 灾难性崩溃：policy 生成垃圾，且无法恢复

**TRPO 方案**：约束 $D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$。需要二阶方法（成本高）。

### 第 5 步：PPO 的 Clipped Surrogate（一阶近似）

PPO 用一个** clipped 目标**取代硬性 KL 约束，仅使用一阶 gradient 即可达到类似行为：

$$
\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]}
$$

**Gradient 的推导**：

令 $L_t = \min(r_t \hat{A}_t,\; \bar{r}_t \hat{A}_t)$，其中 $\bar{r}_t = \text{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon)$。

$$
\nabla_\theta L_t = \begin{cases}
\nabla_\theta r_t(\theta) \cdot \hat{A}_t & \text{if } r_t \hat{A}_t < \bar{r}_t \hat{A}_t \text{ (unclipped term is smaller)} \\
0 & \text{if } r_t \hat{A}_t \geq \bar{r}_t \hat{A}_t \text{ (clipped term is smaller, gradient = 0)}
\end{cases}
$$

展开各条件：

- **当 $\hat{A}_t > 0$ 且 $r_t < 1+\epsilon$**：Gradient 正常传播——policy 被鼓励提升 $\pi_\theta(a_t\mid s_t)$。
- **当 $\hat{A}_t > 0$ 且 $r_t \geq 1+\epsilon$**：Gradient 为**零**——policy 提升已足够，停止继续推。
- **当 $\hat{A}_t < 0$ 且 $r_t > 1-\epsilon$**：Gradient 正常传播——policy 被鼓励降低 $\pi_\theta(a_t\mid s_t)$。
- **当 $\hat{A}_t < 0$ 且 $r_t \leq 1-\epsilon$**：Gradient 为**零**——policy 降低已足够，停止继续推。

### 第 6 步：完整的 PPO 更新规则

将 clipped policy loss、value loss 与熵奖励组合起来：
$$
\boxed{\theta_{k+1} = \theta_k + \alpha \cdot \nabla_\theta \left[L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H[\pi_\theta]\right]}
$$

其中：
$$
L^{\text{VF}}(\theta) &= \left(V_\theta(s_t) - V_t^{\text{target}}\right)^2 & &\text{(value function regression loss)} \\
H[\pi_\theta] &= -\sum_a \pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t) & &\text{(entropy of the policy)}
$$

> **小结：它为何有效**
>
> 1. **Policy gradient 定理**给出了改进 policy 的方向。
> 2. **重要性采样**让我们能在多个 epoch 中复用来自 $\pi_{\theta_{\text{old}}}$ 的数据。
> 3. **Clipping** 防止重要性权重变得极端，保持更新安全。
> 4. **$\min$ 算子**保证我们始终在 (clipped, unclipped) 中取更保守的一项——对改进设置悲观下界。
> 5. **结果**：以概率 1 实现单调改进，且只用一阶 gradient。无需 Hessian、共轭梯度或线搜索。

## Rollout Buffer 与 Rollout

在 PPO 中，数据管理依赖一个被称为 **Rollout Buffer** 的专用短期存储系统。与无限期地在 replay buffer 中保存经验的离策略算法（如 DQN）不同，PPO 需要一种短暂存在的结构，以满足其在策略的数学约束。

### 什么是 Rollout？

**Rollout**（轨迹）是 Agent 在环境中执行当前 policy 所产生的一段交互序列：

- **过程**：Agent 观察 state、选择 action、获得 reward，然后转移到下一个 state。重复进行固定步数，或直到 episode 结束。
- **在大语言模型/RLHF 中**：一次 rollout 即从数据集中取出一条 prompt，让语言模型一个 token 接一个 token 地生成完整序列，直到遇到文本结束标记。每个 token 就是一个“step”。

### Rollout Buffer

Rollout buffer 临时存储 rollout 阶段收集到的全部数据。对每个生成的 token/step，它记录：
$$
\boxed{\mathcal{B} = \left\{ \left(s_t,\; a_t,\; \log\pi_{\theta_{\text{old}}}(a_t\mid s_t),\; r_t,\; V(s_t)\right) \right\}_{t=1}^{T}}
$$

- $s_t, a_t, r_t$：步 $t$ 的 state、所采取的 action 与 reward。
- $\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)$：在生成该 action 的那一个 policy 下取该 action 的对数概率（计算 ratio 时需要）。
- $V(s_t)$：Value function 给出的基线预测（计算 GAE advantage 时需要）。

### Rollout Buffer 的生命周期

该 buffer 按严格的三阶段时钟周期运行：

1. **收集**：当前 policy 与环境交互，用新鲜轨迹填满 buffer（对一个 70B 模型，batch=128，max_tokens=512：单次 rollout 可达 65K 个 token 级 transition）。
2. **训练**：对各轨迹计算 GAE advantage。使用 clipped 目标在 mini-batch 上跑 $K$ 个 epoch（通常 3--10）的 gradient 下降，更新 policy 权重。
3. **清空**：整个 buffer 被**彻底清空**。由于 PPO 是 on-policy 的，旧 policy 产生的数据无法安全地复用于下一轮更新——比率 $r_t(\theta)$ 会变得过期，clipping 保证也会失效。

> **Rollout Buffer 与 Replay Buffer 的区别**
>
> **Replay Buffer**（DQN、SAC）：Off-policy。无限期地存储数百万条 transition。随机采样。数据跨多次更新复用。
>
> **Rollout Buffer**（PPO、GRPO）：On-policy。存储一个 batch 的轨迹。使用若干 epoch 后被完全丢弃。每个周期都需要新鲜数据。
>
> 这就是为什么 PPO 需要持续生成——每次更新后 buffer 被清空，需要新的 rollout。这也让生成瓶颈（占 wall-clock 时间 60--70%）尤为痛苦。

> **RLHF 场景中的 vLLM**
>
> 在 RLHF 训练中，vLLM 被用于**生成阶段**（占 wall-clock 时间 60--70%）。Policy 模型生成 rollout，随后由 reward 模型打分。其主要优势：
>
> - **批量生成**：跨多条 prompt 并行生成 256+ 条回答。
> - **显存效率**：可容纳更多并发生成 $\rightarrow$ 在生成瓶颈期 GPU 利用率更高。
> - **前缀共享**：当每条 prompt 生成 $N=8$ 条回答时（如 GRPO），prompt 的 KV 只算一次并被 8 条共享——无冗余 prefill。
> - **集成**：OpenRLHF、TRL 等框架将 vLLM 作为生成后端，把生成 worker（vLLM）与训练 worker（DeepSpeed/FSDP）分离。

## 用于 RLHF 的 PPO：完整循环


![fig_025_fig25.png]({{ site.baseurl }}/figures/fig_025_fig25.png)

> **70B Chat 模型的具体 PPO 步骤**
>
> **设置**：128 条 prompt 一个 batch，policy 为 Llama-3-70B，最大生成 512 个 token。
>
> **第 1 步——生成**：采样 128 条回答（temperature=0.7，top-p=0.9）。耗时占 60%。
>
> **第 2 步——打分**：Reward 模型对每个 (prompt, response) 对打分。取值范围：0.2--0.95。
>
> **第 3 步——KL**：计算逐 token 的 KL：$\text{KL}_t = \log\pi_\theta(y_t\mid y_{<t}) - \log\pi_\text{ref}(y_t\mid y_{<t})$。跨 token 的均值 KL：通常 3--8。
>
> **第 4 步——最终 reward**：$R = r_\text{RM} - 0.05 \times \text{mean\_KL}$（只在最后一个 token 给）。
>
> **第 5 步——GAE**：使用 value head 的预测，为每个 token 位置计算 $\hat{A}_t$。对 advantage 做 whitening（零均值、单位方差）。
>
> **第 6 步——更新**：在大小为 16 的 mini-batch 上跑 4 个 epoch 的 SGD。Clip 比率 $\epsilon = 0.2$，梯度范数 clip 至 1.0。
>
> **结果**：Policy 每步约提升 0.005 的胜率。10K 步后：相对于 SFT 绝对提升 5--10%。

> **大语言模型 RL 中的 Tokenization 陷阱**
>
> 在计算逐 token KL 惩罚与 advantage 时，请记住：tokenization 决定了什么算一“步”。单个概念性 action（例如输出“2024”）依据 tokenizer 的不同可能跨越 1--4 个 token。这带来若干微妙问题：
>
> - **KL 记账**：相同的语义内容若被切成不同数量的 token，逐 token KL 加总会得出不同总额（例如，被切成更多子词的稀有词会获得更高的总 KL 惩罚）。
> - **Credit 分配**：GAE 按 token 位置分配 advantage——但语义“决策”常跨越多个 token。模型其实只在词首 token 真正“决定”；后续子词 token 基本是确定性的。
> - **Reward 放置**：只在最后一个 token 给 reward 意味着所有先前 token 必须通过 GAE 向后回传 credit——越长的回答信号被稀释得越厉害。
>
> **缓解措施**：一些系统按序列长度对 KL 做归一化，使用词级 reward shaping，或在语义边界（而非最末 token）处施加 reward。

## 细节机制：Logits 与 Policy 更新


![PPO 端到端流程：从 prompt batch 出发，经过生成、reward 打分、KL 计算、advantage 估计，到 clipped policy 更新。反馈环显示更新后的 policy 又被用于下一次生成。]({{ site.baseurl }}/figures/fig_026_fig26.png)

PPO 在内存中维护两份不同的参数状态，它们共享同一神经网络拓扑，但在优化过程中持有不同的权重值：

> **核心架构：两个网络**
>
> 1. **Policy 网络（$\pi_\theta$）：**由权重 $\theta$ 参数化的、处于活动状态的实时网络。优化过程中通过反向传播持续更新。
> 2. **旧 Policy 网络（$\pi_{\theta_{\text{old}}}$）：**由权重 $\theta_{\text{old}}$ 参数化的冻结快照。在单个优化周期内充当静态锚，防止 policy 漂移过快。

### 阶段 1：Rollout（数据收集）

在数据收集期间，Agent 与环境交互 $T$ 步。在每个时间步 $t$：

1. 环境给出当前 state/观测 $s_t$（对大语言模型而言：prompt + 到目前为止已生成的 token）。
2. $s_t$ 被送入当前网络快照（$\theta_{\text{old}}$）。
3. 网络输出原始未归一化的值——**logits** $z_{\text{old}}$——一个长度为 $\lvert V \rvert$（词表大小 32K--128K）的向量。
4. 通过 Softmax 计算概率：
$$
\boxed{P(a \mid s_t) = \text{Softmax}(z_{\text{old}}) = \frac{\exp(z_{\text{old}, a})}{\sum_{j=1}^{\lvert V \rvert} \exp(z_{\text{old}, j})}}
$$
5. 从 $P(a \mid s_t)$ 中采样一个 action $a_t$（下一个 token），并将 transition 元组 $\langle s_t, a_t, r_t, s_{t+1} \rangle$ 连同 $\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ 一起存入 rollout buffer。

> **为何要存对数概率？**
>
> 在 rollout 阶段把 $\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ 作为标量保存，可以避免在优化期间重新运行冻结的网络。每个 mini-batch 因此可省下一次完整 forward pass——对 70B 模型而言意义重大。

### 阶段 2：优化循环（Mini-Batch 更新）

一旦 rollout buffer 填满，PPO 就在 mini-batch 上跑 $K$ 个 epoch（通常 3--10）。在每个 gradient 步中，使用所存 state $s_t$ 为两个 policy 同时生成 logits：

**旧 Policy 评估**（冻结）：
$$
z_{\text{old}} = f(s_t; \theta_{\text{old}}) \quad \longrightarrow \quad \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) = \text{LogSoftmax}(z_{\text{old}})[a_t]
$$

*实现捷径：复用 rollout 时保存的标量，而非重新计算。*

**实时 Policy 评估**（更新中）：
$$
z_{\text{new}} = f(s_t; \theta) \quad \longrightarrow \quad \log \pi_\theta(a_t \mid s_t) = \text{LogSoftmax}(z_{\text{new}})[a_t]
$$

由于每次 mini-batch gradient 步后 $\theta$ 都会更新，$z_{\text{new}}$ 会在整个优化循环中持续变化，而 $z_{\text{old}}$ 则始终保持不变。

### 从 Logits 到概率比

PPO 的核心比率衡量某个 action 在新 policy 下相对旧 policy 的概率变化：
$$
\boxed{r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}}
$$

为了避免直接除原始概率带来的数值下溢/上溢灾难，计算在**对数空间**中进行：
$$
\log \pi_\theta(a_t \mid s_t) &= \text{LogSoftmax}(z_{\text{new}})[a_t] \\
\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) &= \text{LogSoftmax}(z_{\text{old}})[a_t]
$$

比率由差值的指数恢复：
$$
\boxed{r_t(\theta) = \exp\!\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)}
$$

该比率随后被代入 PPO 的 clipping 目标：
$$
\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\!\left(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta),\, 1{-}\epsilon,\, 1{+}\epsilon)\,\hat{A}_t\right) \right]}
$$

> **Clipping 的工作方式**
>
> - 若 $\hat{A}_t > 0$（好 action）：比率被 clip 至 $1+\epsilon$——不会对好 action 过度利用。
> - 若 $\hat{A}_t < 0$（坏 action）：比率被 clip 至 $1-\epsilon$——不会对坏 action 过度惩罚。
> - $\min(\cdot)$ 保证我们始终取更保守的估计。
>
> 结果：在一个信任域内实现单调改进——不会出现灾难性崩溃。

### PPO 权重生命周期


**$\theta$ 与 $\theta_{\text{old}}$ 在 PPO 训练各阶段的演化。**
| **阶段** | **实时 $\theta$** | **旧 $\theta_{\text{old}}$** | **比率 $r_t(\theta)$** |
| --- | --- | --- | --- |
| 1. Rollout 起始 | 活动副本 | 同一活动副本 | 恒为 $1.0$（按定义） |
| 2. Batch 第 1 步 | 计算 gradient | 冻结 | $1.0$（初始步） |
| 3. Batch 第 $N$ 步 | 正在修改（$\theta \neq \theta_{\text{old}}$） | 冻结 | 偏离 $1.0$（如 $1.06$、$0.94$） |
| 4. Clipping 生效 | 受 $\epsilon$ 限制 | 冻结 | 被锁在边界（$1 \pm \epsilon$） |
| 5. 优化结束 | 高度优化后 | 被丢弃 | 不适用 |
| 6. 下一周期 | $\theta \rightarrow \theta_{\text{old}}$ | 接收最新的 $\theta$ | 重置回 $1.0$ |

### 连续 Action 空间扩展

对于连续 action 空间（在大语言模型中不常见，但对机器人 RL 很重要），网络输出的是分布参数而非离散 logits：

- 预测的均值向量 $\mu$
- 预测的标准差向量 $\sigma$

对数概率由高斯对数 PDF 计算：
$$
\boxed{\log \pi(a_t \mid s_t) = -\frac{1}{2}\left(\frac{a_t - \mu}{\sigma}\right)^{\!2} - \log(\sigma) - \frac{1}{2}\log(2\pi)}
$$

比率 $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$ 的计算方式完全相同，并被代入同一个 clipping 目标。

## TRL 实现

HuggingFace 的 TRL 库 [[160]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-vonwerra2022trl)] 提供了面向大语言模型的所有主流 RL 方法的生产级实现。

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from peft import LoraConfig

# 模型初始化
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
    peft_config=LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","v_proj","k_proj","o_proj"])
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# PPO 配置，包含全部关键超参
ppo_config = PPOConfig(
    learning_rate=1.5e-6,        # 低学习率，利于稳定
    batch_size=128,              # 每步的 prompt 数
    mini_batch_size=16,          # 梯度累计的单位
    ppo_epochs=4,                # 每个 batch 的 epoch 数（复用数据）
    gamma=1.0,                   # 不做折扣（单轮）
    lam=0.95,                    # GAE lambda
    cliprange=0.2,               # PPO epsilon
    cliprange_value=0.2,         # value function 的 clip 范围
    vf_coef=0.1,                 # value loss 系数
    init_kl_coef=0.05,           # 初始 KL 惩罚
    target_kl=6.0,               # 自适应 KL 目标
    whiten_rewards=True,         # 对 advantage 做归一化
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
)

ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer,
    dataset=prompt_dataset, data_collator=collator)

# 训练循环
for batch in ppo_trainer.dataloader:
    # 1. 生成回答
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True
    )
    # 2. 用 reward 模型打分
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    rewards = [torch.tensor(reward_model.score(q, r)) for q, r in zip(batch["query"], texts)]
    # 3. PPO 更新（内部完成 KL、GAE、clipping）
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # 监控指标：stats["ppo/mean_scores"]、stats["ppo/policy/approx_kl"]
```

## 关键超参

| **参数** | **典型值** | **设错的影响** |
| --- | --- | --- |
| `cliprange` | 0.2 | 过低：学不到东西。过高：不稳定。 |
| `init_kl_coef` | 0.01--0.1 | 过低：reward hacking。过高：卡在 SFT 状态。 |
| `target_kl` | 4--8 | 自适应控制器的目标值；越低越保守。 |
| `ppo_epochs` | 4 | 过多：在单 batch 上过拟合。过少：浪费生成算力。 |
| `learning_rate` | $1{-}5 \times 10^{-6}$ | 过高：灾难性遗忘（Catastrophic Forgetting）。 |
| `batch_size` | 64--256 | 越大：梯度更平滑，但生成算力更多。 |
| `temperature` | 0.7--1.0 | 越低：探索越少。越高：advantage 越嘈杂。 |

