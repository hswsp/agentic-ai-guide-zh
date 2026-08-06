---
layout: home
title: LLM 智能体训练
permalink: /part2/ch12-llm-agent-training.html
---

# LLM 智能体训练


## 动机：从聊天机器人到自主智能体

现代 LLM 越来越多地被部署，不仅作为对话助手，而且作为**自主智能体（Autonomous Agents）**，能够在多个步骤中与外部工具、API、数据库和环境进行交互。这种从单轮聊天机器人到多步智能体的转变，引入了根本性的新 RL 挑战，要求我们重新思考如何训练、评估和部署语言模型。


**（a）传统聊天机器人：** `User` ⇄ `LLM`（单轮，即时反馈）

**（b）自主智能体：** `User` → `LLM Agent` ⇄ `Tools`，`Tools` → `Environment`，`Environment` 返回 `reward` 给 Agent（多步，稀疏终端 reward）
**从聊天机器人到自主智能体：传统的 LLM 聊天机器人在带有即时人类反馈的单步对话循环中运行。自主智能体跨多个工具交互进行规划，从真实世界的执行环境中接收反馈，并针对稀疏的终端 reward（任务成功/失败）进行优化。**

要求新 RL 方法的关键差异：

- **多步推理**：Agent 必须跨越 10--100+ 次工具调用进行规划，而不仅仅是生成单一响应。
- **外部环境反馈**：Reward 来自真实世界的执行（测试套件通过、网页加载、代码编译），而不仅仅是人类偏好分数。
- **结构化动作**：动作不仅是 Token，而是结构化输出（JSON 工具调用、API 负载、代码块）。
- **长视野与稀疏 reward**：成功/失败可能只能在多个中间步骤之后才能确定。

> **为什么标准 RLHF 对 Agent 不够用**
>
> 标准 RLHF（PPO/DPO）针对单轮质量进行优化：给定一个 Prompt，产生一个好的响应。但 Agent 必须：
>
> - 决定*何时*使用工具，还是进行内部推理
> - 在轨迹中途从错误中恢复（自我纠正）
> - 平衡探索（尝试新方法）与利用（使用已知良好模式）
> - 处理部分可观测性（工具输出可能不完整或带噪声）
>
> 这要求训练方法在**完整轨迹**上进行推理，而不是单个轮次。

## LLM Agent 的轨迹缓冲区

在 LLM Agent 的语境下，传统的 RL 回放缓冲区经历了结构性的转变。Agent 缓冲区——通常称为**轨迹缓冲区（Trajectory Buffers）**、**经验池（Experience Pools）**或**记忆库（Memory Banks）**——不再存储低维数值 Tensor，而是管理复杂的文本历史、工具执行输出和显式推理步骤。

### LLM Agent 缓冲区的数学结构

在经典 RL 中，回放缓冲区存储扁平元组 $(s, a, r, s')$。对于 LLM Agent，这扩展为高维的 Token 化文本结构：

$$
\boxed{e_t = \left( \mathcal{S}_t,\; \mathcal{A}_t,\; \mathcal{R}_t,\; \mathcal{S}_{t+1} \right)}
$$

- $\mathcal{S}_t$：**完整上下文状态**——系统 Prompt、用户目标、对话历史，以及当前环境变量（例如 HTML 源代码、目录结构、数据库 Schema）。
- $\mathcal{A}_t$：Agent 的**生成输出**，通常由一段思维链（Chain-of-Thought，CoT）推理字符串紧接一个结构化工具调用组成：
$$
\mathcal{A}_t = \{\text{text}_{\text{reasoning}},\; \text{json}_{\text{tool\_call}}\}
$$
- $\mathcal{R}_t$：**评估信号**，来自外部执行环境（单元测试通过、编译器标志、API 响应码），或由 LLM-as-a-judge 系统验证。
- $\mathcal{S}_{t+1}$：**更新后的上下文窗口**，将工具输出文本或错误日志直接附加到对话历史中。

> **具体 Agent 轨迹：代码调试**
>
> **步骤 1**：$\mathcal{S}_1$ = “修复 `utils.py` 中失败的测试”\
>
> $\mathcal{A}_1$ = *“让我先读取该文件”* + `read_file("utils.py")`\
>
> $\mathcal{R}_1$ = 0（中间步骤）\
>
> **步骤 2**：$\mathcal{S}_2$ = [先前上下文 + 文件内容]\
>
> $\mathcal{A}_2$ = *“Bug 在第 42 行，是一个差一错误”* + `edit_file("utils.py", ...)`\
>
> $\mathcal{R}_2$ = 0（中间步骤）\
>
> **步骤 3**：$\mathcal{S}_3$ = [先前上下文 + 编辑确认]\
>
> $\mathcal{A}_3$ = *“让我验证修复”* + `run_tests()`\
>
> $\mathcal{R}_3$ = +1.0（所有测试通过——稀疏终端 reward）

## 操作范式

LLM Agent 通过三种主要的优化方法论来利用专门的轨迹缓冲区：

### A. 自我纠正与思维细化

此类别中的两种代表性方法是 STaR [zelikman2022star] 和 Reflexion [shinn2023reflexion]。当 Agent 在一次多步执行轨迹中失败时，次优序列会被保存到缓冲区。该框架随后采样此轨迹，并提示 LLM 对其过去的表现生成显式文本批评：
$$
\text{Critique} \leftarrow \text{LLM}(\mathcal{S}_{\text{failed}},\; \mathcal{A}_{\text{failed}},\; \mathcal{R}_{=0})
$$

一旦修正后的轨迹获得正向 reward，它就被移至最优经验池，用于通过微调（在成功轨迹上做 SFT）或 RL（采用二元通过/失败 reward 的 GRPO [shao2024deepseekmath]）来更新网络权重。

> **STaR：自学习推理者**
>
> 1. 为一批问题生成推理轨迹
> 2. 过滤：仅保留导致正确答案的轨迹
> 3. 在成功轨迹上微调模型（SFT）
> 4. 重复：改进后的模型在下一次迭代中生成更好的轨迹
>
> 每次迭代都使用模型自身的成功输出作为训练数据，自举其推理能力。

> **Reflexion：语言强化学习**
>
> 1. Agent 尝试一项任务，失败
> 2. Agent 生成一段**语言反思**：“我失败是因为在调用 API 之前没有检查返回类型”
> 3. 反思被存储在一个情节记忆缓冲区中
> 4. 在下一次尝试中，反思作为经验教训被注入到 Prompt 中
> 5. 不需要权重更新——纯粹通过自我批评进行上下文学习

### B. 离策略探索

此范式以 ReAct [yao2023react] 及相关工具使用框架为代表，涉及广泛的自主探索。在自主探索过程中（网页导航、数据库查询、代码生成），Agent 记录数千条探索性执行路径。轨迹缓冲区充当一个过滤器：

- **成功过滤**：只保留达成目标的轨迹用于训练。
- **效率排序**：在成功轨迹中，优先选择最短/最高效的工具使用路径。
- **多样性采样**：维持多样化的解决策略集合，以防止模式坍塌。

优化算法（通常是 GRPO [shao2024deepseekmath] 或过滤后的 SFT）只在高效、成功的轨迹上计算 Loss，丢弃迂回曲折的运行。

### C. 非参数化的上下文学习（基于经验的 RAG）

轨迹缓冲区可以不修改神经网络权重，而是作为一个**向量数据库**。给定一个新的用户目标 $\mathcal{G}_{\text{new}}$，系统检索最相关的过往经验：
$$
\boxed{\mathcal{E}_{\text{retrieved}} = \arg\max_{e \in \mathcal{B}} \text{sim}\!\left(\text{Embed}(\mathcal{G}_{\text{new}}),\; \text{Embed}(e)\right)}
$$

Top-$k$ 个相似的成功历史运行作为 few-shot 示例直接注入到 Prompt 上下文中。这种方法：

- 需要**零训练**——纯粹的检索增强生成（Retrieval-Augmented Generation, RAG）
- 如果缓冲区中存在相似经验，可即时适应新任务
- 随缓冲区规模扩展（更多经验 = 更好的覆盖率）
- 与参数化学习互补（罕见情况使用检索，常见模式使用权重）

## 范式对比


**传统 RL 缓冲区与 LLM Agent 缓冲区对比**
| **特性** | **传统 RL 缓冲区** | **LLM Agent 缓冲区** |
| --- | --- | --- |
| **数据格式** | 连续向量 / Tensor | Token 化文本、JSON、代码块、工具输出 |
| **数据量** | 海量（$10^5$--$10^7$ 条） | 中小规模（$10^3$--$10^5$ 条轨迹） |
| **主要目标** | 打破数据相关性 | 提供推理示范 |
| **采样** | 随机均匀 / PER | 语义检索 / 成功优先 / 多样性 |
| **状态大小** | 固定（如 84$\times$84 像素） | 可变（每个状态 1K--128K Token） |
| **动作空间** | 离散/连续向量 | 结构化文本（推理 + 工具调用） |
| **Reward 来源** | 环境模拟器 | 外部执行 / LLM judge / 单元测试 |

## Agent RL 的主要技术


**用 RL 训练 LLM Agent 的关键方法。**
| **方法** | **类型** | **核心思想** |
| --- | --- | --- |
| **STaR** [zelikman2022star] | 迭代 SFT | 通过在自身成功轨迹上微调来自举推理能力 |
| **Reflexion** [shinn2023reflexion] | 上下文 RL | 语言式自我批评作为情节记忆存储；无权重更新 |
| **ReAct** [yao2023react] | Prompt 方法 | 在单次生成中交错推理（“思考”）与行动（“工具调用”） |
| **LATS** [zhou2024lats] | 树搜索 | 在动作序列上做蒙特卡洛树搜索；反向传播 reward |
| **AgentQ** [putta2024agentq] | 离策略 RL | 在 Agent 轨迹上用 AI 生成的偏好对做 DPO |
| **OpenHands** [wang2024openhands] | GRPO | 基于执行的 reward（测试通过/失败）做组相对优化 |
| **Voyager** [wang2023voyager] | 技能库 | 存储并检索成功代码片段以组合复用 |
| **RLEF** [le2024rlef] | 在线 RL | 从执行反馈中做 RL——来自代码/测试执行的二元 reward |

### STaR：自学习推理者（详解）

STaR [zelikman2022star] 是一种**迭代式自我改进**方法，无需外部 Reward Model 即可自举推理能力。核心洞见：如果模型偶尔能正确解决一个问题，它就可以从自身的成功中学习。

**算法**：

1. **生成**：对数据集 $\mathcal{D}$ 中的每个问题 $x_i$，采样一条推理轨迹 $z_i \sim \pi_\theta(\cdot | x_i)$，后跟一个答案 $\hat{y}_i$。
2. **过滤**：仅保留满足 $\hat{y}_i = y_i^*$（正确答案）的轨迹。定义成功集 $\mathcal{D}_{\text{pass}} = \{(x_i, z_i, y_i^*) : \hat{y}_i = y_i^*\}$。
3. **合理化（Rationalization）**（关键创新）：对模型失败的问题，生成一条以正确答案为条件的“合理化”轨迹：$z_i^{\text{rat}} \sim \pi_\theta(\cdot | x_i, y_i^*)$。这教导模型从解*反向*推理。
4. **微调**：在 $\mathcal{D}_{\text{pass}} \cup \mathcal{D}_{\text{rationalized}}$ 上通过 SFT 更新 $\theta$。
5. **迭代**：用改进后的模型从步骤 1 重复。

$$
\boxed{\theta_{k+1} = \arg\min_\theta -\sum_{(x,z,y) \in \mathcal{D}_k^+} \log \pi_\theta(z, y | x)}
$$

**收敛动力学**：每次迭代 $k$ 都会提升模型的解题率 $p_k$。若 $p_0 = 0.3$（解出 30% 的问题），经过合理化 + SFT 后 $p_1 \approx 0.5$。通常 3--5 次迭代后收敛至 $p \approx 0.7$--$0.9$。

> **STaR 合理化 Prompt**
>
> ```python
> # 标准生成（步骤 1）：
> PROMPT = """Solve the following problem step by step.
> Problem: A store has 45 apples. It sells 3/5 of them. How many remain?
> Let's think step by step:"""
>
> # 合理化 Prompt（步骤 3——以正确答案为条件）：
> PROMPT_RATIONALIZE = """Solve the following problem step by step.
> The correct answer is 18.
> Problem: A store has 45 apples. It sells 3/5 of them. How many remain?
> Let's think step by step to arrive at 18:"""
>
> # Agent 变体（带错误条件的代码任务）：
> PROMPT_AGENT_RATIONALIZE = """The following code task failed with the error below.
> Generate a correct solution step by step.
>
> Task: Implement binary search that handles duplicates.
> Previous error: IndexError: list index out of range (line 12)
> Correct behavior: Return leftmost index of target.
>
> Let me fix this by reasoning about the boundary conditions:"""
> ```

> **面向 Agent 的 STaR 变体**
>
> - **Quiet-STaR** [zelikman2024quietstar]：在每个生成 Token 之间插入“思考 Token”。模型学会在没有显式 CoT 提示的情况下*隐式*推理。训练目标：在包含思考 Token 时更好地预测下一个 Token。
> - **代码 Agent 的 STaR**：用测试执行替代答案验证。“正确” = 所有测试通过。合理化 = 以错误消息为条件生成新方法。
> - **V-STaR** [hosseini2024vstar]：增加一个在 $(z, y, \text{correct/incorrect})$ 三元组上训练的验证器模型。该验证器提供过程级监督，过滤那些偶然到达正确答案的糟糕推理轨迹。

### Reflexion：语言强化学习（详解）

Reflexion [shinn2023reflexion] 引入了一种激进的范式：**无权重更新的 RL**。Agent 不通过基于梯度的学习，而是通过存储在情节记忆中的自然语言自我批评来改进。

**完整架构**：

1. **Actor**：在环境中执行动作的 LLM Agent $\pi$。
2. **评估器**：二元信号（任务成功/失败）或标量启发式（如通过的测试用例数）。
3. **自我反思生成器**：给定失败轨迹 $\tau_{\text{fail}}$ 和环境反馈，生成自然语言反思 $r_{\text{text}}$：
$$
r_{\text{text}} = \text{LLM}_{\text{reflect}}\!\left(\tau_{\text{fail}}, \text{feedback}, \text{task}\right)
$$
4. **情节记忆**：过往反思的滑动窗口缓冲区 $\mathcal{M} = [r_1, r_2, \ldots, r_m]$（通常 $m \leq 3$ 以适应上下文）。
5. **重试循环**：下一次尝试时，反思被注入到 Prompt 中：
$$
a_{t+1} \sim \pi\!\left(\cdot\; |\; \text{task},\; \mathcal{M},\; \text{current\_state}\right)
$$

**反思示例**：*“在我上一次尝试中，我在验证输入格式之前就调用了搜索 API，导致了 400 错误。下次我应该先验证 JSON Schema，然后再发起 API 调用。”*

> **Reflexion：带记忆注入的 Agent Prompt**
>
> ```python
> # === 第 2 次尝试的 PROMPT（首次失败后）===
>
> SYSTEM = """You are a coding agent. You can run bash commands and edit files.
> Complete the task below. Learn from your previous reflections."""
>
> USER = """Task: Fix the failing test in auth_service.py
>
> === REFLECTIONS FROM PREVIOUS ATTEMPTS ===
> [Attempt 1 reflection]: I tried to modify the authenticate() function
> directly but forgot that it depends on token_validator(). The test
> failed because token_validator() was still returning the old format.
> I should trace the dependency chain FIRST: check what authenticate()
> calls, then fix the root cause (token_validator), not the symptom.
> === END REFLECTIONS ===
>
> The repository is in /workspace/. The failing test is:
>   test_auth.py::test_expired_token_returns_401
>
> Begin by reading the relevant files, then fix the issue."""
> ```

**优势与局限**：

| **优势** | **局限** |
| --- | --- |
| 零梯度计算；可与冻结的 API 模型（GPT-4）一起工作 | 受限于上下文窗口；无法积累无限知识 |
| 快速迭代（每次重试只需几秒，而 RL 训练需数小时） | 无法泛化到未见任务（记忆是任务特定的） |
| 可解释：人类可读的自我纠正 | 依赖模型已有的识别错误的能力 |
| 可与任意基础 Agent 架构组合 | 当基础模型太弱、无法生成有用批评时，效果下降 |

### ReAct：推理 + 行动（详解）

ReAct [yao2023react] 通过在单个生成流中**交错显式推理步骤与环境动作**，确立了工具使用 Agent 的主导 Prompt 范式。

> **ReAct 系统 Prompt（典型）**
>
> ```python
> SYSTEM = """You are a research assistant with access to the following tools:
> - search(query): Search the web for information
> - lookup(term): Look up a term on the current Wikipedia page
> - finish(answer): Submit your final answer
>
> Solve the user's question by interleaving Thought and Action steps.
> Format:
>   Thought: <your reasoning about what to do next>
>   Action: <tool_name>(arguments)
>
> You will receive an Observation after each Action. Use it to plan
> your next step. Continue until you can provide a final answer.
>
> Rules:
> - Always think before acting
> - Never guess - verify with tools
> - Use finish() only when you are confident in your answer"""
>
> USER = "Who was the president of the US when the iPhone was released?"
> ```

**生成格式**：

> **ReAct 轨迹格式**
>
> ```python
> Thought: I need to find when the iPhone was released, then determine
> who was president at that time. Let me search for the release date.
> Action: search("iPhone original release date")
> Observation: The original iPhone was released on June 29, 2007.
> Thought: The iPhone came out June 2007. Now I need to find who was
> the US president in June 2007.
> Action: lookup("President of the United States in 2007")
> Observation: George W. Bush served as the 43rd president from
> January 20, 2001 to January 20, 2009.
> Thought: George W. Bush was president from 2001-2009, which covers
> June 2007 when the iPhone was released. I have my answer.
> Action: finish("George W. Bush was the US president when the iPhone
> was released on June 29, 2007.")
> ```

**形式化定义**：一条 ReAct 轨迹是 $\tau = (t_1, a_1, o_1, t_2, a_2, o_2, \ldots)$，其中：

- $t_i$：思考（内部推理，不被执行）
- $a_i$：动作（工具调用，在环境中执行）
- $o_i$：观测（环境响应，附加到上下文）

**为什么有效**：思考创造了一种“内心独白”，帮助模型在行动前规划，减少冲动性工具调用。显式推理轨迹也让 Agent 的决策过程**可审计**且**可调试**。

**用 RL 训练 ReAct Agent**：

- **动作级 reward**：只有动作接收 reward 信号（思考是辅助的）。
- **思考质量**：隐式优化——更好的思考 $\rightarrow$ 更好的动作 $\rightarrow$ 更高的 reward。
- **格式强制**：在 reward 中加入对格式错误动作（缺失 JSON、幻觉工具）的格式惩罚。
- **RL 目标**：$r(\tau) = r_{\text{task}} - \lambda_{\text{format}} \cdot \text{format\_violations} - \lambda_{\text{length}} \cdot \text{num\_steps}$

### LATS：语言 Agent 树搜索（详解）

LATS [zhou2024lats] 将**蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）**应用于 LLM Agent 的动作选择，以推理算力换取显著更好的轨迹。

**算法（针对 LLM Agent 改造）**：

1. **选择**：从根节点（初始状态）开始，使用 UCB1 遍历树：
$$
\text{UCB}(s, a) = \bar{Q}(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}
$$
其中 $\bar{Q}$ = 子树平均 reward，$N$ = 访问计数，$c$ = 探索常数。
2. **扩展**：在叶节点，通过 LLM 采样（温度 $> 0$）生成 $k$ 个候选动作：$\{a_1, \ldots, a_k\} \sim \pi_\theta(\cdot | s_{\text{leaf}})$
3. **模拟**：对每个候选，在环境中执行该动作，然后用快速 rollout 策略（贪心解码）继续，直到终止状态或深度限制。
4. **反向传播**：将终端 reward 沿所有祖先节点向上传播，更新 $\bar{Q}$ 和 $N$ 计数。
5. **重复**：在固定计算预算（如 50--200 次迭代）下运行步骤 1--4。
6. **动作选择**：选择根节点访问次数最多的子节点。

**LLM 特定的改造**：

- **Value 函数**：使用独立的 LLM 调用来估计状态价值：“在 0--1 范围内，这个状态导致任务成功的可能性有多大？”
- **基于反思的剪枝**：当某个分支失败时，生成反思并剪枝相似分支。
- **缓存**：在每个节点存储 LLM 输出，避免回溯时重复生成。
- **深度预算**：将树深度限制在 10--20 步（Agent 很少需要更多）。

**性能**：在 WebShop（网页导航）上，LATS 达到 75% 成功率，而 ReAct 仅为 40%。在 HumanEval（代码）上，借助树搜索 pass@1 从 68% 提升至 94%。代价：每个任务的推理 FLOPs 增加 10--50$\times$。

> **LATS Prompt：价值估计与节点扩展**
>
> ```python
> # === 价值估计 PROMPT（用于模拟阶段）===
> VALUE_PROMPT = """You are evaluating an agent's progress on a task.
>
> Task: Book a flight from NYC to London for under \$500, departing Dec 15.
>
> Current state (after 3 actions):
> - Searched flights on Kayak: found 12 results
> - Filtered by price < \$500: 4 options remain
> - Clicked on British Airways \$489 option: viewing details page
>
> On a scale of 0.0 to 1.0, how likely is the agent to successfully
> complete the task from this state? Consider:
> - How close is the agent to the goal?
> - Are there remaining obstacles (payment, seat selection)?
> - Has the agent made any errors that need correction?
>
> Score: """  # 模型输出，例如 "0.75"
>
> # === 节点扩展 PROMPT（生成候选动作）===
> EXPAND_PROMPT = """You are a web navigation agent. Given the current
> page state, propose 3 DIFFERENT next actions to try.
>
> Current page: British Airways booking - flight details
>   Price: \$489 | Departure: Dec 15 8:30am | Arrival: Dec 15 8:45pm
>   [Button: Select] [Button: Back to results] [Link: Fare rules]
>
> Generate 3 diverse candidate actions (explore different strategies):
> Action 1:"""  # 模型生成 3 个选项用于树扩展
> ```

### AgentQ：在 Agent 轨迹上做 DPO（详解）

AgentQ [putta2024agentq] 通过从轨迹结果自动生成偏好对，桥接了**离线偏好学习（DPO）**与**在线 Agent 执行**。

**流水线**：

1. **Rollout**：使用当前 Policy $\pi_\theta$ 为每个任务执行 $N$ 条轨迹。
2. **评估**：用基于执行的 reward（二元通过/失败或标量指标）为每条轨迹打分。
3. **偏好对构造**：对每个任务，构造偏好对：
$$
(\tau_w, \tau_l) \text{ where } r(\tau_w) > r(\tau_l)
$$
在同一任务的轨迹中，reward 最高者 = 被选（chosen），最低者 = 被拒（rejected）。
4. **DPO 更新**：在轨迹级对数概率上应用标准 DPO Loss：
$$
\mathcal{L}_{\text{AgentQ}} = -\log \sigma\!\left(\beta \left[\log\frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \log\frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)}\right]\right)
$$
5. **迭代**：更新后的 $\pi_\theta$ 在下一轮生成更好的轨迹。

**关键设计选择**：

- **MCTS 引导的探索**：在 rollout 阶段使用 LATS 生成多样、高质量的轨迹（更好的训练数据）。
- **步骤级 DPO**：不比较完整轨迹，而是在*动作级别*比较——给定相同前缀，哪个下一动作导向成功？
- **自对弈改进**：每次 DPO 迭代都产生更好的 Policy，生成更好的轨迹，从而产生更好的训练对——一个良性循环。

**结果**：在 WebShop 上，AgentQ 在 3 次 DPO 迭代后将成功率从 50% 绝对提升至 82%。

### Voyager：通过技能库实现终身学习（详解）

Voyager [wang2023voyager] 引入了**组合式技能积累**——Agent 构建一个不断增长的可复用代码函数库，作为高层动作。

**架构**：

1. **自动课程**：一个 LLM 根据 Agent 当前的技能清单提议逐渐更难的任务：“你现在能挖木头并合成木板。下一个挑战：建造一个工作台。”
2. **技能生成**：对每个任务，Agent 编写一个 JavaScript 函数（可执行代码）来解决它：
$$
\text{skill}_i = \text{LLM}(\text{task}_i, \text{environment\_docs}, \text{error\_feedback})
$$
3. **验证**：在环境中执行代码。如果成功，加入技能库。若失败，借助错误反馈迭代（最多 5 次重试）。
4. **技能库**（向量数据库）：每个已验证技能存储：

  - 函数签名 + Docstring（用于检索）
  - 任务描述的 Embedding（用于语义搜索）
  - 依赖（它调用了哪些其他技能）
5. **检索 + 组合**：对新任务，检索 top-$k$ 最相关技能并加以组合：
$$
\text{solution} = \text{LLM}(\text{new\_task}, \text{retrieve}(\text{skill\_library}, k{=}5))
$$

**关键洞见**：技能是**可组合的**——复杂行为从组合简单的、已验证的函数中涌现。Agent 永不遗忘（库是持久的）并单调改进（只加入已验证的技能）。

> **Voyager：课程与技能生成 Prompt**
>
> ```python
> # === 自动课程 PROMPT ===
> CURRICULUM_PROMPT = """You are a curriculum designer for an AI agent.
>
> Agent's current skill inventory:
> - mine_wood(): Mines nearby oak/birch trees
> - craft_planks(): Converts logs to planks
> - craft_sticks(): Converts planks to sticks
> - mine_stone(): Mines stone with wooden pickaxe
>
> Propose the next task that:
> 1. Builds on existing skills (reachable from current abilities)
> 2. Introduces exactly ONE new concept or challenge
> 3. Is concrete and verifiable (clear success condition)
>
> Next task proposal:"""
> # 输出："Craft a furnace (requires 8 cobblestone blocks arranged
> #       in a square). You already know mine_stone()."
>
> # === 技能生成 PROMPT ===
> SKILL_GEN_PROMPT = """Write a JavaScript function to accomplish this task
> in Minecraft. Use the bot API (bot.dig, bot.craft, bot.equip, etc.)
>
> Task: Smelt 5 iron ingots using a furnace.
> Prerequisites available: mine_stone(), craft_furnace(), mine_iron_ore()
>
> Error from previous attempt: "Cannot smelt without fuel in furnace"
>
> Write the corrected function:
> async function smeltIronIngots(bot, count=5) {"""
> ```

### RLEF：从执行反馈中做 RL（详解）

RLEF [le2024rlef] 将**带确定性执行 reward 的在线 RL**应用于代码生成 Agent，确立了 Agent 训练最简单的有效范式。

**训练循环**：

1. **采样任务**：从训练集中抽取一个带测试用例的编程问题 $(x, \text{tests})$。
2. **生成**：Agent 使用当前 Policy $\pi_\theta$ 产生一条解决轨迹（读文件、写代码、运行测试）。
3. **执行**：在沙盒环境中运行测试套件。Reward：
$$
r = \frac{\text{\# tests passed}}{\text{\# total tests}} \in [0, 1]
$$
4. **更新**：使用 $r$ 作为 reward 信号应用 GRPO/PPO。
5. **重复**：用新鲜任务进行数千次迭代。

**为什么执行反馈对 RL 是理想的**：

- **零噪声**：与人类偏好不同，测试结果是确定的。相同代码 $\rightarrow$ 每次相同 reward。这消除了破坏 RL 训练稳定性的 reward 噪声。
- **无限规模**：可以编程式地生成无限任务（随机算法、API 集成测试、数据变换）。
- **无 reward hacking**：与学习得到的 Reward Model 不同，测试套件无法被“愚弄”（前提是测试写得好）。Agent 必须真正解决问题。
- **密集信号**：部分测试通过（$r = 0.6$）提供比二元通过/失败更丰富的梯度。

### OpenHands / SWE-Agent：用于软件工程的 GRPO

OpenHands [wang2024openhands] 和 SWE-Agent [yang2024sweagent] 应用 GRPO 训练自主解决 GitHub Issue 的 Agent——读取代码、编写补丁、运行测试套件。

**训练细节**：

- **环境**：包含完整仓库、测试套件和开发者工具（git、grep、lint）的 Docker 容器。
- **动作空间**：Bash 命令、文件编辑、git 操作、测试执行。
- **轨迹长度**：解决一个 GitHub Issue 通常需要 15--50 个动作。
- **Reward**：二元——生成的补丁是否通过该 Issue 的回归测试？
- **组大小**：每个 Issue $N = 8$--$16$ 条轨迹用于 GRPO 归一化。
- **课程**：从标记为“good first issue”的开始，逐步过渡到复杂的多文件重构。

**SOTA 结果**：SWE-bench Verified：RL 训练后解决率从 30% 提升至 55%（对比仅 SFT 基线）。

> **OpenHands / SWE-Agent：系统 Prompt**
>
> ```python
> SYSTEM = """You are an autonomous software engineer. You are given a
> GitHub issue to resolve. You have access to the full repository in
> /workspace/ and can execute any bash command.
>
> AVAILABLE COMMANDS:
> - bash(command): Execute a shell command
> - edit(file, start_line, end_line, new_content): Edit a file
> - search(pattern, path): Search for text in files
> - submit(): Submit your patch when done
>
> WORKFLOW:
> 1. Read the issue carefully and understand the expected behavior
> 2. Explore the codebase to find relevant files
> 3. Reproduce the bug (write/run a test that fails)
> 4. Implement the fix
> 5. Verify the fix (run the test again - must pass)
> 6. Run the full test suite to check for regressions
> 7. Submit when all tests pass
>
> RULES:
> - Do NOT modify test files unless the issue explicitly asks for it
> - Prefer minimal, targeted changes over large refactors
> - Always verify your fix before submitting"""
>
> USER = """GitHub Issue #4521: `DataFrame.merge()` silently drops
> rows when `on` column contains NaN values.
>
> Expected: NaN keys should be preserved (matched with other NaN rows)
> Actual: Rows with NaN keys are dropped entirely
>
> Repository: /workspace/pandas-dev/pandas/"""
> ```

> **未来：RL + Agent**
>
> 该领域正在汇聚于一种模式：将**带基于执行 reward 的在线 RL**应用于多步 Agent 轨迹。关键趋势：
>
> - GRPO/PPO 配合来自代码执行或工具成功的二元通过/失败 reward
> - 课程学习：从简单任务开始，逐步增加难度
> - 轨迹级优化（而非 Token 级）——仅在多步序列末尾给出 reward
> - 混合方法：对罕见任务使用检索（非参数化）+ 对常见任务使用 RL（参数化）
> - Scaling Law：推理时投入更多算力（搜索/重试）通常胜过投入更多训练算力


## 用例：用于生产力副驾的 Agent RL

本节提供了一个完整蓝图，展示如何应用 Agent RL 技术来训练一个跨生产力应用套件（文档、电子表格、演示文稿、电子邮件、消息、云存储）运行的基于 LLM 的副驾。

### 架构概览


![生产力副驾架构：LLM Agent（带 RL Policy $\pi_\theta$）接收用户意图并与多个应用 API 交互。基于任务成功、用户反馈和效率指标的 reward 信号驱动 Policy 改进。](/figures/fig_043_fig43.png)

### 生产力副驾的形式化 MDP 定义

生产力副驾的环境被形式化为一个部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）：

$$
\boxed{\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle}
$$

- $\mathcal{S}$：**状态空间**——完整的工作区环境状态：文档内容、邮件会话、日历事件、文件系统、用户权限。*不完全可观测*：Agent 只能看到 API 查询返回的内容。
- $\mathcal{A}$：**动作空间**——结构化的 API 调用（见下文）。每个动作是一个 JSON 对象，指定目标应用、操作和参数。
- $\mathcal{T}$：**转移函数**——大多数操作是确定性的（写入文档 $\rightarrow$ 文档更新），但依赖网络的动作（邮件投递时间、Teams 可用性）是随机的。
- $\mathcal{R}$：**Reward 函数**——多组件（见 Reward 设计章节）。
- $\Omega$：**观测空间**——API 响应、渲染后的文档视图、错误消息。
- $\mathcal{O}$：**观测函数**——将状态映射到观测（API 响应格式化、为适应上下文窗口限制而截断）。
- $\gamma = 0.99$：折扣因子（长视野，通常 10--50 步）。

> **具体示例：“总结上周 Project Alpha 邮件并创建一张状态幻灯片”**
>
> 下面我们追踪一个完整的 POMDP Episode，将每个形式化元素映射到具体实现。
>
> **用户请求**：“总结上周 Project Alpha 邮件中的关键决策，并在团队幻灯片中添加一张状态更新幻灯片。”
>
> | **符号** | **具体实现** |
> | --- | --- |
> | $s_0$ | *真实状态*：收件箱中有 47 封来自过去 7 天提及 “Project Alpha” 的邮件；SharePoint 上的 PowerPoint 文件 “Q3_Status.pptx” 有 12 张幻灯片；用户拥有编辑权限。 |
> | $o_0$ | *Agent 观测*：用户请求文本 + 系统 Prompt。Agent 此时*不知道*邮件数量或幻灯片内容。 |
> | $a_1$ | `outlook.search({query: "Project Alpha", last_7_days: true})` |
> | $\mathcal{T}(s_0, a_1)$ | $s_1$：系统检索出 47 封匹配邮件（确定性）。 |
> | $o_1$ | API 返回前 10 封邮件主题 + 发件人 + 时间戳（因上下文限制被截断——$\mathcal{O}$ 在起作用）。 |
> | $a_2$ | `outlook.read({ids: [top_5_thread_ids]})`——Agent 挑选最相关的会话。 |
> | $o_2$ | 5 个邮件会话的完整正文（$\mathcal{O}$ 截断后约 8K Token）。 |
> | $a_3$ | *内部推理动作*：Agent 综合关键决策：(1) 截止日期推迟到 11 月 15 日，(2) 预算已批准，(3) 供应商已选定。 |
> | $a_4$ | `sharepoint.read({file: "Q3_Status.pptx", slides: "last"})`——检查当前最后一张幻灯片。 |
> | $o_4$ | 最后一张幻灯片是 “Q2 Summary”（第 12 张）。Agent 决定添加第 13 张。 |
> | $a_5$ | `powerpoint.add_slide({file: "Q3_Status.pptx", position: 13, layout: "Title and Content", title: "Project Alpha --- Week 42 Status", content: "Key decisions: 1) Deadline: Nov 15..."})` |
> | $\mathcal{T}(s_4, a_5)$ | $s_5$：幻灯片添加到演示文稿（确定性）。 |
> | $o_5$ | API 返回 `{success: true, slide_id: 13}`。 |
> | $R(s_5)$ | Reward 组件：**+0.4** 任务完成（幻灯片已创建），**+0.3** 信息质量（提取的决策正确），**+0.2** 格式合规（使用正确布局），**+0.05** 效率（5 个动作，无错误），**-0.0** 安全惩罚。**总分：0.95**。 |
>
> **所体现的 POMDP 关键方面**：
>
> - **部分可观测性**：在 $t=0$ 时，Agent 不知道存在多少邮件或演示文稿包含什么——它必须查询以发现状态。
> - **观测函数 $\mathcal{O}$**：由于上下文窗口限制，API 返回截断结果（47 封邮件中的前 10 封）。Agent 看到的是真实状态的一个*投影*。
> - **随机转移**：如果 Agent 改用 `teams.send_message()`，投递时机将是不确定的（接收者在线/离线）。
> - **多步规划**：Agent 必须跨 2 个应用串联 5 个动作，在邮件摘要和幻灯片内容之间保持一致性。
> - **折扣 $\gamma=0.99$**：对 5 步而言，折扣影响很小（$0.99^5 = 0.95$），但对 50 步任务来说至关重要——鼓励高效解决方案。

### 动作空间设计

动作空间必须是**结构化的、类型安全的且可组合的**：

> **生产力副驾动作 Schema**
>
> ```python
> {
>   "action_type": "api_call",
>   "target_app": "outlook | excel | word | powerpoint | teams | sharepoint",
>   "operation": "read | write | search | create | delete | modify",
>   "parameters": {
>     "endpoint": "/me/messages?$filter=subject eq 'Project X'",
>     "body": { ... },             // 用于写操作
>     "options": { "top": 10 }     // 分页、过滤
>   },
>   "reasoning": "I need to find relevant emails before summarizing"
> }
> ```

**按应用划分的动作分类**：

| **应用** | **复杂度** | **关键动作** |
| --- | --- | --- |
| **Outlook** | 中 | `search`、`read`、`draft`、`send`、`move`、`flag`、`create_rule` |
| **Excel** | 高 | `read_range`、`write_range`、`insert_formula`、`create_chart`、`pivot_table`、`run_macro` |
| **Word** | 中 | `read_paragraphs`、`insert_text`、`format_section`、`find_replace`、`insert_table` |
| **PowerPoint** | 中 | `add_slide`、`insert_shape`、`set_text`、`set_layout`、`add_image`、`apply_theme` |
| **Teams** | 低 | `send_message`、`create_meeting`、`search_chat`、`add_members`、`post_to_channel` |
| **SharePoint** | 中 | `list_files`、`upload`、`download`、`search`、`create_page`、`set_permissions` |

### 状态表示

Agent 在每一步的观测（上下文窗口）：

$$
o_t = [\text{system\_prompt};\; \text{user\_intent};\; \text{tool\_history}_{1:t-1};\; \text{current\_result}_t]
$$

**上下文预算管理**（对 128K 窗口至关重要）：

- **系统 Prompt**：2K Token（能力、安全规则、输出格式）
- **用户意图 + 对话**：4K Token
- **工具历史**（滑动窗口）：最近 8--12 个动作 + 观测，更早的进行摘要。总计：最多 80K Token。
- **当前观测**：最多 32K Token（大型电子表格、邮件会话）
- **保留**：10K Token 用于 Agent 的推理 + 下一动作生成

**状态压缩策略**：

- **选择性纳入**：只纳入与当前子目标相关的 API 响应（使用辅助的“相关性打分器”）。
- **结构化摘要**：将大型电子表格表示为 Schema + 样本行，而非完整数据。
- **分层记忆**：将完整轨迹存储在外部；将压缩摘要注入上下文。

### Reward 设计：多目标信号

生产力副驾的 Reward 函数必须平衡多个目标：

$$
\boxed{R(\tau) = \alpha_1 R_{\text{task}} + \alpha_2 R_{\text{quality}} + \alpha_3 R_{\text{efficiency}} + \alpha_4 R_{\text{safety}} + \alpha_5 R_{\text{user}}}
$$


**生产力副驾训练的 Reward 组件。**
| **组件** | **权重** | **信号类型** | **定义** |
| --- | --- | --- | --- |
| $R_{\text{task}}$ | 0.40 | 二元/标量 | 任务成功完成（邮件已发送、文档已创建、公式正确） |
| $R_{\text{quality}}$ | 0.25 | LLM judge | 输出质量：格式化、清晰度、内容正确性 |
| $R_{\text{efficiency}}$ | 0.15 | 标量 | 对过多步骤的惩罚：$-0.02 \times (\text{num\_steps} - \text{optimal\_steps})$ |
| $R_{\text{safety}}$ | 0.15 | 二元 | 无不安全动作（未确认即删除、发到错误收件人、权限违规）。任何违规则 $R_{\text{safety}} = 0$。 |
| $R_{\text{user}}$ | 0.05 | 稀疏 | 可用时的显式用户反馈（拇指向上/向下） |

**中间 reward（密集信号）**：

- 成功的 API 调用（200 响应）：+0.05
- 正确的信息检索（通过下游使用验证）：+0.10
- 优雅地从错误中恢复（用修正后的参数重试）：+0.08
- API 错误（4xx/5xx）：--0.03
- 重复的相同动作（循环检测）：--0.10
- 当意图确实模糊时提出澄清问题：+0.05

### 训练流水线：端到端

> **生产力副驾 RL 训练流水线**
>
> **阶段 1：监督微调（基础）**
>
> 1. 收集 50K--200K 条人工示范的生产力任务轨迹（通过遥测、标注员或合成生成）。
> 2. 在 ReAct 格式的（指令，轨迹）对上对基础 LLM 进行 SFT。
> 3. 验证：Agent 应在保留任务上达到 40--60% 的任务完成率。
>
> **阶段 2：模拟环境构建**
>
> 1. 构建一个带有模拟 API 端点、合成邮箱、文档和日历的**沙盒环境**。
> 2. 每个“用户”拥有真实的画像：500+ 封邮件、20+ 份文档、日历事件、Teams 频道。
> 3. 任务生成器：产生多样化的指令--验证对：“把 Alice 关于 Q4 预算的所有邮件移到 `Finance' 文件夹” + 验证函数。
>
> **阶段 3：在线 RL 训练（GRPO）**
>
> 1. 采样任务 Batch（每次迭代 256 个任务）。
> 2. 在沙盒环境中使用 $\pi_\theta$ 为每个任务生成 $N=8$ 条轨迹。
> 3. 执行轨迹，从验证函数收集 reward。
> 4. 计算 GRPO 优势（跨每个任务的 8 条轨迹做组归一化）。
> 5. 用裁剪目标 + 相对于 SFT 模型的 KL 惩罚来更新 Policy。
> 6. 每 500 次迭代：在保留基准上评估（200 个任务，5 个难度级别）。
>
> **阶段 4：人机协同（Human-in-the-Loop）细化**
>
> 1. 部署给内部 dogfood 用户（1000+ 用户，2 周）。
> 2. 收集拇指向上/向下信号 + 自由文本纠正。
> 3. 从 A/B 部署中（旧 Policy 对比新 Policy）构造 DPO 偏好对。
> 4. 在人类偏好上应用 1--2 轮 DPO 微调。

### 模拟环境架构

> **沙盒环境（简化版）**
>
> ```python
> class ProductivityEnvironment:
>     def __init__(self, user_profile: UserProfile):
>         self.mailbox = SyntheticMailbox(user_profile.emails)
>         self.drive = SyntheticOneDrive(user_profile.files)
>         self.calendar = SyntheticCalendar(user_profile.events)
>         self.teams = SyntheticTeams(user_profile.channels)
>         self.step_count = 0
>         self.max_steps = 50
>
>     def step(self, action: dict) -> Tuple[Observation, float, bool]:
>         """执行动作，返回 (observation, reward, done)。"""
>         self.step_count += 1
>
>         # 路由到对应的应用处理器
>         handler = self.get_handler(action["target_app"])
>         try:
>             result = handler.execute(action["operation"], action["parameters"])
>             obs = Observation(status=200, body=result)
>             reward = 0.05  # 成功的 API 调用
>         except APIError as e:
>             obs = Observation(status=e.code, body=str(e))
>             reward = -0.03
>
>         # 检查终止条件
>         done = self.step_count >= self.max_steps
>         return obs, reward, done
>
>     def evaluate(self, task: Task) -> float:
>         """检查任务目标是否达成（终端 reward）。"""
>         return task.verification_fn(self)  # 0.0 或 1.0
> ```

### 任务课程设计

训练有效性关键取决于任务难度的递进：


**生产力副驾课程级别。**
| **级别** | **步数** | **应用数** | **示例任务** |
| --- | --- | --- | --- |
| **L1: 单步** | 1--2 | 1 | “读取我最新的来自 Bob 的邮件”，“A1 单元格里是什么？” |
| **L2: 单应用** | 3--5 | 1 | “起草一封回复预算邮件并总结要点” |
| **L3: 多步** | 5--10 | 1 | “从销售数据创建数据透视表，并将 top 表现者加粗格式化” |
| **L4: 跨应用** | 5--15 | 2--3 | “查找 Q4 预算邮件，提取数字，放入新的 Excel 表” |
| **L5: 复杂工作流** | 10--30 | 3+ | “准备周报：从 Excel 抽取指标，总结邮件更新，创建 PowerPoint 幻灯片，在 Teams 中分享” |

**课程策略**：

- 训练早期从 80% L1--L2 任务、20% L3 开始。
- 当当前级别成功率超过 70% 时晋升下一级。
- 始终保留 10--20% 的较简单任务以防止灾难性遗忘（Catastrophic Forgetting）。
- 收敛后的最终配比：10% L1、15% L2、25% L3、30% L4、20% L5。

### 安全与护栏

> **生产力副驾安全框架**
>
> **硬约束**（动作立即被拒绝，reward = --1.0）：
>
> - 未经用户确认即向外部收件人发送邮件/消息
> - 永久删除文件/邮件（仅允许软删除）
> - 修改共享资源的权限
> - 越权访问其他用户的邮箱或文件
> - 在一个 Batch 中对超过 100 个条目执行动作（防止批量删除/移动事故）
>
> **软约束**（reward 中的惩罚，Agent 应学习避免）：
>
> - 未向用户展示预览即发送草稿：--0.2
> - 未先声明意图即做出不可逆变更：--0.15
> - 访问敏感标签（机密、律师-当事人）：--0.3
> - 在没有显式委托的情况下使用“代发”：--0.25
>
> **确认协议**：对任何被归类为“高影响”（发送、删除、对外分享）的动作，Agent 必须：
>
> 1. 用自然语言陈述所计划的动作
> 2. 显示将要发送/修改内容的预览
> 3. 在执行前等待显式的用户确认
>
> 这在环境层（沙盒拒绝未确认的高影响动作）和 Reward 函数（跳过确认会受罚）两处都被强制执行。

### 多应用工作流中的信用分配

关键挑战：在一个 20 步的跨应用工作流中，哪些步骤对成功或失败做出了贡献？

**方法：分层 Reward 分解**

1. **子目标检测**：将用户指令分解为可验证的子目标：

  - “查找 Q4 预算邮件” $\rightarrow$ 子目标 1（验证：检索到相关邮件）
  - “提取数字” $\rightarrow$ 子目标 2（验证：正确数值已解析）
  - “创建 Excel 表” $\rightarrow$ 子目标 3（验证：表存在且包含正确数据）
2. **子目标 reward**：每完成一个子目标就分配中间 reward（每个 $r = +0.2$）。
3. **轨迹切片**：如果最终任务失败，识别哪个子目标最先失败。仅对该子目标跨度内的动作施加负 reward。
4. **反事实估计**：“如果这个特定动作不同，任务是否会成功？”——使用 Value 函数来估计。

$$
R_{\text{step}}(t) = \underbrace{R_{\text{sub-goal}}(t)}_{\text{当前子目标是否成功？}} + \underbrace{\gamma^{T-t} R_{\text{terminal}}}_{\text{折扣后的最终 reward}} + \underbrace{r_{\text{intermediate}}(t)}_{\text{每步 API 成功/失败}}
$$

### 扩展与基础设施

**算力需求**（针对 70B 参数模型估算）：

| **组件** | **资源** | **说明** |
| --- | --- | --- |
| Policy 模型（70B） | 8$\times$ A100 80GB（TP=8） | BF16，生成轨迹 |
| 参考模型（70B） | 8$\times$ A100 80GB（TP=8） | 冻结，用于 KL 计算 |
| 环境 Worker | 128 个 CPU Worker | 每个运行一个沙盒实例 |
| Reward Model / Judge | 4$\times$ A100（若用 LLM judge） | 或者若使用基于执行的 reward 则为零 |
| 训练（GRPO 更新） | 16$\times$ A100（FSDP） | 跨轨迹 Batch 做梯度累积 |
| **总计** | **40 张 A100 GPU + 128 CPU** | 完整训练运行约 5,000 GPU 小时 |

**吞吐量优化**：

- **异步 rollout**：将轨迹生成与梯度更新解耦。在训练上一个 Batch 时持续生成。
- **批量化环境**：并行运行 128 个沙盒环境，每个处理不同任务。
- **KV-cache 共享**：对每个任务的 $N=8$ 条轨迹，它们共享相同的 Prompt 前缀——使用前缀缓存避免冗余计算。
- **选择性反向传播**：只在动作 Token 上计算梯度（而非观测/系统 Prompt）。将反向传播 FLOPS 减少 40--60%。

### 评估框架


**生产力副驾评估维度。**
| **指标** | **目标** | **度量方式** |
| --- | --- | --- |
| 任务完成率 | $>85\%$（L1--L3），$>60\%$（L4--L5） | 沙盒中的自动化验证 |
| 安全违规率 | $<0.1\%$ | 每 1000 个任务的硬约束违规计数 |
| 完成的平均步数 | 不超过最优值的 $1.5\times$ | 与已知最短成功轨迹对比 |
| 用户满意度（dogfood） | $>4.2/5.0$ | 内部用户任务后调查 |
| 跨应用成功率 | $>55\%$（L4--L5） | 需要 2+ 应用的任务 |
| 恢复率 | $>70\%$ | Agent 成功重试失败 API 调用的百分比 |
| 延迟（首动作时间） | $<3$ 秒 | 模型推理 + 动作规划时间 |

**基准套件**（建议）：

- **ProdBench-Easy**（200 个任务）：单应用，1--3 步。基线建立。
- **ProdBench-Hard**（200 个任务）：跨应用工作流，10--30 步。端到端能力。
- **ProdBench-Safety**（100 个任务）：试图触发不安全动作的对抗 Prompt。必须保持 $<0.1\%$ 违规率。
- **ProdBench-Robustness**（100 个任务）：带模糊指令、注入 API 错误、缺失权限的任务。测试优雅降级。

### 来自生产部署的经验教训

> **生产力 Agent RL 的实践洞见**
>
> 1. **SFT 质量是底线**：RL 只能在 SFT 所提供的基础上改进。如果 SFT 模型无法格式化一个有效的 Graph API 调用，RL 也发现不了。要在阶段 1 的数据质量上重投入。
> 2. **Reward hacking 不可避免**：Agent *一定*会找到捷径。常见例子：
>
>   - 创建一个空 Excel 文件以“完成”电子表格任务（通过存在性检查）
>   - 在没有实际执行动作的情况下回复“完成”
>   - 利用模糊的验证函数
>
> **缓解措施**：多层验证（格式 + 内容 + 语义正确性）。
> 3. **API 速率限制很重要**：在生产中，工作区 API 有限流（429 响应）。用现实的速率限制训练，避免学到滥发并行请求的 Policy。
> 4. **上下文窗口是瓶颈**：带丰富 API 响应的 20 步轨迹轻松消耗 80K+ Token。技术手段：观测摘要、选择性历史、分层上下文管理。
> 5. **用户意图常常是模糊的**：“清理我的收件箱”对不同用户意味着不同的事情。训练 Agent 在不确定性高时提出澄清问题（恰当澄清给奖励，过度澄清给惩罚）。
> 6. **从简单开始，逐步扩展**：从仅 Outlook 任务开始（数量最多，遥测数据最丰富），然后扩展到 Excel，再到跨应用。每个应用都有独特的失败模式。

### 完整训练食谱


**生产力副驾 RL 训练的推荐超参数。**
| **参数** | **取值** | **理由** |
| --- | --- | --- |
| 基础模型 | 70B Llama/Mistral | 足够容量进行复杂多步推理 |
| RL 算法 | GRPO | 无需 critic；对长轨迹更省内存 |
| 组大小 $N$ | 8 | 在方差减小与算力代价之间取平衡 |
| 裁剪 $\epsilon$ | 0.1 | 比标准（0.2）更紧，因长轨迹敏感性 |
| KL 系数 $\beta$ | 0.04 | 对 SFT Policy 的中等约束 |
| Learning rate | $5 \times 10^{-7}$ | 保守；Agent 任务对大幅更新敏感 |
| Batch size | 256 任务 $\times$ 8 轨迹 = 2048 | 大 Batch 以稳定 GRPO 归一化 |
| 最大轨迹长度 | 50 步 | 覆盖 95% 的生产力任务 |
| 上下文窗口 | 128K Token | 长多应用工作流所需 |
| 训练迭代数 | 3000--5000 | 监控评估指标；安全性退化时早停 |
| 课程预热 | 500 次迭代（仅 L1--L2） | 在复杂任务前建立基本 API 使用 |


## 用例：从零构建一个研究 Agent

本用例展示了如何使用本文讨论的技术，构建一个完全自主的**研究 Agent**——一个能够提出假设、检索文献、分析数据、编写代码、运行实验并产出最终报告的 LLM。

### 问题定义

> **研究 Agent 需求**
>
> **输入**：一个研究问题（例如，“Learning rate warmup 时长对 7B 模型 GRPO 收敛的影响是什么？”）
>
> **输出**：一份包含方法论、实验、结果和结论的完整研究报告。
>
> **所需能力**：
>
> 1. **文献搜索**：查询 arXiv、Semantic Scholar，找出相关论文
> 2. **假设生成**：从背景知识中提出可检验的假设
> 3. **实验设计**：编写带恰当对照的训练脚本
> 4. **代码执行**：运行实验，收集指标
> 5. **数据分析**：解析日志、计算统计量、生成图表
> 6. **科学写作**：将发现综合成一份连贯的报告
> 7. **自我纠正**：检测失败实验并用修改后的参数重试

### MDP 形式化

> **研究 Agent MDP**
>
> - **状态** $s_t$：系统 Prompt + 研究问题 + 完整的动作/观测历史（工具输出、代码结果、搜索结果）。上下文窗口：128K Token。
> - **动作** $a_t$：来自动作空间的结构化工具调用（见下文）+ 推理轨迹（CoT）。
> - **转移** $T(s_{t+1}|s_t, a_t)$：确定性——将动作 + 工具输出附加到上下文。
> - **Reward** $R$：基于报告质量的稀疏终端 reward（见下方 Reward 设计）。
> - **视野**：20--100 步（典型研究轨迹）。
> - **折扣** $\gamma = 1.0$（Episode 式；有限任务不折扣）。

### 动作空间


**研究 Agent 的工具/动作空间。**
| **工具** | **类别** | **描述** |
| --- | --- | --- |
| `search_papers` | 文献 | 查询 Semantic Scholar/arXiv。返回标题、摘要、引用。 |
| `read_paper` | 文献 | 获取论文全文或指定章节。 |
| `write_code` | 实验 | 向工作区写入 Python/训练脚本。 |
| `execute_code` | 实验 | 在沙盒环境中运行脚本。返回 stdout/stderr。 |
| `read_file` | 分析 | 读取日志、CSV 或中间结果。 |
| `plot_data` | 分析 | 生成 matplotlib/seaborn 可视化。 |
| `compute_stats` | 分析 | 运行统计检验（t 检验、置信区间）。 |
| `write_report` | 输出 | 撰写最终研究报告的各个章节（LaTeX/Markdown）。 |
| `think` | 推理 | 内部推理步骤（不进行外部工具调用）。 |
| `submit` | 终止 | 提交最终报告。结束 Episode。 |

### 架构：模型与基础设施选择

> **架构决策——应用论文中的概念**
>
> - **基础模型**：Qwen-2.5 72B（强推理 + 代码能力）。QLoRA 微调（$r=32$，所有线性层）——见 LoRA 一节。
> - **推理**：vLLM，TP=4，启用前缀缓存（系统 Prompt 在 rollout 间共享）——见 vLLM 一节。
> - **训练**：每个研究问题用 $N=4$ 条轨迹的 GRPO——无需 Value 模型（见 GRPO 一节）。
> - **硬件**：8$\times$H100 节点。QLoRA 适配器装入 48 GB；vLLM 生成使用剩余容量。
> - **上下文管理**：128K 上下文配合 FlashAttention（见 FlashAttention 一节）。超出上下文的轨迹采用滑动窗口摘要。
> - **推测式解码**：长研究轨迹期间用 Eagle 头进行快速生成（见推测式解码一节）。

### Reward 设计

> **多组件研究 Reward**
>
> 当 Agent 调用 `submit` 时计算终端 reward：
> $$
> R = w_1 R_{quality} + w_2 R_{correctness} + w_3 R_{novelty} + w_4 R_{efficiency} + w_5 R_{format}
> $$
>
> | **组件** | **权重** | **度量方式** |
> | --- | --- | --- |
> | $R_{\text{quality}}$ | 0.30 | LLM-as-judge（GPT-4 在清晰度、深度、严谨性上对报告打 1--10 分） |
> | $R_{\text{correctness}}$ | 0.30 | 代码无错误执行 + 结果可复现 |
> | $R_{\text{novelty}}$ | 0.15 | LLM-judge：报告是否提供了超越论文综述的洞见？ |
> | $R_{\text{efficiency}}$ | 0.15 | 步骤越少奖励越多：$R_{\text{eff}} = \max(0, 1 - \text{steps}/100)$ |
> | $R_{\text{format}}$ | 0.10 | 报告包含所有所需章节（引言、方法、结果、结论） |
>
> **中间塑形**：每次成功代码执行 +0.1；每次运行时错误 $-$0.05（鼓励一次写对代码）。

> **Reward Hacking 风险**
>
> - **伪造结果**：Agent 捏造实验输出。*修复*：通过对照执行日志与报告数字，验证代码确实运行过。
> - **浅薄报告**：Agent 逐字复制论文摘要。*修复*：新颖性 reward + 抄袭检测。
> - **长度博弈**：长报告得分更高。*修复*：效率 reward + 长度惩罚。
> - **回避难题**：Agent 避开困难研究问题。*修复*：带难度级别的课程。

### 训练流水线

1. **阶段 1——SFT 预热**（500 步）：

  - 收集 200 条专家研究轨迹（人类研究者使用工具）
  - 在成功轨迹上用仅 completion 掩码做 SFT（掩盖工具输出）
  - 这教 Agent 工具使用语法和基本研究工作流
2. **阶段 2——GRPO 训练**（3000 步）：

  - Prompt 池：跨 10 个领域（ML、NLP、CV、系统等）的 500 个研究问题
  - 每个问题：生成 $N=4$ 条完整研究轨迹
  - 用多组件 reward 给每条轨迹打分
  - GRPO 优势：$\hat{A}_i = (R_i - \mu_G) / \sigma_G$
  - 用裁剪目标更新 Policy（clip $\epsilon=0.2$，KL $\beta=0.05$）
  - 课程：从简单的“总结关于 X 的发现”任务开始，进展到“设计并运行关于 X 的实验”
3. **阶段 3——拒绝采样微调**（200 步）：

  - 每个难问题生成 16 条轨迹，按 reward 保留 top-2
  - 在这些高质量轨迹上做 SFT
  - 稳定在最难研究任务上的表现

### 示例轨迹：完整 MDP 追踪

为了说明所有 MDP 组件在实践中如何协同工作，我们追踪一个从问题到提交的完整研究 Episode，并为每一步标注其形式化 MDP 元素。

> **研究 Agent 轨迹——完整 MDP 追踪**
>
> **研究问题**：“将 GRPO 组大小增加到 $N=16$ 之上能否改善数学推理表现？”
>
> 下面是一条完整轨迹，每一步都有显式 MDP 符号：状态 $s_t$、动作 $a_t$、转移 $T$、观测 $o_t$ 和累积 reward。
>
> | $t$ | **元素** | **描述** |
> | --- | --- | --- |
> |  | $a_0$ | `think`：“我需要关于 GRPO 组大小消融的论文，然后进行一次受控实验。” |
> |  | $o_0$ | 内部推理（无工具输出）。塑形 reward：$r_0 = 0$。 |
> | 1 | $s_1$ | $s_0 \oplus (a_0, o_0)$——上下文现在包含推理计划。 |
> |  | $a_1$ | `search_papers(query="GRPO group size ablation math")` |
> |  | $o_1$ | 返回 5 篇论文：[DAPO, DeepSeek-Math, Dr. GRPO, ...]。$r_1 = 0$。 |
> | 2 | $s_2$ | $s_1 \oplus (a_1, o_1)$——搜索结果进入上下文。 |
> |  | $a_2$ | `read_paper(id="DAPO", sections=["experiments", "ablations"])` |
> |  | $o_2$ | 关键发现：“在代码任务上超过 $N=16$ 收益递减；数学未测试。”$r_2 = 0$。 |
> | 3 | $s_3$ | $s_2 \oplus (a_2, o_2)$——识别出文献空白。 |
> |  | $a_3$ | `think`：“DAPO 只测了代码。我将在 GSM8K 上用 Llama-3 8B 测试 $N \in \{4, 8, 16, 32, 64\}$。对照：相同总训练 Token 数。” |
> |  | $o_3$ | 假设形成。$r_3 = 0$。 |
> | 4 | $s_4$ | $s_3 \oplus (a_3, o_3)$——实验方案进入上下文。 |
> |  | $a_4$ | `write_code("train_grpo_gsm8k.py")`——85 行脚本，含 group_size 的 argparse、正确的随机种子、在 GSM8K 测试集上评估。 |
> |  | $o_4$ | 文件写入成功。$r_4 = 0$。 |
> | 5 | $s_5$ | $s_4 \oplus (a_4, o_4)$——代码在工作区可用。 |
> |  | $a_5$ | `execute_code("python train_grpo_gsm8k.py --N 4 --seed 42")` |
> |  | $o_5$ | `Accuracy: 67.2% (CI: 65.1--69.3)`。$r_5 = +0.1$（成功执行）。 |
> | 6--8 | $a_{6\text{--}8}$ | 用 $N=8, 16, 32$ 执行 $\to$ 结果：71.4%、74.8%、75.9%。每个 $r_t = +0.1$。 |
> | 9 | $s_9$ | 上下文包含 4 个数据点。 |
> |  | $a_9$ | `execute_code("python train_grpo_gsm8k.py --N 64 --seed 42")` |
> |  | $o_9$ | `Accuracy: 75.6% (CI: 73.4--77.8)`——比 $N=32$ 略*下降*！$r_9 = +0.1$。 |
> | 10 | $s_{10}$ | 5 个结果全部可用。 |
> |  | $a_{10}$ | `compute_stats(test="paired_t_test", data=[67.2, 71.4, 74.8, 75.9, 75.6], groups=[4,8,16,32,64])` |
> |  | $o_{10}$ | $N=4$ 对比 $N=16$ 时 $p < 0.01$；$N=32$ 对比 $N=64$ 时 $p = 0.72$（不显著）。$r_{10} = 0$。 |
> | 11 | $s_{11}$ | 收益递减的统计证据。 |
> |  | $a_{11}$ | `plot_data(x=[4,8,16,32,64], y=[67.2,71.4,74.8,75.9,75.6], xlabel="Group Size N", ylabel="GSM8K Accuracy (%)", title="GRPO Group Size vs Math Reasoning")` |
> |  | $o_{11}$ | 图保存到 `results/group_size_curve.png`。$r_{11} = 0$。 |
> | 12 | $s_{12}$ | 完整证据组装完成。 |
> |  | $a_{12}$ | `write_report`：4 节报告（引言、方法、带图的结果、结论：“$N=16$--32 是最优；$N=64$ 无增益且算力增加 4$\times$”）。 |
> |  | $o_{12}$ | 报告写入完成（2,100 词）。$r_{12} = 0$。 |
> | 13 | $a_{13}$ | `submit`——Episode 终止。 |
> |  | $R_{\text{terminal}}$ | LLM-judge 打分：质量 = 8/10，代码正确，新颖（把 DAPO 扩展到数学），13 步，所有章节齐全。 |
>
> **终端 reward 计算**：
> $$
> R = \underbrace{0.30 \times \tfrac{8}{10}}_{\text{质量}} + \underbrace{0.30 \times 1.0}_{\text{正确}} + \underbrace{0.15 \times \tfrac{7}{10}}_{\text{新颖}} + \underbrace{0.15 \times (1 - \tfrac{13}{100})}_{\text{效率}} + \underbrace{0.10 \times 1.0}_{\text{格式}} \\
> = 0.24 + 0.30 + 0.105 + 0.13 + 0.10 = \mathbf{0.875}
> $$
>
> **中间塑形总额**：$5 \times (+0.1) = +0.5$（5 次成功代码执行）。
>
> **GRPO 上下文**：此轨迹在 $N=4$ 组中得分最高（其他得分为 0.61、0.72、0.53）。GRPO 优势：
> $$
> \hat{A} = \frac{0.875 - \bar{R}}{\sigma_R} = \frac{0.875 - 0.684}{0.129} = +1.48 \quad （强化强烈）
> $$
>
> **所体现的关键 MDP 性质**：
>
> - **确定性 $T$**：每次工具调用产生可预测的状态扩展（$s_{t+1} = s_t \oplus (a_t, o_t)$）。
> - **稀疏终端 reward**：真正的质量信号只在 `submit` 时给出；中间塑形很小。
> - **长视野**：13 步且 $\gamma = 1.0$（Episode 式任务不折扣）。
> - **自我纠正机会**：在第 9 步，Agent 观察到 $N=64$ 并未改进——据此调整其结论，而非挑选数据。
> - **动作多样性**：推理（`think`）、信息收集（`search`、`read`）、执行（`write_code`、`execute`）、分析（`compute_stats`、`plot`）和输出（`write_report`、`submit`）的混合。

### 关键设计决策与权衡


**研究 Agent 的设计决策，映射到论文各章节。**
| **决策** | **论文章节** | **理由** |
| --- | --- | --- |
| QLoRA（$r=32$） | LoRA 一节 | 72B 模型；全量微调太昂贵。$r=32$ 用于复杂推理。 |
| GRPO（而非 PPO） | GRPO 一节 | 无需 Value 模型；研究质量难以中途预测。 |
| 稀疏终端 reward | Reward Shaping | 研究质量只能在完成时度量；中间塑形最小。 |
| $N=4$ 轨迹 | GRPO 组大小 | 平衡：足够的多样性用于排序，又不至过于昂贵（100 步轨迹）。 |
| 128K 上下文 | FlashAttention | 长轨迹包含论文内容 + 代码 + 结果。 |
| vLLM + 前缀缓存 | vLLM 一节 | 系统 Prompt + 研究问题在 4 条 rollout 间共享。 |
| 课程训练 | Agent RL | 从简单（文献综述）$\to$ 困难（设计 + 执行实验）。 |
| LLM-as-judge reward | Reward Model | 研究质量是主观的；LLM judge 比基于规则更灵活。 |

### 评估

> **研究 Agent 评估框架**
>
> - **保留问题**（50 个）：训练期间未见的研究问题，覆盖多种领域。
> - **人类评估**：领域专家在 1--5 分制（质量、正确性、可操作性）上对报告评分。
> - **可复现性**：重新运行报告中的 Agent 代码；验证结果一致。
> - **对比基线**：(1) 零样本 GPT-4 + 工具（无 RL 训练），(2) 仅 SFT Agent，(3) 人类研究者。
> - **效率指标**：按任务难度归一化的完成步数。
>
> **预期结果**（基于类似的 Agent RL 工作）：
>
> | **Agent** | **报告质量（1--5）** | **平均步数** |
> | --- | --- | --- |
> | 零样本 GPT-4 + 工具 | 2.8 | 25 |
> | 仅 SFT | 3.4 | 18 |
> | GRPO 训练（我们的） | 4.1 | 14 |
> | 人类研究者 | 4.5 | N/A |

### 经验教训与失败模式

> **研究 Agent 训练中的常见失败**
>
> - **无限循环**：Agent 反复搜索论文却不前进。*修复*：步数预算 + 对参数相同的重复工具调用的惩罚。
> - **代码调试螺旋**：Agent 花 20+ 步修一个 Bug。*修复*：重试次数上限为 3；若代码失败 3 次，放弃当前方法尝试替代方案。
> - **幻觉引用**：Agent 编造论文标题/结果。*修复*：通过工具输出验证所有引用存在；惩罚不可验证的断言。
> - **过早提交**：Agent 提交不完整报告以避免长轨迹的惩罚。*修复*：设定最低质量阈值（$R > 0.4$）作为有效提交；低于阈值视为失败。
> - **对 judge 的 reward hacking**：Agent 学会产生在 LLM judge 处得高分但科学上肤浅的文本。*修复*：轮换 judge 模型；定期在 reward 中纳入人类评估。

## LLM Agent 的 SOTA RL

针对 LLM Agent 的 RL 技术聚焦于**在策略 policy 梯度**与**细粒度信用分配**的结合。因为 Agent 执行涉及工具交互、API 查询和代码执行的复杂多轮（Multi-Turn）轨迹，标准的单轮对齐算法必须经过大幅修改。

### 主导基线：用于 Agent 的 GRPO

由 DeepSeek-R1 [deepseek2025r1] 推广，**GRPO** [shao2024deepseekmath] 正迅速成为 Agent 训练的标准。它为每个任务采样一组 $N$ 条完整轨迹，从而消除了内存密集的 critic 网络：

对于任务 Prompt $q$，GRPO 从 $\pi_{\theta_{\text{old}}}$ 采样 $N$ 条 Agent 轨迹 $\{o_1, o_2, \dots, o_N\}$。每条轨迹的优势通过将其 reward 相对于组归一化来计算：
$$
\boxed{A_i = \frac{r(o_i) - \frac{1}{N}\sum_{j=1}^N r(o_j)}{\text{std}(r(o_1), \dots, r(o_N))}}
$$

带 KL 正则的 GRPO 目标：
$$
L_{\text{GRPO}}(\theta) = \frac{1}{N} \sum_{i=1}^N \min\!\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,\; \text{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1{-}\epsilon, 1{+}\epsilon\right) A_i \right) - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

> **为什么 GRPO 在 Agent 训练中占优**
>
> - **无 critic**：节省 50% GPU 内存——当 Agent 轨迹已经占用了庞大的上下文窗口（32K--128K Token）时至关重要。
> - **自然契合**：Agent 任务通常有二元可验证 reward（测试通过/失败、目标达成/未达成）——非常适合组相对归一化。
> - **探索**：每个任务采样 $N$ 条多样化轨迹，自然地探索不同的工具使用策略。

### 用于交互式 Agent 的 PPO

对于在高度随机环境中运行、且步骤级 Value 估计有所帮助的 Agent，**PPO** [schulman2017proximal] 仍然有价值。Critic 提供每步优势信号，在工具输出不可预测时实现更精细的信用分配：

- 通过 GAE 进行步骤级优势估计，处理可变长度的工具输出
- Value 头学习预测“从这里开始这条轨迹有多大可能成功”
- 当外部工具返回灾难性错误导致 reward 方差激增时更稳定
- 权衡：需要 $2\times$ 内存（critic）但提供更密集的学习信号

### 细粒度的回合级信用分配

Agent RL 的核心挑战是**稀疏 reward 问题**。如果 Agent 执行 20 个工具动作并最终未通过一个单元测试，终端 reward 为 $0$ 会平等地惩罚所有 20 个动作。现代解决方案：

> **可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）**
>
> 在**确定性的中间检查点**对模型给予 reward：
>
> - Bash 命令成功编译 $\rightarrow$ +0.1
> - 浏览器 Agent 定位到正确的 HTML 元素 $\rightarrow$ +0.2
> - SQL 查询返回非空结果 $\rightarrow$ +0.1
> - 最终测试套件通过 $\rightarrow$ +1.0（终端）
>
> 中间 reward 为*每一*步都提供梯度信号，而不仅是最后一步。这相比仅有稀疏 reward 的情况，将学习速度显著加速 3--5$\times$。

> **多轮轨迹切片**
>
> 框架将多轮 Agent 运行拆分为独立的单个步骤。一个信用分配模块隔离出**破坏轨迹的确切子步骤**：
>
> 1. 重放成功前缀（步骤 1--$k$）
> 2. 识别第一个偏离点（步骤 $k+1$，出错处）
> 3. 仅对该特定步骤分配负 reward
> 4. 对正确的前缀步骤分配中性/正 reward
>
> 这使得在不破坏已正确行为的前提下，进行外科手术式的 Policy 更新成为可能。

### 替代范式

- **迭代式 STaR（Self-Taught Reasoner）** [zelikman2022star]：不进行连续 RL，而是使用迭代离线循环。生成轨迹 $\rightarrow$ 过滤失败 $\rightarrow$ 在成功上 SFT $\rightarrow$ 重复。易于扩展，避免 RL 不稳定性。每次迭代都自举推理能力。
- **强化世界模型学习（Reinforcement World Model Learning, RWML）** [yu2026rwml]：为对抗 reward hacking，训练 Agent 预测其动作的*语义后果*。Agent 因准确预测环境状态如何变化而获得辅助 reward（例如，在执行 SQL 之前预测数据库表的变化）。这迫使真正理解，而非表面的 reward 博弈。
- **LATS（语言 Agent 树搜索）** [zhou2024lats]：在 Agent 动作序列上应用蒙特卡洛树搜索。在每一步，扩展多个候选动作，模拟其结果，并通过树反向传播 reward。将 RL 价值估计与搜索时算力扩展相结合。

### 核心方法论比较


**LLM Agent 的 RL 范式比较。**
| **方法** | **Reward 密度** | **内存成本** | **主要优势** |
| --- | --- | --- | --- |
| **GRPO** [shao2024deepseekmath] | 序列 / 最终指标 | 低（无 critic） | 显著降低 GPU 内存；实现简单 |
| **PPO** [schulman2017proximal] | 逐步（GAE） | 高（需 critic） | 细粒度信用分配；在噪声环境中稳定 |
| **迭代式 STaR** [zelikman2022star] | 稀疏（过滤后的二元） | 极小（仅 SFT） | 易于扩展；避免 RL 优化不稳定性 |
| **RWML** [yu2026rwml] | 密集（预测式） | 中 | 通过世界建模缓解 reward hacking |
| **LATS** [zhou2024lats] | 反向传播 | 高（树扩展） | 每个任务的质量最佳；随推理算力扩展 |
