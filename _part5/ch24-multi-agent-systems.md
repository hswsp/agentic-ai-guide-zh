---
layout: home
title: 多智能体系统（Multi-Agent Systems）
permalink: /part5/ch24-multi-agent-systems.html
---

# 多智能体系统（Multi-Agent Systems）

## 动机：为什么需要多个 Agent？

人工智能的历史在很多方面就是一部规模演进史。早期 AI 系统是单体式的：单一程序、单一知识库、单一推理引擎。随着问题日益复杂，研究者发现，没有任何单一 Agent——无论能力多强——能够高效地处理一个丰富、开放式任务的方方面面。这一洞见在分布式 AI 与多智能体系统（Multi-Agent System, MAS）研究中早有定论 [weiss1999multiagent, wooldridge2009introduction]，而在大语言模型时代又被赋予了新的紧迫性。

> **核心直觉**
>
> 单个 LLM 无论多大，本质上都是一个通才。一支由专门化 LLM 组成的团队，每个 LLM 聚焦于一个狭窄的子问题并交流结果，在复杂的多维度任务上能够胜过通才——正如一支人类专家团队在复杂工程项目上胜过单个通才。

从单体 Agent 转向*Agent 社会*（agent societies）的根本动机有四个：

**专门化（Specialization）。**

不同的子任务受益于不同的能力、Prompt 策略，乃至不同的基础模型。代码生成 Agent 可以在编程语料上微调；事实核查 Agent 可以借助检索工具锚定事实；创意写作 Agent 可以通过 Prompt 激发风格多样性。强迫单个 Agent 在所有这些方面同时出色，既低效，往往也不可能。

**并行化（Parallelism）。**

许多现实世界的任务可以分解为可并发执行的独立子任务。一个需要文献综述、数据分析和报告撰写的研究流水线可以让这三者并行进行，从而显著降低墙钟时间。串行的单 Agent 处理是一个瓶颈，而多 Agent 并行化正好消除了它。

**鲁棒性（Robustness）。**

单个 Agent 是单点故障。如果它产生幻觉、陷入循环或给出细微错误的答案，将无人核验。多 Agent 系统引入了冗余：第二个 Agent 可以验证、批判，或独立地重新推导结果。在输出被信任之前，对抗性 Agent 可以主动探测其弱点。

**涌现能力（Emergent Capabilities）。**

也许最引人入胜的是，Agent 群体可以展现出任何单一 Agent 都不具备的能力。通过辩论、协商和迭代精化，多 Agent 系统能够得到超越任一单独 Agent 所能产出的解决方案——这是一种社会性生物涌现智能的计算类比。

> **历史脉络**
>
> 多智能体系统研究可以追溯到 1980 年代，其奠基性工作包括分布式问题求解 [durfee1989distributed]、合同网协议（Contract Net Protocol） [smith1980contract] 以及 FIPA Agent 通信标准 [fipa2002acl]。向基于 LLM 的 Agent 转变让这些经典思想在新的载体上重获新生：以前是手工编码、做符号推理的 Agent，现在则是其"认知"从学习到的神经表示中涌现出来的 Agent。核心的架构模式——层级、市场、黑板、消息传递——依然惊人地相关。

从单体 Agent 到 Agent 社会的转变映射了复杂系统中一个更广泛的模式：随着问题空间的增长，分布式、模块化架构始终优于集中式、单体式架构。问题不再是*是否*使用多个 Agent，而是*如何*组织它们。

## 多智能体架构

多智能体系统的拓扑结构——Agent 之间如何连接、权威如何在它们之间流动——是最具决定性的架构选择。目前已经形成了四种典型模式，各有不同的权衡。

### 集中式（Supervisor/Manager）架构

在集中式架构中，由单一的*Orchestrator* Agent（也称为 supervisor、manager 或 planner）持有全局状态，分解任务，将子任务委派给工作 Agent，并汇总它们的结果。其拓扑是**中心辐射式（hub-and-spoke）**：所有通信都通过中心节点流动。

![集中式（Supervisor）架构。Manager 将任务委派给专门化的 Worker，并汇总它们的输出。所有通信均通过中央枢纽流动。]({{ site.baseurl }}/figures/fig_069_centralized-arch.png)

Manager 的职责包括：

- **任务路由**：决定哪个 Worker 最适合每个子任务
- **上下文管理**：为每个 Worker 提供全局上下文中相关的子集
- **结果聚合**：将 Worker 输出综合为一个连贯整体
- **错误处理**：检测 Worker 故障并重新路由或重试

> **LangGraph 中的 Supervisor 模式**
>
> ```python
> from langgraph.graph import StateGraph, START, END
> from typing import TypedDict, Literal
>
> class TeamState(TypedDict):
>     task: str
>     plan: str
>     code: str
>     tests: str
>     review: str
>     next_agent: str
>     final_output: str
>
> def supervisor_node(state: TeamState) -> TeamState:
>     # 中央 orchestrator：决定下一步应调用哪个 agent。
>     """Central orchestrator: decides which agent to invoke next."""
>     messages = [
>         {"role": "system", "content": SUPERVISOR_PROMPT},
>         {"role": "user",   "content": f"Task: {state['task']}\n"
>                                       f"Plan: {state.get('plan','')}\n"
>                                       f"Code: {state.get('code','')}\n"
>                                       f"Tests: {state.get('tests','')}\n"
>                                       "Which agent should act next? "
>                                       "Options: planner, coder, tester, reviewer, FINISH"}
>     ]
>     response = llm.invoke(messages)
>     return {**state, "next_agent": response.content.strip()}
>
> def route(state: TeamState) -> Literal["planner","coder","tester","reviewer","__end__"]:
>     return state["next_agent"] if state["next_agent"] != "FINISH" else END
>
> builder = StateGraph(TeamState)
> builder.add_node("supervisor", supervisor_node)
> builder.add_node("planner",    planner_node)
> builder.add_node("coder",      coder_node)
> builder.add_node("tester",     tester_node)
> builder.add_node("reviewer",   reviewer_node)
>
> builder.add_edge(START, "supervisor")
> builder.add_conditional_edges("supervisor", route)
> for agent in ["planner", "coder", "tester", "reviewer"]:
>     builder.add_edge(agent, "supervisor")   # 始终回到 supervisor
>
> graph = builder.compile()
> ```

> **集中式架构的权衡**
>
> **优点：**控制流简单；责任清晰；易于调试（所有决策集中在一处）；实现直接。
>
> **缺点：**单点故障——如果 Manager 产生幻觉或混乱，整个系统就会失败；在高负载下 Manager 会成为瓶颈；Manager 的上下文窗口必须容纳全局状态，限制了可扩展性。

### 去中心化（Peer-to-Peer）架构

在去中心化架构中，Agent 之间直接交互，没有中央协调者。其拓扑是**网状（mesh）**：任意 Agent 可以与任意其他 Agent 通信。协调通过局部交互涌现，而非通过全局规划。

![去中心化（peer-to-peer）架构。Agent 直接通信；协调从局部交互中涌现。]({{ site.baseurl }}/figures/fig_070_decentralized-arch.png)

对等系统中的涌现式协调通过以下机制产生：

- **协商（Negotiation）**：Agent 为任务或资源进行竞标
- **Stigmergy（共识主动性）**：Agent 修改共享状态，其他 Agent 观察该状态（见 "协调机制" 节）
- **流言协议（Gossip protocols）**：Agent 在网络中传播信息
- **局部共识（Local consensus）**：小组 Agent 在无全局协调的情况下达成一致

> **去中心化架构的权衡**
>
> **优点：**对个别 Agent 故障具有韧性；随着 Agent 数量增加可自然扩展；无瓶颈。
>
> **缺点：**难以调试——涌现行为难以追踪；当 Agent 对状态的视图不一致时可能产生冲突；朴素消息传递下协调开销以 $O(n^2)$ 增长；难以保证全局一致性。

### 层级式（Hierarchical）架构

层级式架构将集中式模式推广为具有多层管理的**树形结构**。顶层 Orchestrator 委派给特定领域的子 Manager，后者再委派给专门化 Worker。这映射了大型企业的组织结构。

![层级式架构。顶层 Orchestrator 委派给领域子 Manager，后者再委派给专门化 Worker。虚线箭头表示升级路径。]({{ site.baseurl }}/figures/fig_071_hierarchical-arch.png)

层级式系统的关键特征：

- **委派链（Delegation chains）**：权威与上下文沿树向下流动；结果向上流动
- **升级路径（Escalation paths）**：Worker 可以将无法解决的问题升级给其 Manager
- **领域隔离（Domain isolation）**：子 Manager 维护特定领域的上下文，减轻顶层 Orchestrator 的认知负担
- **作用域限制（Scope limitation）**：每个 Agent 只需了解其直接上级与下级

企业的类比十分贴切：CEO（顶层 Orchestrator）制定战略；VP（子 Manager）将战略转化为领域计划；一线贡献者（Worker）执行。层级结构在实现规模化的同时保留了责任归属。

### 蜂群（Swarm）架构

受生物系统（蚁群、鸟群）启发的蜂群架构，由许多遵循简单局部规则的**松耦合 Agent**组成，能在没有任何中央协调者或全局状态的情况下产生复杂的全局行为。

OpenAI 的 **Swarm** 框架 [openai2024swarm]（现已被 OpenAI Agents SDK 取代，但其概念原语依然有影响力）通过两个原语将其落地：

- **Routines（例程）**：Agent 为完成子任务所遵循的指令序列
- **Handoffs（移交）**：一个 Agent 将控制权（以及相关上下文）转交给另一个 Agent

> **OpenAI Swarm：Routines 与 Handoffs**
>
> ```python
> from swarm import Swarm, Agent
>
> client = Swarm()
>
> def transfer_to_billing():
>     # Handoff：将控制权移交给计费专员。
>     """Handoff: transfer control to the billing specialist."""
>     return billing_agent
>
> def transfer_to_technical():
>     # Handoff：将控制权移交给技术支持 agent。
>     """Handoff: transfer control to the technical support agent."""
>     return technical_agent
>
> triage_agent = Agent(
>     name="Triage Agent",
>     instructions="""You are a customer service triage agent.
>     Determine the nature of the customer's issue:
>     - For billing questions, transfer to billing.
>     - For technical issues, transfer to technical support.
>     - For general questions, answer directly.""",
>     functions=[transfer_to_billing, transfer_to_technical],
> )
>
> billing_agent = Agent(
>     name="Billing Specialist",
>     instructions="You handle billing inquiries. "
>                  "Access account data and resolve payment issues.",
>     functions=[lookup_account, process_refund],
> )
>
> technical_agent = Agent(
>     name="Technical Support",
>     instructions="You resolve technical issues. "
>                  "Diagnose problems and provide step-by-step solutions.",
>     functions=[run_diagnostics, escalate_to_engineering],
> )
>
> # 无全局状态——每个 agent 仅在其局部上下文中运行
> response = client.run(
>     agent=triage_agent,
>     messages=[{"role": "user", "content": "My invoice is wrong"}]
> )
> ```

> **Swarm 的性质**
>
> - **无全局状态**：每个 Agent 只维护自己的局部上下文窗口
> - **局部决策**：路由决策由当前 Agent 做出，而非由中央规划者
> - **通过集体行为完成任务**：复杂任务通过一连串 handoff 完成，每个 Agent 贡献其专长
> - **轻量**：无编排开销；Agent 在 handoff 之间是无状态的

## 协调机制

Agent 如何协调——如何共享信息、分工以及解决冲突——与拓扑同等重要。六种典型的协调机制适用于基于 LLM 的多智能体系统。

### 共享状态（全局黑板）

**黑板架构（blackboard architecture）** [hayes1985blackboard] 提供一个所有 Agent 都可以读写的共享数据结构。在 LLM 系统中，它通常实现为一个共享字典、数据库或结构化文档。

```python
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class BlackboardEntry:
    value: Any
    author: str
    timestamp: float
    confidence: float = 1.0

class Blackboard:
    # 用于多 agent 协调的线程安全共享状态。
    """Thread-safe shared state for multi-agent coordination."""

    def __init__(self):
        self._data: Dict[str, BlackboardEntry] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable]] = {}

    def write(self, key: str, value: Any, author: str,
              confidence: float = 1.0) -> bool:
        # 写入黑板；置信度更高的条目在冲突中获胜。
        """Write to blackboard; higher-confidence entries win conflicts."""
        with self._lock:
            existing = self._data.get(key)
            if existing and existing.confidence > confidence:
                return False  # 冲突：已有条目获胜
            import time
            self._data[key] = BlackboardEntry(
                value=value, author=author,
                timestamp=time.time(), confidence=confidence
            )
            self._notify(key, value)
            return True

    def read(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            return entry.value if entry else None

    def subscribe(self, key: str, callback: Callable):
        # Agent 订阅特定 key 上的变化。
        """Agents subscribe to changes on specific keys."""
        self._subscribers.setdefault(key, []).append(callback)

    def _notify(self, key: str, value: Any):
        for cb in self._subscribers.get(key, []):
            cb(key, value)
```

### 消息传递（Message Passing）

消息传递是 LLM Agent 最自然的协调机制：Agent 通过相互发送结构化文本消息来通信。关键的设计决策包括：

- **消息格式**：结构化（JSON schema）、自然语言或混合形式
- **路由**：直接（Agent 到 Agent）、广播或基于主题的发布/订阅
- **会话线程**：在多轮对话（Multi-Turn）交流中维持上下文
- **确认（Acknowledgment）**：发送方是否需要接收/处理的确认

### 规划与分解

Manager Agent 将高层任务分解为子任务的**有向无环图（directed acyclic graph, DAG）**，将每个子任务分配给合适的 Worker，并跟踪依赖关系。这是经典分层任务网络（hierarchical task network, HTN）规划在多 Agent 场景下的类比。

```python
from dataclasses import dataclass, field
from typing import List, Optional
import asyncio

@dataclass
class Task:
    id: str
    description: str
    assigned_to: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"   # pending | running | done | failed
    result: Optional[str] = None

class TaskDAG:
    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def ready_tasks(self) -> List[Task]:
        # 返回依赖已全部完成的任务。
        """Return tasks whose dependencies are all completed."""
        return [
            t for t in self.tasks.values()
            if t.status == "pending"
            and all(self.tasks[d].status == "done"
                    for d in t.dependencies)
        ]

    async def execute(self, agent_pool: dict):
        while any(t.status != "done" for t in self.tasks.values()):
            ready = self.ready_tasks()
            if not ready:
                await asyncio.sleep(0.1)
                continue
            # 并行执行就绪任务
            await asyncio.gather(*[
                self._run_task(t, agent_pool[t.assigned_to])
                for t in ready
            ])

    async def _run_task(self, task: Task, agent):
        task.status = "running"
        try:
            task.result = await agent.execute(task.description)
            task.status = "done"
        except Exception as e:
            task.status = "failed"
            raise
```

### 投票与共识

当多个 Agent 产生相互冲突的输出时，投票机制将它们的响应聚合为一个决策。常见方案包括：

- **多数投票（Majority voting）**：出现最多的答案获胜；适用于事实性问题
- **加权投票（Weighted voting）**：历史记录或置信度更高的 Agent 获得更大权重
- **基于辩论的裁决**：Agent 为各自立场辩论；由一个裁判 Agent 做出决定
- **Delphi 方法**：迭代轮次中，Agent 在看到他人的推理后修改自己的答案

形式化地，给定 $n$ 个 Agent 产生输出 $\{o_1, \ldots, o_n\}$ 及其权重 $\{w_1, \ldots, w_n\}$，加权共识为：

$$o^* = \arg\max_{o} \sum_{i=1}^{n} w_i \cdot \mathbf{1}[o_i = o]$$

对于连续输出（例如概率估计），适用加权平均：

$$\hat{p} = \frac{\sum_{i=1}^{n} w_i \cdot p_i}{\sum_{i=1}^{n} w_i}$$

### 基于市场的协调

市场机制通过**拍卖与竞标**分配任务和资源。合同网协议（Contract Net Protocol） [smith1980contract] 是最古老的多 Agent 协调机制之一，本质上是一种任务拍卖：

1. *Manager* 广播一条带有需求的任务公告
2. *Contractor*（承包者）Agent 提交竞标（能力声明 + 成本估算）
3. Manager 将合同授予最佳竞标者
4. 中标的承包者执行并报告结果

在 LLM 系统中，竞标可以用自然语言（如"我能用 3 步以高置信度完成此任务"）或结构化格式表达。市场机制在资源受限、需要最小化 API 成本的场景下尤为有效。

### Stigmergy：通过环境进行的间接通信

**Stigmergy（共识主动性）** [grasse1959reconstruction] 用一种更简单的机制取代显式的 Agent 间消息传递：每个 Agent 在工作时以副作用方式修改共享环境，其他 Agent 对这些修改而非直接信号做出反应。经典例证是觅食的蚂蚁在返程路径上沉积信息素；后续蚂蚁强化成功路径，无需任何蚂蚁"开口"说话。

在 LLM 多智能体系统中，Stigmergy 表现为：

- **共享文档**：Agent 写入共享文档；其他 Agent 读取并在其基础上扩展
- **代码仓库**：一个 Agent 提交代码；另一个 Agent 读取并扩展
- **标注层**：Agent 对共享产物进行标注（高亮错误、添加评论）
- **任务队列**：Agent 从共享队列中添加和消费任务

Stigmergy 实现了无显式通信开销的协调——Agent 只需观察共享环境的状态并据此行动。

## 通信协议

有效的多智能体系统需要定义良好的通信协议：约定的 Agent 间消息格式、语义和模式。（关于标准化的 Agent 间协议，参见第 "智能体到智能体通信" 章。）

### 结构化消息格式

LLM Agent 之间的消息应该是结构化的，以支持可靠解析与路由。一个最小的消息 schema：

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

PerformativeType = Literal[
    "inform",    # 共享信息
    "request",   # 请求执行某动作
    "propose",   # 提议某种行动方案
    "accept",    # 接受提议
    "reject",    # 拒绝提议
    "query",     # 提出问题
    "confirm",   # 确认收到/完成
    "failure",   # 报告失败
]

class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str          # 将相关消息分组
    sender: str                   # Agent 标识
    receiver: str                 # 目标 agent（或 "broadcast"）
    performative: PerformativeType
    content: str                  # 自然语言内容
    metadata: Dict[str, Any] = {} # 结构化载荷
    reply_to: Optional[str] = None  # 所回复的 message_id
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_llm_prompt(self) -> str:
        # 将消息渲染为接收方 agent 的 prompt 片段。
        """Render message as a prompt fragment for the receiving agent."""
        return (
            f"[MESSAGE from {self.sender}]\n"
            f"Type: {self.performative}\n"
            f"Content: {self.content}\n"
            + (f"Metadata: {self.metadata}\n" if self.metadata else "")
        )
```

### Performative 类型（受 FIPA-ACL 启发）

借鉴 FIPA Agent 通信语言 [fipa2002acl]，并针对 LLM Agent 进行现代化改造：

| Performative | 语义 | 示例用途 |
| --- | --- | --- |
| `inform` | 发送方相信 $\phi$ 为真 | 共享研究发现 |
| `request` | 发送方希望接收方执行 $\alpha$ | 委派子任务 |
| `propose` | 发送方提议计划 $\pi$ | 建议某种方案 |
| `accept` | 接收方同意提议 | 确认任务分配 |
| `reject` | 接收方拒绝提议 | 拒绝不兼容的任务 |
| `query` | 发送方希望知道 $\phi$ | 请求澄清 |
| `confirm` | 发送方确认 $\phi$ 已发生 | 确认完成 |
| `failure` | 发送方未能完成 $\alpha$ | 报告错误 |

### 上下文共享策略

多 Agent 通信中的一个关键挑战是**上下文管理**：每个 Agent 需要多少历史？三种策略：

- **完整历史**：将整个会话历史传递给每个 Agent。信息最丰富但代价高昂；上下文窗口会迅速被填满。
- **摘要**：由一个摘要 Agent 将先前的交流压缩为紧凑的摘要。高效但有损；重要细节可能被丢弃。
- **相关摘录**：通过语义搜索仅检索最相关的历史消息。在成本与信息量之间取得平衡；需要检索机制。

> **上下文共享经验法则**
>
> 对短对话（$<$10 轮）使用**完整历史**；对中等长度对话使用**摘要**；对长时运行的 Agent 会话使用**检索增强的摘录**。始终原文保留最近的 $k$ 条消息，以保持即时上下文。

## 角色设计与专门化

Agent 角色的设计——其能力、人格化设定与职责——既是艺术也是科学。设计良好的角色能促成专门化；设计糟糕的角色则会带来混乱与冗余。

### 定义 Agent 角色

LLM 多智能体系统中常见的角色：

| 角色 | 主要能力 | 典型工具 |
| --- | --- | --- |
| Researcher | 信息收集与综合 | 网页搜索、RAG、数据库 |
| Planner | 任务分解与调度 | 无（仅推理） |
| Coder | 代码生成与调试 | Code interpreter、linter |
| Reviewer | 质量评估与批判 | 无（仅推理） |
| Tester | 测试生成与执行 | 测试运行器、覆盖率工具 |
| Writer | 文本生成与编辑 | 语法检查器、风格指南 |
| Critic | 对抗式评估 | 无（仅推理） |
| Orchestrator | 协调与委派 | 所有 Agent 接口 |

### 基于能力 vs. 基于角色的分配

任务分配的两种理念：

- **基于角色**：根据预定义的角色标签分配任务。简单且可预测；当任务跨多个角色时可能不是最优。
- **基于能力**：根据对每个 Agent 能力相对于任务需求的动态评估来分配。更灵活；需要一个能力登记与匹配机制。

### 动态角色重分配

在长时运行的系统中，静态角色分配会变得次优。动态重分配允许 Agent 基于以下因素承担新角色：

- 当前负载（负载均衡）
- 近期任务上展现出的表现
- 任务需求的变化
- 需要顶替的 Agent 故障

### 以人格化设计实现思维多样性

一种微妙但强大的技巧：给 Agent 赋予**鲜明的人格设定（personas）**，以鼓励多元视角。与其使用五个完全相同的"助手"Agent，不如设计：

- 一位强调机会的*乐观主义者*
- 一位质疑假设的*怀疑论者*
- 一位聚焦实现的*务实派*
- 一位着眼长远的*远见者*
- 一位反向论辩的*唱反调者（devil's advocate）*

这种思维多样性受"六顶思考帽"（Six Thinking Hats） [debono1985six] 等技巧启发，能减少群体思维，产出更稳健的集体推理。

> **角色冲突的化解**
>
> 当 Agent 的职责出现重叠时，冲突就会产生。可用显式的**优先级规则**来化解：定义每类任务由哪个角色优先处理。或者使用一个**元 Agent（meta-agent）**，其唯一职责是冲突仲裁。绝不要让角色冲突隐式存在——它们会以相互矛盾的输出或死循环的形式出现。

## 面向 LLM 的多 Agent 模式

除了架构拓扑之外，还有若干**交互模式**已被证明对基于 LLM 的多智能体系统特别有效。（这些模式与第 "Agent 设计模式" 章中的单 Agent 设计模式互为补充。）

### 辩论模式（Debate Pattern）

多个 Agent 为不同立场辩论；一个裁判 Agent 评估论点并做出决定。已有研究表明辩论能提升事实准确性并减少幻觉 [du2023improving]。

```python
async def debate_round(question: str, agents: list, judge: Agent,
                       rounds: int = 2) -> str:
    # 运行多 agent 辩论，并返回裁判的判决。
    """Run a multi-agent debate and return the judge's verdict."""
    positions = {a.name: await a.generate_position(question)
                 for a in agents}

    for round_num in range(rounds):
        # 每个 agent 看到他人立场并可反驳
        rebuttals = {}
        for agent in agents:
            others = {k: v for k, v in positions.items()
                      if k != agent.name}
            rebuttals[agent.name] = await agent.rebut(
                question, positions[agent.name], others
            )
        positions = rebuttals

    # 裁判评估所有最终立场
    verdict = await judge.evaluate(question, positions)
    return verdict
```

### 反思模式（Reflection Pattern）

一个 Agent 生成输出；第二个 Agent 对其进行批判；第一个 Agent 根据批判进行修订。这实现了一个"生成—批判—修订"循环，迭代地提升质量。

```python
async def reflection_loop(task: str, generator: Agent,
                          critic: Agent, max_rounds: int = 3) -> str:
    draft = await generator.generate(task)

    for _ in range(max_rounds):
        critique = await critic.critique(task, draft)
        if critique.is_satisfactory:
            break
        draft = await generator.revise(task, draft, critique.feedback)

    return draft
```

### 分工模式（Division of Labor Pattern）

任务被分解为可并行执行的独立子任务。结果由一个综合 Agent 聚合。该模式可在"高度并行（embarrassingly parallel）"任务上最大化吞吐量。

### 流水线模式（Pipeline Pattern）

Agent 构成一条顺序处理链：每个 Agent 转换前一个 Agent 的输出。类似于 Unix 管道。适用于具有明确顺序依赖的任务（例如：研究 $\to$ 大纲 $\to$ 草稿 $\to$ 编辑 $\to$ 排版）。

### 集成模式（Ensemble Pattern）

多个 Agent 独立求解同一问题；由某种选择机制挑出最佳答案（best-of-$N$），或聚合答案（mixture-of-experts 风格）。以算力为代价提升可靠性。

$$o^* = \arg\max_{o \in \{o_1,\ldots,o_N\}} \text{score}(o, \text{task})$$

其中 $\text{score}$ 可以是一个 Reward 模型、一个裁判 LLM 或一个验证器。

### 师生模式（Teacher-Student Pattern）

能力更强的 Agent（teacher）引导能力较弱的 Agent（student）完成任务，提供提示、纠正和解释。该模式在推理阶段实现知识蒸馏，也可用于微调 student Agent。

### 红队模式（Red Team Pattern）

一个对抗性 Agent（red team）主动尝试在其他 Agent 的输出中寻找弱点、错误或安全违规。Red team Agent 被 Prompt 引导以尽可能挑剔与富有创造性的方式发起攻击。该模式对安全关键应用至关重要。

> **Red Team Agent 的 Prompt**
>
> ```python
> RED_TEAM_PROMPT = """You are a red team agent. Your job is to find
> flaws, errors, biases, safety violations, and failure modes in the
> following output. Be adversarial, creative, and thorough.
>
> Consider:
> 1. Factual errors or hallucinations
> 2. Logical inconsistencies
> 3. Safety and ethical concerns
> 4. Edge cases the solution doesn't handle
> 5. Ways a malicious user could exploit this output
> 6. Unintended consequences
>
> Output: {agent_output}
>
> Provide a detailed critique with specific examples of each flaw found."""
> ```

## 用强化学习训练多智能体系统

用 RL 训练多智能体系统会引入超出单 Agent RL 的挑战。根本难点在于：每个 Agent 的环境都包含其他正在学习的 Agent，从任一单 Agent 的视角看，环境是**非平稳的（non-stationary）**。

### 数学形式化

多智能体系统可形式化为一个**马尔可夫博弈（Markov Game）**，也称随机博弈（stochastic game） [shapley1953stochastic]：

$$\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, \mathcal{T}, \{R^i\}_{i \in \mathcal{N}}, \gamma \rangle$$

其中 $\mathcal{N} = \{1, \ldots, n\}$ 是 Agent 集合，$\mathcal{S}$ 是共享状态空间，$\mathcal{A}^i$ 是 Agent $i$ 的动作空间，$\mathcal{T}: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \Delta(\mathcal{S})$ 是状态转移函数，$R^i: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \mathbb{R}$ 是 Agent $i$ 的 Reward 函数，$\gamma$ 是折扣因子。

每个 Agent $i$ 都力图最大化其期望折扣回报：

$$J^i(\pi^1, \ldots, \pi^n) = \mathbb{E}_{\pi^1,\ldots,\pi^n}\left[\sum_{t=0}^{\infty} \gamma^t R^i(s_t, a_t^1, \ldots, a_t^n)\right]$$

### 独立学习（Independent Learning）

最简单的方法：每个 Agent $i$ 将其他 Agent 视为环境的一部分，使用标准单 Agent RL（例如 PPO、REINFORCE）独立优化自己的 Policy $\pi^i$。

$$\nabla_{\theta^i} J^i \approx \mathbb{E}\left[\nabla_{\theta^i} \log \pi^i(a^i_t | o^i_t) \cdot \hat{A}^i_t\right]$$

> **非平稳性问题**
>
> 独立学习违反了马尔可夫性假设：随着其他 Agent 更新其 Policy，Agent $i$ 看到的状态转移和 Reward 分布都会发生变化。这会导致训练不稳定、震荡乃至无法收敛。独立学习在简单合作任务上实际可行，但在竞争或复杂合作场景下表现挣扎。

### 集中训练、去中心化执行（CTDE）

集中训练、去中心化执行（Centralized Training, Decentralized Execution, CTDE） [lowe2017multi, rashid2018qmix] 是合作型多 Agent RL 的主流范式。训练阶段，集中式 critic 可访问全局状态 $s$ 与所有 Agent 的动作 $\mathbf{a} = (a^1, \ldots, a^n)$。执行阶段，每个 Agent 仅基于其局部观测 $o^i$ 行动。

Agent $i$ 的集中式 critic：

$$Q^i_\phi(s, \mathbf{a}) = Q^i_\phi(s, a^1, \ldots, a^n)$$

Agent $i$ 的去中心化 actor：

$$\pi^i_{\theta^i}(a^i | o^i)$$

带有集中式 critic 的策略梯度：

$$\nabla_{\theta^i} J^i = \mathbb{E}\left[\nabla_{\theta^i} \log \pi^i(a^i | o^i) \cdot Q^i_\phi(s, \mathbf{a})\right]$$

CTDE 在训练时解决了非平稳性问题（集中式 critic 看到完整的联合状态），同时在推理时保留了去中心化执行（无需通信）。

### 通信学习（Communication Learning）

与使用固定通信协议不同，Agent 可以**学习要通信什么**。在可微通信框架 [sukhbaatar2016learning, das2019tarmac] 中，Agent 产生连续的通信向量 $m^i_t$ 并传递给其他 Agent：

$$a^i_t, m^i_t = \pi^i_{\theta^i}(o^i_t, \{m^j_{t-1}\}_{j \neq i})$$

通信向量通过联合 Reward 信号的反向传播进行端到端优化。对于 LLM Agent，这一思想被近似为训练 Agent 产生最大化任务表现的结构化自然语言消息。

### 涌现通信（Emergent Communication）

当 Agent 仅依靠 Reward 信号（没有预定义语言）从零训练时，它们可以发展出**涌现通信协议** [lazaridou2020emergent]：编码任务相关信息的共享符号系统。虽然在科学上令人着迷，但 LLM 系统中的涌现通信通常并不可取——我们希望 Agent 使用人类可理解的语言进行通信。

### 自我对弈（Self-Play）

在竞争或混合动机场景下，**自我对弈（self-play）** [silver2017mastering] 通过让 Agent 与自身的副本对抗来训练。这会自动产生课程式学习：随着 Agent 的提升，其对手（自身的早期版本）变得越来越难以击败。

对于 LLM Agent，自我对弈被用于：

- 红队 vs. 蓝队训练
- 辩论训练（Agent 相互辩论）
- 谈判训练（Agent 相互谈判）

### 基于种群的训练（Population-Based Training）

**基于种群的训练（Population-Based Training, PBT）** [jaderberg2019human] 维护一个具有不同 Policy、超参数和专长的多样化 Agent 种群。Agent 会被定期评估；表现不佳的 Agent 会被表现优秀 Agent 的变异副本所替换。

对于多 Agent LLM 系统，PBT 实现了：

- 有效角色专长的自动发现
- 对个别 Agent 故障的鲁棒性（多样化种群）
- 通过种群多样性避开局部最优

### 社会福利与 Nash 均衡

在多 Agent 场景下，最优性的概念比单 Agent 场景更复杂。两个关键的解概念：

**Nash 均衡（Nash Equilibrium）**：一个联合 Policy $(\pi^{1*}, \ldots, \pi^{n*})$，使得没有任何 Agent 可以通过单方面偏离来提高其期望回报：

$$J^i(\pi^{i*}, \pi^{-i*}) \geq J^i(\pi^i, \pi^{-i*}) \quad \forall i, \forall \pi^i$$

其中 $\pi^{-i}$ 表示除 $i$ 之外所有 Agent 的联合 Policy。

**社会福利最大化（Social Welfare Maximization）**：优化所有 Agent 回报之和：

$$\max_{\pi^1,\ldots,\pi^n} \sum_{i=1}^{n} J^i(\pi^1, \ldots, \pi^n)$$

在完全合作场景下（所有 Agent 共享同一 Reward），社会福利最大化是合适的目标。在竞争场景下，Nash 均衡是相关的解概念。大多数现实世界的多 Agent LLM 系统是**混合动机（mixed-motive）**的：Agent 之间的目标部分一致、部分冲突。

> **延伸阅读：多 Agent RL 的博弈论基础**
>
> 对于有兴趣了解多智能体系统博弈论基础的读者：
>
> - **Shoham & Leyton-Brown** [shoham2008multiagent] —— 全面的教科书，涵盖 Agent 系统的 Nash 均衡、机制设计与社会选择理论。
> - **Zhang et al.** [zhang2021multiagent] —— 综述了在合作、竞争与混合场景下具有收敛保证的多 Agent RL 算法。
> - **Nisan et al.** [nisan2007algorithmic] —— 算法博弈论的权威参考，涵盖拍卖、均衡计算与无政府代价（price of anarchy）。

## 挑战与解决方案

### 协调开销

每条 Agent 间消息都会消耗 Token——也就意味着消耗时间与金钱。在朴素实现中，即使不必要，Agent 也会不停通信。

> **何时*不要*通信**
>
> - 当信息已经存在于共享黑板中时
> - 当接收方 Agent 当前任务并不需要该信息时
> - 当消息会重复已经发送过的信息时
> - 当任务简单到单个 Agent 即可完成时
>
> **准则**：仅当信息的期望价值超过消息成本时才进行通信。

量化通信成本：若一条消息消耗 $c$ 个 Token，接收方 Agent 的任务价值为 $v$，则仅当任务价值的期望提升 $\Delta v > c \cdot \text{cost\_per\_token}$ 时才进行通信。

### 冗余 vs. 效率

多个 Agent 可能独立求解同一子问题，从而浪费算力。解决方案：

- **重复检测（Duplicate detection）**：开始任务前先检查黑板上是否已有结果
- **结果缓存**：以语义键存储已完成的子任务结果以便检索
- **任务锁**：将任务标记为"进行中"以防止重复执行

### 归因（Attribution）

当多智能体系统成功或失败时，是哪个 Agent 应当负责？归因对以下方面至关重要：

- RL 的 Reward 分配（功劳分配问题，credit assignment problem）
- 调试与改进
- 信任校准（决定该依赖哪些 Agent）

**反事实功劳分配（counterfactual credit assignment）**方法通过提问"若该 Agent 采取不同行动，结果会改变多少？"来估计每个 Agent 的贡献。

$$\text{credit}^i = J(\pi^1, \ldots, \pi^n) - J(\pi^1, \ldots, \pi^{i}_{\text{default}}, \ldots, \pi^n)$$

### 可扩展性（Scalability）

朴素的消息传递在 Agent 数量上呈 $O(n^2)$ 扩展。解决方案：

- **层级通信**：Agent 仅在自身子树内通信
- **基于主题的发布/订阅**：Agent 仅订阅相关主题的消息
- **稀疏通信图**：仅连接需要交互的 Agent
- **异步通信**：Agent 不阻塞等待响应

### 涌现行为与安全性

多智能体系统可能展现出意料之外的涌现行为——Agent 之间的交互产生没有任何单个 Agent 被设计来产出的结果。这既是特性（涌现能力），也是风险（涌现失败）。

> **多智能体系统中的安全关切**
>
> - **Prompt 注入级联**：对一个 Agent 的恶意输入会在系统中传播
> - **Reward hacking**：Agent 找到违背意图的意外方式来最大化 Reward
> - **合谋（Collusion）**：在竞争场景下，Agent 可能发展出隐式的合谋策略
> - **放大（Amplification）**：一个 Agent 中的错误或偏差会被下游 Agent 放大
>
> 始终包含一个**安全监控 Agent（safety monitor agent）**，它观察所有 Agent 间通信，一旦检测到不安全行为即可中止系统。

### 评估（Evaluation）

评估多智能体系统需要多层级的指标：

| 层级 | 指标 | 示例 |
| --- | --- | --- |
| 系统 | 任务完成率 | 正确完成任务的百分比 |
| 系统 | 端到端延迟 | 从任务到最终输出所用时间 |
| 系统 | 总 Token 成本 | 所有 Agent 消耗的 Token 总量 |
| Agent | 个体准确率 | 单个 Agent 的任务成功率 |
| Agent | 通信效率 | 有用消息 / 总消息数 |
| Agent | 贡献分数 | 反事实功劳（式 "归因"） |
| 涌现 | 协调质量 | 任务重叠/缺口的程度 |

## 现实世界中的多智能体应用

### 软件开发团队

一支多 Agent 软件开发团队映射了真实的工程组织：

```python
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class SoftwareTeamState:
    requirements: str
    architecture: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None
    review_feedback: Optional[str] = None
    final_code: Optional[str] = None
    approved: bool = False

class SoftwareDevelopmentTeam:
    """
    多 agent 软件团队：
      Architect -> Coder -> Tester -> Reviewer -> （迭代或交付）
    """

    def __init__(self, llm_factory):
        self.architect = llm_factory(
            system_prompt="""You are a software architect. Given requirements,
            produce a clear technical design: components, interfaces, data
            structures, and implementation plan."""
        )
        self.coder = llm_factory(
            system_prompt="""You are an expert software engineer. Given a
            technical design, write clean, well-documented, production-ready
            code. Follow best practices for the language."""
        )
        self.tester = llm_factory(
            system_prompt="""You are a QA engineer. Given code, write
            comprehensive tests: unit tests, edge cases, integration tests.
            Identify potential bugs and failure modes."""
        )
        self.reviewer = llm_factory(
            system_prompt="""You are a senior code reviewer. Evaluate code
            for correctness, security, performance, and maintainability.
            Provide specific, actionable feedback. Approve only if excellent."""
        )

    async def build(self, requirements: str,
                    max_iterations: int = 3) -> SoftwareTeamState:
        state = SoftwareTeamState(requirements=requirements)

        # 阶段 1：架构
        state.architecture = await self.architect.invoke(
            f"Requirements:\n{requirements}\n\nProduce technical design."
        )

        for iteration in range(max_iterations):
            # 阶段 2：实现
            prompt = (f"Design:\n{state.architecture}\n\n"
                      + (f"Previous feedback:\n{state.review_feedback}\n\n"
                         if state.review_feedback else "")
                      + "Write the implementation.")
            state.code = await self.coder.invoke(prompt)

            # 阶段 3：测试
            state.tests = await self.tester.invoke(
                f"Code:\n{state.code}\n\nWrite comprehensive tests."
            )

            # 阶段 4：评审
            review = await self.reviewer.invoke(
                f"Code:\n{state.code}\n\nTests:\n{state.tests}\n\n"
                "Review this code. End with APPROVED or NEEDS_REVISION."
            )

            if "APPROVED" in review:
                state.final_code = state.code
                state.approved = True
                break
            else:
                state.review_feedback = review

        return state

    async def run(self, requirements: str) -> str:
        state = await self.build(requirements)
        if state.approved:
            return f"# Final Implementation\n\n{state.final_code}"
        else:
            return f"# Best Attempt (not approved)\n\n{state.code}"
```

### 研究团队

一个研究团队 Agent 社会映射了学术合作：

- **Literature Reviewer**：检索并综合现有工作
- **Hypothesis Generator**：提出新的研究方向
- **Experimentalist**：通过执行代码设计并运行实验
- **Statistician**：分析结果并评估显著性
- **Writer**：将发现综合成连贯的报告

### 客户服务系统

一个分级客户服务系统：

- **Router**：分类进入的请求并路由到专员
- **Billing Specialist**：处理支付与账户问题
- **Technical Specialist**：解决产品/服务问题
- **Escalation Agent**：处理需要人类判断的复杂情形

### 创意团队

一条创意生产流水线：

- **Brainstormer**：无自我审查地产生多样化想法
- **Drafter**：将最有前景的想法发展成完整草稿
- **Editor**：在清晰度、风格与连贯性上精修草稿
- **Critic**：提供对抗式反馈以加强作品

## 架构对比

| 架构 | 可扩展性 | 可调试性 | 协调成本 | 容错性 | 最适用场景 |
| --- | --- | --- | --- | --- | --- |
| 集中式（Supervisor） | M | H | L | L | 简单流水线；任务分解清晰；小团队 |
| 去中心化（P2P） | H | L | H | H | 动态环境；对韧性要求高；大规模 |
| 层级式 | H | M | M | M | 企业级工作流；跨多个领域的复杂任务 |
| Swarm | H | L | L | H | 客户服务路由；简单的 handoff 链 |
| Pipeline | M | H | L | L | 顺序处理；阶段依赖清晰 |
| Ensemble | L | H | H | H | 高风险决策；可靠性优先于效率 |

> **如何选择架构**
>
> - **独立子任务** $\rightarrow$ 并行架构（分工、ensemble）。
> - **具有清晰依赖的顺序任务** $\rightarrow$ pipeline。
> - **需要容错** $\rightarrow$ 避免集中式；优先选择层级式或去中心化。
> - **可调试性关键** $\rightarrow$ 集中式或 pipeline（所有决策可追溯）。
> - **$<$5 个 Agent** $\rightarrow$ 集中式最简单。**$>$20 个 Agent** $\rightarrow$ 层级式或 swarm。
>
> 在实践中，大多数生产系统采用**层级式**架构：顶层 Supervisor 委派给特定领域的子 Supervisor，后者管理由专门化 Worker 组成的小团队。

## 小结

多智能体系统代表着我们部署 LLM 方式的根本性转变：从孤立的助手转向由专门化 Agent 构成的协作社会。本节的关键洞见：

> **多智能体系统：要点回顾**
>
> 1. **架构很重要**：Agent 连接的拓扑决定了可扩展性、可调试性与容错性。应根据任务结构与运维需求做出选择。
> 2. **协调是有代价的**：每条 Agent 间消息都消耗 Token。设计通信协议时应在保持必要信息流的同时最小化开销。
> 3. **专门化带来质量**：在复杂任务上，具有专注角色与定制 Prompt 的 Agent 始终胜过通才 Agent。
> 4. **RL 训练很难**：多 Agent RL 引入非平稳性、功劳分配挑战与涌现行为。CTDE 是目前合作场景下的最佳实践。
> 5. **安全需要显式设计**：多智能体系统会放大错误并展现意外的涌现行为。安全监控必须作为一等架构关切。
> 6. **从简开始**：从集中式 Supervisor 模式起步，衡量其局限，仅在必要时才向更复杂的架构演进。

多 Agent LLM 系统领域正在快速演进。本文所述的模式与技术代表了当前的最先进水平，但新的架构、协调机制与训练算法正在不断涌现。无论具体实现如何演变，这些基础原则——专门化、协调、涌现行为，以及效率与鲁棒性之间的张力——都将持续保持其相关性。
