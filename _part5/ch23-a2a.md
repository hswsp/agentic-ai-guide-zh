---
layout: home
title: 智能体到智能体通信（Agent-to-Agent, A2A）
permalink: /part5/ch23-a2a.html
---

随着大语言模型从孤立的助手演化为由专业化 Agent 组成的协作网络，*Agent 之间如何对话*这一问题已变得与单个 Agent 内部如何推理同等重要。本章将介绍那些使多 Agent 系统能够协调、委派并共同解决任何单一 Agent 都无法独立处理的问题的协议、模式与工程实践。

## 动机：为何 Agent 必须相互通信

> **专业化的必然性**
>
> 单一的通才型 Agent 面临一个根本性矛盾：知识的广度与能力的深度。真实世界的任务——例如法律文档审阅、多步科学研究、企业级软件开发——同时需要两者。Agent 间通信通过让一个*由专业化 Agent 组成的网络*进行协作来化解这一矛盾：每个 Agent 贡献自己的长处，同时把短板委派出去。

推动结构化 Agent 间通信需求的力量主要有以下几股：

**认知负载与上下文限制。**

每个 LLM 都在有限的上下文窗口内运行。复杂的工作流——涉及数百份文档、Tool 调用和推理步骤——很快会超出单一 Agent 的内存承载力。通过将任务分解到多个 Agent，每个 Agent 都在可管理的上下文中工作，而编排型 Agent 只需维护高层状态。

**专业化与专家能力。**

不同的 Agent 可针对特定领域进行微调、Prompt 设计或 Tool 配置：例如可访问编译器和测试运行器的 `CodeAgent`、可访问判例数据库的 `LegalAgent`、配备统计库的 `DataAgent`。将子任务路由给合适的专家可同时提升质量与效率。

**并行性与吞吐量。**

彼此独立的子任务可以同时分派给多个 Agent。一个研究编排器可能并行地把文献检索扇出到五个专业化 Agent，然后再综合它们的结果——大幅缩短墙钟时间。

**故障隔离与韧性。**

当某个 Agent 失败时，设计良好的多 Agent 系统可以换用另一个 Agent 重试、回退到更简单的方案，或者升级到人工介入——而不会让整个工作流崩溃。

**委派与交接。**

随着上下文转移，长时间运行的任务可能需要在 Agent 之间交接。最初的 `PlannerAgent` 分解目标，将子任务交给 `ExecutorAgents`，最后由 `ReviewerAgent` 校验输出——每个 Agent 只接收它所需要的那部分上下文。

> **A2A 通信的核心需求**
>
> 1. **可发现性**：Agent 必须能够找到其他 Agent 并理解它们的能力。
> 2. **互操作性**：由不同团队或厂商构建的 Agent 必须使用共同的协议进行通信。
> 3. **异步性**：长时间运行的任务不得阻塞调用方；结果应通过回调或轮询返回。
> 4. **安全性**：Agent 之间必须相互认证，并强制执行授权边界。
> 5. **可观测性**：每一次消息交换都必须可追踪，以便调试和审计。

## Google A2A 协议

2025 年 4 月，Google（在 50 余家技术合作伙伴的共同贡献下）发布了 **智能体到智能体（A2A）协议** [[360]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-google-a2a-2025)]，这是一份用于 AI Agent 之间互操作通信的开放规范。该协议随后被捐赠给 **Linux Foundation**，截至 2026 年支持机构已超过 150 家。A2A 围绕一组核心原则进行设计，使其区别于早期的临时方案。

### 设计理念

A2A 规范明确提出了五条指导原则（根据官方规范 [[360]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-google-a2a-2025)] §1.2 改编）：

> **A2A 设计原则**
>
> **不透明执行（Opaque execution）**
>
> 调用方 Agent 永远不会探查远程 Agent 的内部——它们只通过声明的接口进行交互。目标究竟是 GPT-4、Gemini 还是基于规则的系统，对协议而言无关紧要，从而真正实现异构 Agent 生态。
>
> **企业级就绪（Enterprise readiness）**
>
> 认证（OAuth 2.0、API 密钥、JWT）、审计日志和合规要求并非事后补丁——它们从一开始就在协议层面被集成。
>
> **模态无关性（Modality agnosticism）**
>
> 一条消息可以同时包含文本、二进制文件和结构化 JSON 负载，从而无需扩展协议即可支持处理图像、音频、代码或文档的 Agent。
>
> **基于现有标准的简洁性（Simplicity via existing standards）**
>
> A2A 不发明新的传输层，而是复用 HTTP/HTTPS 配合 JSON-RPC 2.0 消息、用于流式传输的服务器发送事件（Server-Sent Events, SSE），以及作为备选绑定的 gRPC——这些都是每个基础设施团队已经在运行的技术。
>
> **异步优先的任务模型（Async-first task model）**
>
> 长时间运行的操作是常态而非例外。推送通知与轮询都是一等机制，因此调用方无需保持连接长达数小时。

### Agent Card

A2A 可发现性的基础是 **Agent Card**——一份托管在固定端点（`/.well-known/agent.json`）上的机器可读 JSON 清单。它声明该 Agent 能做什么、如何认证、把任务发送到何处——类似于 OpenAPI 规范，但面向的是自主 Agent 而非 REST 端点。

> **Agent Card 结构**
>
> ```python
> # 托管于 https://agent.example.com/.well-known/agent.json 的 Agent Card
> agent_card = {
>     "name": "DataAnalysisAgent",
>     "description": "Analyzes structured datasets, produces statistical summaries, "
>                    "generates visualizations, and answers data questions.",
>     "url": "https://agent.example.com/a2a",
>     "version": "1.2.0",
>     "capabilities": {
>         "streaming": True,
>         "pushNotifications": True,
>         "stateTransitionHistory": True
>     },
>     "authentication": {
>         "schemes": ["Bearer", "ApiKey"]
>     },
>     "skills": [
>         {
>             "id": "statistical-analysis",
>             "name": "Statistical Analysis",
>             "description": "Compute descriptive statistics, run hypothesis tests, "
>                            "fit regression models on tabular data.",
>             "tags": ["statistics", "data", "analysis", "regression"],
>             "examples": [
>                 "What is the correlation between columns A and B?",
>                 "Run a t-test comparing these two groups.",
>                 "Fit a linear regression predicting sales from ad spend."
>             ],
>             "inputModes": ["text", "data"],
>             "outputModes": ["text", "data", "file"]
>         },
>         {
>             "id": "visualization",
>             "name": "Data Visualization",
>             "description": "Generate charts, plots, and dashboards from data.",
>             "tags": ["charts", "plots", "visualization", "dashboard"],
>             "examples": [
>                 "Create a bar chart of monthly revenue.",
>                 "Plot the distribution of customer ages."
>             ],
>             "inputModes": ["text", "data"],
>             "outputModes": ["file", "text"]
>         }
>     ],
>     "defaultInputModes": ["text"],
>     "defaultOutputModes": ["text"]
> }
> ```

Agent Card 支持*基于能力的路由*：编排型 Agent 可以从注册中心获取卡片，将子任务在语义上匹配到最合适的 Agent，并据此分派——整个过程无需任何硬编码的路由逻辑。

### 任务生命周期

A2A 把所有工作都建模为 **Task**。一个任务会沿着一个明确定义的状态机推进：

`submitted`

客户端已发出任务；服务器已确认接收。

`working`

Agent 正在主动处理。客户端可以进行轮询，或等待 SSE 事件。

`input-required`

Agent 在继续之前需要来自用户或调用方 Agent 的额外信息（例如澄清问题、缺失的凭据）。

`completed`

任务成功完成；结果可在响应中获取。

`failed`

发生了不可恢复的错误；错误消息会说明原因。

`rejected`

Agent 拒绝了该任务（例如超出其能力范围或未获授权）。该状态于 A2A v1.0 加入。

`canceled`

任务被中止，可能由客户端或服务器发起。

### 基于 Server-Sent Events 的流式传输

对于会产生增量输出的任务（例如正在撰写的长报告、正在生成的代码文件），A2A 使用 **服务器发送事件（Server-Sent Events, SSE）**。客户端打开一个持久 HTTP 连接，接收 JSON 事件流：

> **SSE 事件流示例**
>
> ```python
> # 每个 SSE 事件携带一个 TaskStatusUpdateEvent 或 TaskArtifactUpdateEvent
> # 下面是"撰写一份研究报告"任务的事件流示例：
>
> # 事件 1：状态更新
> data: {
>   "id": "task-abc123",
>   "status": {"state": "working"},
>   "final": false
> }
>
> # 事件 2：部分产物（流式文本）
> data: {
>   "id": "task-abc123",
>   "artifact": {
>     "parts": [{"type": "text", "text": "## Introduction\n\nRecent advances in..."}],
>     "index": 0,
>     "append": false,
>     "lastChunk": false
>   },
>   "final": false
> }
>
> # 事件 3：追加更多文本
> data: {
>   "id": "task-abc123",
>   "artifact": {
>     "parts": [{"type": "text", "text": " reinforcement learning have shown..."}],
>     "index": 0,
>     "append": true,   # 追加到已有产物
>     "lastChunk": false
>   },
>   "final": false
> }
>
> # 终止事件：任务完成
> data: {
>   "id": "task-abc123",
>   "status": {"state": "completed"},
>   "final": true
> }
> ```

### 长时间运行任务的推送通知

当任务可能持续数分钟乃至数小时，保持一个常开的 SSE 连接并不现实。A2A 支持**推送通知**：客户端注册一个 webhook URL，服务器会随着任务的推进通过 POST 发送状态更新。

```python
# 客户端在提交任务时注册推送通知端点
task_request = {
    "id": "task-xyz789",
    "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Analyze Q3 sales data and produce a report."}]
    },
    "pushNotification": {
        "url": "https://my-orchestrator.example.com/webhooks/a2a",
        "token": "secret-hmac-token-for-verification",
        "authentication": {
            "schemes": ["Bearer"],
            "credentials": "eyJhbGciOiJIUzI1NiJ9..."
        }
    }
}
# 当任务在不同状态之间转换时，
# 服务器会将 TaskStatusUpdateEvent 对象 POST 到该 webhook URL。
```

### 消息格式

A2A 消息由一个 **role**（`user` 或 `agent`）加上一组带类型的 **parts**（文本、文件或结构化数据）组成。完整的消息模式、多模态示例以及上下文传递指南详见第 "消息格式与 Schema" 节。

### 认证与授权

A2A 支持多种认证方案，这些方案在 Agent Card 中声明，并按每次请求执行：

- **Bearer Token（JWT/OAuth 2.0）**：企业部署的标准方案；令牌携带的 scope 限定了调用方 Agent 可以请求的内容。
- **API 密钥**：适用于内部或可信环境的更简单方案。
- **双向 TLS（Mutual TLS, mTLS）**：用于高安全部署的基于证书的认证。
- **OpenID Connect**：联合身份机制，使跨组织的 Agent 通信成为可能。

> **强制执行授权 scope**
>
> 接收任务的 Agent 不仅要验证*谁*在调用（认证），还要验证*它被允许请求什么*（授权）。一个 `ReportingAgent` 可能接受任何已认证 Agent 的只读数据查询，但将写操作限定给持有特定 OAuth scope 的 Agent。未能强制执行这一点会在多 Agent 系统中制造权限提升漏洞。

## 通信模式

多 Agent 系统会根据任务的性质、延迟要求以及参与 Agent 的数量，采用多种不同的通信模式。

### 请求-响应（Request-Response）

最简单的模式：Agent A 把任务发给 Agent B，并等待完整的响应。适用于结果需要在继续之前获得的、简短且定义明确的子任务。

### 流式传输（Streaming）

Agent A 打开一个 SSE 连接；Agent B 在产生部分结果时以流的形式发送出来。非常适合长文本生成（报告、代码）、实时协作或渐进式 UI 更新。

> **流式传输模式的应用场景**
>
> 编排器请求 `WritingAgent` 起草一份 10 页的技术文档。编排器无需等待 2 分钟以获得完整文档，而是在每一节写完时即流式接收，从而让 `ReviewAgent` 在后续章节仍在生成时就开始审阅前面的章节——这种流水线将总延迟降低了 40--60\%。

### 多轮交互（Multi-Turn）

某些任务需要迭代式精化。Agent 进入 `input-required` 状态，编排器提供澄清信息，然后任务恢复执行。这与人类协作流程相似：起草 $\to$ 反馈 $\to$ 修改。

```python
# 多轮：编排器处理 input-required 状态
async def run_multiturn_task(client, initial_message):
    task = await client.send_task(message=initial_message)

    while task.status.state not in ("completed", "failed", "canceled"):
        if task.status.state == "input-required":
            # Agent 需要澄清
            clarification_needed = task.status.message
            print(f"Agent asks: {clarification_needed}")

            # 编排器生成或转发澄清回复
            user_reply = await get_clarification(clarification_needed)

            # 发送回复以继续任务
            task = await client.send_task(
                task_id=task.id,
                message={"role": "user",
                         "parts": [{"type": "text", "text": user_reply}]}
            )
        else:
            # 仍在执行 —— 延迟后再轮询
            await asyncio.sleep(2)
            task = await client.get_task(task.id)

    return task
```

### 广播（Broadcast）

编排器将同一条消息同时发送给多个 Agent——适用于发布通告、分发共享上下文，或触发并行独立的工作流。

### 发布-订阅（Publish-Subscribe, Pub-Sub）

Agent 订阅事件通道（例如 `new-document-uploaded`、`model-retrained`）。当事件触发时，所有订阅的 Agent 都会收到通知。这种方式解耦了生产者与消费者，并支持响应式、事件驱动的架构。

### 协商（Negotiation）

两个 Agent 互相交换提案与反提案，以便在计划、资源分配或方案上达成一致。常见于 Agent 具有不同目标或约束的多 Agent 规划系统中。

> **协商模式**
>
> `PlannerAgent` 提出一个 5 步研究计划。`ResourceAgent` 回复说第 3 步（运行一次大规模仿真）将超出计算预算。`PlannerAgent` 反提议一次规模缩小的仿真。`ResourceAgent` 批准。随后达成一致的计划被分派给执行型 Agent。

### 基于拍卖的任务分配（Auction-Based Task Allocation）

编排器宣布带需求的任务；候选 Agent 提交竞标（预计完成时间、置信度、成本）；编排器把任务授予中标方。这使得在一组 Agent 上进行动态的、市场化的负载均衡成为可能。

| 模式 | 延迟 | 最佳适用场景 |
| --- | --- | --- |
| 请求-响应 | 低 | 简短、定义明确的子任务 |
| 流式传输 | 低（首 Token） | 长文本生成、实时 UI |
| 多轮 | 中 | 需要澄清的模糊任务 |
| 广播 | 低 | 共享上下文分发 |
| 发布-订阅 | 可变 | 事件驱动的响应式工作流 |
| 协商 | 中--高 | 资源受限的规划 |
| 拍卖 | 中 | 动态负载均衡 |

## Agent 发现与路由

在一个 Agent 能够与另一个 Agent 通信之前，它必须先*找到*对方。Agent 发现就是定位可以处理特定任务的 Agent 的过程。

### Agent 注册中心

**Agent 注册中心**是一种目录服务，用于索引 Agent Card 并提供搜索与查找 API。常见的部署模型有两种：

**中心化注册中心（Centralized Registry）**

单一权威注册中心（例如企业服务目录）索引所有 Agent。运维简单，但会形成单点故障，且可能无法扩展到跨组织部署。

**联合注册中心（Federated Registry）**

由多个注册中心组成，每个对特定领域或组织具有权威性，并通过跨注册中心搜索协议互通。更具韧性和隐私保护，但需要标准化的联合协议。

### 基于能力的路由

编排器不会硬编码 Agent 的 URL，而是进行**基于能力的路由**：先向注册中心查询匹配所需技能的 Agent，再从中选择最合适的。

```python
class AgentRouter:
    """根据能力匹配将任务路由到 Agent。"""

    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self._cache: dict[str, list[AgentCard]] = {}

    async def find_agents(self, required_skill: str,
                          tags: list[str] | None = None) -> list[AgentCard]:
        """向注册中心查询具备所需技能的 Agent。"""
        params = {"skill": required_skill}
        if tags:
            params["tags"] = ",".join(tags)
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.registry_url}/agents", params=params)
            return [AgentCard(**card) for card in resp.json()["agents"]]

    async def route(self, task_description: str) -> AgentCard:
        """在语义上将任务描述匹配到最合适的可用 Agent。"""
        # 对任务描述进行嵌入
        task_embedding = await embed(task_description)

        # 获取所有已注册的 Agent
        all_agents = await self.find_agents(required_skill="*")

        # 按任务与 Agent 描述的余弦相似度对每个 Agent 评分
        scored = []
        for agent in all_agents:
            agent_embedding = await embed(agent.description)
            score = cosine_similarity(task_embedding, agent_embedding)
            scored.append((score, agent))

        # 返回得分最高的 Agent
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
```

### 等价 Agent 间的负载均衡

当多个 Agent 提供相同能力时，路由器必须分配负载。常用策略包括：

- **轮询（Round-robin）**：在所有可用 Agent 之间平均分发任务。
- **最少负载（Least-loaded）**：路由到活跃任务最少的 Agent（需要健康/指标端点）。
- **延迟感知（Latency-aware）**：路由到近期响应时间最低的 Agent。
- **亲和性路由（Affinity-based）**：将相关任务路由到同一个 Agent，以利用已缓存的上下文。

### 版本管理与兼容性

Agent Card 包含 `version` 字段。编排器应指定最低版本要求，并在只有旧版本可用时进行优雅降级。推荐使用语义化版本（Semantic Versioning）[[361]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-preston2024semver)]（`MAJOR.MINOR.PATCH`）：破坏性接口变更递增 `MAJOR`，新增能力递增 `MINOR`。

> **长生命周期系统中的版本偏斜**
>
> 在生产环境的多 Agent 系统中，不同 Agent 可能在不同时间被升级，从而产生版本偏斜（Version Skew）。一个针对 Agent Card v2.1 编译的编排器，可能遇到仍运行 v1.3 的 Agent。务必实现向后兼容的消息处理，并明确测试跨版本场景。

## 消息格式与 Schema

### 结构化与非结构化消息

A2A 支持从完全非结构化（纯文本）到完全结构化（带类型的 JSON Schema）的整个谱系。选择何种格式取决于参与的 Agent：

| 消息类型 | 优点 | 缺点 |
| --- | --- | --- |
| 纯文本 | 灵活、可读性强、易于生成 | 难以可靠解析，无 Schema 校验 |
| 结构化 JSON | 机器可解析、可校验、带类型 | 需要事先约定 Schema，灵活性较差 |
| 混合（文本 + 数据） | 人类可读的意图 + 机器可解析的负载 | 构造和解析更复杂 |

### 多模态消息

A2A 消息的结构是一个 **role**（`user` 或 `agent`）外加一组带类型的 **parts**：

| Part 类型 | 字段 | 应用场景 |
| --- | --- | --- |
| `TextPart` | `text: string` | 自然语言指令、响应 |
| `FilePart` | `mimeType`、`uri` 或 `bytes` | 文档、图像、音频、代码文件 |
| `DataPart` | `data: object` | 结构化 JSON（Tool 结果、Schema） |

现代 Agent 越来越多地处理非文本模态。A2A 的 `FilePart` 支持任意 MIME 类型，从而可以构建丰富的多模态工作流：

> **多模态 A2A 消息：数据分析**
>
> ```python
> # 一条同时包含文本指令、数据负载和文件的消息
> message = {
>     "role": "user",
>     "parts": [
>         {
>             "type": "text",
>             "text": "Analyze the attached CSV and the schema below. "
>                     "Identify anomalies and produce a summary report."
>         },
>         {
>             "type": "file",
>             "mimeType": "text/csv",
>             "uri": "https://storage.example.com/data/sales_q3.csv"
>         },
>         {
>             "type": "data",
>             "data": {
>                 "schema": {
>                     "columns": ["date", "region", "product", "revenue", "units"],
>                     "types":   ["date", "string", "string", "float", "int"]
>                 },
>                 "expectedRowCount": 15000,
>                 "anomalyThreshold": 3.0  # z 分数阈值
>             }
>         }
>     ]
> }
> ```

> **多模态 A2A 消息：图像分析**
>
> ```python
> # 多模态消息：文本 + 图像 + 结构化数据
> multimodal_message = {
>     "role": "user",
>     "parts": [
>         {"type": "text",
>          "text": "Describe what is wrong with this chart and suggest fixes."},
>         {"type": "file",
>          "mimeType": "image/png",
>          "bytes": base64.b64encode(chart_image_bytes).decode()},
>         {"type": "data",
>          "data": {
>              "chartType": "bar",
>              "dataSource": "Q3 Revenue by Region",
>              "knownIssues": ["y-axis does not start at zero",
>                              "missing error bars"]
>          }}
>     ]
> }
> ```

### 上下文传递：哪些该共享，哪些应保留

多 Agent 系统中一个关键的设计决策是*上下文范围控制*：要把多少对话历史和内部状态传递给子 Agent。

> **上下文范围控制原则**
>
> **最小化上下文（Minimal Context）**
>
> 只传递子 Agent 完成其任务所必需的内容。可降低 Token 用量、延迟以及敏感信息泄露的风险。
>
> **摘要化上下文（Summarized Context）**
>
> 不要传递原始对话历史，而是传递结构化摘要：目标、约束、已做出的决策以及相关事实。
>
> **私有状态（Private State）**
>
> 内部推理、中间草稿以及用户的个人身份信息（PII），除非确有需要，通常*不应*转发给子 Agent。
>
> **关联 ID（Correlation IDs）**
>
> 始终传递 `correlationId`，以便子 Agent 的行为能够在日志和审计轨迹中追溯到发起的工作流。

### 会话串联与关联 ID

在复杂工作流中，许多任务可能同时处于进行中。**关联 ID** 用于跨 Agent 串联相关任务：

```python
import uuid

class WorkflowContext:
    """在多 Agent 工作流中携带关联元数据。"""

    def __init__(self, workflow_id: str | None = None):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id: str | None = None

    def child_context(self) -> "WorkflowContext":
        """为子任务创建子上下文。"""
        child = WorkflowContext(workflow_id=self.workflow_id)
        child.parent_span_id = self.span_id
        return child

    def to_metadata(self) -> dict:
        return {
            "x-workflow-id": self.workflow_id,
            "x-span-id": self.span_id,
            "x-parent-span-id": self.parent_span_id
        }

# 使用方式：在每次 A2A 任务提交时附加该元数据
ctx = WorkflowContext()
task = await client.send_task(
    message=message,
    metadata=ctx.to_metadata()
)
# 子任务使用子上下文以便追踪
sub_ctx = ctx.child_context()
```

## 协调协议

除了点对点通信，多 Agent 系统还能从更高层次的**协调协议**中受益——这些是支持集体决策与问题求解的结构化交互模式。

### 合同网协议（Contract Net Protocol）

**合同网协议（Contract Net Protocol, CNP）** [[362]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-smith1980contract)] 是一种经典的多 Agent 协调机制，可被改造用于基于 LLM 的系统：

1. **宣告（Announcement）**：管理者 Agent 向所有潜在承包者 Agent 广播任务公告，其中包含任务需求和评估标准。
2. **竞标（Bidding）**：承包者 Agent 根据自身能力评估任务，并提交包含预计完成时间、置信度和资源需求的竞标。
3. **授标（Award）**：管理者选择中标方（对于并行子任务则可选择多个中标方），并授予合同。
4. **执行与汇报（Execution and Reporting）**：承包者执行任务并将结果汇报给管理者。

> **合同网协议实现**
>
> ```python
> import dataclasses
>
> class ContractNetManager:
>     """为任务分配实现合同网协议（Contract Net Protocol）。"""
>
>     async def allocate_task(self, task: Task,
>                             candidate_agents: list[AgentCard]) -> AgentCard:
>         # 阶段 1：向所有候选 Agent 宣告任务
>         announcement = {
>             "type": "task-announcement",
>             "task": dataclasses.asdict(task),
>             "deadline": (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat(),
>             "evaluationCriteria": ["confidence", "estimatedTime", "cost"]
>         }
>
>         # 阶段 2：收集竞标
>         bids = await asyncio.gather(*[
>             self._request_bid(agent, announcement)
>             for agent in candidate_agents
>         ], return_exceptions=True)
>
>         valid_bids = [(agent, bid) for agent, bid in zip(candidate_agents, bids)
>                       if not isinstance(bid, Exception) and bid is not None]
>
>         if not valid_bids:
>             raise RuntimeError(f"No agents bid on task {task.id}")
>
>         # 阶段 3：授予最佳竞标方（置信度最高、用时最少）
>         def score_bid(agent_bid):
>             _, bid = agent_bid
>             return bid["confidence"] - 0.1 * bid["estimatedSeconds"]
>
>         winner_agent, winning_bid = max(valid_bids, key=score_bid)
>
>         # 通知中标方与落标方
>         await self._award_contract(winner_agent, task)
>         await asyncio.gather(*[
>             self._reject_bid(agent, task.id)
>             for agent, _ in valid_bids if agent != winner_agent
>         ])
>
>         return winner_agent
>
>     async def _request_bid(self, agent: AgentCard,
>                            announcement: dict) -> dict | None:
>         """请求某个 Agent 对任务进行竞标。"""
>         try:
>             result = await self.client.send_task(
>                 agent_url=agent.url,
>                 message={"role": "user",
>                          "parts": [{"type": "data", "data": announcement}]}
>             )
>             return result.artifacts[0].parts[0]["data"]
>         except Exception:
>             return None
> ```

### 黑板系统（Blackboard Systems）

**黑板系统** [[309]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hayes1985blackboard)]提供一个共享工作区（即"黑板"），Agent 在上面发布部分解、观察和假设。其他 Agent 监视黑板，在能够提供价值时进行贡献——这是一种*机会主义*的问题求解方式。

黑板系统非常适合那些求解路径事先未知、不同 Agent 在不同阶段做出贡献的问题——例如科学假设生成、复杂调试或多源情报分析。

### 共识协议（Consensus Protocols）

当多个 Agent 必须就一项决策达成一致（例如执行哪个计划、结果是否正确），**共识协议**提供了结构化的投票机制：

**简单多数投票（Simple Majority Voting）**

每个 Agent 投票；获得 $> 50\%$ 选票的选项胜出。速度快，但当 Agent 共享同一基础模型时，容易受到相关性错误的影响。

**加权投票（Weighted Voting）**

按 Agent 的置信度或历史准确率对选票加权。更稳健，但需要经过良好校准的置信度估计。

**基于法定人数（Quorum-Based）**

一项决策需要 $n$ 个 Agent 中至少 $k$ 个达成一致。提供容错能力：最多可有 $n-k$ 个 Agent 失败或不同意而不会阻塞流程。

**德尔菲方法（Delphi Method）**

Agent 先投票，看到匿名化的结果后修改投票，并反复迭代直至收敛。可减少锚定偏差，鼓励真正的深入审议。

```python
async def quorum_vote(agents: list[AgentCard], question: str,
                      options: list[str], quorum: int) -> str | None:
    """在 Agent 之间进行法定人数投票。返回获胜选项或 None。"""
    votes = await asyncio.gather(*[
        ask_agent_to_vote(agent, question, options)
        for agent in agents
    ])

    counts: dict[str, int] = {}
    for vote in votes:
        if vote in options:
            counts[vote] = counts.get(vote, 0) + 1

    # 返回第一个达到法定人数的选项
    for option, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= quorum:
            return option
    return None  # 未达到法定人数
```

### 领导者选举（Leader Election）

在动态的多 Agent 系统中，可能需要在运行时选举一位**领导者**（编排器）——例如原编排器失败时，或当 Agent 在没有预先指派协调者的情况下自组织时。经典的分布式系统算法（Bully、Ring）可被改造用于 Agent 网络：Agent 间交换能力分数或优先级令牌，以选举出当前可用 Agent 中能力最强的作为领导者。

## A2A 与 MCP：互补的协议

A2A 与 **模型上下文协议（Model Context Protocol, MCP）** [[323]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic-mcp-2024)] 之间的关系常被误解。这两种协议是*互补*的，并非竞争关系：

> **核心区别**
>
> - **MCP** 是*纵向*协议：它把一个 Agent 向下延伸到数据库、API、文件系统和代码执行器的世界。只有 Agent 在推理；MCP 端点是确定性服务。
> - **A2A** 是*横向*协议：它把一个具备推理能力的 Agent 与另一个相连。两端都是能够推理、规划和使用 Tool 的智能体。

| 维度 | MCP | A2A |
| --- | --- | --- |
| **参与方** | Agent $\leftrightarrow$ Tool/资源 | Agent $\leftrightarrow$ Agent |
| **智能性** | 仅一端（Agent）具备智能 | 双方均具备智能 |
| **状态性** | 通常为无状态 Tool 调用 | 带生命周期的有状态任务 |
| **流式传输** | 有限（Tool 结果） | 一等的 SSE 流式传输 |
| **发现机制** | Tool 清单 | Agent Card |
| **认证模型** | 服务端控制 | 双向认证、OAuth 2.0 |
| **典型延迟** | 毫秒级 | 秒到分钟级 |
| **典型用例** | "搜索网络"、"运行 SQL" | "委派给专家 Agent" |

### 何时使用哪一种

- 当远程端点是一个确定性函数时使用 **MCP**：数据库查询、API 调用、代码执行沙箱。Agent 完全掌控交互过程。
- 当远程端点需要对请求进行*推理*时使用 **A2A**：解释模糊指令、做出判断、使用自有 Tool 或进行多轮对话。
- 在同一系统中**同时使用两者**：编排 Agent 通过 A2A 将任务委派给专家 Agent，而每个专家 Agent 通过 MCP 访问自己的 Tool。

### 组合架构

在生产环境的多 Agent 系统中，A2A 与 MCP 在不同层次上协同工作：**A2A** 处理 Agent 之间的委派与协调（同级之间的横向通信），而 **MCP** 处理每个 Agent 到其 Tool 与数据源的连接（与能力的纵向集成）。这种关注点分离是构建可扩展 Agent 架构的关键。

![A2A 与 MCP 组合架构。编排器通过 A2A 将任务委派给专家 Agent；每个 Agent 则通过 MCP 服务器访问自己的 Tool。]({{ site.baseurl }}/figures/fig_068_combined-a2a-mcp.png)

- **用 A2A 进行委派**：当一个 Agent 需要它本身不具备的能力时，会通过 A2A 任务消息委派给另一个 Agent。每个 Agent 都是带有自身 Agent Card 的自包含服务。
- **用 MCP 访问 Tool**：每个 Agent 通过 MCP 服务器连接其 Tool。这意味着 Tool 永远不会直接暴露给其他 Agent——只能通过所属 Agent 的接口访问。
- **信任边界的分离**：编排器信任专家 Agent（通过 A2A 认证进行核实）。每个专家信任其自有的 MCP 服务器（本地或已认证）。不存在可传递的 Tool 访问。
- **独立扩缩容**：代码密集型负载可扩展 CodeAgent 实例；数据型负载可扩展 DataAgent。编排器始终保持轻量。

## 多 Agent 系统中的安全与信任

多 Agent 系统带来了独特的安全挑战。当 Agent A 委派给 Agent B、Agent B 又委派给 Agent C 时，信任链必须被精心管理。

### Agent 身份验证

每个 Agent 都必须具有可验证的身份。可选方案包括：

- 由可信身份提供方签发的 **JWT 令牌** [[363]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rfc7519)]，携带 Agent ID、颁发者和过期时间。接收方 Agent 使用提供方的公钥进行验证。
- 由内部 CA 颁发的 **mTLS 证书** [[364]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rfc8705)]，同时提供认证与传输加密。
- 适用于不存在单一可信权威的跨组织场景的 **去中心化标识符（Decentralized Identifiers, DIDs）** [[365]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-w3c-did-2022)]。

### 消息完整性与加密

- 所有 A2A 通信都应通过 **TLS 1.3** [[366]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rfc8446)] 进行，以防止窃听和中间人攻击。
- 对于敏感负载，**端到端加密**（例如 JWE）可以确保中间基础设施（负载均衡器、代理）无法读取消息内容。
- **消息签名**（JWS）提供不可否认性：接收方 Agent 可以证明某条特定消息确实来自某位特定发送者。

### 授权 scope

并非每个 Agent 都应该能够要求其他任何 Agent 做任何事。OAuth 2.0 授权 scope [[367]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rfc6749)] 定义了边界：

```python
# DataAgent 的 OAuth 2.0 scope 示例
SCOPES = {
    "data:read":        "Read data from connected databases",
    "data:write":       "Write or modify data in connected databases",
    "data:export":      "Export data to external systems",
    "analysis:run":     "Execute statistical analyses",
    "analysis:schedule":"Schedule recurring analyses",
    "admin:config":     "Modify agent configuration"
}

# ReportingAgent 可能仅持有：data:read, analysis:run
# ETL 流水线 Agent 可能持有：data:read, data:write, data:export
# 只有人类管理员持有：admin:config

class A2AServer:
    def verify_authorization(self, token: str, required_scope: str) -> bool:
        """验证调用方 Agent 是否持有所需 scope。"""
        claims = jwt.decode(token, self.public_key, algorithms=["RS256"])
        granted_scopes = claims.get("scope", "").split()
        if required_scope not in granted_scopes:
            raise PermissionError(
                f"Caller lacks required scope '{required_scope}'. "
                f"Granted: {granted_scopes}"
            )
        return True
```

### 审计轨迹与问责

> **问责鸿沟**
>
> 在一条 Agent 委派链中，*谁*应对某个行为负责可能会变得模糊。如果 Agent A 让 Agent B 删除一个文件，而 Agent B 照做了，那么应由谁负责？每一次 A2A 交互都必须记录：调用方 Agent 的身份、任务描述、所用授权令牌、时间戳以及结果。这种审计轨迹对于事件响应、合规和调试都至关重要。

每个 A2A 服务器都应输出结构化的审计日志：

```python
@dataclass
class A2AAuditEvent:
    timestamp: str          # ISO 8601
    workflow_id: str        # 顶层工作流的关联 ID
    span_id: str            # 当前任务的 span
    parent_span_id: str     # 调用任务的 span（用于委派链）
    caller_agent_id: str    # 经验证的调用方 Agent 身份
    callee_agent_id: str    # 当前 Agent 的身份
    task_id: str
    skill_invoked: str
    authorization_scopes: list[str]
    outcome: str            # "completed" | "failed" | "rejected"
    duration_ms: int
    error_code: str | None
```

## 实现示例：多 Agent 研究工作流

下面的示例演示了一个完整的、基于 A2A 的多 Agent 研究工作流：一个 `OrchestratorAgent` 分解研究问题，将子任务委派给专家 Agent，并综合它们的结果。

```python
"""
使用 A2A 协议实现的多 Agent 研究工作流。
演示内容：Agent Card、A2A 客户端/服务器、任务生命周期、
多轮交互以及 Agent 间交接。
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# -- 数据模型 ------------------------------------------------------------------

class Part(BaseModel):
    type: str           # "text" | "file" | "data"
    text: str | None = None
    data: dict | None = None
    mimeType: str | None = None
    uri: str | None = None

class Message(BaseModel):
    role: str           # "user" | "agent"
    parts: list[Part]

class TaskStatus(BaseModel):
    state: str          # submitted | working | input-required | completed | failed
    message: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

class Artifact(BaseModel):
    parts: list[Part]
    index: int = 0
    append: bool = False
    lastChunk: bool = True

class Task(BaseModel):
    id: str
    status: TaskStatus
    messages: list[Message] = []
    artifacts: list[Artifact] = []
    metadata: dict = {}

# -- A2A 客户端（HTTP/REST 绑定） ---------------------------------------------
# 注意：A2A v1.0 定义了三种协议绑定：JSON-RPC 2.0、gRPC，
# 以及 HTTP+JSON/REST。此示例为提高可读性使用 REST 绑定。

class A2AClient:
    """用于向兼容 A2A 的 Agent 发送任务的客户端。"""

    def __init__(self, agent_url: str, auth_token: str):
        self.agent_url = agent_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

    async def get_agent_card(self) -> dict:
        """获取该 Agent 的能力卡片。"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.agent_url}/.well-known/agent.json",
                headers=self.headers
            )
            resp.raise_for_status()
            return resp.json()

    async def send_task(self, message: Message,
                        task_id: str | None = None,
                        metadata: dict | None = None) -> Task:
        """提交一个任务，返回初始的 Task 对象。"""
        payload = {
            "id": task_id or str(uuid.uuid4()),
            "message": message.model_dump(),
            "metadata": metadata or {}
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.agent_url}/tasks/send",
                json=payload,
                headers=self.headers,
                timeout=30.0
            )
            resp.raise_for_status()
            return Task(**resp.json())

    async def stream_task(self, message: Message,
                          metadata: dict | None = None) -> AsyncIterator[dict]:
        """提交任务并以 SSE 事件流的形式接收结果。"""
        payload = {
            "id": str(uuid.uuid4()),
            "message": message.model_dump(),
            "metadata": metadata or {}
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.agent_url}/tasks/sendSubscribe",
                json=payload,
                headers={**self.headers, "Accept": "text/event-stream"},
                timeout=300.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        yield event_data
                        if event_data.get("final"):
                            break

    async def get_task(self, task_id: str) -> Task:
        """轮询任务状态。"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.agent_url}/tasks/{task_id}",
                headers=self.headers
            )
            resp.raise_for_status()
            return Task(**resp.json())

    async def wait_for_completion(self, task: Task,
                                  poll_interval: float = 2.0) -> Task:
        """轮询直至任务到达终止状态。"""
        terminal_states = {"completed", "failed", "canceled"}
        while task.status.state not in terminal_states:
            await asyncio.sleep(poll_interval)
            task = await self.get_task(task.id)
        return task

# -- A2A 服务器（FastAPI） ----------------------------------------------------

class ResearchAgent:
    """
    一个专门的研究 Agent，能够检索文献并对给定主题
    的研究结果进行总结。
    """

    AGENT_CARD = {
        "name": "ResearchAgent",
        "description": "Searches academic literature and synthesizes research findings.",
        "url": "https://research-agent.example.com/a2a",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        "authentication": {"schemes": ["Bearer"]},
        "skills": [{
            "id": "literature-search",
            "name": "Literature Search",
            "description": "Search and summarize academic papers on a topic.",
            "tags": ["research", "literature", "academic", "papers"],
            "examples": [
                "Summarize recent papers on transformer attention mechanisms.",
                "What does the literature say about RLHF for code generation?"
            ],
            "inputModes": ["text"],
            "outputModes": ["text", "data"]
        }]
    }

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.app = FastAPI(title="ResearchAgent A2A Server")
        self._register_routes()

    def _register_routes(self):
        @self.app.get("/.well-known/agent.json")
        async def agent_card():
            return self.AGENT_CARD

        @self.app.post("/tasks/send")
        async def send_task(request: Request):
            body = await request.json()
            task = await self._create_and_run_task(body)
            return task.model_dump()

        @self.app.post("/tasks/sendSubscribe")
        async def send_subscribe(request: Request):
            body = await request.json()
            return StreamingResponse(
                self._stream_task(body),
                media_type="text/event-stream"
            )

        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            return self.tasks[task_id].model_dump()

    async def _create_and_run_task(self, body: dict) -> Task:
        task_id = body.get("id", str(uuid.uuid4()))
        message = Message(**body["message"])

        task = Task(
            id=task_id,
            status=TaskStatus(state="submitted"),
            messages=[message],
            metadata=body.get("metadata", {})
        )
        self.tasks[task_id] = task

        # 异步执行
        asyncio.create_task(self._execute_task(task_id))
        return task

    async def _execute_task(self, task_id: str):
        task = self.tasks[task_id]
        task.status = TaskStatus(state="working")

        try:
            # 从消息中提取研究问题
            question = task.messages[0].parts[0].text

            # 模拟文献检索（实际生产中替换为真实搜索 Tool）
            await asyncio.sleep(1)  # 模拟延迟
            findings = await self._search_literature(question)

            # 生成产物
            task.artifacts = [Artifact(parts=[
                Part(type="text", text=findings["summary"]),
                Part(type="data", data={"papers": findings["papers"],
                                        "query": question})
            ])]
            task.status = TaskStatus(state="completed")

        except Exception as e:
            task.status = TaskStatus(state="failed", message=str(e))

        self.tasks[task_id] = task

    async def _search_literature(self, question: str) -> dict:
        """占位实现：生产环境中调用真实的搜索 API。"""
        return {
            "summary": f"Based on a search of recent literature regarding "
                       f"'{question}', key findings include: ...",
            "papers": [
                {"title": "Attention Is All You Need", "year": 2017,
                 "relevance": 0.95},
                {"title": "RLHF: Training Language Models to Follow Instructions",
                 "year": 2022, "relevance": 0.88}
            ]
        }

    async def _stream_task(self, body: dict) -> AsyncIterator[str]:
        task = await self._create_and_run_task(body)

        # 推送状态更新
        yield f"data: {json.dumps({'id': task.id, 'status': {'state': 'submitted'}, 'final': False})}\n\n"
        yield f"data: {json.dumps({'id': task.id, 'status': {'state': 'working'}, 'final': False})}\n\n"

        # 等待完成
        while task.status.state not in ("completed", "failed", "canceled"):
            await asyncio.sleep(0.5)
            task = self.tasks[task.id]

        # 推送产物
        if task.artifacts:
            for part in task.artifacts[0].parts:
                event = {
                    "id": task.id,
                    "artifact": {
                        "parts": [part.model_dump()],
                        "index": 0,
                        "append": False,
                        "lastChunk": True
                    },
                    "final": False
                }
                yield f"data: {json.dumps(event)}\n\n"

        # 最终状态
        yield f"data: {json.dumps({'id': task.id, 'status': task.status.model_dump(), 'final': True})}\n\n"

# -- 编排器：多 Agent 工作流 --------------------------------------------------

class ResearchOrchestrator:
    """
    编排一个多 Agent 研究工作流：
    1. 将研究问题分解为子问题
    2. 将每个子问题分派给 ResearchAgent
    3. 将结果综合为最终报告
    """

    def __init__(self, research_agent_url: str, auth_token: str):
        self.research_client = A2AClient(research_agent_url, auth_token)
        self.workflow_id = str(uuid.uuid4())

    async def run(self, research_question: str) -> str:
        print(f"[Orchestrator] Starting workflow {self.workflow_id}")
        print(f"[Orchestrator] Question: {research_question}")

        # 第 1 步：分解为子问题
        sub_questions = self._decompose(research_question)
        print(f"[Orchestrator] Decomposed into {len(sub_questions)} sub-questions")

        # 第 2 步：并行分派子问题
        tasks = await asyncio.gather(*[
            self.research_client.send_task(
                message=Message(role="user", parts=[Part(type="text", text=q)]),
                metadata={"workflowId": self.workflow_id, "subQuestion": i}
            )
            for i, q in enumerate(sub_questions)
        ])

        # 第 3 步：等待所有任务完成
        completed_tasks = await asyncio.gather(*[
            self.research_client.wait_for_completion(task)
            for task in tasks
        ])

        # 第 4 步：检查失败情况
        failed = [t for t in completed_tasks if t.status.state == "failed"]
        if failed:
            print(f"[Orchestrator] Warning: {len(failed)} sub-tasks failed")

        # 第 5 步：综合结果
        findings = []
        for task, question in zip(completed_tasks, sub_questions):
            if task.status.state == "completed" and task.artifacts:
                summary = task.artifacts[0].parts[0].text
                findings.append(f"### {question}\n{summary}")

        report = self._synthesize(research_question, findings)
        print(f"[Orchestrator] Workflow complete. Report: {len(report)} chars")
        return report

    def _decompose(self, question: str) -> list[str]:
        """将复杂问题分解为聚焦的子问题。"""
        # 生产环境中：使用 LLM 进行分解
        return [
            f"What are the foundational methods for: {question}?",
            f"What are the most recent advances in: {question}?",
            f"What are the open challenges and limitations in: {question}?"
        ]

    def _synthesize(self, question: str, findings: list[str]) -> str:
        """将子结论综合为一份连贯的报告。"""
        # 生产环境中：使用 LLM 进行综合
        sections = "\n\n".join(findings)
        return f"# Research Report: {question}\n\n{sections}"

# -- 入口 ---------------------------------------------------------------------

async def main():
    orchestrator = ResearchOrchestrator(
        research_agent_url="https://research-agent.example.com/a2a",
        auth_token="eyJhbGciOiJSUzI1NiJ9..."
    )
    report = await orchestrator.run(
        "Reinforcement learning from human feedback for large language models"
    )
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
```

## 小结

> **关键要点：Agent 到 Agent 通信**
>
> 1. **A2A 让规模化的专业化成为可能**：通过将任务路由给专家 Agent，多 Agent 系统可同时获得深度与广度。（第 "多智能体系统" 章将深入介绍多 Agent 架构。）
> 2. **Google 的 A2A 协议**为可互操作的 Agent 通信提供了一个面向生产的开放标准，涵盖 Agent Card、任务生命周期管理、SSE 流式传输和企业级认证。
> 3. **通信模式**从简单的请求-响应，到复杂的协商和基于拍卖的分配——可根据任务复杂度和延迟需求来选择。
> 4. **A2A 与 MCP 互补**：A2A 连接 Agent 与 Agent；MCP 连接 Agent 与 Tool。大多数生产系统两者并用。
> 5. **安全不可妥协**：Agent 身份验证、授权 scope 和审计轨迹在任何多 Agent 部署中都不可或缺。
> 6. **协调协议**（合同网、黑板、共识）在简单委派之外，提供了用于集体决策的结构化机制。
> 7. **基于关联 ID 的可观测性**对于调试和审计跨越多个 Agent 与 Tool 的复杂多 Agent 工作流至关重要。

> **A2A 中的开放研究问题**
>
> - Agent 应如何处理来自层级中多个编排器的*相互冲突的指令*？哪些冲突解决机制最为有效？
> - Agent 能否通过经验*学习*出更好的路由与委派策略，而不再依赖静态的能力声明？
> - 如何防范*Prompt 注入攻击*——即恶意 Agent 通过在消息中嵌入对抗性指令来操纵下游 Agent？
> - 上下文传递的合理*隐私边界*在哪里——子 Agent 应该看到多少对话历史，又如何在技术上强制执行这些边界？
> - 当 Agent 网络扩展到成百上千个 Agent 时，如何在不形成瓶颈或破坏一致性的前提下维持*连贯的全局状态*？
