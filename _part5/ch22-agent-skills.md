---
layout: home
title: Agent Skills
permalink: /part5/ch22-agent-skills.html
---

随着 Agent 从单体式的 prompt-加-工具系统演化为模块化架构，一个关键的设计问题浮现出来：*Agent 的能力应如何被组织、发现与组合？*答案越来越收敛到 **skill**（技能）这一概念——一些离散、可复用的行为单元，可以在不重新训练的前提下被加载、组合与替换。

这一思路由 Voyager [[216]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wang2023voyager)] 推广开来：该工作展示了一个在 Minecraft 中运行的 LLM Agent 可以不断积累一个可执行代码 skill 的库，每个 skill 都经过验证并被存储以供后续复用。同样的原则也适用于生产级 Agent：skill 以可组合、可版本化的形式封装领域专长，其可扩展性远超任何单个 prompt 所能容纳。Skill 通常会包装 MCP server（见第 "MCP" 章）以获得工具访问能力，将 skill 抽象与标准化工具层连接起来。

## 什么是 Skill？

一个 **skill** 是一个自包含的能力模块，赋予 Agent 在某个特定领域或任务上的专长。与仅暴露单个函数的原始 tool 不同，一个 skill 涵盖：

- **系统 prompt 增强**：注入到 Agent 上下文中的领域特定指令、约束与人设要素。
- **Tool 绑定**：该 skill 所需的一个或多个 tool（API、MCP server、本地命令）。
- **知识**：Agent 正确执行该 skill 所需的参考资料、示例或 few-shot 演示。
- **工作流逻辑**：引导 Agent 完成复杂任务的多步骤流程、决策树或条件流。
- **安全护栏**：该 skill 特有的安全约束、输出格式要求与校验规则。

> **Skill vs. Tool vs. Agent**
>
> | 概念 | 范围 | 示例 |
> | --- | --- | --- |
> | **Tool** | 单次函数调用 | `web_search(query)` |
> | **Skill** | 完整的能力（prompts + tools + 知识） | "Research Analyst" skill |
> | **Agent** | 拥有多个 skill 的自主实体 | 一个编程助手 |
>
> Tool 是一把锤子。Skill 是知道*如何搭起房屋的框架*。Agent 则是决定使用哪些 skill 的木匠。

## Skill 架构模式

### 静态 Skill 加载

最简单的模式：根据配置在 Agent 初始化时加载 skill。Agent 始终可以访问其全部 skill。

```python
# 伪代码 —— 与具体框架无关的模式
agent = Agent(
    model="claude-sonnet-4-20250514",
    skills=["code-review", "documentation", "testing"],
    # 每个 skill 都会向 Agent 添加 prompts、tools 与知识
)
```

**优点：**简单、可预测、低延迟。

**缺点：**未被使用的 skill 会浪费上下文窗口；无法扩展到数百个 skill。

### 动态 Skill 发现

Agent 根据当前任务选择激活哪些 skill。一个 skill 路由器（通常是一个轻量分类器或基于 embedding 的匹配器）来判断相关性：

```python
# 伪代码 —— 与具体框架无关的模式
relevant_skills = skill_router.match(
    user_request=message,
    available_skills=skill_registry,
    max_skills=3
)
agent.activate(relevant_skills)
```

**优点：**可扩展到大型 skill 库；上下文效率高。

**缺点：**路由错误可能漏掉相关 skill；引入额外延迟。

### 层级化 Skill 组合

Skill 可以依赖于其他 skill，从而构成一个有向无环图（DAG）。一个高层 skill（例如"部署应用"）可能会调用一些子 skill（"跑测试"、"构建 Docker 镜像"、"更新 DNS"）：

- Skill 显式声明其依赖
- 编排器在执行前解析依赖图
- 子 skill 可被多个父 skill 共享

## 案例研究：Anthropic 的 Agent 设计

Anthropic 对 Agent 架构的方法 [[330]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic2024buildingagents)] 是关于生产环境中基于 skill 的 Agent 设计最清晰的阐述之一。其设计哲学强调**简单优于复杂**、**可组合的构建块优于单体框架**。（这些模式也会在第 "Agent 设计模式" 章从编排视角进行讨论。）

### 核心原则

1. **从最简单的方案起步。**在尝试更简单的方案（单次 LLM 调用、检索 + 生成）并确认其不足之前，不要急于使用 Agent 化模式。
2. **Workflow 与 Agent 的区分。**Anthropic 将以下两者加以区分：

- **Workflows**：对 LLM 调用进行预先编排——确定性的控制流，在特定节点包含 LLM 步骤。更可预测、更易调试。
- **Agents**：由 LLM 动态决定下一步该做什么——工具选择、迭代次数与停止条件都由模型驱动。更灵活，但更难控制。

3. **以增强型 LLM 作为原子单元。**原语从来不是裸模型——它总是与其检索源、可调用工具和持久化记忆捆绑在一起的模型。这一组合单元在实践中就是"配备了 skill 的模型"。

### 构建块模式

Anthropic 总结了五种可组合的 workflow 模式，可作为 skill 模板使用：

| 模式 | 机制 | 适用场景 |
| --- | --- | --- |
| **Prompt Chaining** | 顺序的 LLM 调用，每一步的输出作为下一步的输入。步骤之间设有关卡，校验中间结果。 | 具有清晰分解的多步骤转换任务 |
| **Routing** | 一个分类器或 LLM 根据任务类型将输入分发给专门的处理者（skill）。 | 需要不同领域专长的多个独立任务类别 |
| **Parallelization** | 多个 LLM 调用同时进行——可以是分段（拆分任务），也可以是投票（同一任务、汇总结果）。 | 相互独立的子任务；或通过共识获得置信度 |
| **Orchestrator--Workers** | 一个中心 LLM 将任务拆分为子任务，分派给 worker LLM，再综合结果。 | 子任务难以事先预测的复杂任务 |
| **Evaluator--Optimizer** | 一个 LLM 生成，另一个 LLM 评估；反复迭代直至达到质量阈值。 | 具有明确质量标准的任务（代码、写作） |

### 增强型 LLM

在 Anthropic 的框架中，基本单元不是裸模型，而是**增强型 LLM**：

$$\text{Augmented LLM} = \text{Model} + \text{Retrieval} + \text{Tools} + \text{Memory}$$

这与 skill 概念直接对应：每个 skill 都为一个特定任务配置该模型可访问的检索源、tool 与记忆存储。Skill 边界定义了模型在某次特定调用中*能看到什么、能做什么*。

### 实际影响

> **Anthropic 的关键洞见**
>
> 最有效的 Agent 并不是最复杂的，而是**配有好用工具的简单循环**：
>
> ```python
> while not done:
>     action = llm.decide(context, tools)
>     result = execute(action)
>     context.append(result)
>     done = llm.should_stop(context)
> ```
>
> 智能来自（1）模型自身的能力、（2）工具描述的质量、（3）任务表述的清晰度——而非花哨的编排逻辑。Skill 为（2）和（3）提供了结构。

**源自 Anthropic 方法的设计建议：**

- **保持 Agent 循环简单**：避免过度工程化控制流，让模型自己决策。
- **在工具质量上投入**：详尽、无歧义的工具描述比复杂的路由逻辑更有价值。
- **使用结构化输出**：强制模型以可解析的格式（JSON、函数调用）输出决策——可减少 skill 执行错误。
- **内建恢复机制**：Skill 应优雅地处理错误——用不同参数重试、请求澄清，或升级到由人接管。
- **限制单个 skill 的范围**：试图无所不能的 skill 什么也做不好。狭窄、定义明确的 skill 比宽泛的 skill 更易组合。

## Skill 生命周期

1. **发现**：系统识别有哪些可用的 skill（注册表、市场、本地定义）。
2. **选择**：根据用户请求匹配并加载相关 skill。
3. **激活**：将 skill 的 prompt、tool 与知识注入到 Agent 的上下文中。
4. **执行**：Agent 利用该 skill 的能力完成任务。
5. **失活**：移除 skill 上下文，为后续任务释放上下文窗口空间。
6. **学习**：执行结果可能更新该 skill 的 few-shot 示例或微调路由。

## Skill 注册表与市场

生产级 skill 系统需要相应的基础设施：

- **Skill 清单（manifest）**：一份结构化描述（名称、能力、所需工具、输入/输出 schema），用于支持自动发现与路由。
- **版本控制**：Skill 会不断演进；Agent 需要锁定特定版本以保证可复现性。
- **依赖解析**：Skill 可能需要特定的 MCP server、API key 或其他 skill。
- **权限模型**：并非所有 Agent 都应能访问所有 skill（安全性、成本与能力边界考虑）。
- **市场**：组织可以发布、分享与安装 skill——类似于代码的包管理器。

> **Skill 清单示例**
>
> 一份 skill 清单声明了编排器加载并调用一个 skill 所需的一切。目前尚不存在行业标准 schema；下面是一个示意性格式，它涵盖了真实实现（Anthropic MCP、OpenAI 函数规范、LangChain 工具定义）中常见的字段：
>
> ```python
> // 示意性 schema —— 并非任何具体 SDK 的格式
> {
>   "name": "code-review",
>   "description": "Review code changes for bugs, style, and security issues",
>   "version": "2.1.0",
>   "requires": {
>     "tools": ["file_read", "grep", "git_diff"],
>     "mcp_servers": ["github"],
>     "models": ["claude-sonnet-4-20250514"]
>   },
>   "input_schema": {
>     "type": "object",
>     "properties": {
>       "repo": {"type": "string"},
>       "pr_number": {"type": "integer"}
>     }
>   },
>   "prompts": ["skills/code-review/system.md"],
>   "knowledge": ["skills/code-review/style-guide.md"]
> }
> ```

## Skill vs. 微调

一个自然的问题：为什么要使用运行时的 skill 注入，而不是对模型进行微调？

| 维度 | Skill（上下文内） | 微调 |
| --- | --- | --- |
| 部署速度 | 即时 | 数小时--数天 |
| 灵活性 | 运行时可替换/组合 | 训练时固定 |
| 上下文成本 | 占用上下文窗口 | 运行时零成本 |
| 深层行为改变 | 受上下文长度限制 | 深层参数级变化 |
| 多租户 | 不同用户可用不同 skill | 所有用户共用同一模型 |
| 维护 | 更新文本文件 | 在新数据上重新训练 |

在实践中，两种方法是互补的：微调提供*基础能力*（指令跟随、工具使用格式、推理），而 skill 在运行时叠加提供*任务特定的专长*。
