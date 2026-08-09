---
layout: home
title: Agent 设计模式
permalink: /part5/ch19-agent-design-patterns.html
---

构建有效的 Agent 不仅需要强大的模型和一组工具。*架构*——即如何编排 LLM、如何分解任务、控制流如何在各组件间流动——决定了 Agent 是否可靠、可调试且具备成本效益。本章介绍从 Anthropic、OpenAI、Google 以及开源社区的生产部署中沉淀出的经典设计模式。

> **何时使用 Agent 而非工作流**
>
> 并非每个任务都需要一个自主 Agent。关键区别在于：
>
> - **工作流（Workflow）**：预定义的控制流，在特定步骤调用 LLM。可预测、可测试、成本更低。当任务结构已知时使用。
> - **Agent**：LLM 动态决定下一步做什么。灵活、可应对新情况。当任务需要自适应决策时使用。
>
> **从工作流开始。** 只有当任务确实需要动态路由或开放式探索时，才升级到 Agent。

## 工作流模式

以下模式改编自 Anthropic 对 Agent 构建模块的分类 [[330]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic2024buildingagents)]，在*预定义*的控制流中使用 LLM。由系统（而非模型）决定执行顺序。

### Prompt 链（Prompt Chaining）

最简单的模式：将复杂任务拆分为固定序列的 LLM 调用，将一个调用的结果作为上下文传入下一个调用。步骤之间的校验门可在错误向下游传播之前及早捕获它们。

![带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。]({{ site.baseurl }}/figures/fig_060_fig60.png)

**适用场景**：天然顺序化的任务——内容生成、数据转换、多阶段分析。

**关键优势**：每一步可以使用不同的 Prompt、模型或温度。中间结果可检查、可调试。

### 路由（Routing）

由分类器（LLM 或传统方法）检查输入，并分派到专门的处理器。

![路由模式：输入被分类一次，然后由专门的处理器处理。]({{ site.baseurl }}/figures/fig_061_fig61.png)

**适用场景**：不同任务类型对应不同的最佳 Prompt、工具或模型。如客服分诊、多模态输入处理。

### 并行化（Parallelization）

多个 LLM 调用并发运行，由一个程序层合并它们的输出。可分为两个子模式：

- **分段（Sectioning，fan-out）**：将输入划分为互不相交的块并独立处理——例如对一个代码库同时运行安全、性能和风格检查。
- **投票（Voting，冗余）**：用不同随机种子或温度对同一 Prompt 发起 $N$ 次调用，然后通过多数投票 [[331]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wang2022selfconsistency)]、奖励模型打分或 LLM-as-judge 选出最佳结果。

> **并行化示例：代码评审**
>
> 1. **并行调用**：安全评审 $\|$ 性能评审 $\|$ 风格评审
> 2. **聚合**：合并所有发现，去重，按严重程度排序
>
> 延迟为 $\max$（各调用），而非 $\sum$（各调用）。

### Orchestrator-Workers

在该模式下，由 LLM 自身决定如何拆分工作。一个 Orchestrator 模型分析任务、产出子任务计划、将每个子任务分派给 Worker LLM（可能使用不同的 Prompt 或工具），最终将它们的输出合并为一致的结果。与并行化的关键区别在于：分解逻辑是由模型生成的，而非硬编码的。

![Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。]({{ site.baseurl }}/figures/fig_062_fig62.png)

**适用场景**：开放式问题，子任务的数量和性质无法在设计时穷举——例如“重构这个代码库”需要先理解依赖图，再决定要修改哪些文件。

### Evaluator-Optimizer

一个双模型反馈循环 [[228]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-madaan2023selfrefine)]：生成器产出候选输出，由独立的评估器依据显式标准对其打分。若得分低于阈值，则将评估器的评论追加到生成器的上下文中，循环往复，直到达到质量标准或耗尽重试预算。

![Evaluator-optimizer：无需训练的迭代式精化。]({{ site.baseurl }}/figures/fig_063_fig63.png)

**适用场景**：具有明确质量标准的任务——必须通过测试的代码、必须保留语义的翻译、必须符合风格指南的写作。

## 自主 Agent 模式

这些模式把执行流的控制权交给 LLM 本身。

### ReAct（Reason + Act）

最基础的 Agent 模式 [[108]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yao2023react)]。LLM 在思考（内部推理）、行动（工具调用）和观察（处理结果）之间循环交替，直到产出最终答案。

> **ReAct 实现要点**
>
> - **草稿本（Scratchpad）**：“思考”步骤被记录但不向用户展示。
> - **工具解析**：脚手架（harness）从模型输出中抽取结构化的工具调用。
> - **最大迭代次数**：始终对循环设上限（通常 10--25 次迭代）。
> - **终止条件**：模型输出特殊动作（如 `final_answer`），或检测不到任何工具调用。

### 规划型 Agent（Planning Agents）

Agent 在执行前生成显式计划，并可在执行过程中修订计划 [[107]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wang2023planandsolve)]。

| 策略 | 再规划时机 | 特征 |
| --- | --- | --- |
| Plan-then-Execute | 从不 | 简单；对意外结果脆弱 |
| Adaptive | 失败时 | 仅在某步失败时再规划；成本适中 |
| Continuous | 每一步 | 每次观察后完整重估；昂贵但鲁棒 |
| Hierarchical | 子计划完成时 | 高层计划固定；子计划动态生成 |

> **规划型 Agent：研究报告生成**
>
> **用户请求**：“写一份 2 页的报告，比较用于时间序列预测的 transformer 架构。”
>
> **步骤 1 --- 计划生成**（单次 LLM 调用）：
>
> ```python
> plan = [
>     {"id": 1, "task": "Search for recent transformer-based "
>                       "time-series models (2023-2025)",
>      "tool": "search_web", "deps": []},
>     {"id": 2, "task": "Read top 5 papers, extract key methods",
>      "tool": "read_papers", "deps": [1]},
>     {"id": 3, "task": "Build comparison table (architecture, "
>                       "dataset, metrics)",
>      "tool": "none", "deps": [2]},
>     {"id": 4, "task": "Write introduction + methodology section",
>      "tool": "none", "deps": [2]},
>     {"id": 5, "task": "Write results + conclusion",
>      "tool": "none", "deps": [3, 4]},
>     {"id": 6, "task": "Review and polish final report",
>      "tool": "none", "deps": [5]},
> ]
> ```
>
> **步骤 2 --- 带自适应再规划的执行**：Agent 按依赖顺序执行各步骤。第 1 步后，搜索仅返回 3 篇相关论文。Agent 进行*再规划*：新增一个子步骤，将搜索范围扩展到相邻领域（如 PatchTST、iTransformer）。修订后的计划基于扩充后的语料从第 2 步继续。
>
> **关键洞见**：计划是一份*活文档*——它提供结构，但会随观察而调整。harness 将依赖关系跟踪为 DAG，仅执行其前驱已完成的步骤。

### 反思与自我批判（Reflection and Self-Critique）

Agent 暂停以评估自身轨迹并纠正方向：

1. **输出校验**：“这正确吗？我有遗漏吗？”
2. **轨迹回顾**：回顾最近 $k$ 步，识别错误或低效之处。
3. **策略修订**：重新考虑整体方法（“我在解决正确的问题吗？”）。

> **Reflexion：从失败中学习**
>
> **Reflexion** 模式 [[212]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shinn2023reflexion)] 维护一个持久化的“反思记忆”。每次失败后，Agent 写下一段自然语言反思（“我失败是因为没有检查边界情况”）。在下一次尝试中，这些反思被纳入 Prompt——从而实现跨 Episode 的学习而无需更新权重。

### 工具调用模式（Tool-Use Patterns）

Agent 调用工具的方式显著影响其可靠性、延迟和成本。已涌现出五种经典模式 [[320]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schick2023toolformer)]：

| 模式 | 描述 | 示例 |
| --- | --- | --- |
| 单轮（Single-turn） | 每次 LLM 回复一次工具调用 | 带搜索的简单问答 |
| 多工具（Multi-tool） | 单次回复中并行多个工具调用 | 搜索 + 计算 + 格式化 |
| 顺序（Sequential） | 工具输出馈入下一次工具调用 | 搜索 $\to$ 阅读 $\to$ 抽取 |
| 嵌套（Nested） | 工具调用触发另一个 Agent | 代码 Agent 调用测试运行器 |
| 回退（Fallback） | 首选工具失败时尝试备选方案 | API $\to$ 爬取 $\to$ 缓存 |

**单轮工具使用。**

最简模式：模型发出一次工具调用、接收结果、产出最终答案。足以应对事实查询、单位换算或单次 API 查询。harness 恰好进行两次 LLM 调用（一次决定使用哪个工具，一次综合结果）。

**多工具（并行）。**

现代 API（OpenAI、Anthropic）允许模型在单次回复中请求多个工具调用。harness 并发执行它们并一起返回所有结果。对于需要从多个来源获取独立信息的任务（如同时获取股价、天气和日历），这能极大降低延迟。关键约束：这些工具必须*相互独立*（任何工具的输出都不作为另一工具的输入）。

**顺序（流水线）。**

每个工具的输出馈入下一个工具的输入，形成数据流水线。模型基于前一步结果决定下一个工具。在研究工作流中很常见：`search` $\to$ `fetch_page` $\to$ `extract_data` $\to$ `analyze`。harness 必须跟踪不断增长的上下文，并可能需要对中间结果进行摘要以保持在预算之内。

**嵌套（Agent-as-Tool）。**

一次工具调用会唤起一个完全独立的 Agent——拥有自己的 Prompt、工具和上下文。父 Agent 将子 Agent 视为黑盒函数。这实现了专业化：研究 Agent 将代码执行委派给编码 Agent，后者能访问沙箱和测试运行器。Swarm 模式 [[324]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-openai2024swarm)] 通过专业化 Agent 之间的交接（handoff）泛化了这一思路。

**回退（优雅降级）。**

harness 按优先级顺序尝试工具：若首选工具失败（超时、限流、API 错误），则自动回退到备选工具。模型无需感知回退逻辑——harness 透明处理。示例：主搜索 API $\to$ 备用搜索 $\to$ 缓存结果 $\to$ 告知模型搜索不可用。

## 设计原则

以下原则提炼自 Anthropic 的《构建有效 Agent》指南 [[330]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic2024buildingagents)]，适用于所有模式：

1. **保持简单。** 使用能跑通的最简架构。仅在确有必要时增加复杂度。一个能解决问题的 Prompt 链，永远胜过一个“也许能解决”的多 Agent 系统。
2. **透明优于聪明。** 每一步都应可检查。避免隐藏状态或隐式推理。当 Agent 失败时，你需要理解*原因*——不透明的架构使调试无从下手。
3. **提供良好工具。** 文档完善、类型清晰、错误信息明确的工具是效能放大器。描述含糊的工具会被误用；具备精确 Schema 和使用指南的工具会被正确选择。
4. **为失败做规划。** 每次工具调用都可能失败。在 harness 层构建重试逻辑、回退方案和优雅降级，使模型无需推理基础设施故障。
5. **使用结构化输出。** 受约束的生成（JSON Schema、函数调用）防止解析失败。一个产出需要正则解析的自由文本的 Agent 是脆弱的；一个产出经校验 JSON 的 Agent 是鲁棒的。
6. **使用多样化输入测试。** Agent 行为比单轮对话更易变。同一 Prompt 在不同运行中可能产生不同的工具调用序列。进行对抗性测试，覆盖边界情况、模糊请求和畸形输入。

## 模式选择指南

选择正确的模式取决于三个因素：(1)任务结构的可预测性如何，(2)延迟和成本上能承担多少次 LLM 调用，(3)质量是否需要迭代。可将下表作为决策矩阵——从顶部（最简模式）开始，仅当较简单的模式确实失败时才向下移动。

| 模式 | 复杂度 | LLM 调用次数 | 最适用于 |
| --- | --- | --- | --- |
| Prompt 链 | 低 | $N$（固定） | 顺序任务、内容流水线 |
| 路由 | 低 | 1 + 1 | 多类型输入、分诊 |
| 并行化 | 低 | $N$（并行） | 独立子任务、投票 |
| Orchestrator-workers | 中 | 可变 | 未知的分解结构 |
| Evaluator-optimizer | 中 | 2--10（循环） | 对质量要求高的输出 |
| ReAct | 中 | 3--25（循环） | 通用工具使用、探索 |
| 规划型 Agent | 高 | 5--50+ | 长时程、多步任务 |
| 反思 | 高 | +50% 开销 | 首次尝试常失败的任务 |
| 多 Agent | 高 | 多次 | 复杂领域、专业化 |

这些模式是可组合的：一个规划型 Agent 可以在各步骤内使用 Prompt 链，在评审阶段使用 Evaluator-optimizer，并使用路由将子任务分派给专家。艺术在于知道何时停止增加层次。
