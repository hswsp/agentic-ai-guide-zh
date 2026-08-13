---
layout: home
title: 智能体记忆系统
permalink: /part5/ch17-memory-systems.html
---

## 动机：为什么 Agent 需要记忆

大语言模型本质上是无状态的函数近似器：给定 Prompt $x$，它们产生续写分布 $$p_\theta(y \mid x)$$。每一次推理调用都从零开始。*上下文窗口*（context window）——模型可以关注的有限 Token 序列——是生成时唯一可用的信息。对短小、自包含的任务而言这已足够；但对长周期的智能体任务而言，这是一个根本性瓶颈。

> **上下文窗口瓶颈**
>
> 设 $L$ 为最大上下文长度（例如 GPT-4o 的 $L = 128{,}000$ 个 Token）。单个 Token 大约编码 4 个字符；一本典型书籍约含 $\sim\!500{,}000$ 词 $\approx 670{,}000$ 个 Token。即便不计成本，一个运行数日的自主 Agent 所累积的观察、工具输出和推理轨迹*不可能*塞进任何固定窗口。记忆系统正是对这一物理约束的工程回应。

当 Agent 缺乏持久化记忆时，会出现三种截然不同的失败模式：

1. **上下文的灾难性遗忘（Catastrophic Forgetting）。** 一旦事件滚出上下文窗口便不可挽回地丢失。Agent 无法回溯一万个 Token 前所做的决策。
2. **无法从经验中学习。** 没有情景式（episodic）存储，每个 Episode 对 Agent 都像是第一次。成功的策略无法复用，错误则反复发生。
3. **缺乏个性化。** 用户偏好、领域事实、关系历史必须在每次会话中重新建立，导致用户体验与效率双双下降。

> **记忆作为认知架构**
>
> 认知科学在生物智能体中区分出多种记忆系统 [[293]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-tulving1985memory), [294]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-squire1992declarative)]：*工作记忆*（对信息的主动操控）、*情景记忆*（自传式事件）、*语义记忆*（世界知识）以及*程序记忆*（技能与习惯）。有效的智能体 AI 系统从类似的划分中获益——并非因为我们要模拟神经科学，而是因为这些类别真实地反映了截然不同的*访问模式*、*更新频率*和*检索机制*。

形式化地，我们把 Agent 建模为一个元组 $$\mathcal{A} = (\pi_\theta, \mathcal{M}, \mathcal{R}, \mathcal{W})$$，其中 $$\pi_\theta$$ 是 Policy（LLM），$\mathcal{M}$ 是记忆存储，$\mathcal{R}: \mathcal{Q} \times \mathcal{M} \to \mathcal{D}$ 是把查询映射到检索结果的检索函数，$\mathcal{W}: \mathcal{M} \times \mathcal{E} \to \mathcal{M}$ 是用新经验 $\mathcal{E}$ 更新记忆的写入函数。在每一步 $t$，Agent 观察到 $$o_t$$，检索相关上下文 $$c_t = \mathcal{R}(o_t, \mathcal{M})$$，并采取行动：

$$
a_t \sim \pi_\theta\!\left(\cdot \;\middle\vert\; [s_t;\, c_t;\, h_t]\right),
$$

其中 $$s_t$$ 是当前的系统 Prompt，$$c_t$$ 是检索到的记忆，$$h_t$$ 是近期的上下文历史。行动之后，Agent 可写入新信息：$$\mathcal{M} \leftarrow \mathcal{W}(\mathcal{M},\, (o_t, a_t, r_t))$$。

## 记忆类型分类

![智能体记忆系统的四分类法，与认知科学的划分相对应。每种记忆类型具有截然不同的访问模式、更新频率和检索机制。]({{ site.baseurl }}/figures/fig_052_memory-taxonomy.png)

### 工作记忆（短期）

工作记忆是 Agent 的*活动工作空间*：当前正在被操控的信息。在 LLM Agent 中它对应于：

- **草稿板（Scratchpads）。** 在产出最终答案之前写入专用缓冲区的中间推理步骤（例如思维链 Chain-of-Thought [[103]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wei2022chain)]、scratchpad [[295]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-nye2021show)]）。
- **思维链缓冲区。** 在答案 Token $a$ 之前生成的推理 Token 序列 $$z_1, z_2, \ldots, z_k$$，建模为 $$p(a \mid x) = \sum_z p(a \mid x, z)\,p(z \mid x)$$。
- **对话上下文。** 保留在上下文窗口中的近期轮次历史 $$[(u_1, a_1), \ldots, (u_t, a_t)]$$。

工作记忆是*快速的*（零检索延迟——它已经在上下文中）、*易失的*（上下文清空时即丢失），且*容量受限*（受 $L$ 约束）。

### 情景记忆（基于经验）

情景记忆存储*具体的过往事件*，并按上下文与时间索引。对 Agent 而言：

- **过往交互。** 先前对话、任务尝试及其结果的完整或摘要记录。
- **成功轨迹。** 可作为少样本范例被检索、以服务于未来相似任务的高 Reward 动作序列。
- **失败案例。** 带有根因标注的错误记录，使 Agent 能够避免重复犯错。
- **检索增强的情景回忆。** 给定新任务 $q$，检索最相似的 $k$ 个过去 Episode $$\{e_i\}_{i=1}^k$$ 并加入上下文。

情景记忆通常实现为基于 Episode 摘要 Embedding 的向量存储（见「基于 RAG 的记忆」一节）。

### 语义记忆（世界知识）

语义记忆编码与具体 Episode 解耦的*一般性事实与概念*：

- **事实性知识。** 实体、属性与关系（例如 “Paris 是 France 的首都”）。
- **领域概念。** 与 Agent 任务领域相关的定义、分类法与本体。
- **知识图谱。** 结构化表示 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点 $v \in \mathcal{V}$ 为实体，边 $e \in \mathcal{E}$ 为带类型的关系。

与情景记忆不同，语义记忆是*上下文无关的*：水在 $100^\circ$C 沸腾这一事实，无论何时何地学到都成立。

### 程序记忆（技能）

程序记忆编码*如何做事*——已被自动化的技能与动作模式：

- **习得的工具使用模式。** 对哪种任务调用哪个 API、如何格式化输入、如何处理错误。
- **动作序列。** 多步过程（例如 “部署代码：跑测试 $\to$ 构建镜像 $\to$ 推送 $\to$ 更新 manifest”）。
- **作为记忆的 Policy。** 模型权重 $\theta$ 本身即编码了程序性知识；在成功轨迹上微调是程序记忆巩固的一种形式。

> **记忆类型的分类示例**
>
> 一个协助软件开发的 Agent 会用到：
>
> - **工作记忆**：当前正在编辑的文件、刚收到的错误信息。
> - **情景记忆**：“上周我在模块 X 第 42 行加了一个空值检查，修复了一个类似的 `NullPointerException`。”
> - **语义记忆**：“Python 的 `asyncio.gather` 并发运行协程；除非设置 `return_exceptions=True`，否则异常会向上传播。”
> - **程序记忆**：标准调试工作流——复现 $\to$ 隔离 $\to$ 提假设 $\to$ 验证 $\to$ 修复。

## 记忆架构

### 基于 RAG 的记忆

检索增强生成（Retrieval-Augmented Generation, RAG） [[109]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lewis2020retrieval)] 是 LLM Agent 外部记忆的主流范式。记忆存储 $\mathcal{M}$ 是文档集合 $$\{d_i\}_{i=1}^N$$；检索把查询 $q$ 映射到一个有序子集。

**Embedding 存储与向量数据库。**

每个文档 $$d_i$$ 由 Embedding 模型 $\phi$ 编码：$$\mathbf{v}_i = \phi(d_i) \in \mathbb{R}^{D}$$。查询同样被编码：$\mathbf{q} = \phi(q)$。检索按相似度返回 top-$k$ 文档：

$$
\text{Retrieve}(q, \mathcal{M}, k) = \underset{S \subseteq [N],\, \lvert S \rvert=k}{\arg\max} \sum_{i \in S} \text{sim}(\mathbf{q}, \mathbf{v}_i),
$$

其中 $\text{sim}(\cdot,\cdot)$ 通常为余弦相似度。近似最近邻（Approximate Nearest-Neighbor, ANN）索引（FAISS [[263]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-johnson2019billion)]、HNSW [[264]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-malkov2018efficient)]、ScaNN [[296]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-guo2020scann)]）使得 $N \sim 10^7$ 量级下仍然可行。

**检索策略。**

- **稠密检索（Dense retrieval）。** 查询和文档都由神经编码器编码（例如 DPR [[262]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-karpukhin2020dense)]、`text-embedding-3-large`）。能捕捉语义相似度，但需要 GPU 推理。
- **稀疏检索（Sparse retrieval）。** 基于 Token 重叠的 BM25 或 TF-IDF。快速、可解释、在精确关键词匹配上表现强劲。
- **混合检索（Hybrid retrieval）。** 通过倒数排名融合（Reciprocal Rank Fusion, RRF）合并稠密与稀疏分数：

$$
\text{RRF}(d, k) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)},
$$

其中 $k=60$ 是平滑常数。混合检索一致地优于单独使用任一种 [[297]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chen2022hybrid)]。

**重排序（Re-ranking）。**

交叉编码器（cross-encoder）重排序器 $$f_\psi(q, d) \in [0,1]$$ 与查询联合打分每个被检索文档，以 $O(k)$ 次前向传播的代价换取更高准确率。整体流水线为：用 ANN 检索 $k' \gg k$ 个候选，再用交叉编码器重排序，返回 top $k$。

> **检索引发的幻觉风险**
>
> RAG 并不能消除幻觉——反而可能*引入*幻觉。如果被检索的文档过时、错误或只是表面相关，模型可能自信地把错误信息纳入回答。务必包含来源元数据（来源、时间戳、置信度），并考虑加入忠实性（faithfulness）验证步骤。

### 基于摘要的记忆

当原样存储过于昂贵或过于嘈杂时，*摘要化*在存储之前先压缩信息。

**渐进式摘要（Progressive Summarization）。**

在每一步 $t$，Agent 维护一个运行中的摘要 $$S_t$$。当新信息 $$e_t$$ 到达时：

$$
S_{t+1} = \text{LLM}\!\left(\texttt{``Summarize: [}S_t\texttt{] + [}e_t\texttt{]''}\right).
$$

这样把记忆规模保持在 $O(1)$，但有丢失细节的风险。

**分层压缩（Hierarchical Compression）。**

将记忆组织为层级 $$L_0 \supset L_1 \supset \cdots \supset L_K$$，其中 $$L_0$$ 是原文，每个 $$L_{i+1}$$ 都是 $$L_i$$ 的摘要。检索首先访问 $$L_K$$（压缩程度最高、最快），按需逐层深入。这呼应了 Forte [[298]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-forte2022building)] 的*渐进式摘要*技术。

**何时摘要、何时原样存储。**

- 原样存储：精确事实、代码片段、数值结果、用户原话。
- 摘要存储：叙事性上下文、推理链、冗余观察。
- 直接丢弃：噪声、无信息量的失败工具调用。

### 基于图的记忆

**知识图谱。**

知识图谱 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$ 把事实存储为三元组 $(h, r, t)$，其中 $h, t \in \mathcal{V}$ 是实体，$r \in \mathcal{R}$ 是关系。Agent 可通过 SPARQL [[299]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-harris2013sparql)]、Cypher [[300]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-francis2018cypher)] 或自然语言到图的翻译进行查询。

**实体-关系抽取。**

新观察由抽取模型 $$\text{IE}: \text{text} \to \{(h_i, r_i, t_i)\}$$ 解析并合并到 $\mathcal{G}$ 中。共指消解（coreference resolution）与实体链接（entity linking）保证一致性。

**GraphRAG。**

GraphRAG [[278]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-edge2024local)] 在 RAG 之上叠加图遍历：给定一个查询，先检索种子实体，然后通过 $k$ 跳邻域遍历来挖掘 Embedding 相似度未直接匹配到的相关事实。这对多跳推理尤其强大：

$$
\text{GraphRetrieve}(q, \mathcal{G}, k) = \bigcup_{v \in \text{seeds}(q)} \mathcal{N}_k(v, \mathcal{G}),
$$

其中 $$\mathcal{N}_k(v, \mathcal{G})$$ 是 $v$ 的 $k$ 跳邻域。

**时序知识图谱。**

事实具有有效期：$$(h, r, t, [t_\text{start}, t_\text{end}])$$。时序知识图谱 [[301]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lacroix2020tensor)] 支持类似 “2023 年 OpenAI 的 CEO 是谁？” 的查询，而不会把过去与现在的状态混淆在一起。

### 键值记忆网络（Key-Value Memory Networks）

可微分记忆网络 [[302]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-weston2014memory), [303]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-sukhbaatar2015end)] 把记忆表示为一组键值对 $$\{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^M$$，并通过基于软 Attention 的检索访问：

$$
\alpha_i = \text{softmax}\!\left(\frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{D}}\right), \qquad
  \mathbf{c} = \sum_{i=1}^M \alpha_i \mathbf{v}_i.
$$

检索得到的上下文 $\mathbf{c}$ 是查询的可微函数，从而支持端到端训练。现代 Transformer 的 Attention 就是这种机制的特例。在智能体场景中，记忆槽可以通过梯度下降更新，也可以通过显式的写入操作更新。

### MemGPT 与虚拟上下文管理

MemGPT [[304]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-packer2023memgpt)] 引入了类似操作系统虚拟内存的*虚拟上下文*抽象。记忆被组织为多层：

**页入/页出策略（Page-In / Page-Out）。**

Agent 基于以下因素决定*哪些*记忆要提升到热上下文（page-in），*哪些*要逐出（page-out）：

- **近期性（Recency）：** 最近访问过的条目更有可能再次被需要。
- **相关性（Relevance）：** 与当前查询相似度高的条目。
- **重要性（Importance）：** 写入时被标记为高重要性的条目。

**自驱动的记忆管理。**

在 MemGPT 中，LLM 自身把记忆管理的函数调用（`memory_search`、`memory_insert`、`memory_delete`）作为其动作空间的一部分发出。这使记忆管理成为*可学习的行为*而非硬编码的 Policy——天然适合作为 RL 训练的目标（见「用强化学习训练记忆系统」一节）。

## 记忆操作

### 写入：把信息提交到记忆

并非每条观察都应被存储。写入决策本质上是一个过滤问题：

$$
\text{Write}(e) = \mathbf{1}\!\left[\text{importance}(e) > \tau\right],
$$

其中 $\tau$ 为阈值，$\text{importance}(e)$ 可以是：

- **惊讶度（Surprise）：** $$-\log p_\theta(e \mid \text{context})$$——出乎意料的事件信息量更大。
- **Reward 信号：** 与高 $$\lvert r_t \rvert$$（正或负）相关联的事件值得记住。
- **LLM 自评：** 提示模型在 1--10 的尺度上为重要性打分。

**冲突检测。**

在写入新事实 $$f_\text{new}$$ 之前，应检查它与已有记忆是否冲突：

$$
\text{Conflict}(f_\text{new}, \mathcal{M}) = \exists\, f \in \mathcal{M} : \text{Contradicts}(f_\text{new}, f).
$$

冲突检测可以通过自然语言推理（NLI）模型实现，也可以通过提示 LLM 来完成。一旦发生冲突，Agent 必须决定：覆盖、带时间戳并保留两者，或者标记为需要人工审查。

**记忆格式与粒度。**

除了*存什么*，*怎么存*也至关重要。记忆条目从原子事实到冗长对话记录不等，各有不同的取舍：

| 格式 | 优点 | 缺点 |
| --- | --- | --- |
| **原子事实**（“用户偏好 Python。”） | 检索精确；可组合；去重与冲突检测容易 | 失去上下文；存在抽取错误；对细微信息脆弱 |
| **结构化笔记**（A-MEM [[305]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-xu2025amem)]） | 元数据丰富（标签、链接）；支持图遍历；在精确性与上下文之间取得平衡 | 写入成本更高；需要预先设计 schema |
| **摘要式 Episode**（MemGPT [[304]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-packer2023memgpt)]） | 保留叙事连贯性；紧凑；适合多轮（Multi-Turn）推理 | 摘要有损；难以局部更新 |
| **原始对话记录** | 无损；无抽取错误；支持精确引用 | 存储量大；检索嘈杂；扫描代价高 |

实际生产系统往往会组合多种粒度 [[306]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chhikara2025mem0)]：抽取原子事实以支持精确召回，维护摘要式 Episode 以保留叙事性上下文，并把原始对话记录存入冷存储以满足可审计性需求。Generative Agents 架构 [[307]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-park2023generative)] 将观察存储为原子的“记忆对象”，附带自然语言描述、重要性分数与时间戳——同时支持精确检索与时序推理。

**设计准则。**

- **粒度匹配查询类型。** 如果用户问的是事实性问题（“我的 API key 是什么？”），原子事实更佳；如果问的是上下文性问题（“我们当时为什么决定用 Redis？”），就需要 Episode 摘要。
- **在可承受的前提下用尽可能细的粒度存储**，然后在其上构建粗粒度视图。把原子事实摘要化很容易；但从有损摘要中恢复原子是不可能的。
- **记录来源。** 每条记忆都应回链至来源（对话轮次、文档、工具输出），以便 Agent 可校验、用户可审计。

### 读取 / 检索

**查询的构造。**

检索查询 $q$ 未必要直接使用原始观察。更好的策略包括：

- **HyDE（假设性文档 Embedding，Hypothetical Document Embeddings）** [[274]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-gao2022precise)]： 先生成一个假设性的答案，对其做 Embedding，然后以该 Embedding 作为查询。
- **查询扩展（Query expansion）：** 生成查询的多个改写，并取检索结果的并集。
- **回退式 Prompt（Step-back prompting）：** 在检索之前先把具体查询抽象为一个更一般的问题。

**时间衰减与近期性偏置。**

更久远的记忆可能相关性更低。一个带时间权重的评分是：

$$
\text{score}(d, q, t) = \lambda \cdot \text{sim}(\mathbf{q}, \mathbf{v}_d) + (1-\lambda) \cdot \exp\!\left(-\frac{t - t_d}{\tau_\text{decay}}\right),
$$

其中 $$t_d$$ 是记忆的创建时间，$$\tau_\text{decay}$$ 控制衰减速率。Generative Agents 论文 [[307]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-park2023generative)] 采用了类似的近期性加权检索。

### 更新：冲突解决与巩固

记忆巩固（consolidation）会合并相关记忆，以减少冗余并浮现出更高层的模式：

$$
\mathcal{M}' = \text{Consolidate}(\mathcal{M}) = \text{Cluster}(\mathcal{M}) \cup \text{Summarize}(\text{Cluster}(\mathcal{M})).
$$

**遗忘机制。**

生物记忆会遗忘；人工记忆也应如此。可用策略包括：

- **LRU 驱逐：** 容量超限时移除最近最少使用的条目。
- **重要性加权的遗忘：** $p(\text{forget}\,\mid\,d) \propto \exp(-\text{importance}(d))$。
- **间隔重复（Spaced repetition）：** 反复被访问的记忆保留得更久，遵循指数遗忘曲线 [[308]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-ebbinghaus1885memory)]。

### 反思：元认知操作

反思（reflection） [[307]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-park2023generative), [212]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shinn2023reflexion)] 是一种高阶记忆操作：Agent 读取自己的记忆并生成*洞见*：

$$
\text{Reflect}(\mathcal{M}) \to \{i_1, i_2, \ldots\} \subset \mathcal{M}_\text{semantic},
$$

其中每条洞见 $$i_j$$ 都是从多条情景记忆中归纳出的更高层抽象。

> **反思的实践（Reflexion）**
>
> 在三次解决某个编程题失败之后，Agent 进行反思：
>
> 1. 从情景记忆中检索这三次失败的 Episode。
> 2. 生成一条洞见：“我总是忘记处理输入列表为空的边界情况。”
> 3. 把这条洞见存入语义记忆。
> 4. 在下一次尝试中检索这条洞见，并显式地检查空输入。
>
> 这就是 Reflexion [[212]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shinn2023reflexion)] 的核心机制：通过自我反思的“言语化强化学习”。

**反思结果存放在哪里？**

反思从情景记忆*读取*，但向语义记忆*写入*。所产生的洞见是上下文无关的概括（“始终检查空输入”），并非具体 Episode 的记录——因此属于语义记忆 $$\mathcal{M}_\text{semantic}$$。然而在反思过程本身中，中间推理（检索到的 Episode + 综合 Prompt + 生成的洞见）占用的是*工作记忆*（上下文窗口）。一言以蔽之：

- **输入**：情景记忆（具体的过往事件）
- **计算**：工作记忆（上下文中的主动推理）
- **输出**：语义记忆（持久化的、归纳后的洞见）

这与生物记忆巩固相对应——情景经验在睡眠与反思中逐渐被转化为语义知识。

## 多轮对话中的记忆

### 用户建模与偏好追踪

持久化的用户模型 $\mathcal{U}$ 存储：

- **显式偏好：** 用户明确表达的喜好/厌恶、沟通风格偏好。
- **隐式偏好：** 从行为中推断（例如用户总是要求 Python 代码、偏好简洁回答）。
- **专业水平：** 从词汇与问题复杂度中推断出的领域知识水平。
- **目标与上下文：** 进行中的项目、当前任务、组织角色。

用户模型在每次交互后更新：

$$
\mathcal{U}_{t+1} = \text{Update}(\mathcal{U}_t,\, (u_t, a_t, \text{feedback}_t)).
$$

### 会话连续性

没有记忆，每次对话都从“冷启动”开始。有了会话记忆后：

1. 在会话开始时，检索用户模型 $\mathcal{U}$ 与近期会话摘要。
2. 注入个性化的系统 Prompt：“你正在帮助 Alice，一位从事分布式训练项目的高级 ML 工程师。上次会话你帮她调试了一个 Gradient 同步问题。”
3. 在会话结束时，对会话进行摘要并更新 $\mathcal{U}$。

### 通过记忆实现个性化

个性化同时提升*效率*（更少的澄清问题）与*质量*（针对用户专业度校准的回答）。核心技术包括：

- **自适应详尽度：** 根据用户的历史互动调整回答长度。
- **领域引导（Domain priming）：** 从语义记忆中预置相关领域上下文。
- **主动回忆：** 无需用户提示即主动浮现相关的历史交互（“你上个月问过这个话题，当时我们的结论是……”）。

> **隐私与记忆**
>
> 持久化的用户记忆带来了显著的隐私问题。Agent 必须：（1）在存储个人信息前获取明确同意；（2）提供检视与删除已存储记忆的机制；（3）在多用户部署中强制执行访问控制；（4）遵守数据留存法规（GDPR、CCPA）。记忆系统应秉持“默认隐私（privacy-by-default）”原则进行设计。

## 多智能体系统中的记忆

当多个 Agent 协作完成一个共享任务时，记忆变成了一种*协调机制*——而不仅是个人的知识库。负责任务分解的规划 Agent 必须把子目标传达给执行 Agent；批评者（critic）Agent 必须访问到与被评估 Agent 相同的上下文；研究型 Agent 团队必须避免重复工作。如果没有共享记忆，Agent 之间就只能通过直接消息传递一切，造成带宽瓶颈，并在对话滚出上下文时丢失信息。共享记忆通过提供所有 Agent 都可读写的、持久化且可查询的底座来解决这一问题——把隐式协调（“我希望对方记得”）转换为显式状态（“答案就在 blackboard 上”）。

### 共享记忆池

在多智能体系统中，Agent 可以在共享记忆存储 $$\mathcal{M}_\text{shared}$$ 之外，再各自保留私有存储 $$\mathcal{M}_i$$：

$$
\text{context}_i(t) = \mathcal{R}(\mathcal{M}_i, q_i) \cup \mathcal{R}(\mathcal{M}_\text{shared}, q_i).
$$

共享记忆使*隐式协调*成为可能：Agent $A$ 写入一个发现；Agent $B$ 无需显式通信即可检索到。

### Blackboard 架构

*blackboard*（黑板）模式 [[309]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hayes1985blackboard)] 是一种经典的多智能体协调机制：

每个 Agent 都从 blackboard 读取并向其写入。一个*控制器*监控 blackboard，并在某个 Agent 的前置条件满足时将其激活。这把 Agent 彼此解耦：它们通过共享状态而非直接消息进行通信。

### 共享知识中的共识与冲突

当多个 Agent 同时写入共享记忆时，冲突难以避免。常用的解决策略包括：

- **Last-write-wins（后写者胜出）：** 简单但会丢失信息。
- **版本化记忆：** 维护所有写入的历史；Agent 可以查询任一版本。
- **投票 / 共识：** 在某个事实被提交之前，要求 $n$ 个 Agent 中有 $k$ 个达成一致。
- **置信度加权合并：** $$f_\text{merged} = \sum_i w_i f_i$$，其中 $$w_i$$ 是 Agent $i$ 的置信度。
- **指定权威：** 把不同记忆区域的所有权分配给特定的 Agent。

> **开放问题：分布式记忆一致性**
>
> 在并发写入、网络分区与对抗性 Agent 的存在下，多智能体系统应当如何维持记忆一致性？经典的分布式系统方案（Paxos、Raft）可以套用但代价高昂。带有界过时（bounded staleness）的近似一致性对许多智能体任务可能已经足够——但合适的取舍仍是一个开放的研究问题。

## 用强化学习训练记忆系统

### 记忆操作的奖励信号

记忆操作（读、写、更新、反思）可以被视为 RL 框架中的动作。挑战在于设计能激励*有用*记忆行为的 Reward 信号：

- **任务 Reward 回传。** 如果某次记忆检索导致了正确答案，则把功劳归于该检索动作。稀疏但毫不含糊。
- **检索精度 Reward。** $$r_\text{retrieve} = \text{Relevance}(d_\text{retrieved}, \text{task})$$，由一个学习得到的相关性模型估计。
- **记忆效率 Reward。** 对不必要的写入施加惩罚：$$r_\text{write} = -\lambda \cdot \mathbf{1}[\text{write}]$$，从而鼓励选择性存储。
- **一致性 Reward。** 奖励内部一致（无矛盾）的记忆状态。

在第 $t$ 步对一次记忆操作 $$m_t$$ 的组合 Reward：

$$
r_t^{\text{mem}} = r_t^{\text{task}} + \alpha \cdot r_t^{\text{retrieve}} + \beta \cdot r_t^{\text{write}} + \gamma \cdot r_t^{\text{consistency}}.
$$

### 学习“记什么”

“记什么”是一个元学习挑战：Agent 必须学到一个能最大化未来任务表现的写入 Policy $$\pi_\text{write}(e)$$。其难点在于：

1. 一条记忆的价值只有在未来才会显现（延迟 Reward）。
2. 写入时未来可能查询的空间是未知的。
3. 记忆之间相互影响：存储 $e$ 的价值依赖于 $\mathcal{M}$ 中已有的其他内容。

常见的方法有：

- **事后重标记（Hindsight relabeling）** [[310]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-andrychowicz2017hindsight)]。一个 Episode 成功后，回溯性地把被检索到的记忆标记为“重要”，并训练写入 Policy 去存储类似条目。
- **元强化学习（Meta-RL）** [[311]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-duan2016rl2)]。在一组任务分布上训练写入 Policy；Policy 学到存储能在任务间泛化的信息。
- **好奇心驱动的存储（Curiosity-driven storage）** [[312]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-pathak2017curiosity)]。存储令人意外（预测误差高）的观察，因为它们更可能蕴含信息。

### 记忆增强的 Policy 优化

联合优化 Policy 与其记忆系统的思想可追溯到可微分记忆网络 [[313]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-graves2016hybrid)]，并由 REALM [[292]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-guu2020realm)] 推广到检索增强的 LLM。一个记忆增强 Agent 的完整 Policy Gradient 目标为：

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \gamma^t r_t\right] - \lambda \cdot \mathcal{L}_\text{mem}(\phi),
$$

其中 $\theta$ 是 LLM 参数，$\phi$ 是记忆系统参数（例如检索模型权重），$$\mathcal{L}_\text{mem}$$ 是对记忆复杂度的正则项。

> **关键洞见：把记忆作为可学习的归纳偏置**
>
> 用 RL 训练记忆操作，让 Agent 得以发展出面向具体任务的记忆策略。一个编码 Agent 学会存储 API 签名；一个研究型 Agent 学会存储引文链；一个客服 Agent 学会存储用户投诉模式。记忆系统由此成为一种为 Agent 领域量身打造的*可学习归纳偏置*。

## 记忆方案的对比

| 架构 | 容量 | 检索 | 更新成本 | 可训练 | 最适合 |
| --- | --- | --- | --- | --- | --- |
| 上下文内（工作记忆） | $O(L)$ Token | 0 ms | 免费 | 通过微调 | 短任务、主动推理 |
| 稠密 RAG [[109]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lewis2020retrieval)] | $O(10^7)$ 文档 | 10--50 ms | $O(1)$ Embedding | 仅编码器 | 语义检索、问答 |
| 稀疏（BM25） [[261]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-robertson2009probabilistic)] | $O(10^8)$ 文档 | 1--5 ms | $O(\lvert d \rvert)$ 索引 | 否 | 关键词检索、法律/医疗 |
| 混合 RAG [[297]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chen2022hybrid)] | $O(10^7)$ 文档 | 15--60 ms | $O(1)$ Embedding | 仅编码器 | 通用检索 |
| 摘要 | 无上限 | 0 ms（在上下文内） | $O(\lvert e \rvert)$ 次 LLM 调用 | 通过微调 | 长对话、叙事 |
| 知识图谱 [[301]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lacroix2020tensor)] | $O(10^9)$ 三元组 | 5--100 ms | $O(1)$ 插入 | Embedding 层 | 结构化事实、多跳 |
| 键值记忆网络 [[303]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-sukhbaatar2015end)] | $O(M)$ 槽位 | $O(M)$ Attention | 一次 Gradient 更新 | 完全可训 | 端到端可微任务 |
| MemGPT 分层 [[304]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-packer2023memgpt)] | 无上限 | 0--100 ms | 混合 | 通过 RL | 长周期 Agent、助手 |
| Graph RAG [[278]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-edge2024local)] | $O(10^7)$ 节点 | 20--200 ms | $O(1)$ 插入 | 仅编码器 | 复杂推理、社区结构 |

## 评估记忆系统

评估智能体记忆颇具挑战，因为记忆操作的质量只能*间接*显现——通过长时间跨度的下游任务表现。一个对所存事实有完美召回率的记忆系统，若检索出无关上下文或撑爆 LLM 的上下文窗口，仍可能整体失败。

### 评估维度

LongMemEval [[314]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wu2024longmemeval)] 提出长期记忆系统必须展现的五项核心能力：

1. **信息抽取。** 系统能否从对话轮次中识别并存储显著事实？通过事实召回率衡量：有多少真值事实可从记忆中恢复？
2. **跨会话推理。** 系统能否综合分散在多次过往会话中的信息？例如：“根据我们上周和昨天的对话，项目范围发生了哪些变化？”
3. **时序推理。** 系统能否正确回答与时间相关的查询？例如“在组织调整*之前*我说我的优先事项是什么？”这就要求区分不同时间的状态。
4. **知识更新。** 当事实发生变化时（用户搬到新城市、偏好改变），记忆能否反映最新状态，同时保留历史？
5. **拒答（Abstention）。** 当系统并无相关记忆时，它能否正确地说“我不知道”，而不是幻觉出一段看似合理却完全是捏造的回忆？

### 基准（Benchmarks）

| 基准 | 发表会议/期刊 | 规模 | 关注点 |
| --- | --- | --- | --- |
| LongMemEval [[314]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wu2024longmemeval)] | ICLR 2025 | 500 个问题，可扩展的历史 | 五项记忆能力；多会话对话 |
| LOCOMO [[315]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-maharana2024locomo)] | EMNLP 2024 | 多会话对话 | 对话之上的单跳、时序、多跳与开放域问答 |
| InfiniteBench [[316]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-zhang2024infinitebench)] | ACL 2024 | 100K+ Token 上下文 | 长上下文召回，并非专门为记忆设计，但能测试上限 |

### 度量指标

**记忆层面的指标。**

- **记忆召回率（Memory Recall）：** $\frac{\text{可从记忆中检索到的真值事实数}}{\text{真值事实总数}}$。衡量存储的完备程度。
- **记忆精确率（Memory Precision）：** $\frac{\text{top-}k\text{ 检索中相关条目数}}{k}$。衡量检索的噪声。
- **延迟（Latency）：** 从查询到取回上下文的耗时（p50 与 p95）。
- **Token 效率：** 每次查询注入上下文的 Token 总数。越低越好——多余的上下文既降低 LLM 准确率又增加成本。

**下游指标。**

- **回答准确率：** 在记忆条件下最终回答的正确性（EM、F1 或 LLM-as-judge）。
- **忠实性（Faithfulness）：** 回答是否准确反映记忆中的内容，而没有捏造？
- **个性化质量：** 用户满意度，通过偏好评分或在记忆增强与无记忆系统之间做 A/B 测试来衡量。
- **自相矛盾率：** 系统给出与此前所述事实不一致回答的频率。

**运维层面的指标。**

- **写入选择性：** 触发记忆写入的轮次比例。过高 $\to$ 噪声；过低 $\to$ 信息缺口。
- **过时率（Staleness）：** 尽管已存在更新，仍检索到过时事实的频率。
- **存储增长率：** 每小时交互所存储的 Token 数。无界增长是不可持续的。

> **评估鸿沟**
>
> 大多数记忆论文都在短基准（10--50 个会话）上做评估。真实的生产 Agent 要运行数月、跨越数千个会话。长周期评估——记忆漂移、矛盾累积与存储膨胀在那里成为主要失败模式——仍是一个开放挑战。从业者应在基准分数之外，对运维指标进行纵向监控。

## 实现模式

### 基于 Embedding 的向量存储记忆

最常见的记忆模式把条目存储为 Embedding 向量，并附带元数据（时间戳、重要性分数、标签）。检索把余弦相似度与时间衰减结合起来，使近期且重要的记忆优先浮现；去重与 LRU 驱逐则把存储规模约束在上限以内。

```python
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class MemoryEntry:
    """带元数据的单条记忆条目。"""
    content: str
    embedding: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    source: str = "agent"

class VectorMemoryStore:
    """
    带时间衰减的稠密+稀疏混合记忆存储。
    支持基于重要性的加权检索与 LRU 驱逐。
    """

    def __init__(
        self,
        embed_fn,           # 可调用对象：str -> np.ndarray
        max_entries: int = 10_000,
        decay_rate: float = 0.01,   # 每小时
        recency_weight: float = 0.3,
    ):
        self.embed_fn = embed_fn
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.recency_weight = recency_weight
        self.entries: list[MemoryEntry] = []

    # -- 写入 --------------------------------------------------------------

    def write(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        check_duplicates: bool = True,
    ) -> MemoryEntry:
        """提交一条新记忆，若已达容量上限则驱逐。"""
        if check_duplicates and self._is_duplicate(content):
            return None  # 跳过近似重复条目

        embedding = self.embed_fn(content)
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            importance=importance,
            tags=tags or [],
        )

        if len(self.entries) >= self.max_entries:
            self._evict()

        self.entries.append(entry)
        return entry

    def _is_duplicate(self, content: str, threshold: float = 0.95) -> bool:
        """检查是否已存在近似重复项。"""
        if not self.entries:
            return False
        emb = self.embed_fn(content)
        sims = self._cosine_similarities(emb)
        return float(np.max(sims)) > threshold

    def _evict(self):
        """移除最不重要且最久未访问的条目。"""
        now = datetime.now()
        scores = []
        for e in self.entries:
            age_hours = (now - e.timestamp).total_seconds() / 3600
            recency = np.exp(-self.decay_rate * age_hours)
            score = e.importance * (1 - self.recency_weight) \
                  + recency * self.recency_weight
            scores.append(score)
        worst_idx = int(np.argmin(scores))
        self.entries.pop(worst_idx)

    # -- 检索 -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        recency_boost: bool = True,
    ) -> list[MemoryEntry]:
        """
        混合检索：稠密相似度 + 时间近期性。
        返回按组合分数排序的前 k 条。
        """
        if not self.entries:
            return []

        q_emb = self.embed_fn(query)
        dense_scores = self._cosine_similarities(q_emb)

        now = datetime.now()
        combined = []
        for i, (entry, d_score) in enumerate(
            zip(self.entries, dense_scores)
        ):
            if recency_boost:
                age_h = (now - entry.timestamp).total_seconds() / 3600
                recency = np.exp(-self.decay_rate * age_h)
                score = (1 - self.recency_weight) * d_score \
                      + self.recency_weight * recency
            else:
                score = d_score
            combined.append((score, i))

        combined.sort(reverse=True)
        top_k = [self.entries[i] for _, i in combined[:k]]

        # 更新访问元数据
        for entry in top_k:
            entry.access_count += 1
            entry.last_accessed = now

        return top_k

    def _cosine_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        """对所有已存 Embedding 进行向量化余弦相似度计算。"""
        matrix = np.stack([e.embedding for e in self.entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        return matrix_norm @ q_norm

    # -- 反思 ------------------------------------------------------------

    def reflect(self, llm_fn, k: int = 10) -> list[str]:
        """
        元认知反思：检索近期记忆，综合出更高层的洞见，并写回存储。
        """
        if len(self.entries) < 3:
            return []

        # 检索近期高重要性的记忆
        recent = sorted(
            self.entries, key=lambda e: e.timestamp, reverse=True
        )[:k]
        context = "\n".join(f"- {e.content}" for e in recent)

        # 请 LLM 生成洞见
        prompt = (
            "Given these recent memories, extract 2-3 high-level "
            "insights or patterns:\n" + context
        )
        raw_insights = llm_fn(prompt)

        # 把每条洞见作为高重要性记忆存储
        insights = []
        for line in raw_insights.strip().split("\n"):
            line = line.strip().lstrip("-*").strip()
            if len(line) > 20:
                self.write(
                    f"[INSIGHT] {line}",
                    importance=0.9,
                    check_duplicates=True,
                )
                insights.append(line)
        return insights

    def get_stats(self) -> dict:
        """返回记忆统计信息，便于监控。"""
        return {
            "total_entries": len(self.entries),
            "avg_importance": float(
                np.mean([e.importance for e in self.entries])
            ) if self.entries else 0.0,
            "oldest_entry": min(
                (e.timestamp for e in self.entries), default=None
            ),
        }
```

### 分层记忆管理器

受 MemGPT [[304]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-packer2023memgpt)] 启发，这一模式把记忆组织为三层：*热（hot）*（在上下文内，立即可访问）、*温（warm）*（向量存储，可快速检索）和*冷（cold）*（归档，容量无上限）。条目根据访问频率与重要性自动升级或降级——类似 CPU 的缓存层级。

```python
from enum import Enum
from collections import OrderedDict

class MemoryTier(Enum):
    HOT  = "hot"    # 在上下文内：立即可访问
    WARM = "warm"   # 向量存储：可快速检索
    COLD = "cold"   # 归档：慢但容量无上限

class HierarchicalMemoryManager:
    """
    受 MemGPT 启发的三层记忆管理器。
    热层是 LRU 缓存；温层是向量存储；
    冷层是只追加的归档存储。
    """

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        hot_capacity: int = 20,     # 热层最大条目数
        warm_capacity: int = 5_000,
        llm_summarize_fn=None,      # 用于摘要的可调用对象
    ):
        self.vector_store = vector_store
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.summarize = llm_summarize_fn

        # 热层：用 OrderedDict 实现 LRU 语义
        self.hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        # 冷层：只追加列表（生产环境中通常是一个数据库）
        self.cold: list[MemoryEntry] = []

    # -- 页入（Page-in）：把温层提升到热层 ---------------------------------------

    def page_in(self, query: str, k: int = 3) -> list[MemoryEntry]:
        """
        从温层检索并提升到热层。
        如有必要会驱逐热层中最近最少使用的条目。
        """
        candidates = self.vector_store.retrieve(query, k=k)
        promoted = []
        for entry in candidates:
            key = entry.content[:64]  # 用前缀作为键
            if key not in self.hot:
                if len(self.hot) >= self.hot_capacity:
                    self._evict_hot()
                self.hot[key] = entry
                self.hot.move_to_end(key)
            promoted.append(entry)
        return promoted

    def _evict_hot(self):
        """把热层中的 LRU 条目驱逐回温层。"""
        # OrderedDict：第一个元素即为 LRU
        key, entry = self.hot.popitem(last=False)
        # 重新插入温层（实际上已存在，只需更新访问信息）
        # 在真实系统中我们会更新温层中的元数据

    # -- 带层级分配的写入 ------------------------------------------

    def write(
        self,
        content: str,
        importance: float = 0.5,
        tier: MemoryTier = MemoryTier.WARM,
    ) -> MemoryEntry:
        """写入到合适的层级。"""
        if tier == MemoryTier.HOT:
            entry = MemoryEntry(
                content=content,
                embedding=self.vector_store.embed_fn(content),
                importance=importance,
            )
            key = content[:64]
            if len(self.hot) >= self.hot_capacity:
                self._evict_hot()
            self.hot[key] = entry
            return entry

        elif tier == MemoryTier.WARM:
            return self.vector_store.write(content, importance=importance)

        else:  # 冷层
            entry = MemoryEntry(
                content=content,
                embedding=np.array([]),  # 冷层不存 Embedding
                importance=importance,
            )
            self.cold.append(entry)
            return entry

    # -- 摘要与压缩 ---------------------------------------------

    def compress_hot_to_warm(self) -> Optional[str]:
        """
        对热层内容做摘要并把摘要写入温层。
        在热层已满且有新的重要内容到来时调用。
        """
        if not self.hot or not self.summarize:
            return None

        hot_contents = "\n".join(
            f"- {e.content}" for e in self.hot.values()
        )
        summary = self.summarize(
            f"Summarize these memory entries concisely:\n{hot_contents}"
        )
        self.vector_store.write(summary, importance=0.7)
        return summary

    # -- 统一检索 --------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """
        从所有层级检索，热层优先。
        最多返回按相关性排序的 k 条。
        """
        results = []

        # 1. 检查热层（精确匹配 + 语义匹配）
        q_emb = self.vector_store.embed_fn(query)
        for entry in self.hot.values():
            if entry.embedding.size > 0:
                sim = float(
                    np.dot(q_emb, entry.embedding)
                    / (np.linalg.norm(q_emb) * np.linalg.norm(entry.embedding) + 1e-8)
                )
                if sim > 0.7:
                    results.append((sim + 1.0, entry))  # +1 的热层加成

        # 2. 从温层检索
        warm_results = self.vector_store.retrieve(query, k=k)
        for entry in warm_results:
            results.append((0.5, entry))

        # 3. 去重并排序
        seen = set()
        final = []
        for score, entry in sorted(results, reverse=True):
            key = entry.content[:64]
            if key not in seen:
                seen.add(key)
                final.append(entry)
            if len(final) >= k:
                break

        return final

    def get_hot_context(self) -> str:
        """把热层格式化为上下文字符串返回。"""
        if not self.hot:
            return ""
        lines = ["[Memory Context]"]
        for entry in list(self.hot.values())[-10:]:  # 最后 10 条
            lines.append(f"  * {entry.content}")
        return "\n".join(lines)
```

### 记忆增强的 Agent 循环

这一模式由 MemGPT [[304]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-packer2023memgpt)] 提出，并在 CoALA 框架 [[317]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-sumers2023coala)] 中得到形式化。它通过一个*读—行动—反思—写入*的循环把记忆系统接入 Agent 的推理流程：响应之前，Agent 检索相关记忆；响应之后，Agent 决定要存什么。LLM 输出中的特殊 Token 会触发记忆操作，从而把对自身持久化的控制权交给模型本身。

```python
import re
from typing import Any

class MemoryAugmentedAgent:
    """
    一个具有完整“读—行动—反思—写入”记忆循环的 LLM Agent。
    实现 MemGPT 风格的自驱动记忆管理。
    """

    SYSTEM_PROMPT = """You are a memory-augmented AI assistant.
You have access to persistent memory across conversations.
At each turn you may issue memory commands:
  [MEMORY_SEARCH: <query>]  - retrieve relevant memories
  [MEMORY_WRITE: <content>] - store important information
  [MEMORY_REFLECT]          - synthesize insights from memory

Always think step by step. Use memory to avoid repeating mistakes
and to personalize your responses."""

    def __init__(
        self,
        llm_fn,                         # 可调用对象：messages -> str
        memory_manager: HierarchicalMemoryManager,
        importance_threshold: float = 0.6,
        max_memory_tokens: int = 1500,
    ):
        self.llm = llm_fn
        self.memory = memory_manager
        self.importance_threshold = importance_threshold
        self.max_memory_tokens = max_memory_tokens
        self.conversation_history: list[dict] = []

    # -- Agent 主循环单步 ----------------------------------------------------

    def step(self, user_message: str) -> str:
        """
        Agent 的完整单步：
        1. 检索相关记忆
        2. 构造增强 Prompt
        3. 生成响应（可能包含记忆命令）
        4. 执行记忆命令
        5. 反思与巩固
        6. 把响应返回给用户
        """

        # 步骤 1：检索相关记忆
        memories = self.memory.retrieve(user_message, k=5)
        memory_context = self._format_memories(memories)

        # 步骤 2：构造增强 Prompt
        messages = self._build_messages(user_message, memory_context)

        # 步骤 3：生成响应
        raw_response = self.llm(messages)

        # 步骤 4：执行响应中的所有记忆命令
        clean_response, memory_ops = self._parse_memory_commands(
            raw_response
        )
        self._execute_memory_ops(memory_ops, user_message, clean_response)

        # 步骤 5：自动写入重要信息
        self._auto_write(user_message, clean_response)

        # 步骤 6：更新对话历史
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": clean_response}
        )

        return clean_response

    # -- 记忆检索与格式化 -----------------------------------

    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for i, m in enumerate(memories, 1):
            age = (datetime.now() - m.timestamp).days
            lines.append(
                f"  [{i}] (importance={m.importance:.1f}, "
                f"{age}d ago) {m.content}"
            )
        return "\n".join(lines)

    def _build_messages(
        self, user_message: str, memory_context: str
    ) -> list[dict]:
        system = self.SYSTEM_PROMPT
        if memory_context:
            system += f"\n\n{memory_context}"
        system += f"\n\n{self.memory.get_hot_context()}"

        messages = [{"role": "system", "content": system}]
        # 包含最近的对话历史（最后 6 轮）
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": user_message})
        return messages

    # -- 记忆命令解析 ---------------------------------------------

    def _parse_memory_commands(
        self, response: str
    ) -> tuple[str, list[dict]]:
        """从响应中抽取并移除记忆命令。"""
        ops = []
        patterns = {
            "search":  r"\[MEMORY_SEARCH:\s*(.+?)\]",
            "write":   r"\[MEMORY_WRITE:\s*(.+?)\]",
            "reflect": r"\[MEMORY_REFLECT\]",
        }
        clean = response
        for op_type, pattern in patterns.items():
            for match in re.finditer(pattern, response, re.DOTALL):
                content = match.group(1) if op_type != "reflect" else None
                ops.append({"type": op_type, "content": content})
                clean = clean.replace(match.group(0), "").strip()
        return clean, ops

    def _execute_memory_ops(
        self,
        ops: list[dict],
        user_msg: str,
        response: str,
    ):
        """执行 LLM 发出的记忆命令。"""
        for op in ops:
            if op["type"] == "search":
                results = self.memory.retrieve(op["content"], k=3)
                # 把结果页入热层，以便立即使用
                self.memory.page_in(op["content"], k=3)

            elif op["type"] == "write":
                self.memory.write(
                    op["content"],
                    importance=0.8,  # 显式写入 = 重要
                    tier=MemoryTier.WARM,
                )

            elif op["type"] == "reflect":
                self._reflect()

    # -- 自动写入启发式 -----------------------------------------------

    def _auto_write(self, user_msg: str, response: str):
        """
        在无显式命令的情况下自动存储重要信息。
        使用简单的启发式：当响应包含事实、决策或用户偏好时即写入。
        """
        importance_keywords = [
            "remember", "important", "note that", "you prefer",
            "your name is", "decided to", "the answer is",
            "key insight", "learned that",
        ]
        combined = (user_msg + " " + response).lower()
        if any(kw in combined for kw in importance_keywords):
            summary = f"User: {user_msg[:100]} | Agent: {response[:200]}"
            self.memory.write(
                summary,
                importance=self.importance_threshold,
                tier=MemoryTier.WARM,
            )

    # -- 反思 --------------------------------------------------------

    def _reflect(self):
        """
        元认知反思：从近期记忆中综合出洞见。
        将高层洞见写回语义记忆。
        """
        recent = self.memory.retrieve("recent important events", k=10)
        if len(recent) < 3:
            return  # 不够支撑反思

        recent_text = "\n".join(f"- {m.content}" for m in recent)
        insight_prompt = [
            {"role": "system", "content": "You extract high-level insights."},
            {"role": "user", "content":
                f"Based on these memories, what are 2-3 key insights?\n"
                f"{recent_text}\nRespond with bullet points only."},
        ]
        insights = self.llm(insight_prompt)
        # 把每条洞见作为高重要性的语义记忆存储
        for line in insights.split("\n"):
            line = line.strip().lstrip("*-").strip()
            if len(line) > 20:
                self.memory.write(
                    f"[INSIGHT] {line}",
                    importance=0.9,
                    tier=MemoryTier.WARM,
                )
```

> **读—行动—反思—写入循环**
>
> 记忆增强的 Agent 循环实现了一个四阶段的认知周期：
>
> 1. **读（Read）：** 在行动之前，检索相关记忆以指引响应。
> 2. **行动（Act）：** 基于检索到的上下文生成响应。
> 3. **反思（Reflect）：** 定期从已累积的记忆中综合出更高层的洞见。
> 4. **写入（Write）：** 选择性地把重要的新信息提交到持久化存储中。
>
> 这一循环呼应了军事战略中的 *observe-orient-decide-act*（OODA）循环，以及认知心理学中的 *encode-store-retrieve*（编码—存储—检索）模型。关键洞见在于：记忆并非被动的存储库，而是认知过程的*主动参与者*。

## 智能体记忆的最新进展

上文描述的记忆系统奠定了基础模式。近期若干工作进一步推进了边界：

### CoALA：面向语言 Agent 的认知架构

Sumers 等人 [[317]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-sumers2023coala)] 提出了 *面向语言 Agent 的认知架构*（Cognitive Architectures for Language Agents, CoALA），这是一个借用认知科学与符号 AI 原则、用于组织日益庞杂的 LLM Agent 生态的统一框架。CoALA 将一个语言 Agent 拆解为：

- **模块化记忆**：工作记忆（上下文窗口）、情景记忆（过往经验）、语义记忆（世界知识）以及程序记忆（动作 schema）——与「记忆类型分类」一节的分类体系相对应。
- **结构化的动作空间**：内部动作（推理、检索、记忆写入）与外部动作（工具调用 Tool Calling、与环境交互）。
- **决策循环**：一个广义的“感知—规划—行动”循环，其中带有显式的检索与写入步骤。

CoALA 的贡献与其说是一个新系统，不如说是一种*设计语言*：它提供了一种系统化的方式来分析已有 Agent 并识别缺失能力，对从业者而言是一个有用的参考架构。

### Mem0：生产级记忆层

Mem0 [[306]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chhikara2025mem0)] 致力于弥合研究型记忆系统与生产部署之间的鸿沟。核心思路：

- **自动抽取**：Mem0 并不依赖 LLM 显式发出记忆写入命令，而是自动从对话轮次中抽取显著事实，并将其巩固到持久化存储中。
- **基于图的记忆**：除了扁平的向量存储之外，Mem0 还在抽取出的实体与事实之上维护一张*关系图*，以支持多跳记忆查询（“用户在项目 Y 的语境下，对话题 X 说过什么？”）。
- **记忆压缩**：冗余或被取代的事实会自动合并，使记忆存储保持紧凑与最新。

在 LOCOMO 基准上，Mem0 相对 OpenAI 的基线记忆取得了 26% 的相对提升，与全上下文方案相比，p95 延迟降低 91%，Token 成本降低 $>$90%。

### 睡眠时计算（Sleep-Time Compute）：离线记忆处理

Lin 等人 [[318]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lin2025sleeptime)] 提出了 *sleep-time compute*（睡眠时计算）范式，让 Agent 在用户交互的*间隔之间*处理并巩固记忆，而不是只在查询时才计算。其类比是生物的睡眠——大脑在睡眠中巩固记忆并预先建立有用的联想。

**工作机制。**

在空闲时段（“睡眠”）中，Agent 会：

1. 基于当前上下文预测未来可能出现的查询。
2. 预先计算推理链、摘要与结构化表示。
3. 把这些预计算产物存储下来，以便测试时（test-time）推理可检索并复用。

**效果。**

在推理类基准上，sleep-time compute 把达成等价准确率所需的测试时计算降低约 $5\times$。如果在同一上下文的多个相关查询上进行摊销，平均每次查询的成本下降 $2.5\times$。当用户查询是*可预测*的——即上下文强约束了可能被问到的问题时——该方法最为有效。

> **把记忆巩固视作离线 RL**
>
> Sleep-time compute 可以看作*离线 Policy 改进*：在空闲时间，Agent 利用已经收集到的数据（过往交互）改进其记忆表示（Policy），无需与环境进行新的交互。这与离线 RL 方法相通——后者从一个静态的轨迹数据集学习。

### A-MEM：受 Zettelkasten 启发的智能体记忆

A-MEM [[305]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-xu2025amem)] 引入了一种借鉴 *Zettelkasten* 方法的记忆系统——该方法是一种基于密集互联原子笔记的笔记系统——为 LLM Agent 实现动态、自组织的记忆。

**关键设计原则。**

- **结构化笔记。** 每条记忆条目都不是原始文本块，而是一条带有多种结构化属性的*笔记*：上下文描述、关键词、标签，以及指向相关笔记的显式链接。这些元数据使检索比单纯的 Embedding 相似度更丰富。
- **动态链接。** 当一条新记忆被加入时，系统会分析已有记忆，识别语义上有意义的关联并建立双向链接。结果是一张*知识网络*，而不是扁平的列表。
- **记忆演化。** 关键之处在于，新增笔记可以*触发对已有笔记的更新*——随着 Agent 理解的加深，对它们的上下文表示与属性进行细化。这让记忆成为一个随时间不断演进的活体结构，而非静态归档。
- **Agent 驱动的组织。** 与固定 schema 的记忆系统不同，A-MEM 让 LLM 自身决定如何组织、链接与更新记忆——使组织结构能够自适应任务领域。

**效果。**

在六个基础模型、多种多会话推理任务上，A-MEM 稳定优于扁平向量存储、基于摘要的记忆与图数据库方案，说明记忆*如何*组织与*存什么*同等重要。

## 小结

智能体记忆系统是有能力的 AI Agent 的基础组件，用以应对有限上下文窗口这一根本限制。我们已综述：

- 一个**四分类法**（工作记忆、情景记忆、语义记忆、程序记忆），对应认知科学，并反映了截然不同的工程需求。
- **五大架构家族**：基于 RAG、基于摘要、基于图、键值网络，以及分层虚拟上下文（MemGPT）。
- **四种核心操作**：写入（含重要性打分与冲突检测）、读取/检索（含时间衰减与查询扩展）、更新（含冲突解决与巩固），以及反思（元认知洞见生成）。
- **多轮与多智能体扩展**：用户建模、会话连续性、共享记忆池与 blackboard 架构。
- **记忆系统的 RL 训练**：记忆操作的 Reward 信号、学习“记什么”，以及记忆增强的 Policy 优化。

该领域仍在快速演进。关键的开放挑战包括：（1）*记忆落地（memory grounding）*——确保被检索到的记忆被忠实地纳入回答，而非被忽略或被幻觉所覆盖；（2）*可扩展的一致性*——在大型多智能体系统中维持连贯的共享记忆；（3）*保护隐私的记忆*——在不损害用户数据的前提下实现个性化。随着上下文窗口的增长，上下文内记忆与外部记忆之间的边界会发生移动，但对*有选择、有结构、可检索*的信息存储的根本需求将一直存在。
