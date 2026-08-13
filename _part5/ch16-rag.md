---
layout: home
title: 检索增强生成（RAG）
permalink: /part5/ch16-rag.html
---

检索增强生成（Retrieval-Augmented Generation, RAG）[[109]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lewis2020retrieval)] 已成为在生产环境中部署大型语言模型时最具实际影响力的技术之一。RAG 不再仅依赖训练时编码进模型权重中的知识，而是为 LLM 配备一个动态、可更新的外部记忆——使其能够在广泛的知识密集型任务中给出准确、有据可查且可验证的响应。

## 动机与问题陈述

> **为什么 LLM 需要外部知识**
>
> 大型语言模型以*参数化*方式存储知识——在训练过程中压缩进数十亿个权重之中。这带来了三个根本性局限：
>
> 1. **幻觉**：当查询超出模型可靠知识边界时，模型会自信地生成听起来合理但事实错误的陈述。
> 2. **知识过时**：训练数据存在截止日期；模型无法了解训练之后发生的事件、论文或产品更新。
> 3. **领域特异性**：通用模型缺乏对专有代码库、内部文档、专业法规或企业数据的深入了解。

### 参数化知识 vs. 非参数化知识

我们可以将两种知识来源之间的区别形式化。令 $$\mathcal{M}_\theta$$ 表示参数为 $\theta$ 的语言模型，$$\mathcal{D} = \{d_1, d_2, \ldots, d_N\}$$ 为外部文档语料库。在每种范式下，给定查询 $q$ 生成回答 $a$ 的概率为：

$$
\begin{aligned}
  P_{\text{parametric}}(a \mid q) &= P_{\mathcal{M}_\theta}(a \mid q) \\[6pt]
  P_{\text{RAG}}(a \mid q, \mathcal{D}) &= \sum_{d \in \mathcal{D}} P_{\mathcal{M}_\theta}(a \mid q, d)\,
    P_{\text{ret}}(d \mid q, \mathcal{D})
\end{aligned}
$$

其中 $$P_{\text{ret}}(d \mid q, \mathcal{D})$$ 是关于文档的检索分布。RAG 对检索到的证据进行边缘化，使生成过程接地于非参数化知识。

> **图书馆类比**
>
> 可以把参数化 LLM 想象成一位记住了一整座庞大图书馆但已毕业多年的学者。RAG 为这位学者发放了一张图书证——他们可以实时查阅资料、引用来源，并在需要查证文献时坦然承认，而不是凭记忆瞎猜。

### 何时选择 RAG vs. 微调 vs. 长上下文

| 标准 | RAG | 微调 | 长上下文 | RAG + 微调 |
| --- | --- | --- | --- | --- |
| 知识频繁更新 | ✓ | $\times$ | $\times$ | ✓ |
| 需要引用 / 接地 | ✓ | $\times$ | ✓ | ✓ |
| 专有大型语料库 | ✓ | $\times$ | $\times$ | ✓ |
| 适配风格 / 格式 | $\times$ | ✓ | $\times$ | ✓ |
| 教授新推理技能 | $\times$ | ✓ | $\times$ | ✓ |
| 语料库可放入上下文窗口 | $\times$ | $\times$ | ✓ | $\times$ |
| 要求低延迟 | $\times$ | ✓ | $\times$ | $\times$ |

> **常见误解**
>
> RAG *不是*微调的替代品。微调教会模型*如何*推理和响应；RAG 则提供*推理的内容*。两者是互补的。一个经过良好指令跟随微调的模型比基础模型更能有效利用检索到的上下文。

## 核心 RAG 架构

一个标准的 RAG 系统由两个阶段构成：处理和存储文档的**离线索引流水线**，以及为查询提供服务的**在线检索—生成流水线**。

### 完整流水线图

![端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。]({{ site.baseurl }}/figures/fig_050_rag_arch.png)

### 索引流水线

**文档加载。**

文档以异构格式抵达（PDF、HTML、Markdown、DOCX、代码）。加载器提取干净的文本，并保留元数据（来源 URL、页码、章节标题、时间戳），这些元数据会与 Embedding 一并存储，用于过滤和引用。

**分块（Chunking）。**

长文档必须切分成可放入 Embedding 模型上下文窗口（通常为 512 个 Token）并保持语义连贯的块。分块策略是 RAG 系统设计中影响最大的决策之一（参见分块策略一节）。

**Embedding。**

每个块 $$c_i$$ 通过 Embedding 模型 $$f_\phi$$ 编码为稠密向量 $$\mathbf{e}_i = f_\phi(c_i) \in \mathbb{R}^d$$。这些向量与原始文本和元数据一同存储到向量数据库中。

### 检索

给定查询 $q$，检索步骤将其编码为 $$\mathbf{q} = f_\phi(q)$$，并通过余弦相似度找出最相似的 $k$ 个块：

$$
  \text{sim}(\mathbf{q}, \mathbf{e}_i) = \frac{\mathbf{q} \cdot \mathbf{e}_i}{\|\mathbf{q}\|\,\|\mathbf{e}_i\|}
$$

返回 top-$k$ 块 $$\mathcal{C}_k = \{c_{(1)}, \ldots, c_{(k)}\}$$ 作为上下文。

### 生成

检索到的块被注入到一个 Prompt 模板中：

```python
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY
the provided context. If the context does not contain enough information,
say so explicitly. Cite your sources using [Doc N] notation."""

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    context_str = "\n\n".join(
        f"[Doc {i+1}] (Source: {c['source']}, Page: {c.get('page','N/A')})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""{SYSTEM_PROMPT}

Context:
{context_str}

Question: {query}

Answer:"""
```

## 检索方法

### 稀疏检索：BM25 与 TF-IDF

稀疏检索方法将文档和查询表示为词汇表上的高维稀疏向量。给定查询 $q$（含词项 $$t_1, \ldots, t_n$$）时，针对文档 $d$ 的经典 BM25 评分函数 [[261]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-robertson2009probabilistic)] 为：

$$
  \text{BM25}(d, q) = \sum_{i=1}^{n} \text{IDF}(t_i) \cdot
    \frac{f(t_i, d) \cdot (k_1 + 1)}{f(t_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{\lvert d \rvert}{\text{avgdl}}\right)}
$$

其中 $$f(t_i, d)$$ 为词频，$\lvert d \rvert$ 为文档长度，$\text{avgdl}$ 为平均文档长度，$$k_1 \in [1.2, 2.0]$$、$b = 0.75$ 为可调参数。

> **稀疏检索仍然占优的场景**
>
> - **精确关键词匹配**：产品代号、错误代码、专有名词、罕见词项
> - **低资源领域**：稠密模型的训练数据不足
> - **可解释性**：容易调试某个文档为何被检索出来
> - **速度**：无需 GPU；借助倒排索引可扩展到数十亿文档
> - **未登录词**：Embedding 训练时未见过的新术语

### 稠密检索：DPR

稠密段落检索（Dense Passage Retrieval, DPR）[[262]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-karpukhin2020dense)] 使用两个独立的、基于 BERT 的编码器——一个*查询编码器* $$E_Q$$ 和一个*段落编码器* $$E_P$$——通过对比 Loss 进行训练，使相关的查询—段落对在 Embedding 空间中彼此靠近。

**双编码器（Bi-Encoder）架构。**

$$
  \text{sim}(q, p) = E_Q(q)^\top E_P(p)
$$

**使用 Batch 内负样本进行训练。**

给定包含 $B$ 个查询—段落对的 Batch $$\{(q_i, p_i^+)\}_{i=1}^B$$，对比 Loss 将 Batch 中所有其他段落视为负样本：

$$
  \mathcal{L}_{\text{DPR}} = -\frac{1}{B} \sum_{i=1}^{B}
    \log \frac{\exp\!\left(E_Q(q_i)^\top E_P(p_i^+) / \tau\right)}
              {\sum_{j=1}^{B} \exp\!\left(E_Q(q_i)^\top E_P(p_j) / \tau\right)}
$$

其中 $\tau$ 是温度超参数。硬负样本（词面相似但语义无关的段落）对训练强检索器至关重要。

**近似最近邻搜索。**

在大规模场景下，对数百万个 Embedding 进行穷举搜索不可行。FAISS [[263]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-johnson2019billion)]（Facebook AI Similarity Search）提供了高效的近似最近邻（Approximate Nearest Neighbor, ANN）搜索，所用方法包括：

- **IVF（倒排文件索引，Inverted File Index）**：将向量聚类到 Voronoi 单元中；只搜索邻近的单元
- **HNSW（分层可导航小世界，Hierarchical Navigable Small World）** [[264]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-malkov2018efficient)]：基于图的索引，搜索复杂度 $O(\log N)$
- **PQ（乘积量化，Product Quantization）**：压缩向量以减少内存占用

### 基于倒数排名融合的混合检索

混合检索结合稀疏和稠密得分。一个简单的线性组合为：

$$
  s_{\text{hybrid}}(d, q) = \alpha \cdot s_{\text{dense}}(d, q) + (1-\alpha) \cdot s_{\text{sparse}}(d, q)
$$

然而，来自不同系统的分数无法直接比较。**倒数排名融合（Reciprocal Rank Fusion, RRF）** [[265]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-cormack2009reciprocal)] 通过在排名而非分数上操作来规避这个问题：

$$
  \text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}
$$

其中 $\mathcal{R}$ 是排名列表集合（如 BM25 排名和稠密排名），$$\text{rank}_r(d)$$ 是文档 $d$ 在列表 $r$ 中的排名，$k = 60$ 是平滑常数，用于削弱排名极高的文档带来的影响。

> **RRF 计算示例**
>
> 假设 BM25 将文档 $d$ 排在第 3 位，而稠密检索将其排在第 7 位。当 $k = 60$ 时：
>
> $$
> \text{RRF}(d) = \frac{1}{60 + 3} + \frac{1}{60 + 7} = \frac{1}{63} + \frac{1}{67} \approx 0.0159 + 0.0149 = 0.0308
> $$
>
> 而一个在两个列表中都排第 1 位的文档得分为 $\frac{1}{61} + \frac{1}{61} \approx 0.0328$。

### 学习式稀疏检索：SPLADE 与 SPLADEv2

> **为什么需要 SPLADE？**
>
> 传统的稀疏检索（BM25）依赖精确的词面匹配——当查询说 “car” 而文档说 “automobile” 时它就会失败。稠密检索（DPR）能捕捉语义但缺乏可解释性，查询时需要 GPU，且索引体积大。**SPLADE** 兼具二者的优点：稀疏向量（像 BM25 一样支持快速的倒排索引查找）配合可学习的语义扩展（像稠密模型一样处理同义词和相关概念）。

**SPLADE（v1）—— 核心思想。**

SPLADE（Sparse Lexical and Expansion Model）[[266]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-formal2021splade)] 使用预训练的掩码语言模型（如 BERT/DistilBERT），为每个文档或查询生成一个覆盖*整个词汇表*的稀疏向量。关键洞见在于：MLM 头部已经知道文本中每个位置在语义上与哪些词相关——SPLADE 把这种知识重新利用为词项重要性权重。

**架构。**

给定输入文本 $$x = [x_1, \ldots, x_n]$$：

1. 通过 Transformer 编码器，并经由 MLM 头部得到上下文表示 $\mathbf{H} \in \mathbb{R}^{n \times \lvert \mathcal{V} \rvert}$
2. 跨位置聚合，并应用一个饱和激活：

$$
  w_t(x) = \log\!\left(1 + \text{ReLU}\!\left(\max_{i \in [1,n]} \mathbf{H}_i[t]\right)\right)
$$

其中 $$\mathbf{H}_i[t]$$ 是输入位置 $i$ 处针对词汇 Token $t$ 的 MLM logit。

- $\log(1 + \cdot)$ 饱和防止任何单个词项占主导（类似 BM25 中的 TF 饱和）
- ReLU 保证稀疏性——绝大多数词汇项的权重为零
- 跨位置的 $\max$ 池化为每个词项从文本任意位置中捕获最强信号
- **扩展**：即使是原文中*未出现*的 Token 也可获得非零权重（例如关于 “neural networks” 的文档可能为 “deep learning”、“AI”、“backpropagation” 赋权）

**打分。**

查询和文档分别映射为稀疏向量 $\mathbf{w}^q, \mathbf{w}^d \in \mathbb{R}^{\lvert \mathcal{V} \rvert}$。相关性得分就是一个简单的点积：

$$
  s(q, d) = \sum_{t \in \mathcal{V}} w_t^q \cdot w_t^d
$$

由于两个向量都是稀疏的（在 30K 词汇表中通常只有 20--200 个非零项），这可以使用标准倒排索引（Lucene、Anserini）高效计算——查询时无需 GPU。

**训练。**

SPLADE 通过对比学习（Batch 内负样本 + 硬负样本）进行训练，并附加两个正则项：

$$
  \mathcal{L} = \mathcal{L}_{\text{contrastive}} + \lambda_q \|\mathbf{w}^q\|_1 + \lambda_d \|\mathbf{w}^d\|_1
$$

查询和文档表示上的 $$L_1$$ 惩罚鼓励稀疏性——若没有它们，模型将学到稠密表示，从而背离设计初衷。

**SPLADEv2 —— 关键改进。**

SPLADEv2 [[267]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-formal2021spladev2)] 引入了若干改进，显著提升了效率与效果：

1. **从交叉编码器蒸馏**：SPLADEv2 不只用二元相关性标签训练，而是借助交叉编码器教师（如 MonoT5 [[268]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-nogueira2020document)]）提供软相关性得分，从而获得更丰富的训练信号：

$$
    \mathcal{L}_{\text{distill}} = \text{KL}\!\left(\sigma(s_{\text{student}}) \,\|\, \sigma(s_{\text{teacher}})\right)
$$

2. **分离的查询/文档编码器**：SPLADEv2 为查询与文档设置不同的稀疏度目标。查询被鼓励*更稀疏*（查找更快），而文档可以稍稠密（离线预计算）：

$$
    \lambda_q > \lambda_d \quad \text{(e.g., } \lambda_q = 3 \times 10^{-4},\; \lambda_d = 1 \times 10^{-4}\text{)}
$$

3. **FLOPS 正则化**：SPLADEv2 不再使用简单的 $$L_1$$，而是引入了一个感知 FLOPS 的正则项，直接惩罚预期检索代价：

$$
    \mathcal{L}_{\text{FLOPS}} = \sum_{t \in \mathcal{V}} \left(\overline{a}_t^q\right)^2 + \sum_{t \in \mathcal{V}} \left(\overline{a}_t^d\right)^2
$$

其中 $$\overline{a}_t$$ 是 Batch 中词项 $t$ 的平均激活。该项会惩罚在许多文档上都非零的词项（倒排列表长 = 检索慢）。

4. **高效骨干**：使用 DistilBERT（66M 参数）替代 BERT-base（110M），编码时间减半，质量损失极小。

> **SPLADE 与 SPLADEv2 对比**
>
> | 方面 | SPLADE（v1） | SPLADEv2 |
> | --- | --- | --- |
> | 训练信号 | 二元相关性 + 硬负样本 | 交叉编码器蒸馏 |
> | 稀疏度控制 | $$L_1$$ 正则化 | 感知 FLOPS 的正则化 |
> | 查询/文档对称性 | 同一编码器、同一 $\lambda$ | 非对称（查询更稀疏） |
> | 骨干 | BERT-base（110M） | DistilBERT（66M） |
> | MRR@10（MS MARCO [[269]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-bajaj2016msmarco)]） | 34.0 | 36.8 |
> | 平均每文档非零项数 | $\sim$200 | $\sim$120（更稀疏 40%） |

> **何时使用 SPLADE**
>
> - **适合使用 SPLADE/v2 的场景**：你需要在查询时不依赖 GPU 完成语义检索；你的基础设施已有倒排索引（Elasticsearch、Lucene）；或你需要可解释的相关性得分（可以查看是哪些扩展词项匹配上了）。
> - **更适合使用稠密检索的场景**：你有 GPU 预算用于查询编码；需要多语言支持（稠密模型的迁移性更好）；或者查询非常短（1--2 个词时扩展帮助较小）。
> - **最佳实践**：用 SPLADEv2 作为第一阶段检索器 + 在 top-$k$ 上使用交叉编码器重排序器。这种组合可在更低延迟下匹敌甚至超越稠密检索流水线。

### ColBERT：晚期交互

ColBERT [[270]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-khattab2020colbert)] 将查询和文档编码为 Token 级 Embedding 的*集合*，并使用 *MaxSim* 算子进行打分：

$$
  s(q, d) = \sum_{i \in \lvert \mathbf{q} \rvert} \max_{j \in \lvert \mathbf{d} \rvert} \mathbf{q}_i^\top \mathbf{d}_j
$$

这种晚期交互（Late Interaction）机制比单向量双编码器更具表达力，又因为文档 Embedding 可离线预计算而远比交叉编码器更快。

**架构。**

查询编码器 $$E_Q$$ 和文档编码器 $$E_D$$ 都是基于 BERT 的模型，产生*逐 Token* 的 Embedding（而不是单个 [CLS] 向量）。每个 Token Embedding 通过一个线性层投影到更低维度（通常为 128）：

$$
\begin{aligned}
  \mathbf{q}_i &= \text{Linear}(E_Q(q)_i) \in \mathbb{R}^{128}, \quad i = 1, \ldots, \lvert q \rvert \\
  \mathbf{d}_j &= \text{Linear}(E_D(d)_j) \in \mathbb{R}^{128}, \quad j = 1, \ldots, \lvert d \rvert
\end{aligned}
$$

**训练。**

ColBERT 在正、负段落上以成对的 Softmax 交叉熵 Loss 进行训练。给定查询 $q$、正段落 $d^+$ 和一组负段落 $$\{d^-_1, \ldots, d^-_N\}$$：

$$
  \mathcal{L}_{\text{ColBERT}} = -\log \frac{\exp(s(q, d^+))}{\exp(s(q, d^+)) + \sum_{k=1}^{N} \exp(s(q, d^-_k))}
$$

其中 $s(q, d)$ 是上文公式中的 MaxSim 分数。负样本来自：

- **Batch 内负样本**：同一训练 Batch 中的其他段落（无额外成本、数量充足）
- **硬负样本**：由 BM25 检索出的、词面相似但语义无关的段落（对质量影响最大）
- **蒸馏负样本**（ColBERTv2 [[271]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-santhanam2022colbertv2)]）：使用交叉编码器教师挖掘最难的负样本，并将其得分蒸馏到 ColBERT 中

**索引与服务。**

在索引阶段，所有文档 Token 的 Embedding 都被预计算并存储（在 ColBERTv2 中可选地使用残差量化压缩）。在查询阶段，只需实时编码查询 Token，并与已存储的文档 Embedding 计算 MaxSim。这种分离带来：

- **离线文档编码**：编码一次，服务于多次查询
- **PLAID 索引** [[271]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-santhanam2022colbertv2)]：对文档 Embedding 聚类，用聚类中心进行初步候选检索，然后只在候选集上计算精确的 MaxSim——延迟可降低 5--10$\times$
- **索引大小**：每文档 $\lvert d \rvert \times 128$ 个浮点数（比单向量方法更大，但通过量化可压缩到约每维度 2 字节）

### 检索方法对比

| 方法 | 延迟 | 准确率 | 索引大小 | GPU | 最佳适用场景 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF [[272]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-sparckjones1972idf)] | 极低 | 低 | 小 | 无需 | 基线、精确匹配 |
| BM25 [[261]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-robertson2009probabilistic)] | 极低 | 中等 | 小 | 无需 | 关键词搜索、罕见词项 |
| DPR / 双编码器 [[262]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-karpukhin2020dense)] | 低 | 高 | 大 | 需要 | 语义相似度 |
| SPLADE [[266]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-formal2021splade)] | 低 | 高 | 中等 | 需要 | 兼顾精度与速度的混合方案 |
| ColBERT [[270]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-khattab2020colbert)] | 中等 | 极高 | 极大 | 需要 | 高精度检索 |
| 交叉编码器 [[273]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-nogueira2019passage)] | 高 | 最高 | N/A | 需要 | 对 top-$k$ 重排序 |
| 混合（RRF） [[265]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-cormack2009reciprocal)] | 低 | 极高 | 大 | 需要 | 生产系统 |

## 分块策略

分块是将文档切分为以下段落的过程：(1) 足够小，能放入 Embedding 模型的上下文窗口；(2) 语义连贯；(3) 即使被单独检索出来，也包含足够的上下文以发挥作用。

### 带重叠的固定大小分块

最简单的策略：每 $W$ 个 Token 切分一次，相邻块之间保留 $O$ 个 Token 的重叠。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # 每个块的 Token 数
    chunk_overlap=64,     # 重叠量，用于在边界处保留上下文
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)
```

**重叠公式**：对于长度为 $L$ 个 Token 的文档，块的数量为：

$$
  N_{\text{chunks}} = \left\lceil \frac{L - O}{W - O} \right\rceil
$$

### 语义分块

语义分块不在固定间隔处切分，而是在通过相邻句子间 Embedding 相似度检测出的*话题边界*处切分：

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # 或 "standard_deviation"
    breakpoint_threshold_amount=95,          # 在最不相似的前 5% 处切分
)
chunks = chunker.split_documents(documents)
```

### 基于文档结构的分块

对于结构化文档（Markdown、HTML、代码），在自然边界处切分：

- **Markdown**：在 `##` 标题处切分，保留章节上下文
- **HTML**：在 `<section>`、`<article>`、`<p>` 标签处切分
- **代码**：在函数/类定义处切分，并在每个块中保留 import 语句
- **表格**：将整个表格保留为单个块；切勿在行中间切分

### 父—子分块

这是一种将检索粒度与生成上下文解耦的强大模式：

1. **索引较小的子块**（如 128 个 Token）以实现精确检索
2. **将较大的父块**（如 512 个 Token）返回给 LLM 以提供更丰富的上下文

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)
```

### 块大小的经验指南

| 使用场景 | 推荐块大小 | 重叠 |
| --- | --- | --- |
| 事实型问答（精确事实） | 128--256 Token | 20--32 Token |
| 摘要 / 综合 | 512--1024 Token | 64--128 Token |
| 代码检索 | 完整函数 | 无 |
| 法律 / 监管文档 | 段落级 | 1 句 |
| 对话 / 聊天 | 256--512 Token | 32--64 Token |

## 高级 RAG 模式

### 查询变换

用户的原始查询往往含糊、过短，或与文档语言匹配不佳。查询变换技术在搜索之前改进检索。

**HyDE（假设性文档 Embedding，Hypothetical Document Embeddings）** [[274]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-gao2022precise)]。

不直接对查询做 Embedding，而是生成一个*假设性回答*并对其做 Embedding：

$$
  \hat{d} = \text{LLM}(q), \quad \mathbf{e}_{\text{query}} = f_\phi(\hat{d})
$$

直觉在于：假设性回答与真实文档处于相同的语言风格中，从而缩小查询—文档之间的分布差距。

**后退式提示（Step-Back Prompting）。**

对于具体问题，先生成一个更通用的“后退”问题，对两者都进行检索，再合并上下文。例如：“乙醇在 2 atm 下的沸点是多少？” $\to$ 后退问题：“哪些因素影响液体的沸点？”

**多查询生成。**

生成 $M$ 种多样化的查询改写，对每个改写都进行检索，并对结果取并集：

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(temperature=0.7),
    include_original=True,   # 同时也对原始查询进行检索
)
# 内部生成 3 个查询变体，对每个进行检索并去重
docs = retriever.get_relevant_documents(query)
```

### 重排序

在初步检索出 top-$k$ 候选后，*交叉编码器*重排序器会对每个查询—文档对联合打分（同时关注两者），以更高延迟为代价产生更准确的相关性得分：

$$
  s_{\text{cross}}(q, d) = \text{CrossEncoder}([q; d])
$$

交叉编码器无法用于第一阶段检索（没有预计算的文档 Embedding），但非常适合对小规模候选集（通常 $k = 20$--$100$）进行重排序。

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank(query: str, docs: list[str], top_n: int = 5) -> list[str]:
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_n]]
```

### 上下文压缩

检索到的块往往在相关段落周围包含许多无关句子。上下文压缩使用 LLM 仅提取相关部分：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
compressed_docs = compression_retriever.get_relevant_documents(query)
```

### Self-RAG

Self-RAG [[275]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-asai2023selfrag)] 训练单个模型完成以下任务：(1) 决定*是否*进行检索，(2) 在带或不带检索的条件下生成，(3) 使用特殊的反思 Token *批评*自己的输出：

- `[Retrieve]`：模型是否应检索更多段落？
- `[IsRel]`：检索到的段落是否与查询相关？
- `[IsSup]`：生成的陈述是否可由检索到的段落支持？
- `[IsUse]`：整体回复是否有用？

模型经过端到端训练，与回复一起预测这些 Token，从而实现对检索与自我评分的细粒度控制。

### CRAG：纠正式 RAG

CRAG [[276]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yan2024crag)] 增加了一个*检索评估器*，对检索到的文档打分并触发纠正动作：

1. 检索 top-$k$ 文档
2. 对每个文档打分：**正确** / **模糊** / **错误**
3. 如果所有文档都为错误或模糊 $\to$ 回退到网页搜索
4. 如果部分文档正确 $\to$ 使用知识精炼（剔除无关句子）
5. 基于精炼后的上下文生成回答

### 自适应 RAG

自适应 RAG（Adaptive RAG） [[277]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-jeong2024adaptive)] 根据预测出的查询复杂度将其路由到不同的检索策略：

- **不检索**：模型可凭参数化知识回答的简单事实型查询
- **单步 RAG**：对中等复杂度的查询采用标准的“先检索后生成”
- **多步 RAG**：针对复杂的多跳问题进行迭代检索

一个在查询复杂度标签上训练的轻量级分类器对每个传入查询进行路由。

### Graph RAG

Microsoft 的 Graph RAG [[278]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-edge2024local)] 从文档语料库构建*知识图谱*，并使用社区检测生成分层摘要：

1. **实体抽取**：LLM 从每个块中抽取实体与关系
2. **图构建**：构建图 $G = (V, E)$，其中节点是实体、边是关系
3. **社区检测**：在多种分辨率下应用 Leiden 算法寻找社区
4. **社区摘要**：LLM 为每个社区生成摘要
5. **查询**：对全局查询，对社区摘要执行 map-reduce；对局部查询，使用标准向量搜索

> **何时使用 Graph RAG**
>
> Graph RAG 擅长处理需要跨大量文档综合信息的*全局*查询（“该语料库的主要主题有哪些？”），但构建与维护成本高。标准 RAG 更适合*局部*查询（“文档 X 关于话题 Y 说了什么？”）。

### RAG-Fusion

RAG-Fusion [[279]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rackauckas2023ragfusion)] 从原始查询生成多条搜索查询，对每条都进行检索，再使用 RRF（上文公式）融合排名列表：

```python
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """使用 RRF 融合多个有序文档列表。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def rag_fusion(query: str, retriever, llm, n_queries: int = 4) -> str:
    # 第 1 步：生成查询变体
    variants = generate_query_variants(query, llm, n=n_queries)
    # 第 2 步：对每个变体进行检索
    all_ranked = [retriever.retrieve(q) for q in [query] + variants]
    # 第 3 步：使用 RRF 融合
    fused_docs = reciprocal_rank_fusion(all_ranked)
    # 第 4 步：生成回答
    return generate_answer(query, fused_docs[:5], llm)
```

## 高效 RAG 解码：REFRAG

RAG 的一个实际瓶颈是*解码延迟*：拼接到 LLM 上下文中的检索段落往往很长但相关性稀疏，从而拉高首 Token 时延（Time-to-First-Token, TTFT）和 KV-cache 内存。REFRAG [[280]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lin2025refrag)] 观察到，由于检索到的段落是独立来源的（通过重排序中的多样性或去重），它们的 Attention 模式呈*块对角*——大多数跨段落 Attention 接近于零。这种稀疏性意味着解码阶段对 RAG 上下文的大部分计算其实是不必要的。

**压缩—感知—展开（Compress--Sense--Expand）框架。**

REFRAG 通过三阶段解码策略利用这种结构：

1. **压缩**：用紧凑的摘要（如对每个段落块的 Key/Value 进行均值池化）替换检索段落的完整 KV 表示，大幅降低内存占用。
2. **感知**：在每个解码步骤上，对压缩后的表示执行轻量级 Attention，识别哪些段落块与当前 Token 相关。
3. **展开**：仅为被选中的块重建完整的 KV 项，对稀疏的活跃集合执行精确 Attention。

**结果。**

在基于 LLaMA 的模型上，REFRAG 实现了最高 $30.85\times$ 的 TTFT 加速（比此前的稀疏 Attention 基线提升 $3.75\times$），且困惑度无损。在固定内存预算下，它还将有效上下文长度扩展了 $16\times$。这些收益在 RAG、多轮对话以及长文档摘要任务中均成立。

> **REFRAG 为何对智能体 RAG 至关重要**
>
> 智能体 RAG（见下文「智能体 RAG」）需要为每个查询进行*多轮*检索，延迟会累积。REFRAG 这类高效解码方法是必备的基础设施：通过确保每一轮的解码代价相对上下文长度是次线性的，它们使迭代式的检索—推理—生成循环在大规模下具备可行性。

## 智能体 RAG

### 动机：静态 RAG 的局限

标准 RAG 遵循固定的“先检索后生成”模式。这在以下情况下会失败：

- **多跳问题**：“是谁创办了在 2023 年收购 OpenAI 主要竞争对手的那家公司？”这类问题需要串联多次检索
- **歧义查询**：正确的检索策略取决于已找到的内容
- **异构来源**：不同子问题需要不同的知识库
- **迭代精炼**：初次检索可能揭示需要换一种查询方式

> **将 RAG 视为马尔可夫决策过程**
>
> 智能体 RAG 将检索视为一个序贯决策问题。*状态*是当前的上下文（查询 + 至今为止检索到的文档）；*动作*包括检索、推理、生成和停止；*Reward* 是答案的正确性。Agent 学习一种关于何时检索以及检索什么的 Policy。

### 智能体 RAG 架构

![智能体 RAG 控制流。Agent 在返回答案前迭代地进行规划、检索、判断充分性，并自检接地性。]({{ site.baseurl }}/figures/fig_051_agentic_rag.png)

### 多源路由

智能体 RAG 系统可以将子查询路由到专门化的知识源。其核心洞见是：不同问题类型需要不同的检索后端——没有任何单一索引能在所有方面都表现出色。

**为什么要路由？**

设想一个金融分析师助手处理四种查询：

- “我们公司的带薪休假（PTO）政策是什么？” $\rightarrow$ **向量数据库**（内部文档）
- “美联储昨天宣布了什么？” $\rightarrow$ **网页搜索**（实时）
- “按地区展示 Q3 营收” $\rightarrow$ **SQL 数据库**（结构化数据）
- “我们的鉴权中间件如何校验 Token？” $\rightarrow$ **代码索引**（代码库）

“从单一索引扁平化检索”的做法要么错过答案，要么返回无关段落。路由在检索开始前，为*合适的子问题选择合适的工具*。

**路由策略。**

按复杂度递增的三种主要方法：

1. **基于规则的路由。**通过关键词触发（例如，SQL 关键字 $\rightarrow$ 数据库，URL 模式 $\rightarrow$ 网页）。快速且可解释，但对歧义查询非常脆弱。
2. **基于分类器的路由。**由一个轻量模型（如微调过的 BERT 分类器，或在查询 Embedding 上的逻辑回归）预测最佳来源。延迟低（$<$10 毫秒），可基于路由日志训练，但需要带标签数据。
3. **基于 LLM 的路由。**LLM 本身在一次结构化输出调用中决定来源（见下面的代码清单）。最为灵活——可以处理新颖的查询类型并解释其推理——但会增加一次 LLM 调用的延迟。

> **路由器即可学习的 Policy**
>
> 多源路由在最简单的形式下是一个*分类*问题，在最丰富的形式下则是一个*规划*问题。当被建模为 RL Policy——其中状态是查询加上对话历史，动作是来源的选择（以及可选的查询改写），Reward 是下游答案质量——时，路由器可以通过 Policy Gradient 技术进行端到端优化。

**实践考量。**

- **回退链**：若主要来源返回的结果置信度较低，则尝试次优来源。
- **并行扇出**：对于歧义查询，同时从多个来源检索，并通过倒数排名融合合并结果。
- **成本感知**：网页搜索和 API 调用可能存在金钱开销或速率限制；路由器应将其纳入考量。
- **可观测性**：记录每个路由决策及其推理过程——对调试和再训练至关重要。

```python
from enum import Enum
from pydantic import BaseModel

class KnowledgeSource(str, Enum):
    VECTOR_DB   = "vector_db"    # 内部文档
    WEB_SEARCH  = "web_search"   # 实时网页
    SQL_DB      = "sql_db"       # 结构化数据
    CODE_INDEX  = "code_index"   # 代码库
    API         = "api"          # 外部 API

class RouteDecision(BaseModel):
    source: KnowledgeSource
    refined_query: str
    reasoning: str

def route_query(query: str, llm) -> RouteDecision:
    """使用 LLM 决定该查询哪个知识源。"""
    prompt = f"""Given the query: "{query}"

Decide which knowledge source to use:
- vector_db: for internal documents, policies, past reports
- web_search: for current events, recent information
- sql_db: for numerical data, statistics, structured records
- code_index: for code examples, API documentation
- api: for real-time data (weather, stock prices, etc.)

Return a JSON with: source, refined_query, reasoning."""

    return llm.with_structured_output(RouteDecision).invoke(prompt)
```

### 完整的智能体 RAG 实现

前面几节介绍了各个独立组件——路由、检索、评估。一个完整的智能体 RAG 系统将它们编排为一个*有状态节点的图*，其中控制流取决于中间结果。下面的实现使用 LangGraph 将四个节点串成一个循环：

1. **Plan（规划）**：将用户查询分解为子查询（每个信息需求对应一个）。
2. **Retrieve（检索）**：将每个子查询路由到合适的来源并取回文档。
3. **Evaluate（评估）**：判断累积的上下文是否足以回答原始查询。
4. **Generate（生成）**：综合检索到的文档生成带引用的最终答案。

关键设计模式是*条件循环*：评估之后，Agent 要么进入生成阶段（如果上下文足够，或迭代预算耗尽），要么带着精炼后的子查询回到检索阶段。这与一个在信息收集动作上运行的 RL Agent 所遵循的“感知—行动—评估”循环如出一辙。

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

class AgentState(TypedDict):
    query: str
    sub_queries: list[str]
    retrieved_docs: Annotated[list[dict], operator.add]
    context_sufficient: bool
    answer: str
    iterations: int
    max_iterations: int

def plan_node(state: AgentState) -> AgentState:
    """将查询分解为子查询。"""
    sub_queries = decompose_query(state["query"])
    return {**state, "sub_queries": sub_queries, "iterations": 0}

def retrieve_node(state: AgentState) -> AgentState:
    """为当前的子查询检索文档。"""
    new_docs = []
    for sq in state["sub_queries"]:
        source = route_query(sq)
        docs = retrieve_from_source(sq, source)
        new_docs.extend(docs)
    return {**state, "retrieved_docs": new_docs,
            "iterations": state["iterations"] + 1}

def evaluate_node(state: AgentState) -> AgentState:
    """评估检索到的上下文是否充分。"""
    sufficient = evaluate_context_sufficiency(
        query=state["query"],
        docs=state["retrieved_docs"]
    )
    return {**state, "context_sufficient": sufficient}

def generate_node(state: AgentState) -> AgentState:
    """基于检索到的上下文生成回答。"""
    answer = generate_with_citations(
        query=state["query"],
        docs=state["retrieved_docs"]
    )
    return {**state, "answer": answer}

def should_retrieve(state: AgentState) -> str:
    if state["context_sufficient"]:
        return "generate"
    if state["iterations"] >= state["max_iterations"]:
        return "generate"  # 放弃继续检索，用现有内容生成
    return "retrieve"

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("plan",     plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan",     "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges("evaluate", should_retrieve,
    {"retrieve": "retrieve", "generate": "generate"})
workflow.add_edge("generate", END)

agent = workflow.compile()

# 运行
result = agent.invoke({
    "query": "What were the main causes of the 2023 banking crisis?",
    "max_iterations": 3,
    "retrieved_docs": [],
    "iterations": 0,
})
```

### 工具增强的 RAG

智能体 RAG 可以将检索与计算工具结合：

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

@tool
def search_documents(query: str) -> str:
    """搜索内部文档知识库。"""
    docs = vectorstore.similarity_search(query, k=5)
    return "\n\n".join(d.page_content for d in docs)

@tool
def query_database(sql: str) -> str:
    """在分析数据库上执行 SQL 查询。"""
    return db.run(sql)

@tool
def web_search(query: str) -> str:
    """在网上搜索最新信息。"""
    return tavily_client.search(query)

@tool
def execute_python(code: str) -> str:
    """执行 Python 代码进行计算。"""
    return python_repl.run(code)

tools = [search_documents, query_database, web_search, execute_python]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Search-R1：RL 训练的智能体 RAG

上述智能体 RAG 方法依赖*Prompt 工程*的编排——Agent 的搜索行为由指令控制，而非通过训练学习。**Search-R1** [[281]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-jin2025searchr1)] 采取了根本不同的方法：通过强化学习训练 LLM，使其*学会在推理过程中何时、检索什么、检索多少次*。

**核心思想。**

Search-R1 扩展了 DeepSeek-R1 [[156]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-deepseek2025r1)] 推理框架，将搜索引擎查询视为 RL 训练循环中的**Action**。在思维链（Chain-of-Thought，CoT）生成过程中，模型可以输出特殊 Token `<search>query</search>`，触发对搜索引擎的实时检索。检索结果被注入回推理上下文，模型继续生成。

**形式化设定。**

模型生成一个与搜索动作交错的推理轨迹：

$$
\underbrace{\text{think}_1}_{\text{reasoning}} \to \underbrace{\texttt{<search>}q_1\texttt{</search>}}_{\text{action}} \to \underbrace{[\text{results}_1]}_{\text{observation}} \to \text{think}_2 \to \texttt{<search>}q_2\texttt{</search>} \to \cdots \to \text{answer}
$$

整条轨迹（推理 + 搜索 + 最终答案）由一个终端 Reward 打分：最终答案相对于真实标签的正确性。

**训练算法。**

Search-R1 使用组相对策略优化（Group Relative Policy Optimization, GRPO）：

1. **每个问题采样 $N$ 条轨迹**，每条可能包含 0--5 次搜索调用
2. **实时执行搜索**——环境返回真实的搜索引擎结果
3. **对终端答案的正确性打分**（与真实答案的精确匹配或 F1）
4. **计算组相对优势**：$$\hat{A}_i = (R_i - \mu_G) / \sigma_G$$
5. 使用 GRPO 的裁剪目标**更新 Policy**——强化那些有效搜索的轨迹

模型学会：

- **在不确定时才搜索**——避免对已掌握的知识进行不必要的搜索
- **构造有效查询**——学会能返回相关结果的查询措辞
- **多次搜索**——基于初次结果迭代精炼查询
- **整合检索到的上下文**——利用搜索结果支撑或修正其推理

**Search-R1 与基于 Prompt 的智能体 RAG 的差异。**

| 维度 | 基于 Prompt 的智能体 RAG | Search-R1 |
| --- | --- | --- |
| 搜索决策 | Prompt / 启发式 | 通过 RL 学习 |
| 查询构造 | 由 Prompt 指引（“改写查询”） | 端到端训练 |
| 搜索次数 | 固定或推理时由 LLM 决定 | 学习到的最优次数 |
| 训练信号 | 无（冻结模型） | 正确性 Reward |
| 搜索结果整合 | 追加到上下文 | 交错在 CoT 中 |
| 失败恢复 | 启发式重试 | 学习到的退避 / 重构造 |
| 推理开销 | 框架开销（LangGraph） | 原生模型行为 |

**结果。**

在开放域问答基准（NQ [[282]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-kwiatkowski2019natural)]、TriviaQA [[283]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-joshi2017triviaqa)]、HotpotQA [[284]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yang2018hotpotqa)]）上，使用 7B 模型的 Search-R1 优于：

- 标准 RAG（单次检索）准确率高出 15--20%
- 基于 Prompt 的智能体 RAG（ReAct 风格）准确率高出 8--12%
- 接近搭配标准 RAG 的大得多模型（70B）的性能

关键洞见是：**学习“何时以及如何搜索”比拥有一个知道更多内容的大模型更有价值**。一个善于搜索的小模型可以击败不搜索的大模型。

> **Search-R1：范式转变**
>
> 传统 RAG 问的是：“对于这个查询，我应该检索什么？”（一种在生成之前做出的流水线决策）。
>
> Search-R1 问的是：“基于我已经推理出的内容，我是否需要更多信息？如果需要，什么具体问题能填补这个空缺？”（一种在生成*过程中*做出的、可学习的决策）。
>
> 这就像考试前先翻教科书的学生，与在做题中途意识到卡住、随即去查阅参考资料的学生之间的差别。后者更高效、更有针对性。

## 评测

评测一个 RAG 系统比单独评测检索或生成更难，因为错误可能源自流水线的*任何阶段*——而且会逐级累积。完美的生成器无法弥补无关的检索结果，而当生成器产生幻觉或忽略上下文时，完美的检索器也会被浪费。

因此，有效的 RAG 评测在**三个层面**进行：

1. **检索质量**：检索器是否找出了正确的段落？（Recall、Precision、MRR、NDCG）
2. **生成质量**：答案是否正确、忠实于检索上下文且完整？（正确性、忠实度、答案相关性）
3. **端到端质量**：整个系统是否令用户满意？（人类偏好、任务成功率、按延迟调整的效用）

一种常见的失败模式是只优化某一层——例如，通过大 $K$ 最大化 Recall@$K$ 会让上下文塞满边缘相关的段落，反而*降低*生成质量。下文的指标涵盖检索与生成两端，便于从业者诊断瓶颈所在阶段。

### 检索指标

令 $$\mathcal{R}_k$$ 为前 $k$ 名的检索文档集合，$$\mathcal{R}^*$$ 为相关文档集合。

**Recall@K。**

$$
  \text{Recall@}K = \frac{\lvert \mathcal{R}_K \cap \mathcal{R}^* \rvert}{\lvert \mathcal{R}^* \rvert}
$$

**Precision@K。**

$$
  \text{Precision@}K = \frac{\lvert \mathcal{R}_K \cap \mathcal{R}^* \rvert}{K}
$$

**平均倒数排名（Mean Reciprocal Rank, MRR）。**

$$
  \text{MRR} = \frac{1}{\lvert Q \rvert} \sum_{i=1}^{\lvert Q \rvert} \frac{1}{\text{rank}_i}
$$

其中 $$\text{rank}_i$$ 是查询 $i$ 第一个相关文档的排名。

**归一化折损累计增益（NDCG@K）。**

$$
  \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}, \quad
  \text{DCG@}K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}
$$

其中 $$\text{rel}_i \in \{0, 1, 2, \ldots\}$$ 是第 $i$ 个结果的分级相关性，IDCG 是理想（完美）的 DCG。

### 生成指标

**忠实度（Faithfulness）。**

衡量生成的答案是否*接地*于检索到的上下文——即答案中的每一项主张都能归因到某一篇检索到的文档。由 LLM 判官评估：

$$
  \text{Faithfulness} = \frac{\text{\# claims supported by context}}{\text{\# total claims in answer}}
$$

**答案相关性（Answer Relevance）。**

衡量答案是否针对问题作答。计算方式是从答案中生成问题，并衡量其与原始查询的相似度：

$$
  \text{AnswerRelevance} = \frac{1}{N} \sum_{i=1}^{N} \cos\!\left(E(q), E(\hat{q}_i)\right)
$$

其中 $$\hat{q}_i$$ 是从答案中生成的问题。

**上下文精度与召回率。**

$$
\begin{aligned}
  \text{ContextPrecision@}K &= \frac{1}{K} \sum_{k=1}^{K} \text{Precision@}k \cdot \mathbf{1}[\text{doc}_k \text{ is relevant}] \\
  \text{ContextRecall} &= \frac{\text{\# ground-truth claims attributable to context}}{\text{\# total ground-truth claims}}
\end{aligned}
$$

### RAGAs 框架

RAGAs（Retrieval Augmented Generation Assessment） [[285]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-es2023ragas)] 提供了一个使用 LLM 判官的无参考评测框架：

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question":  questions,
    "answer":    generated_answers,
    "contexts":  retrieved_contexts,   # 嵌套列表（每条样本对应一个上下文列表）
    "ground_truth": reference_answers,
})

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
)
print(results.to_pandas())
```

### 常见失败模式

> **需监控的 RAG 失败模式**
>
> 1. **检索遗漏**：相关文档存在于语料库中却未被检索到。原因：分块不佳、Embedding 模型不匹配、查询—文档词汇差距。
> 2. **上下文污染**：检索到的文档包含误导性或矛盾的信息，导致模型生成错误答案。
> 3. **中部迷失（Lost-in-the-Middle）**：LLM 对长上下文的开头和结尾关注更强；中间部分的相关信息可能被忽略 [[286]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-liu2023lost)]。
> 4. **过度检索**：检索块过多会稀释相关信号，并增加延迟与成本。
> 5. **即便检索仍幻觉**：模型忽略检索到的上下文，基于参数化记忆生成内容，尤其当上下文与训练数据相冲突时。
> 6. **引用伪造**：模型将主张归因到并不支持这些主张的文档。

## 生产考量

### Embedding 模型选型

Embedding 模型是 RAG 系统中影响最大的单一组件选择——它决定了检索质量的上限。该领域发展迅速；下表汇总了当前在成本—质量光谱上的可选项。

| 模型 | 维度 | 最大 Token | MTEB 均值 | 访问方式 | 备注 |
| --- | --- | --- | --- | --- | --- |
| *基于 API（托管）* |  |  |  |  |  |
| Voyage `voyage-4-large` | 1024* | 32K | --- | API | 最佳检索质量 |
| OpenAI `text-embedding-3-large` | 3072 | 8191 | 64.6 | API | 套娃式维度（Matryoshka） |
| Cohere `embed-english-v3.0` | 1024 | 512 | 64.5 | API | 支持 int8/二值 |
| Google `text-embedding-005` | 768 | 2048 | --- | API | 集成于 Vertex AI |
| *开放权重（自托管）* |  |  |  |  |  |
| `nvidia/NV-Embed-v2` [[287]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lee2024nvembed)] | 4096 | 32K | 72.3 | 免费 | MTEB #1（2024 年 9 月） |
| `Alibaba-NLP/gte-Qwen2-7B` [[288]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-li2023gte)] | 3584 | 32K | 70.2 | 免费 | Apache-2.0，多语言 |
| `BAAI/bge-m3` [[289]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chen2024bgem3)] | 1024 | 8192 | 65.0 | 免费 | 稠密 + 稀疏 + 多向量 |
| `jinaai/jina-embeddings-v3` | 1024 | 8192 | 66.0 | 免费 | 多语言，LoRA 适配器 |
| `BAAI/bge-large-en-v1.5` [[290]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-xiao2023cpack)] | 1024 | 512 | 64.2 | 免费 | 成熟，生态完善 |

**选型标准。**

- **领域匹配**：专门化模型（如代码领域的 `voyage-code-3`、金融领域的 `voyage-finance-2`）在领域任务上可比通用模型高出 5--15%。
- **上下文长度**：具有 32K Token 上下文的模型（Voyage-4、NV-Embed-v2）可对整篇文档进行 Embedding 而无需分块，简化流水线。
- **套娃式 Embedding（Matryoshka）**：支持灵活输出维度（256--4096）的模型允许你在服务时以质量换取存储/延迟，而无需重新编码。
- **量化支持**：模型层面支持 int8 或二值量化的（Cohere、Voyage）能将索引大小缩减 4--32$\times$，召回率损失极小。
- **多语言**：对非英语或跨语言 RAG，优先选择明确做过多语言训练的模型（BGE-M3、Jina-v3、Voyage-4）。

### 向量数据库对比

| 数据库 | 托管方式 | 规模 | 过滤 | 混合检索 | 最佳适用场景 |
| --- | --- | --- | --- | --- | --- |
| FAISS1 | 自托管 | 数十亿 | 有限 | 否 | 研究、离线 |
| Pinecone2 | 托管 | 数十亿 | 是 | 是 | Serverless，易部署 |
| Weaviate3 | 两者皆可 | 数十亿 | 是 | 是 | GraphQL、多模态 |
| Chroma4 | 自托管 | 百万级 | 是 | 否 | 本地开发、原型 |
| Qdrant5 | 两者皆可 | 数十亿 | 是 | 是 | 高性能 |
| Milvus6 | 两者皆可 | 数十亿 | 是 | 是 | 企业级、GPU 加速 |
| pgvector7 | 自托管 | 百万级 | 是 | 是 | 已有 Postgres 用户 |

来源：5 [qdrant.tech](https://qdrant.tech)；6 [milvus.io](https://milvus.io)；7 [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)

### 延迟优化

1. **预过滤**：在 ANN 搜索之前，使用元数据过滤器（日期范围、类别、来源）缩减搜索空间
2. **近似最近邻**：使用 HNSW 或 IVF 索引代替精确搜索；以约 1% 的召回率损失换取 $10\times$ 加速
3. **Embedding 缓存**：为频繁重复出现的查询缓存 Embedding
4. **异步检索**：并行地从多个来源检索
5. **流式生成**：在检索完成的同时流式输出 LLM 结果
6. **量化**：对 Embedding 使用 int8 或二值量化，以降低内存并提升吞吐

**异步并行检索。**

上述技术 (3) 和 (4) 天然可组合：缓存查询 Embedding，然后将检索请求扇出（fan out）到多个后端并发执行。在多源 RAG 系统中，用户查询可能需要同时从向量数据库、关键词索引和 Web API 中获取结果。顺序检索会累加延迟；并行检索仅需支付*最慢*来源的代价。下面的代码使用 Python 的 `asyncio` 展示了这一模式——`lru_cache` 装饰器确保重复查询完全跳过 Embedding 模型，而 `asyncio.gather` 同时分派所有来源的查询。

```python
import asyncio
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_cached_embedding(text: str) -> list[float]:
    return embedding_model.embed_query(text)

async def parallel_retrieve(
    query: str,
    sources: list[str],
    k: int = 5
) -> list[dict]:
    """并行从多个来源检索。"""
    tasks = [
        asyncio.create_task(retrieve_from_source_async(query, src, k))
        for src in sources
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 扁平化并去重
    all_docs = []
    for r in results:
        if not isinstance(r, Exception):
            all_docs.extend(r)
    return deduplicate_by_content(all_docs)
```

### 增量索引与版本管理

在生产环境中，文档语料库从不是静态的——政策会被修订、新报告每天涌入、过时内容必须移除。一次完整的重新索引（重新分块、重新 Embedding、重新上传）代价高昂并会导致停机。增量索引通过在文档级别应用变更来解决这一问题。

**核心操作。**

- **Upsert（更新插入）**：当一篇文档被创建或更新时，删除该 `doc_id` 对应的所有已有块，对新内容重新分块、Embedding 并插入。这保证不会有过时片段残留。
- **删除 / 过期**：按文档 ID 移除块（显式删除）或按 TTL（针对新闻、市场数据等时效性来源的自动垃圾回收）。
- **版本追踪**：在块元数据中存储 `version` 和 `indexed_at` 时间戳。这支持回滚（从来源恢复旧版本）与可审计性（“模型看到了哪个版本？”）。

**一致性挑战。**

- **Embedding 模型漂移**：升级 Embedding 模型时，新旧向量不兼容。解决方案：(a) 为每个模型版本维护独立索引，在后台迁移；或 (b) 使用 Matryoshka 兼容的模型，通过维度截断保持兼容性。
- **分块边界变化**：更改分块策略会使所有现有块失效。版本元数据让你能识别并选择性地重新索引受影响的文档。
- **最终一致性**：在分布式向量数据库中，新 upsert 的向量可能无法立即被检索到。请将流水线设计为可容忍短暂的索引滞后（通常为秒级到分钟级）。

**实现。**

下面的代码展示了一个最小化的 `RAGIndexManager` 类，封装了 upsert 与过期逻辑，适合包装任何支持元数据过滤的向量存储。

```python
class RAGIndexManager:
    def __init__(self, vectorstore, metadata_store, chunker, embedder):
        self.vs = vectorstore
        self.meta = metadata_store
        self.chunker = chunker
        self.embedder = embedder

    def upsert_document(self, doc_id: str, content: str,
                        metadata: dict) -> None:
        """添加或更新一篇文档，替换旧块。"""
        # 删除该文档现有的块
        self.vs.delete(filter={"doc_id": doc_id})
        # 对新版本分块（向量库内部完成 Embedding）
        chunks = self.chunker.split_text(content)
        self.vs.add_texts(
            texts=chunks,
            metadatas=[{**metadata, "doc_id": doc_id,
                        "version": metadata.get("version", 1),
                        "indexed_at": datetime.utcnow().isoformat()}
                       for _ in chunks],
        )

    def expire_old_documents(self, ttl_days: int = 365) -> int:
        """移除早于 TTL 的文档。"""
        cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
        return self.vs.delete(filter={"indexed_at": {"$lt": cutoff}})
```

## RAG 与微调的协同

### 何时将 RAG 与微调结合

微调与 RAG 解决的是互补的弱点：

- **仅微调**：模型学到风格与格式，但可能在事实上产生幻觉
- **仅 RAG**：模型能访问事实，但可能不知道如何最优地使用它们
- **组合**：对模型进行微调，使其*善于使用检索到的上下文*——引用来源、承认不确定性、忽略无关上下文

### RAFT：检索增强微调（Retrieval-Augmented Fine-Tuning）

RAFT [[291]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-zhang2024raft)] 在混合了相关文档与*干扰*文档的设置下训练模型回答问题，教会模型识别并只使用相关上下文：

1. 对每个训练样本 $$(q, a, d^*)$$，采样 $k-1$ 个干扰文档 $$\{d_i^-\}$$
2. 在 `[q, `$$d^*$$`, `$$d_1^-$$`, \ldots{}, `$$d_{k-1}^-$$`]` $\to$ `[chain-of-thought + a]` 上微调
3. 思维链显式引用 $$d^*$$ 中的内容，教模型对答案进行接地

$$
  \mathcal{L}_{\text{RAFT}} = -\mathbb{E}_{(q,a,d^*,\{d_i^-\})} \left[
    \log P_\theta\!\left(\text{CoT}(d^*) \oplus a \;\middle\vert\; q, d^*, \{d_i^-\}\right)
  \right]
$$

### 检索器—生成器联合训练

为获得极致性能，可以对检索器与生成器进行联合训练。REALM [[292]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-guu2020realm)] 与 RAG [[109]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lewis2020retrieval)] 论文提出了端到端训练方法，让 Gradient 流经检索步骤：

$$
  \nabla_\theta \mathcal{L} = \nabla_\theta \left[
    -\log \sum_{d \in \mathcal{D}} P_\theta(a \mid q, d) \cdot P_\phi(d \mid q)
  \right]
$$

检索器参数 $\phi$ 通过 REINFORCE 估计器更新，或将 $$P_\phi(d \mid q)$$ 视为关于文档的可微 Attention 进行更新。

> **联合训练的挑战**
>
> 检索器—生成器联合训练强大但复杂：(1) 随着 $\phi$ 变化，文档索引必须定期刷新（异步索引刷新），(2) 训练信号稀疏（仅 top-$k$ 文档参与贡献），(3) 在未从预训练检索器仔细初始化的情况下，训练不稳定。

## 各种 RAG 方法的综合对比

| 方法 | 准确率 | 延迟 | 复杂度 | 成本 | 最佳适用场景 |
| --- | --- | --- | --- | --- | --- |
| 朴素 RAG [[109]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lewis2020retrieval)] | 中等 | 低 | 低 | 低 | 原型、简单问答 |
| RAG + 重排序 [[273]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-nogueira2019passage)] | 高 | 中等 | 中等 | 中等 | 生产级问答系统 |
| HyDE [[274]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-gao2022precise)] | 高 | 中等 | 低 | 中等 | 语义不匹配的领域 |
| 多查询 RAG | 高 | 中等 | 中等 | 中等 | 歧义查询 |
| RAG-Fusion [[279]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rackauckas2023ragfusion)] | 高 | 中等 | 中等 | 中等 | 多样化的查询类型 |
| Self-RAG [[275]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-asai2023selfrag)] | 高 | 中等 | 高 | 中等 | 选择性检索 |
| CRAG [[276]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yan2024crag)] | 高 | 中等 | 高 | 高 | 不可靠的语料库 |
| 自适应 RAG [[277]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-jeong2024adaptive)] | 高 | 低--高 | 高 | 中等 | 查询复杂度混合 |
| Graph RAG [[278]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-edge2024local)] | 极高 | 高 | 极高 | 高 | 全局综合查询 |
| 智能体 RAG | 极高 | 高 | 极高 | 高 | 多跳推理 |
| RAFT [[291]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-zhang2024raft)] | 极高 | 低 | 极高 | 极高 | 特定领域部署 |

> **RAG 系统的关键设计问题**
>
> 在为生产环境设计 RAG 系统时，请考虑以下问题：
>
> 1. **查询分布是怎样的？**事实型 vs. 分析型 vs. 多跳查询需要不同的检索策略。
> 2. **语料库有多大、变化多频繁？**数百万文档且频繁更新更适合具备增量索引的托管向量数据库。
> 3. **延迟需求如何？**低于 100ms 的响应排除了重排序和智能体循环；批处理或异步场景则可以承受。
> 4. **接地有多关键？**高风险领域（医疗、法律、金融）需要忠实度评估与引用验证。
> 5. **词汇是否专业化？**特定领域术语可能需要混合检索或经过领域适配的 Embedding 模型。

> **RAG 最佳实践总结**
>
> - **从简单开始**：分块得当的朴素 RAG 往往优于分块糟糕的复杂系统
> - **独立评估检索**：先修好检索，再优化生成
> - **使用混合检索**：BM25 + 稠密检索 + RRF 是一个强大的默认方案
> - **加上重排序**：在 top-20 候选上使用交叉编码器重排序器具有很高的 ROI
> - **监控忠实度**：在生产中使用 LLM 判官跟踪幻觉率
> - **积极缓存**：文档只 Embedding 一次；缓存频繁查询的 Embedding
> - **带重叠地分块**：10--15% 的重叠可防止在边界处丢失信息
> - **存储丰富的元数据**：来源、日期、章节和文档类型能支持强大的预过滤，显著提升精度
