---
layout: home
title: LLM 评估
permalink: /part4/ch14-evaluation.html
---

# LLM 评估

评估是任何严谨机器学习流水线的支柱，但它或许是大语言模型开发中最被低估的组成部分。与经典的监督学习不同——在那里，一个带有真实标签的留出测试集就能提供干净的信号——评估 LLM 需要应对开放式生成、主观质量判断、多步推理链，以及无处不在的基准污染风险。本节系统性地梳理评估领域的全景：从评估类型的分类法、人工标注的机制，到排序指标的数学原理、LLM-as-Judge 的实操要点，再到那些悄无声息地腐蚀评估流水线的陷阱。

> **为何 LLM 评估如此困难**
>
> 1. **输出空间无界。**语言模型可以生成任意字符串；很少存在唯一正确答案。
> 2. **质量是多维的。**有用性、事实性、安全性、连贯性与风格是相互独立的维度，彼此之间可能存在权衡。
> 3. **评估本身就是一个语言任务。**判断回答是否优良需要理解能力，这意味着评估本身会暴露在与生成相同的失败模式之下。

## 评估方案设计

在收集任何一个数据点之前，从业者必须先决定要*什么*以及*如何*测量。一套有原则的分类法能避免一个常见错误：仅凭便利性而非与部署目标的对齐性来选择指标。

### 评估类型分类法

**内在评估与外在评估（Intrinsic vs. Extrinsic）。**

*内在*（Intrinsic）评估孤立地度量模型输出的性质，不参照下游应用。留出语料上的困惑度、相对参考译文的 BLEU 分数、编码基准上的 pass@$k$，都属于内在评估。*外在*（Extrinsic）评估则度量模型对真实任务或系统的影响：将 LLM 接入客服流水线后，工单升级率是否下降？编码助手是否提升了开发者的开发速度？

> **内在—外在差距**
>
> 内在指标廉价且可复现，但往往与真实世界的效用相关性很弱。困惑度更低的模型未必更有用。外在指标昂贵且缓慢，但能直接度量我们真正关心的东西。一套成熟的评估策略会用内在指标做快速迭代，用外在指标做最终验证。

**自动评估与人工评估（Automatic vs. Human）。**

*自动*评估使用确定性函数（BLEU、精确匹配）或学习得到的模型（BERTScore、LLM-as-Judge），无需人工参与即可对输出打分。*人工*评估则由标注者对模型输出进行打分或排序。下表总结了二者的权衡。

| 类型 | 成本 | 速度 | 可复现性 | 效度 |
| --- | --- | --- | --- | --- |
| 自动（基于规则） | 极低 | 极快 | 完美 | 低--中 |
| 自动（基于模型） | 低 | 快 | 高 | 中--高 |
| 众包人工 | 中 | 数天 | 中 | 中 |
| 专家人工 | 高 | 数周 | 低--中 | 高 |
| 外在 / A/B 测试 | 极高 | 数月 | 低 | 极高 |

**基于参考与无参考评估（Reference-Based vs. Reference-Free）。**

基于参考的指标（BLEU、ROUGE、BERTScore）将模型输出与一个或多个金标准参考进行比较。无参考指标（困惑度、LLM-as-Judge、人类偏好）则无需参考即可评估质量。当输出空间过大、无法穷尽收集参考时（例如开放式对话），无参考方法是必不可少的。

### 何时使用何种方法

> **示例：对话助手的评估策略**
>
> **开发阶段：**使用自动指标（困惑度、摘要子任务上的 ROUGE、工具调用上的 pass@$k$）进行快速迭代。每晚在标准评测套件（MMLU、HellaSwag、HumanEval）上运行基准测试。
>
> **发布前阶段：**开展人类偏好研究，将新模型与上一个检查点对比。在多样化的 Prompt 集合上使用 LLM-as-Judge 进行可扩展的成对比较。
>
> **发布后阶段：**监控外在指标（用户满意度分数、任务完成率），并留意生产 Prompt 中的分布漂移。

一个实用的决策框架：

- 若任务有明确正确答案（数学、代码、事实型问答）：使用精确匹配或基于执行的指标。
- 若任务是开放式的但存在参考输出：使用基于参考的指标作为下界，并辅以 LLM-as-Judge。
- 若任务是主观的（有用性、语气、创造性）：使用人工评估或经过良好校准的 LLM 评判者。
- 若任务涉及多步 Agent 行为：使用任务成功率与轨迹效率（见「Agent 任务的指标」一节）。

## 评估数据采集

高质量的评估数据是可信基准的基础。本节涵盖人工标注流水线的设计、标注质量的统计度量，以及在众包与专家标注之间如何取舍。

### 人工标注流水线

一条健壮的标注流水线包含五个阶段：

1. **任务定义。**精确说明标注任务：评分对象是什么、采用何种量表、依据哪些标准。此阶段的歧义会传导为带噪标签。
2. **规范编写。**撰写标注规范，并配以覆盖边界情形的详细示例。在全面部署前先用小规模试点组反复打磨。
3. **标注者招募与培训。**选择具备相应背景知识的标注者。组织一次校准会议，让标注者标注相同样本并讨论分歧。
4. **质量控制。**在标注队列中嵌入带已知标签的金标样本。对在金标样本上准确率低于阈值的标注者进行标记。
5. **聚合。**使用多数投票、平均，或概率模型（如 Dawid--Skene）来合并每条样本的多次标注。

### 标注者间一致性

原始一致率（所有标注者达成一致的样本比例）是一个不充分的度量，因为它没有考虑偶然一致。两个标准的偶然性修正度量是 Cohen's $\kappa$[cohen1960coefficient]（两位标注者）和 Fleiss' $\kappa$[fleiss1971measuring]（多位标注者）。

**Cohen's Kappa。**

给定两位标注者将 $N$ 个样本标注为 $k$ 个类别，令 $p_o$ 为观察到的一致率，$p_e$ 为独立性假设下的期望一致率：

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

其中

$$
p_o = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{annotator 1 agrees with annotator 2 on item } i]
$$

以及

$$
p_e = \sum_{c=1}^{k} p_{1c} \cdot p_{2c}
$$

其中 $p_{jc}$ 是标注者 $j$ 将样本判为类别 $c$ 的比例。Cohen's $\kappa$ 的取值从 $-1$（完全不一致）经 $0$（偶然一致）到 $1$（完全一致）。一般认为 $0.6$ 以上可以接受；$0.8$ 以上为强一致。

**Fleiss' Kappa。**

对于 $n$ 位标注者将 $N$ 个样本标注为 $k$ 个类别，令 $n_{ij}$ 为将样本 $i$ 标注为类别 $j$ 的标注者人数。定义：

$$
\bar{P}_i = \frac{1}{n(n-1)} \sum_{j=1}^{k} n_{ij}(n_{ij} - 1), \qquad \bar{P} = \frac{1}{N}\sum_{i=1}^{N}\bar{P}_i
$$

$$
\bar{P}_j^e = \frac{1}{Nn}\sum_{i=1}^{N} n_{ij}, \qquad P_e = \sum_{j=1}^{k} \left(\bar{P}_j^e\right)^2
$$

$$
\kappa_F = \frac{\bar{P} - P_e}{1 - P_e}
$$

> **Kappa 的局限**
>
> Kappa 对类别分布敏感：当某一类别占据主导时，即使原始一致率很高，kappa 仍可能很低（即*kappa 悖论*）。对于序数量表，加权 kappa（按分歧的距离比例进行惩罚）更为合适。在 LLM 评估中，评分通常采用 1--5 的 Likert 量表，应始终报告加权 kappa。

### 标注规范设计

有效的标注规范通常具备以下若干特征：

- **可操作化的标准。**用具体、可观察的行为替换「有用」这类模糊词汇：例如「回答直接处理了用户的问题，并提供了完成所述任务所需的全部信息」。
- **详细示例。**每个评分等级至少提供两个示例，包括边界情形。
- **决策树。**对于复杂任务，用流程图引导标注者完成一系列二元决策，可降低认知负担并提升一致性。
- **明确的范围。**说明标注者*不*应考虑哪些因素（例如「不要因风格偏好而扣分；只关注事实准确性」）。

### 众包标注与专家标注

| 维度 | 众包 | 专家标注 |
| --- | --- | --- |
| 单条样本成本 | 低（\$0.01--\$0.10） | 高（\$1--\$50） |
| 吞吐量 | 极高 | 低 |
| 领域知识 | 低 | 高 |
| 一致性 | 不稳定 | 高 |
| 适用任务 | 简单偏好、流畅度 | 技术准确性、安全性 |
| 平台 | MTurk、Prolific、Scale AI | 领域专家、内部团队 |
| 质量控制 | 金标样本、注意力检查 | 校准会议、同行复核 |

对于安全关键型评估（例如检测有害输出、评估医疗建议），专家标注是不可妥协的。对于大规模偏好采集（例如构建 Reward 模型训练集），配以严格质量控制的众包往往是唯一可行的选择。

## 评估用合成数据生成

人工标注昂贵且缓慢。合成数据生成利用 LLM 自身大规模生产评估数据。本节涵盖其主要范式。

### 用于校准的 LLM-as-Judge

当用 LLM 生成评估标签时，校准至关重要：评判者的分数必须与人类判断对齐。令 $h_i \in [0,1]$ 为样本 $i$ 的人类偏好分数，$\hat{h}_i$ 为评判者预测分数。校准误差由期望校准误差（Expected Calibration Error, ECE）[guo2017calibration] 度量：

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|
$$

其中 $B_b$ 是第 $b$ 个置信度区间，$\text{acc}(B_b)$ 是该区间内评判者与人类一致的样本比例，$\text{conf}(B_b)$ 是该区间内评判者的平均置信度。

一个良好校准的评判者满足：对所有 $p \in [0,1]$，$\mathbb{E}[\hat{h}_i \mid \hat{h}_i = p] = p$。校准可通过温度缩放改善：把评判者的原始 logit $z$ 替换为 $z/T$，其中 $T$ 在留出的校准集上通过最小化负对数似然来调优。

### Self-Instruct

Self-Instruct[wang2022selfinstruct] 从一个由人工撰写的种子任务集合自举出指令跟随数据。算法如下：

1. 维护一个任务池，初始化为 $175$ 个种子任务。
2. 从任务池中采样 $8$ 个任务，将其作为少样本示例，提示 LLM 生成新任务。
3. 过滤生成的任务：移除近似重复项（与任何已有任务的 ROUGE-L 相似度 $> 0.7$），区分分类与非分类任务，并生成输入--输出实例。
4. 将接受的任务加入任务池。
5. 重复直到达到目标任务池规模。

> **示例：Self-Instruct Prompt 模板**
>
> ```python
> system_prompt = """
> Come up with a series of tasks:
> Task 1: {seed_task_1_instruction}
> Task 2: {seed_task_2_instruction}
> ...
> Task 8: {seed_task_8_instruction}
> Task 9:"""
> ```
>
> 模型补全该 Prompt，生成新的任务指令。随后由另一个 Prompt 为该新任务生成输入--输出对。

### Evol-Instruct

Evol-Instruct[xu2023wizardlm] 通过迭代地将指令改写得更复杂或更多样，来演化一个种子指令集。应用两类演化算子：

- **深度演化（In-depth evolution）：**增加约束、提升推理步骤数、将抽象具体化、加深对领域知识的要求。
- **广度演化（In-breadth evolution）：**在相关但不同的主题上生成新指令，提升主题多样性。

若指令通过淘汰过滤器，则被接受：演化后的指令不能是简单复制、不能包含 "I'm sorry" 或类似的拒绝表述，且长度不能短于原指令。

### Constitutional AI 数据生成

Constitutional AI（CAI）[bai2022constitutional] 通过让模型依据一组原则（即「宪法」）对自身输出进行批评与修订，从而生成偏好数据。流水线如下：

1. **监督学习阶段：**采样一个有害 Prompt，生成初始回答，然后提示模型依据某条宪法原则对回答进行批评并修订。将修订后的回答作为监督微调（SFT）的目标。
2. **RL 阶段：**生成回答对（原始 vs. 修订），用模型标注哪一个更符合宪法，并在这些标签上训练偏好模型。将偏好模型作为 RLHF 的 Reward 信号。

这种方法无需让人工标注有害内容即可生成偏好数据，减少了标注者暴露于令人不适素材的风险。

### 评估数据的蒸馏

一个强大的教师模型（例如 GPT-4）可以生成高质量的评估数据，用于训练较小的评判模型。蒸馏流水线如下：

1. 收集多样化的 Prompt 与模型回答集合。
2. 使用教师模型生成详尽的判断（分数 + 理由）。
3. 在（Prompt、回答、判断）三元组上微调一个较小的模型。
4. 在留出的人工标注上验证学生评判者。

> **蒸馏带来的偏置**
>
> 从单一教师蒸馏得到的学生评判者会继承教师的偏置，包括冗长偏置（偏好更长的回答）、自我增强偏置（如果教师同时也是被评估的模型），以及位置偏置。始终要在独立的人工标注上验证蒸馏所得的评判者。

### Arena 风格的成对生成

Chatbot Arena[zheng2023judging] 通过一个众包对战平台来生成评估数据：用户提交 Prompt 并在两个匿名化的模型回答之间投票选择偏好。这能产生一个大规模、自然多样的成对偏好数据集。关键设计选择包括：

- **匿名化：**隐藏模型身份，防止品牌偏置。
- **用户提交的 Prompt：**保证 Prompt 的多样性以及与真实世界的相关性。
- **平局处理：**用户可声明平局，或表明两个回答都很差。
- **去重：**过滤近似重复的 Prompt，防止常见查询被过度代表。

## 排序任务的指标

当目标是按质量对模型进行排序时，成对比较数据比绝对分数更为可靠。本节推导 LLM 评估中常用的主要排序系统。

### ELO 评分系统

ELO 系统[elo1978rating] 最初为国际象棋而设计，它为每位选手（模型）赋予一个标量评分 $R$，使得选手 $A$ 对阵选手 $B$ 的期望得分为：

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

**推导。**

ELO 模型假设：每位选手在某一局比赛中的表现是一个以其评分为中心的 Logistic 分布所抽取的随机变量。$A$ 击败 $B$ 的概率为：

$$
P(A \succ B) = \sigma\!\left(\frac{R_A - R_B}{s}\right) = \frac{1}{1 + e^{-(R_A - R_B)/s}}
$$

其中 $s = 400/\ln(10) \approx 173.7$ 是一个尺度参数，其选择使得 400 分的差距对应于 $10:1$ 的胜率比。每场比赛后，给定结果 $S_A \in \{0, 0.5, 1\}$（负、平、胜），评分更新如下：

$$
R_A \leftarrow R_A + K(S_A - E_A), \qquad R_B \leftarrow R_B + K(S_B - E_B)
$$

其中 $K$ 是控制学习率的 $K$-因子。在 Chatbot Arena 中使用 $K = 4$。

> **ELO 的直觉**
>
> ELO 本质上是在 Logistic 模型下，对观察结果的对数似然进行随机梯度下降式更新。每场比赛提供一个带噪的 Gradient 信号；$K$-因子控制步长。$K$ 越大，适应越快但噪声越大；$K$ 越小则越稳定，但反映真实实力变化的速度越慢。

**ELO 的 Bootstrap 置信区间。**

由于 ELO 评分依赖于比赛被处理的顺序，置信区间通过 Bootstrap 重采样计算：对对战日志有放回地重采样 $B = 1000$ 次，对每次重采样都从头重新计算 ELO 评分，并以第 2.5 与第 97.5 百分位作为 95\% 置信区间。

### Bradley--Terry 模型

Bradley--Terry（BT）模型[bradley1952rank] 是 ELO 的一种最大似然替代方案。给定 $n$ 个模型，其实力参数为 $\beta_1, \ldots, \beta_n > 0$，模型 $i$ 击败模型 $j$ 的概率为：

$$
P(i \succ j) = \frac{\beta_i}{\beta_i + \beta_j}
$$

给定一组成对结果 $\{(i_k, j_k, y_k)\}_{k=1}^{M}$，其中若 $i_k$ 击败 $j_k$ 则 $y_k = 1$，否则 $y_k = 0$，对数似然为：

$$
\ell(\boldsymbol{\beta}) = \sum_{k=1}^{M} \left[ y_k \log \frac{\beta_{i_k}}{\beta_{i_k} + \beta_{j_k}} + (1-y_k) \log \frac{\beta_{j_k}}{\beta_{i_k} + \beta_{j_k}} \right]
$$

极大似然估计 $\hat{\boldsymbol{\beta}}$ 可通过迭代缩放或梯度上升求得。BT 模型仅在相差一个乘法常数的意义下可辨识；常用的归一化是 $\sum_i \log \beta_i = 0$。在对数空间中令 $\theta_i = \log \beta_i$ 得：

$$
P(i \succ j) = \sigma(\theta_i - \theta_j)
$$

这等价于一个带条目特定截距的 Logistic 回归。在能拿到完整对战历史时，BT 模型优于 ELO，因为它同时利用了所有数据，而不是顺序处理比赛。

### TrueSkill

TrueSkill[herbrich2006trueskill] 是一种贝叶斯式的实力评分系统，将每位选手的实力建模为高斯随机变量 $s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$。选手 $i$ 在某一局中的表现为 $p_i = s_i + \epsilon_i$，其中 $\epsilon_i \sim \mathcal{N}(0, \beta^2)$ 为比赛特定的噪声。当 $p_i > p_j$ 时，选手 $i$ 击败选手 $j$。

观察到 $i \succ j$ 后的后验更新通过期望传播（Expectation Propagation, EP）计算。胜方的关键更新方程为：

$$
\mu_i \leftarrow \mu_i + \frac{\sigma_i^2}{c} \cdot v\!\left(\frac{\mu_i - \mu_j}{c}\right)
$$

$$
\sigma_i^2 \leftarrow \sigma_i^2 \left[1 - \frac{\sigma_i^2}{c^2} \cdot w\!\left(\frac{\mu_i - \mu_j}{c}\right)\right]
$$

其中 $c = \sqrt{2\beta^2 + \sigma_i^2 + \sigma_j^2}$，$v(t) = \phi(t)/\Phi(t)$，$w(t) = v(t)(v(t) + t)$ 为截断高斯修正因子（$\phi$ 与 $\Phi$ 分别是标准正态的 PDF 与 CDF）。TrueSkill 的不确定性估计 $\sigma_i$ 在识别需要更多评估数据的模型时尤其有用。

### 带置信区间的胜率

最简单的排序指标是胜率：模型 $A$ 在成对比较中胜出的比例。给定 $n$ 次比较，其中 $w$ 次胜出，胜率为 $\hat{p} = w/n$。在 $p = 0$ 与 $p = 1$ 附近，Wilson 分数置信区间[wilson1927probable] 的覆盖性能优于朴素的 Wald 区间，因此更被推荐：

$$
\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

其中 95\% 区间对应 $z = 1.96$。对于多方比较，胜率应针对一个固定的基线模型来计算，以确保可比性。

### Chatbot Arena 方法论

Chatbot Arena[zheng2023judging] 将上述要素组合为一个生产规模的评估系统：

1. 用户提交 Prompt，并获得两个匿名化模型的回答。
2. 用户对偏好的回答进行投票（或声明平局）。
3. 投票通过 BT 模型聚合以生成排行榜。
4. 对每个模型的分数报告 Bootstrap 置信区间。
5. 置信区间重叠的模型在统计上被视为无法区分。

截至 2024 年，Chatbot Arena 已累计收集超过一百万条人类偏好投票，是目前公开可得的最大规模 LLM 偏好数据集。

## 生成任务的指标

生成指标量化了在具备参考答案或明确正确性标准的任务上模型输出的质量。

### BLEU

BLEU（Bilingual Evaluation Understudy）[papineni2002bleu] 度量假设译文 $h$ 与一个或多个参考 $\mathcal{R}$ 之间的 $n$-gram 精度：

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

其中 $p_n$ 是修正后的 $n$-gram 精度，$w_n = 1/N$ 为均匀权重，BP 为简短惩罚（brevity penalty）：

$$
\text{BP} = \begin{cases} 1 & \text{if } |h| > |r| \\ e^{1 - |r|/|h|} & \text{if } |h| \leq |r| \end{cases}
$$

其中 $|r|$ 为长度最接近的参考的长度。修正后的 $n$-gram 精度将每个 $n$-gram 的计数截断到其在任一参考中的最大出现次数：

$$
p_n = \frac{\sum_{\text{ngram} \in h} \min\!\left(\text{count}(\text{ngram}, h),\, \max_{r \in \mathcal{R}} \text{count}(\text{ngram}, r)\right)}{\sum_{\text{ngram} \in h} \text{count}(\text{ngram}, h)}
$$

> **BLEU 的局限**
>
> BLEU 是为带多个参考的机器翻译而设计的。对于只有单一参考的开放式生成，即使输出质量很高，BLEU 分数也常常接近零。BLEU 无法捕捉语义相似性，会惩罚合理的复述，并且对分词敏感。仅在存在多个多样化参考且任务输出多样性较低时再使用 BLEU。

### ROUGE

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）[lin2004rouge] 是一类面向召回的指标，专为摘要任务设计：

$$
\text{ROUGE-N} = \frac{\sum_{r \in \mathcal{R}} \sum_{\text{ngram} \in r} \min(\text{count}(\text{ngram}, h), \text{count}(\text{ngram}, r))}{\sum_{r \in \mathcal{R}} \sum_{\text{ngram} \in r} \text{count}(\text{ngram}, r)}
$$

$$
\text{ROUGE-L} = \frac{\text{LCS}(h, r)}{|r|}
$$

其中 LCS 表示最长公共子序列（Longest Common Subsequence）。ROUGE-1 与 ROUGE-2 度量 unigram 与 bigram 的召回；ROUGE-L 则捕捉句子级结构。F-measure 变体在精度与召回之间取得平衡：

$$
\text{ROUGE-N}_F = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 P + R}
$$

当 $\beta = 1$ 时为等权。

### BERTScore

BERTScore[zhang2020bertscore] 使用预训练 BERT 模型的上下文 Embedding 来计算 Token 级相似度。给定假设 Token $\hat{\mathbf{x}} = \langle \hat{x}_1, \ldots, \hat{x}_m \rangle$ 与参考 Token $\mathbf{x} = \langle x_1, \ldots, x_n \rangle$，及其 Embedding $\hat{\mathbf{e}}_i$ 与 $\mathbf{e}_j$：

$$
R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_j \in \mathbf{x}} \max_{\hat{x}_i \in \hat{\mathbf{x}}} \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \|\mathbf{e}_j\|}
$$

$$
P_{\text{BERT}} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{\mathbf{x}}} \max_{x_j \in \mathbf{x}} \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \|\mathbf{e}_j\|}
$$

$$
F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
$$

BERTScore 与人类判断的相关性优于 BLEU 与 ROUGE，尤其在处理复述以及语义等价但词面不同的输出时。使用逆文档频率（IDF）进行重要性加权可进一步提升相关性：

$$
R_{\text{BERT}}^{\text{idf}} = \frac{\sum_{x_j \in \mathbf{x}} \text{idf}(x_j) \max_{\hat{x}_i} \cos(\hat{\mathbf{e}}_i, \mathbf{e}_j)}{\sum_{x_j \in \mathbf{x}} \text{idf}(x_j)}
$$

### METEOR

METEOR[banerjee2005meteor] 通过在 unigram 匹配上计算 F-score，并配合词干化与同义词匹配模块，解决了 BLEU 对召回的盲视：

$$
\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Pen})
$$

其中 $F_{\text{mean}} = \frac{10PR}{R + 9P}$（召回加权的调和平均），碎片化惩罚 $\text{Pen} = 0.5 \cdot (c/u_m)^3$ 对非连续匹配进行惩罚（$c$ 为块数，$u_m$ 为匹配的 unigram 数）。

### 困惑度（Perplexity）

困惑度（Perplexity）衡量语言模型对留出文本序列 $w_1, w_2, \ldots, w_T$ 的预测能力：

$$
\text{PPL}(w_{1:T}) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{1:t-1})\right)
$$

困惑度越低，预测性能越好。困惑度适合在相同分词与测试集上比较模型，但在词表或分词器不同的模型间不能直接比较。在评估中，困惑度最实用的用途是作为一种合理性检查（sanity check）以及用于检测分布漂移。

### 代码任务的 Pass@k

对于代码生成，功能正确性通过在测试用例上执行生成的代码来衡量。pass@$k$ 指标[chen2021evaluating] 估计在 $k$ 个生成样本中至少有一个通过全部测试的概率：

$$
\text{pass@}k = \mathbb{E}_{\text{problems}}\!\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

其中 $n$ 是每个问题生成的样本总数，$c$ 是其中通过测试的样本数。该无偏估计量避免了朴素估计量（精确采样 $k$ 个解并检查是否通过）的高方差问题。在实践中通常生成 $n = 200$ 个样本，并报告 pass@1、pass@10、pass@100。

> **示例：Pass@k 计算**
>
> ```python
> import numpy as np
> from scipy.special import comb
>
> def pass_at_k(n: int, c: int, k: int) -> float:
>     """pass@k 的无偏估计量。
>
>     Args:
>         n: 每个问题生成的样本总数
>         c: 通过全部测试的样本数
>         k: 考虑的样本数
>     """
>     if n - c < k:
>         return 1.0
>     return 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)
>
> # 示例：200 个样本，15 个通过，计算 pass@1、pass@10、pass@100
> for k in [1, 10, 100]:
>     score = pass_at_k(n=200, c=15, k=k)
>     print(f"pass@{k}: {score:.4f}")
> # pass@1:   0.0750
> # pass@10:  0.5391
> # pass@100: 0.9999
> ```

### 精确匹配与 F1

对于抽取式问答（例如 SQuAD），有两个标准指标：

- **精确匹配（Exact Match, EM）：**二元指示量，判断归一化（小写化、去除冠词与标点）后的预测答案字符串是否与任一金标答案完全相同。
- **Token 级 F1：**将预测与金标答案视为 Token 的多重集，并计算 F1 分数：

$$
F1 = \frac{2 \cdot |\text{pred} \cap \text{gold}|}{|\text{pred}| + |\text{gold}|}
$$

对于多答案设置，报告对所有金标答案取最大值后的 F1。

| 指标 | 任务 | 是否无参考？ | 与人类判断的相关性 |
| --- | --- | --- | --- |
| BLEU | 翻译 | 否 | 低--中 |
| ROUGE | 摘要 | 否 | 中 |
| BERTScore | 通用 NLG | 否 | 高 |
| METEOR | 翻译 | 否 | 中--高 |
| Perplexity | 语言模型质量 | 是 | 低 |
| Pass@k | 代码生成 | 否（依赖测试） | 极高 |
| Exact Match | 抽取式问答 | 否 | 极高 |
| Token F1 | 抽取式问答 | 否 | 高 |

## Agent 任务的指标

Agent 化的 LLM 在环境中运作、执行动作序列，并须完成多步任务。标准生成指标在此并不充分；Agent 评估需要能够刻画任务完成度、效率以及中间步骤质量的指标。

### 任务成功率

Agent 任务的主要指标是任务成功率（Task Success Rate, TSR）：Agent 达成指定目标状态的任务比例：

$$
\text{TSR} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{goal}(\tau) \text{ achieved}]
$$

目标的达成通常由一个确定性的 Oracle 来验证（例如检查数据库状态、文件系统状态或测试用例执行结果）。对于允许部分得分的任务，可以定义分级成功度量：

$$
\text{TSR}_{\text{graded}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{score}(\tau) \in [0, 1]
$$

### 轨迹效率

一个成功的 Agent 应以尽可能少的非必要动作完成任务。轨迹效率衡量最优轨迹长度与 Agent 实际轨迹长度之比：

$$
\eta = \frac{L^*}{L_{\text{agent}}}
$$

其中 $L^*$ 是最短成功轨迹的长度（由 Oracle 或人类专家计算得出），$L_{\text{agent}}$ 是 Agent 所执行的动作数。$\eta \in (0, 1]$，$\eta = 1$ 表示最优效率。对于失败的轨迹，$\eta = 0$。

一个互补指标是*冗余率*（redundancy rate）：Agent 动作中不出现在任何最优轨迹里的比例。

### 工具调用准确性

对于调用外部工具（API、代码解释器、搜索引擎）的 Agent，工具调用（Tool Calling）准确性衡量调用的正确性：

$$
\text{TUA} = \frac{\text{\# correct tool calls}}{\text{\# total tool calls}}
$$

一次工具调用被视为正确当且仅当：（a）选择了正确的工具，（b）参数有效，（c）该调用发生在轨迹中合适的时机。当工具选择正确但参数错误时，可酌情给予部分得分。

### 多步推理准确性

对于需要推理链的任务（例如多跳问答、数学解题），步骤级准确性衡量正确推理步骤所占的比例：

$$
\text{SRA} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{|S_\tau|} \sum_{s \in S_\tau} \mathbf{1}[s \text{ is correct}]
$$

其中 $S_\tau$ 是轨迹 $\tau$ 中的推理步骤集合。步骤的正确性可由过程奖励模型（Process Reward Model, PRM）或人工标注来验证。

### SWE-bench 方法论

SWE-bench[jimenez2024swebench] 在真实软件工程任务上评估 LLM：给定一个 GitHub issue 描述与仓库代码库，模型必须生成解决该 issue 的 patch。评估流程如下：

1. 将 issue 描述与相关代码上下文提供给模型。
2. 模型生成一个 patch（统一 diff 格式）。
3. 将 patch 应用到仓库。
4. 执行仓库的测试套件；若全部测试通过，则任务成功。

主要指标为 **% Resolved**：生成的 patch 能通过全部测试的 issue 所占比例。SWE-bench Verified 是一个经过人工标注者验证为可解且无歧义的 500 题精选子集。SWE-bench Lite 则是一个 300 题的子集，专为更快的评估而设计。

> **SWE-bench 关键数据（截至 2024 年）**
>
> - **完整基准：**来自 12 个热门 Python 仓库的 2,294 个任务。
> - **最佳开源 Agent：**约 43\% 解决率（SWE-bench Verified）。
> - **人类表现：**约 87\% 解决率（每个任务 15 分钟）。
> - **评估成本：**基于 API 的模型每个任务约 \$0.25。

### WebArena 方法论

WebArena[zhou2024webarena] 在沙箱浏览器环境中以贴近真实的网页导航任务评估 Agent。该基准包含跨五个 Web 应用（电商、社交论坛、协作开发、内容管理与地图）的 812 个任务。评估方式如下：

- **功能性评估：**通过检查应用状态来验证任务结果（例如「商品是否已加入购物车？」「帖子是否已创建？」）。
- **基于 URL 的评估：**对于导航任务，将最终 URL 与期望 URL 进行比较。
- **基于程序的评估：**由自定义评估脚本检查复杂条件（例如「价格是否小于 \$50？」）。

主要指标是任务成功率。人类表现约为 78\%；当前最先进的 Agent 约为 35--45\%。

| 基准 | 领域 | 任务数 | 评估方式 | SOTA（\%） |
| --- | --- | --- | --- | --- |
| SWE-bench | 软件工程 | 2,294 | 测试执行 | $\sim$43 |
| SWE-bench Lite | 软件工程 | 300 | 测试执行 | $\sim$50 |
| WebArena | 网页导航 | 812 | 状态/URL/程序 | $\sim$40 |
| ALFWorld[shridhar2021alfworld] | 家居任务 | 3,553 | 仿真器状态 | $\sim$90 |
| AgentBench[liu2023agentbench] | 多领域 | 1,091 | 任务特定 | $\sim$45 |

## LLM-as-Judge

LLM-as-Judge[zheng2023judging] 使用一个能力强的 LLM 来评估其他（或同一个）LLM 的输出。该方法无需人工标注就能扩展到大规模评估集合，并能为其判断提供详尽的理由。

### 设置与 Prompt 模板

向评判者提供一个 Prompt、一个或多个模型回答，以及一份评估细则。常见的三种格式：

**点评式评分（Pointwise scoring）。**

评判者对单个回答给出绝对分数：

> **示例：Pointwise Judge Prompt**
>
> ```python
> POINTWISE_PROMPT = """
> You are an expert evaluator. Rate the following response on a scale
> of 1-10 for helpfulness, accuracy, and clarity.
>
> [Question]
> {question}
>
> [Response]
> {response}
>
> Provide your evaluation in the following format:
> Reasoning: <step-by-step analysis>
> Score: <integer from 1 to 10>
> """
> ```

**成对比较（Pairwise comparison）。**

评判者比较两个回答并选出更优者：

> **示例：Pairwise Judge Prompt**
>
> ```python
> PAIRWISE_PROMPT = """
> You are an expert evaluator. Compare the two responses below and
> determine which is better. Consider helpfulness, accuracy, and
> depth of explanation.
>
> [Question]
> {question}
>
> [Response A]
> {response_a}
>
> [Response B]
> {response_b}
>
> Output exactly one of: [[A]], [[B]], or [[C]] (tie).
> Reasoning: <your analysis>
> Verdict: <[[A]], [[B]], or [[C]]>
> """
> ```

**参考引导式评分（Reference-guided scoring）。**

向评判者提供参考答案，并据此对回答进行打分。这对评判者本身可能缺乏可靠知识的事实型任务尤其有用。

### 位置偏置的缓解

LLM 评判者会表现出*位置偏置*（position bias）：系统性地偏好出现在特定位置（首位或末位）的回答。此偏置可能高达 10--15 个百分点。缓解策略：

1. **交换增强：**对每一对回答都以两种顺序（A vs. B 与 B vs. A）进行评判。判断一致则采纳；判断不一致则记为平局。
2. **校准式 Prompt：**显式指令评判者："你的评估不应受到回答呈现顺序的影响。"
3. **评判者集成：**使用多个评判者并配以不同的位置顺序，再聚合其裁决。
4. **强制思维链：**要求评判者在给出裁决前先产出详尽理由，以降低对表层位置线索的依赖。

> **冗长偏置**
>
> LLM 评判者同样表现出冗长偏置：更长的回答会被系统性地偏好，即便额外内容是无关或重复的。缓解方式：指示评判者对不必要的长度进行惩罚，并关注信息的质量而非数量。或者，在评判前将回答截断到固定长度。

### 多评判者面板

单一评判者可能带有系统性偏置。来自不同模型家族的评判者面板能提供更稳健的评估。给定 $J$ 位评判者，其裁决为 $v_1, \ldots, v_J \in \{A, B, \text{tie}\}$，面板裁决由多数投票决定。面板一致率为：

$$
\text{Agreement} = \frac{1}{\binom{J}{2}} \sum_{i < j} \mathbf{1}[v_i = v_j]
$$

对于三人评判者面板，一致裁决（三人全部同意）视为高置信度；2--1 分裂视为中置信度；三方平局视为低置信度。

### LLM 评判者的一致性指标

为验证 LLM 评判者，将其裁决与留出集上的人工标注进行比较。关键指标：

- **一致率：**评判者与人类标注一致的样本比例。
- **Cohen's $\kappa$：**经偶然性修正的一致率（见「Cohen's Kappa」公式）。
- **Spearman's $\rho$：**评判者分数与人类分数之间的秩相关，适用于序数评分。
- **Kendall's $\tau$：**另一种秩相关，对平局更具鲁棒性。

若评判者在一份具有代表性的样本上与人类标注者达到 $\kappa > 0.6$ 且一致率 $> 80\%$，则视为可靠。

### G-Eval 框架

G-Eval[liu2023geval] 是一个面向 LLM 评估的结构化框架，使用思维链（Chain-of-Thought，CoT）提示与 Token 概率加权来产出更可靠的分数。框架如下：

1. **生成评估步骤：**提示 LLM 为评估任务生成详尽的评分细则（例如"列出你评估一篇摘要连贯性时所采取的步骤"）。
2. **以概率加权方式打分：**对每个分值 $s \in \{1, 2, 3, 4, 5\}$，从评判模型获取对数概率 $\log P_\theta(s \mid \text{prompt, steps, response})$。最终分数为以概率加权的平均：

$$
\text{G-Eval score} = \sum_{s=1}^{5} s \cdot \frac{e^{\log P_\theta(s)}}{\sum_{s'=1}^{5} e^{\log P_\theta(s')}}
$$

3. **归一化：**将分数除以最大分值，映射到 $[0, 1]$。

G-Eval 与人类判断的相关性高于直接提示，尤其在连贯性与一致性这类细腻维度上，因为概率加权捕捉了评判者的不确定性，而非强制其作离散选择。

> **为何 G-Eval 有效**
>
> 标准提示要求评判者输出单个 Token（例如"4"），这丢弃了模型的不确定性。G-Eval 读取所有分值 Token 上的概率分布，实质上是在评判者的信念下计算期望分数。这类似于使用后验分布的均值而非众数。

## 评估陷阱

即便是精心设计的评估流水线也可能给出误导性的结果。本节梳理最常见的失效模式。

### 基准污染

当评估数据出现在模型的训练集中时（直接以原文出现，或间接以复述/语义相似形式出现），就发生了基准污染（benchmark contamination）。被污染的模型会取得虚高的分数，无法反映其真实的泛化能力。

**检测方法：**

- **$n$-gram 重叠：**计算与训练语料具有高 $n$-gram 重叠（例如 ROUGE-L $> 0.8$）的评估样本比例。
- **成员推断（Membership Inference）：**使用成员推断攻击来估计每个评估样本曾出现在训练集中的概率。
- **金丝雀串（Canary strings）：**在评估样本中嵌入独特的随机生成字符串，并检查模型是否能将其补全。
- **时间留出：**使用模型训练截止日期之后产生的评估数据。

**缓解：**

- 维护一个永不公开发布的私有测试集。
- 定期用新样本刷新基准。
- 报告训练数据截止日期以及去污染流程。

### 对基准的过拟合

即使没有直接污染，模型也可能通过反复评估与超参调优而被隐式地为特定基准而优化。这是一种*自适应过拟合*（adaptive overfitting）：基准向模型开发决策泄漏了信息。

> **基准的生命周期**
>
> 基准的效用会随着研究社区对其的优化而随时间退化。MMLU[hendrycks2021measuring] 曾是对世界知识的挑战性测试，如今模型在其上已接近人类水平，但这些模型在新的知识任务上仍会失败。新基准应被视为临时的信号源，而非永久的金标准。

### 评估中的 Goodhart 定律

Goodhart 定律指出：*"当一个度量成为目标时，它就不再是好的度量。"*[goodhart1984problems] 在 LLM 评估中，这有多种体现：

- **Reward 欺骗（Reward hacking）：**用基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）训练的模型可能学会利用 Reward 模型，而非真正提升质量。模型可能学会输出冗长、听起来自信但实际上事实错误的回答，这类回答在 Reward 模型上得分却很高。
- **指标博弈：**为最大化 BLEU 或 ROUGE 而微调的模型，可能产出在这些指标上得分良好但对人类用处不大的输出。
- **评判者博弈：**用 LLM-as-Judge 反馈训练的模型可能学到评判者的偏置（例如冗长偏置），而非真正提升质量。

> **抵御 Goodhart 定律的策略**
>
> 1. **指标多样化：**使用来自不同家族的多个指标；一个能博弈某个指标的模型，未必能同时博弈所有指标。
> 2. **留出评估：**保留一些不参与训练或模型选择的评估指标。
> 3. **人工抽查：**定期采样模型输出供人工审阅，独立于自动化指标之外。
> 4. **对抗性评估：**主动探测自动化指标会漏掉的失效模式。
> 5. **外在验证：**定期用外在结果来验证内在指标。

### 其他陷阱

**Prompt 敏感性。**

LLM 的性能可能因评估 Prompt 的微小改动而剧烈变化（例如加入"Think step by step"或更改回答格式）。务必报告所使用的确切 Prompt，并考虑在多种 Prompt 变体上进行评估。

**聚合伪影。**

在难度水平与分数分布各异的任务上对分数取平均，可能产生误导性的聚合指标。一个在简单任务上表现优异但在困难任务上失败的模型，可能与一个表现均衡的模型获得相同的平均分。

**人工评估中的选择偏差。**

人类评估者并不是终端用户的随机样本。众包平台上的标注者与目标用户群体可能在偏好、文化背景与领域知识上存在差异。

**评估—部署失配。**

评估 Prompt 往往比真实用户查询更短、更干净、形式也更规整。在基准 Prompt 上表现良好的模型，在生产环境中遇到嘈杂、含糊的多轮对话（Multi-Turn）时可能显著退化。

> **评估设计的关键问题**
>
> 在部署评估流水线之前，请自问：
>
> 1. 评估指标是否与部署目标对齐？
> 2. 评估数据是否代表目标分布？
> 3. 是否已评估污染与过拟合风险？
> 4. 是否对所有指标都报告了置信区间？
> 5. 评估是否可复现（固定随机种子、Prompt 版本化、公开测试集）？
> 6. 评估是否已与人类判断或外在结果进行过验证？
