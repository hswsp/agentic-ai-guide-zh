---
layout: home
title: 速查手册
permalink: /part6/ch28-quickref.html
---

# 速查手册

本章汇总了关键公式、架构规格、API 参考以及失效模式诊断，便于开发与调试时快速查阅。

## 核心 RL 与对齐公式

$$
\begin{align}
\text{PPO Clip:}&\quad L = \mathbb{E}[\min(r_t\hat{A}_t, \text{clip}(r_t,1{\pm}\epsilon)\hat{A}_t)], \quad r_t = \pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t) \\
\text{DPO:}&\quad L = -\mathbb{E}[\log\sigma(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)})] \\
\text{GRPO:}&\quad \hat{A}_i = (r_i - \mu_G)/\sigma_G, \quad \text{then PPO clip update (no critic)} \\
\text{KTO:}&\quad L = \lambda_w(1 - v(y_w)) + \lambda_l \cdot v(y_l), \quad v = \sigma(\beta\log(\pi_\theta/\pi_\text{ref}) - z) \\
\text{IPO:}&\quad L = \mathbb{E}[(\log(\pi_\theta(y_w)/\pi_\text{ref}(y_w)) - \log(\pi_\theta(y_l)/\pi_\text{ref}(y_l)) - 1/(2\beta))^2] \\
\text{ORPO:}&\quad L = L_\text{SFT}(y_w) - \lambda\log\sigma(\log(\text{odds}(y_w)/\text{odds}(y_l))) \\
\text{GAE:}&\quad \hat{A}_t = \textstyle\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \\
\text{KL Penalty:}&\quad R_\text{total} = r_\phi(x,y) - \beta D_\text{KL}[\pi_\theta(y\mid x)\|\pi_\text{ref}(y\mid x)] \\
\text{RM (Bradley-Terry):}&\quad L = -\mathbb{E}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))] \\
\text{Best-of-N:}&\quad y^* = \arg\max_{y_i \sim \pi_\theta(\cdot\mid x),\, i=1..N} r_\phi(x, y_i)
\end{align}
$$

## Transformer 与架构公式

$$
\begin{align}
\text{Self-Attention:}&\quad \text{Attn}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k}) \cdot V \\
\text{Multi-Head:}&\quad \text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O,\quad \text{head}_i = \text{Attn}(XW_i^Q, XW_i^K, XW_i^V) \\
\text{RoPE:}&\quad f(x_m, m) = x_m e^{im\theta_j}, \quad \theta_j = 10000^{-2j/d} \\
\text{LoRA:}&\quad W' = W_0 + (\alpha/r) \cdot BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k} \\
\text{KD (soft targets):}&\quad L_\text{KD} = (1{-}\alpha)L_\text{CE}(y, \hat{y}) + \alpha\, T^2 \cdot \text{KL}(p_T^\text{teacher} \| p_T^\text{student}) \\
\text{FFN (SwiGLU):}&\quad \text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2
\end{align}
$$

## 解码方法

| 方法 | 公式 / 规则 | 关键参数 |
| --- | --- | --- |
| Greedy | $y_t = \arg\max_v P(v\|y_{<t})$ | --- |
| Beam search | 按联合概率保留 top-$B$ 个部分序列 | $B=4$--$8$ |
| Temperature | $P'(v) = \text{softmax}(\text{logit}_v / T)$ | $T \in [0.1, 1.5]$ |
| Top-$k$ | 仅保留 top-$k$ 个 logit，其余置零后重新归一化 | $k=40$--$100$ |
| Top-$p$ (nucleus) | 保留最小集合 $V'$ 使 $\sum_{v \in V'} P(v) \geq p$ | $p=0.9$--$0.95$ |
| Min-$p$ | 保留满足 $P(v) \geq p_\text{min} \cdot P(v_\text{max})$ 的 token | $p_\text{min}=0.05$--$0.1$ |
| Repetition penalty | 若 $v$ 已出现，则 $\text{logit}_v \leftarrow \text{logit}_v / \theta$ | $\theta=1.1$--$1.3$ |

## 系统与并行

| 公式 | 取值 (70B, BF16) | 说明 |
| --- | --- | --- |
| 模型显存 | $2P$ 字节 | $140$ GB（仅权重） |
| Adam 优化器 | $2P \times 4$ 字节 (m + v) | $280$ GB |
| 完整训练占用 | $\sim 8P$ 字节 | $560$ GB（权重 + 优化器 + 梯度） |
| FSDP 每 GPU 显存 | $8P / N_\text{GPUs}$ | 8 卡时为 $70$ GB |
| 生成算术强度 | $2P / 2P = 1$ FLOP/byte | 严重内存瓶颈 |
| Token 速率（生成） | HBM\_BW $/ (2P)$ | $\sim$14 tok/s (A100, batch=1) |
| TP AllReduce / 层 | $2 \times 2 \cdot \frac{T-1}{T} \cdot bsd$ 字节 | $\sim$188 MB (70B, TP=8) |
| PP 气泡占比 | $(P-1)/(P+M-1)$ | $P$=阶段数，$M$=微批数 |
| MFU | observed\_toks $\times$ 6$P$ / peak\_FLOPS | 目标：$>40\%$ |

## GPU 硬件规格

| GPU | 显存 | 带宽 (HBM) | BF16 TFLOPS | NVLink | 备注 |
| --- | --- | --- | --- | --- | --- |
| A100-80GB | 80 GB HBM2e | 2.0 TB/s | 312 | 600 GB/s | 主力型号，广泛可用 |
| H100-80GB | 80 GB HBM3 | 3.35 TB/s | 989 | 900 GB/s | 当前一代，支持 FP8 |
| H200-141GB | 141 GB HBM3e | 4.8 TB/s | 989 | 900 GB/s | 大上下文 / 更少 GPU |
| B200 | 192 GB HBM3e | 8.0 TB/s | 2250 | 1800 GB/s | 下一代（2025） |

## 超参数范围

| 参数 | 典型范围 | 默认值 | 说明 |
| --- | --- | --- | --- |
| $\beta$ (DPO/KTO) | 0.05--0.5 | 0.1 | 越大越保守 |
| $\epsilon$ (PPO clip) | 0.1--0.3 | 0.2 | 越大更新越激进 |
| $\gamma$ (GAE 折扣) | 0.99--1.0 | 1.0 | 情景式任务使用 1.0 |
| $\lambda$ (GAE) | 0.9--0.99 | 0.95 | 越小偏差越大、方差越小 |
| KL 系数 ($\beta_\text{KL}$) | 0.01--0.2 | 0.05 | 自适应目标 KL $\approx$ 5--8 |
| 学习率 (RLHF) | 1e-7 -- 5e-6 | 5e-7 | 远低于预训练 |
| 学习率 (SFT) | 1e-5 -- 5e-5 | 2e-5 | 标准微调范围 |
| LoRA 秩 $r$ | 8--128 | 16--64 | 越大容量越大、显存越多 |
| LoRA alpha $\alpha$ | $r$ -- $2r$ | $2r$ | 缩放因子；$\alpha/r$ 为有效尺度 |
| 温度（生成） | 0.6--1.0 | 0.7 | 越低候选越同质 |
| 生成数量 $K$ | 4--64 | 4--16 | 用于 GRPO / Online DPO / Best-of-N |
| 梯度裁剪范数 | 0.5--2.0 | 1.0 | 防止梯度爆炸 |

## TRL API 速查

| Trainer | 方法 | 关键配置 | 数据格式 |
| --- | --- | --- | --- |
| `SFTTrainer` | 监督微调 | `packing, max_seq_length` | prompt + completion |
| `RewardTrainer` | 奖励模型 | `center_rewards_coefficient` | prompt + chosen + rejected |
| `PPOTrainer` | PPO | `init_kl_coef, target_kl, cliprange` | prompts（在线生成） |
| `DPOTrainer` | DPO / IPO | `beta, loss_type="sigmoid"/"ipo"` | prompt + chosen + rejected |
| `GRPOTrainer` | GRPO | `num_generations, beta, use_vllm` | prompts + reward_fn |
| `OnlineDPOTrainer` | Online DPO | `num_generations, reward_model_path` | prompts（在线生成） |
| `KTOTrainer` | KTO | `desirable_weight, undesirable_weight` | prompt + completion + label |
| `ORPOTrainer` | ORPO | `beta` | prompt + chosen + rejected |
| Best-of-N (manual) | Best-of-N | `n_samples` | prompts（推理） |

## RAG 流水线公式

$$
\begin{align}
\text{Cosine similarity:}&\quad \text{sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|} \\
\text{Retrieval:}&\quad \mathcal{D}_k = \text{top-}k_{d \in \mathcal{C}} \; \text{sim}(\text{embed}(q),\; \text{embed}(d)) \\
\text{RAG generation:}&\quad P(y\mid q) = P_\text{LLM}(y \;\mid\; q, \mathcal{D}_k) \\
\text{Chunking overlap:}&\quad \text{stride} = \text{chunk\_size} - \text{overlap} \\
\text{Reranker (cross-enc):}&\quad \text{score}(q, d) = \text{MLP}(\text{BERT}([q; d]))
\end{align}
$$

## 智能体设计模式

| 模式 | 结构 | 最适用于 |
| --- | --- | --- |
| ReAct | Think $\to$ Act $\to$ Observe $\to$ 循环 | 通用工具使用型 Agent |
| Plan-and-Execute | Plan $\to$ 执行步骤 $\to$ 修订 | 长程、结构化任务 |
| Supervisor | 路由器 $\to$ 专家 Agent | 多领域、子任务边界清晰 |
| Swarm (handoffs) | Agent 移交控制权 + 上下文 | 客服、问题升级流程 |
| Hierarchical | 委派 Agent 的树状结构 | 复杂任务分解 |
| Human-in-the-loop | Agent $\to$ 审批关卡 $\to$ 继续 | 高风险、不可逆动作 |

## 智能体通信协议

| 协议 | 范围 | 传输方式 | 核心概念 |
| --- | --- | --- | --- |
| MCP | 工具集成 | stdio / HTTP+SSE | 服务端暴露工具；客户端发现并调用 |
| A2A | 智能体到智能体 | HTTP + JSON-RPC | 带生命周期的任务（submitted$\to$working$\to$done） |
| OpenAI Function Calling | 工具使用 | API 负载 | 在 `tools[]` 数组中以 JSON schema 描述 |

## 上下文窗口预算

$$
C \geq \underbrace{S}_{\text{system}} + \underbrace{M}_{\text{memory/RAG}} + \underbrace{T}_{\text{tool defs}} + \underbrace{H}_{\text{history}} + \underbrace{R}_{\text{reserved output}}
$$

**经验法则**（针对 128K 上下文）：

- 系统提示：1--4K token（固定）
- 工具定义：2--8K（随工具数量增长）
- RAG 上下文：4--16K（top-$k$ 个 chunk）
- 历史记录：无界增长 $\rightarrow$ 汇总或截断
- 预留输出：2--8K

## 常见失效模式与修复

| 症状 | 可能原因 | 修复方法 |
| --- | --- | --- |
| 奖励上升、质量下降 | 奖励黑客（Reward Hacking） | RM 集成、长度惩罚、增大 $\beta$ |
| KL 爆炸（$>$15） | 学习率过高或模式崩溃 | 降低学习率、回滚 checkpoint |
| 熵塌缩 | 过早收敛 | 增大熵系数、提高温度 |
| 训练损失 NaN | 梯度爆炸 | 降低学习率、增大梯度裁剪、检查数据 |
| 5K 步后无提升 | Prompt 分布不当 | Goldilocks 过滤（通过率 20--80\%） |
| 基准性能回退 | 对齐税（Alignment Tax） | 减小 RL 预算、改用 LoRA、混入 SFT 数据 |
| 长度单调增长 | RM 中的长度漏洞 | 长度惩罚、用长度控制重训 RM |
| 生成时 OOM | KV cache 溢出 | 减小 batch、增大 TP、使用 PagedAttention |
| Agent 永久循环 | 缺少最大迭代守卫 | 设置 `max_iterations`、加循环检测 |
| 工具调用解析失败 | 输出格式不一致 | 少样本示例、约束解码 |
| RAG 返回无关文档 | embedding / chunking 不佳 | Reranker、混合检索、更小 chunk |
| 多智能体死锁 | 循环依赖 | DAG 约束、每个 Agent 设置超时 |

## 方法选择决策树

1. **是否有成对偏好数据（chosen + rejected）？**

   - 标签噪声大 $\rightarrow$ **IPO**
   - 显存受限、尚未做 SFT $\rightarrow$ **ORPO**
   - 数据干净、算力有限 $\rightarrow$ **DPO**
   - DPO 进入瓶颈、希望加入探索 $\rightarrow$ **Online DPO**

2. **只有二值反馈（点赞 / 点踩）？** $\rightarrow$ **KTO**
3. **有可验证奖励（数学 / 代码）？** $\rightarrow$ **GRPO**
4. **不惜代价追求最高质量？** $\rightarrow$ **PPO**
5. **想要免训练的改进？** $\rightarrow$ **Best-of-N**

## 评估指标

| 指标 | 范围 | 衡量内容 |
| --- | --- | --- |
| Perplexity | $[1, \infty)$ | 模型的“困惑度”；越低语言建模越好 |
| Win Rate（对比基线） | $[0, 1]$ | 评审/人类偏好的输出比例 |
| BLEU | $[0, 1]$ | 与参考文本的 $n$-gram 重叠（偏精确率） |
| ROUGE-L | $[0, 1]$ | 与参考文本的最长公共子序列 |
| Pass@$k$ | $[0, 1]$ | $k$ 个代码样本中至少 1 个通过测试的概率 |
| MMLU / GPQA | $[0, 1]$ | 知识/推理基准上的多选准确率 |
| HumanEval | $[0, 1]$ | 生成代码的功能正确率 |
| Faithfulness (RAG) | $[0, 1]$ | 受检索上下文支持的论断比例 |
| Context Relevancy | $[0, 1]$ | 检索内容中与查询相关的比例 |
| Answer Relevancy | $[0, 1]$ | 答案对问题的回应程度 |

## 推理与测试时扩展

| 方法 | 算力开销 | 机制 |
| --- | --- | --- |
| 思维链（Chain-of-Thought，CoT） | 1.5--3$\times$ token | 在 prompt 中加入 “step by step” |
| 自洽性（Self-Consistency） | $N \times$ 生成 | 采样 $N$ 条 CoT 路径，对最终答案多数表决 |
| 思维树（Tree-of-Thought, ToT） | $B \times D \times$ 生成 | 在推理树上 BFS/DFS；对分支评估 |
| Best-of-$N$ | $N \times$ 生成 | 采样 $N$，用 RM 打分，取最高 |
| Beam search（推理上） | $B \times$ 生成 | 维持 top-$B$ 个部分推理链 |
| 预算强制（Budget forcing） | 可变 | 动态为更难问题分配更多 token |
| 验证（ORM/PRM） | $N \times$ 生成 + 打分 | 生成 $N$ 个解，按结果/过程 RM 排序 |

## 记忆系统类型

| 类型 | 存储 | 使用场景 |
| --- | --- | --- |
| 工作记忆 | 上下文窗口 | 当前对话、即时工具结果 |
| 情景记忆（Episodic Memory） | 向量存储 | 历史交互、用户偏好、会话历史 |
| 语义记忆（Semantic Memory） | 知识图谱 / embedding | 事实、概念、领域知识 |
| 程序性记忆（Procedural Memory） | 技能库 / 代码 | 操作手册、学到的工作流 |

## MCP 速查

| 原语 | 方向 | 有副作用？ | 用途 |
| --- | --- | --- | --- |
| Tools | 客户端 $\to$ 服务端 | 是 | 执行动作（创建、修改、删除） |
| Resources | 客户端 $\to$ 服务端 | 否（只读） | 读取数据（文件、DB 记录、配置） |
| Prompts | 客户端 $\to$ 服务端 | 否 | 常用任务的可复用模板 |
| Sampling | 服务端 $\to$ 客户端 | 否 | 服务端请求客户端调用 LLM 生成 |

**传输方式**：`stdio`（本地子进程）或 `HTTP+SSE`（远程、可流式）。

**发现机制**：客户端在连接初始化时调用 `tools/list`、`resources/list`、`prompts/list`。

**工具注解**：`readOnlyHint`、`destructiveHint`、`idempotentHint`、`openWorldHint`。

## A2A 协议速查

| 概念 | 描述 |
| --- | --- |
| Agent Card | 位于 `/.well-known/agent.json` 的 JSON --- 名称、技能、支持的内容类型 |
| Task | 工作单元：`id`、`status`、`artifacts`。生命周期：submitted $\to$ working $\to$ completed/failed |
| Message | 任务内的通信单元（角色：user/agent；parts：text/file/data） |
| Artifact | Agent 产出的输出（结构化数据、文件、生成内容） |
| Push Notifications | 长任务的 webhook 更新（通过 `tasks/pushNotification/set`） |

**关键端点**：`tasks/send`（创建/更新）、`tasks/get`（轮询状态）、`tasks/sendSubscribe`（SSE 流）。

## 智能体框架对比

| 框架 | 编排方式 | 多 Agent | 最适用于 |
| --- | --- | --- | --- |
| LangGraph | 显式状态图 | 条件路由 | 生产环境：持久化、HITL、精细控制 |
| OpenAI Agents SDK | 声明式 handoff | 基于 handoff | 简单易用：护栏、追踪、快速上手 |
| AutoGen (AG2) | 对话驱动 | GroupChat | 原型开发：代码执行、研究场景 |
| CrewAI | 基于角色的团队 | 顺序 / 并行 | 低代码：快速演示、简单流水线 |
| Google ADK | 会话 + 事件 | 原生 A2A | 企业级：artifact 管理、多模态 |

## 智能体 RL 公式

$$
\begin{align}
\text{Trajectory GRPO:}&\quad \hat{A}_i = (R(\tau_i) - \mu_G)/\sigma_G, \quad R(\tau_i) = \sum_{t} r_t^{(\tau_i)} \\
\text{Agent reward:}&\quad R = w_1 R_\text{task} + w_2 R_\text{efficiency} + w_3 R_\text{safety}, \quad R_\text{eff} = \max(0, 1 - \text{steps}/N_\text{max}) \\
\text{Masking:}&\quad \mathcal{L} = \sum_{t \in \text{agent tokens}} \min(r_t \hat{A}_t,\; \text{clip}(r_t) \hat{A}_t) \quad \text{(mask env outputs)} \\
\text{Pass@}k:&\quad 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}, \quad n = \text{total samples},\; c = \text{correct}
\end{align}
$$

## 智能体安全检查清单

| 威胁 | 层级 | 缓解措施 |
| --- | --- | --- |
| Prompt 注入（直接） | 输入 | 输入校验、指令层级、分隔符 |
| Prompt 注入（间接） | 工具输出 | 将工具输出视为不可信；勿遵循检索文档中的指令 |
| 工具滥用 | 执行 | 最小权限；`destructiveHint` 关卡；沙箱化 |
| 数据外泄 | 输出 | 输出过滤；限制工具仅访问白名单域 |
| 过度自主 | 架构 | 最大迭代次数；成本预算；人工审批关卡 |
| 混淆代理（Confused Deputy） | 多 Agent | 校验任务来源；基于能力的访问控制 |

## 智能体评估指标

| 指标 | 公式 / 定义 | 目标 |
| --- | --- | --- |
| 任务成功率（TSR） | 正确完成数 / 总任务数 | $>85\%$（生产） |
| 完成步数 | 单次成功任务的平均动作数 | 越低越高效 |
| 单任务成本 | 总 token 数 $\times$ 单价 | 视预算而定 |
| 延迟（TTFC） | 从请求到首条有效输出 | 交互场景 $<5$ 秒 |
| 工具调用准确率 | 正确工具选择 / 总调用 | $>90\%$ |
| 恢复率 | 重试成功 / 初次失败 | $>60\%$ |
| 人工升级率 | 需要人工的任务 / 总任务 | $<15\%$ |

## 关键智能体基准

| 基准 | 领域 | 指标 | SOTA (2025) |
| --- | --- | --- | --- |
| SWE-bench Verified | 软件工程 | 解决 issue 百分比 | $\sim$70\% |
| WebArena | Web 浏览 | 任务成功率 | $\sim$40\% |
| OSWorld | 桌面计算机使用 | 任务成功率 | $\sim$25\% |
| GAIA | 通用 AI 助手 | 完全匹配准确率 | $\sim$75\%（L1） |
| Tau-bench | 工具使用可靠性 | 通过率（5 次试验） | $\sim$65\% |
| HumanEval / MBPP | 代码生成 | Pass@1 | $>95\%$ |
