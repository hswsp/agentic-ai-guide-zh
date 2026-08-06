---
layout: home
title: 测验题与详细解答
permalink: /part6/ch27-quiz.html
---

# 测验题与详细解答

本章提供一整套题目，用于检验和巩固你对本书所讲内容的理解。每道题目都瞄准一个关键概念、算法或系统设计决策——正是这种知识把表面熟悉与真正的专长区分开来。请将这些题目用于自我检验：先尝试自己作答，再阅读详细解答。题目从基础概念（LLM 架构、强化学习基础）逐步推进到核心算法（PPO、DPO、GRPO），最终覆盖高级系统设计与智能体 AI 主题。

## 基础题

### Q0a：在 decoder-only Transformer 中，attention 机制的作用是什么？为什么它是因果的？

**答**：attention 机制允许每个 Token 关注（即对其表示做加权组合）其他 Token 的表示。在 decoder-only Transformer 中，attention 是**因果的**（也称为*自回归*）：Token $t$ 只能关注 Token $1, \ldots, t$，绝不能关注未来 Token $t+1, \ldots, T$。

**为什么是因果的？**因为模型从左到右生成文本。推理时，未来 Token 字面意义上还不存在。训练时的因果掩码模拟了这一约束，使模型学会仅用左侧上下文预测每个 Token。在数学上，attention 矩阵被掩盖：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

其中对于 $j > i$（未来位置）有 $M_{ij} = -\infty$，强制将这些 attention 权重置零。

**实践含义**：这使得推理时的 KV-cache 优化成为可能——由于过去 Token 的 keys 和 values 永远不变，可被缓存复用，将每个新 Token 的生成代价从 $O(T^2)$ 降至 $O(T)$。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q0b：解释 FlashAttention。它解决什么问题，又是怎么解决的？

**答**：标准 attention 会计算完整的 $T \times T$ attention 矩阵，需要 $O(T^2)$ 显存，并且是**受显存带宽约束的**——GPU 大部分时间都花在 HBM（慢、大）与 SRAM（快、小）之间搬运数据，而不是做真正的计算。

**FlashAttention 的洞见**：永远不要在 HBM 中物化完整的 attention 矩阵。相反，把计算切分成可放进 SRAM 的分块（tile），用在线 Softmax（Online Softmax）算法*逐块*计算 attention，并只把最终输出写入 HBM。

**关键技术**：

1. **分块（Tiling）**：将 Q、K、V 拆成大小为 $B_r \times B_c$、可装入 SRAM 的块。
2. **在线 Softmax（Online Softmax）**：维护一个滑动的 max 和 sum，以增量方式计算 softmax，无需访问完整一行。
3. **重算（Recomputation）**：在反向传播时，从 Q、K、V 重新计算 attention（开销小），而不存储 $T \times T$ 矩阵（开销大）。

**结果**：HBM 显存复杂度为 $O(T)$（而非 $O(T^2)$），墙钟（wall-clock）时间加速 2--4$\times$，数值输出完全一致（并非近似）。

*复习：第 1--2 章（LLM 架构；系统基础）。*

### Q0c：从高层次看，SFT、RLHF 与 DPO 有什么区别？分别在什么场景下使用？

**答**：

- **SFT（监督微调，Supervised Fine-Tuning）**：训练模型去模仿高质量示范。Loss：在精选数据上做下一个 Token 预测。教会模型*格式*与*风格*。
- **RLHF**：从人类偏好训练一个 reward 模型，然后用 RL（PPO）针对它优化 policy。模型会探索示范数据之外的空间。教会模型*人类偏好什么*。
- **DPO**：跳过 reward 模型。直接在偏好对 $(y_w, y_l)$ 上用对比 loss 优化 policy。目标与 RLHF 相同，但流水线更简单。

**典型流水线**：先做 SFT（提供良好起点），再做 RLHF 或 DPO（细化偏好）。仅做 SFT 倾向产出冗长、过度模棱两可的回答。RLHF/DPO 让输出更直接、更贴合人类意图。

**各自适用场景**：有黄金标准输出 $\rightarrow$ SFT。有偏好对但算力有限 $\rightarrow$ DPO。需要极致质量且能承担基础设施成本 $\rightarrow$ RLHF（PPO）。

*复习：第 5、6 与 10 章（PPO；DPO；SFT 最佳实践）。*

### Q0d：什么是 reward 模型？它如何训练，可能出什么问题？

**答**：reward 模型（Reward Model, RM）是一个神经网络，输入 (prompt, response) 对，输出一个表示质量的标量分数。它在人类偏好数据上训练：给定 $(y_w, y_l)$ 对（其中 $y_w$ 被偏好），RM 学习赋予 $R(y_w) > R(y_l)$。

**训练**：Bradley-Terry loss：$\mathcal{L} = -\log\sigma(R(y_w) - R(y_l))$。架构：通常与 policy 共用同一个 Transformer，将 LM head 替换为标量投影。

**可能出什么问题**：

1. **Reward hacking**：policy 找到了在 RM 上得分很高但实际质量很差的输出（例如过度冗长、重复，或包含 RM 存在偏好的特定短语）。
2. **分布漂移（Distribution Shift）**：RM 是在更早的 policy 输出上训练的。随着训练推进，当前 policy 生成的是 RM 无法准确打分的分布外输出。
3. **标签噪声**：人类标注者意见不一致、疲劳，或使用不一致的标准。这种噪声会传播到 RM 的预测中。
4. **过度自信**：RM 对从未见过的输出赋予极端分数，提供具有误导性的梯度信号。

*复习：第 9 章（Reward 模型训练）。*

### Q0e：解释 RL 中的探索-利用（exploration-exploitation）权衡。它在 LLM 训练中如何体现？

**答**：在 RL 中，Agent 必须在以下两者间取得平衡：

- **利用（Exploitation）**：选择已知能产生高 reward 的动作（贪婪行为）。
- **探索（Exploration）**：尝试可能带来更高 reward（但也可能失败）的新动作。

**在 LLM 训练中**：policy 就是语言模型。“动作”是 Token 的选择。“利用”指生成与已得高分相似的回答；“探索”指尝试新颖的措辞、结构或推理路径。

**表现形式**：

- **生成时的温度**：温度越高 = 探索越多。GRPO 使用温度 1.0 以在每个 group 内获得多样的样本。
- **KL 惩罚**：作为一种反探索的刹车——防止 policy 偏离参考模型过远。否则 policy 可能会塌缩为单一的高 reward 模板（mode collapse）。
- **GRPO 中的 group 采样**：每个 prompt 生成 $G$ 个回答以显式探索输出空间，然后强化高于平均水平的回答。

**张力**：探索过少 $\rightarrow$ 模型陷入局部最优（永远给同一个安全答案）。探索过多 $\rightarrow$ 训练不稳定，质量剧烈波动。

*复习：第 3 与 7 章（RL 简介；GRPO）。*

## 核心算法题

### Q1：解释 PPO 的裁剪目标。为什么它比朴素 policy gradient 更有效？

**答**：朴素 policy gradient：$\nabla J = \mathbb{E}[\nabla\log\pi(a|s) \cdot \hat{A}]$。问题：一个幸运/不幸的样本就能产生巨大的梯度 $\rightarrow$ policy 跳到糟糕区域 $\rightarrow$ 生成乱码 $\rightarrow$ 下一个梯度让情况更糟 $\rightarrow$ 进入无法挽回的“死亡螺旋”。

**PPO 的方案**：将概率比 $r = \pi_\text{new}/\pi_\text{old}$ 裁剪到 $[0.8, 1.2]$。

**机制**：对好的动作（$\hat{A}>0$）：目标为 $\min(r\hat{A}, 1.2\hat{A})$。一旦 $r$ 超过 1.2，便不再有额外收益——防止 policy 在单个样本上过度承诺。对坏的动作（$\hat{A}<0$）：目标为 $\min(r\hat{A}, 0.8\hat{A})$。一旦 $r$ 降至 0.8 以下，惩罚便不再增长——防止灾难性遗忘（Catastrophic Forgetting）。

**关键洞见**：它是对 TRPO 的 KL 约束的一阶近似，但无需昂贵的二阶优化。每次更新对 policy 的改变最多 $\pm$20\%。

**对 LLM 而言**：Token 级比率 $r_t = \pi_\theta(y_t|y_{<t})/\pi_\text{old}(y_t|y_{<t})$ 防止任何单个 Token 的概率发生过大变化，从而维持生成的连贯性。

*复习：第 5 章（PPO）。*

### Q2：从第一原理推导 DPO。它做了哪些假设？

**答**：从 RLHF 目标出发：$\max_\pi \mathbb{E}[r(x,y)] - \beta D_\text{KL}[\pi\|\pi_\text{ref}]$。

**步骤 1**：写出 KKT 条件。最优 policy 有闭式解：$\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$。

**步骤 2**：反解出 reward 的表达：$r(x,y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$。

**步骤 3**：代入 Bradley-Terry 模型 $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$。配分函数 $Z(x)$ 抵消（同一 prompt）。

**步骤 4**：将 $\pi^*$ 替换为 $\pi_\theta$（我们要训练的参数化 policy）：$\mathcal{L} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})]$。

**假设**：

1. Bradley-Terry 偏好模型（成对、无平局、可传递）。
2. 最优 policy 可被 $\pi_\theta$ 实现（容量足够）。
3. 偏好与训练数据来自同一分布（无分布漂移）。
4. 参考模型是固定且合理的。

**当假设失效时**：真实偏好并不可传递，数据在训练中漂移，标签有噪声 $\rightarrow$ 这就是 Online DPO 和 IPO 存在的原因。

*复习：第 6 章（DPO）。*

### Q3：GRPO vs PPO——你会在什么场景下选择哪一个？权衡何在？

**答**：

**GRPO 的优势**：

- 无需 value function：节省了一个模型规模的显存与复杂度。
- 更简单：超参更少，更直观（高于均值 = 好，低于均值 = 坏）。
- 更适合可验证奖励：数学/代码中 $r \in \{0, 1\}$ 给出清晰信号。
- DeepSeek-R1 证明，仅用二元 reward 即可教出涌现的推理能力。

**PPO 的优势**：

- 逐 Token 信用分配：value function 为每个 Token 分配 reward，而非仅在序列级。
- 样本效率更高：GAE 用 value 预测估计 advantage，无需生成 $G$ 个样本。
- 更适合细腻的 reward：当 reward 是连续的、在 Token 间显著变化时。
- 更成熟：在 OpenAI、Anthropic 等公司经过实战检验。

**经验法则**：若 reward 可验证（对/错） $\rightarrow$ GRPO。若 reward 细腻（RM 分数）且追求极致质量 $\rightarrow$ PPO。若算力有限 $\rightarrow$ GRPO（无需训练 critic）。

**算力对比**：GRPO 为每个 prompt 生成 $G$ 个回答（生成量增加 8$\times$），但跳过 value function 训练。净算力相近，但分布不同（生成更多，训练更少）。

*复习：第 5 与 7 章（PPO；GRPO）。*

### Q4：GAE 如何工作？为 LLM 走一个具体例子。

**答**：GAE = $n$-步 TD 误差的加权和：$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$。

**具体例子**：回答有 5 个 Token。仅在末端有 reward（$r_5 = 0.8$）。Value 预测：$V_1=0.5, V_2=0.55, V_3=0.6, V_4=0.65, V_5=0.7$。

TD 误差（$\gamma=1$）：$\delta_1 = 0 + V_2 - V_1 = 0.05$，$\delta_2 = 0 + V_3 - V_2 = 0.05$，$\ldots$，$\delta_5 = 0.8 + 0 - 0.7 = 0.1$。

取 $\lambda = 0.95$：$\hat{A}_5 = 0.1$（仅是最后的 TD 误差），$\hat{A}_4 = 0.05 + 0.95 \times 0.1 = 0.145$，$\hat{A}_3 = 0.05 + 0.95 \times 0.145 = 0.188$，依此类推。

**解读**：Token 3 获得 advantage 0.188，因为它对一个得到高于预期 reward 的序列做出了贡献。更早的 Token 通过指数衰减获得信用。

**对 LLM**：$\gamma=1.0$（所有 Token 都重要，且为有限视野）。Token $t$ 的 advantage 回答：“鉴于此 Token 之后发生的事，这个 Token 的选择比预期更好还是更差？”

*复习：第 5 章（PPO）。*

### Q5：如何防止 reward hacking？给出一种分层防御策略。

**答**：**检测信号**：RM 分数上升但胜率持平或下降；回答长度单调增长；KL 散度 $>$ 15；多样性（unique n-grams）下降；阅读高 reward 输出可发现作弊模式。

**分层防御（按优先级排序）**：

1. **KL 惩罚**（首要）：自适应控制器以 KL $\approx$ 6 为目标。若 KL 上升，$\beta$ 自动增大。防止过度偏离参考模型。
2. **Reward 模型集成**（3--5 个模型）：取分数的 min 或 mean。各模型有不同的盲区——能骗过一个的漏洞很少能骗过所有。
3. **长度惩罚**：$r' = r - c \cdot \max(0, \text{length} - L_\text{target})$。防止“只要更长就得高分”的作弊。
4. **周期性 RM 刷新**：每 2000 步，从当前 policy 生成数据、重新标注，并加入 RM 训练集。模型一找到作弊路径就立即堵上。
5. **基于胜率的停训**：跟踪相对 SFT baseline 的胜率。如果 RM 分数上升但胜率停滞 200+ 步，立即停训。模型在作弊，而非进步。

**检测后恢复**：回滚到最近一个“干净”的 checkpoint。将 $\beta$ 提升 2$\times$。把发现的作弊模式加入 RM 的负样本。

*复习：第 9 与 11 章（Reward 模型训练；系统架构）。*

## 系统设计题

### Q6：为训练 70B 模型设计一个 RLHF 系统。逐组件讲解。

**答**：基于 72 张 A100-80GB GPU 的三集群解耦架构：

**集群 1 ——生成（32 GPU）**：

- 8 个 vLLM 实例，每个 TP=4。PagedAttention + 投机解码（speculative decoding，1B 草稿模型）。
- 连续批处理，最多 256 条在飞序列。INT8 权重以省带宽。
- 输出：(prompt, response, 逐 Token log-probs)。吞吐量：$\sim$500 回答/分钟。
- 无状态：仅从共享存储加载最新权重。恢复极简。

**集群 2 ——打分（8 GPU）**：

- Reward 模型（70B，INT8 = 70GB），4 GPU（TP=4）。
- 参考模型（70B，INT8），4 GPU（TP=4）。计算逐 Token 的 log-probs 以算 KL。
- 输出：reward 分数 + 逐 Token 的 KL。轻量级，批量推理。

**集群 3 ——训练（32 GPU）**：

- Policy 模型用 FSDP（ZeRO-3）。FlashAttention 2。梯度检查点。
- 从 buffer 消费已打分的 experience。PPO 更新：在 mini-batch=16 上跑 4 个 epoch。
- 每 50 步将更新后的权重推送到共享存储（异步、后台传输）。
- 每 100 步做一次异步 checkpoint（非阻塞写入 NVMe + S3 备份）。

**连接结构**：

- 生成 $\rightarrow$ 打分：Ray/Redis 队列（每 batch $\sim$10 MB：Token ID + log-probs）。
- 打分 $\rightarrow$ 训练：经验缓冲（循环式，保存最近 500 步）。
- 训练 $\rightarrow$ 生成：共享并行文件系统（Lustre/GPFS）上的权重存储。每 50 步异步推送 140GB。

**重叠**：当训练处理第 $N$ 步时，生成已经在为第 $N+1$ 步生产数据。这隐藏了生成延迟，相对单体架构吞吐提升 1.3--1.5$\times$。

*复习：第 11 章（系统架构与大规模基础设施）。*

### Q7：如何处理生成瓶颈？量化各种方案的收益。

**答**：生成占 RLHF 总墙钟时间的 60--70\%。根本原因：自回归解码受显存带宽约束（算术强度 $\approx 1$ FLOP/byte，而 A100 的 roofline 是 156 FLOP/byte）。

**按影响力排序的方案**：

**1. 解耦生成与训练**（端到端 1.3--1.5$\times$）：在独立硬件上运行生成，与训练重叠。最大的单一架构性胜利。

**2. vLLM + PagedAttention**（2--4$\times$）：消除内部碎片造成的 60--80\% KV cache 显存浪费。可承载 3--4$\times$ 更大的 batch = 更好的带宽利用。

**3. 连续批处理**（1.5--2$\times$）：不等最长序列结束。立即在腾出的槽位开新序列。保持 GPU 繁忙。

**4. 投机解码（Speculative Decoding）**（2--3$\times$）：1B 草稿模型提议 5 个 Token。70B 模型在一次前向中并行验证全部 5 个！平均接受 3--4 个 $\rightarrow$ 每次前向产出 3--4 个 Token 而非 1 个。

**5. 生成权重用 INT8/FP8**（2$\times$）：将每 Token 的 140GB 权重读取减半。质量损失极小，因为：(a) 我们本就在用温度采样，(b) 仅生成使用 INT8，训练仍是 BF16。

**6. CUDA graphs + kernel fusion**（1.1--1.3$\times$）：消除 Python/CUDA 启动开销。把 layernorm+attention+MLP 融合成更少的 kernel。

**组合后**：$1.5 \times 3 \times 1.5 \times 2.5 \times 2 \times 1.2 = 40\times$ 优于朴素实现。实际中由于边际收益递减，整体大约为朴素实现的 10--20$\times$。

*复习：第 2 与 11 章（系统基础；系统架构）。*

### Q8：解释解耦系统中的权重同步。能容忍多少陈旧度（staleness）？

**答**：

**问题**：生成集群使用 policy 权重生成回答，训练集群更新这些权重。它们位于不同硬件上。如何保持同步？

**为什么完美同步是浪费**：70B BF16 全量同步 = 140GB。InfiniBand 400Gb/s（50GB/s）下，每次同步 2.8s。若每步（每 50--90s）都同步，权重传输占总时间的 3--5\%。可接受，但没必要。

**陈旧度容忍分析**：

- 每步 policy 变化：$\sim$0.1\%（按参数均值差衡量）。
- 50 步：$\sim$5\% 累计漂移。
- PPO 裁剪范围：可处理高达 20\% 的概率比偏离。
- 经验：50 步陈旧度 $\rightarrow$ 质量下降 $<$2\%（按胜率衡量）。

**生产策略**：

1. 每 50 训练步：向共享存储推送一次 BF16 完整 checkpoint（传输 2.8s）。
2. 生成集群：在 batch 之间做非阻塞的权重重载。
3. 增量压缩（可选）：仅发送变更参数（INT8 delta $\approx$ 5GB），作为偏移应用。带宽减少 10$\times$。
4. 超大规模（256+ GPU）：流式同步——后台连续发送小块。平均陈旧度：5--10 步。

**关键细节**：生成阶段计算的 log-probs 使用的是陈旧权重。PPO 比率 $\pi_\text{new}/\pi_\text{old}$ 用这些陈旧 log-probs 作为 $\pi_\text{old}$。这没问题，因为 PPO 本就被设计为处理 off-policy 修正。

*复习：第 11 章（系统架构与大规模基础设施）。*

### Q9：在 512 GPU 规模下如何做到容错？

**答**：在 512 GPU 规模下，MTBF 为 4--8 小时。5 天的训练会经历 15--30 次故障。

**架构级韧性**：

- 生成集群 = 无状态。失败实例在 $<$60s 内重启（只需加载权重，无状态）。
- 训练集群 = 有状态。需要基于 checkpoint 的恢复。
- 打分集群 = 无状态。与生成相同。

**Checkpoint 策略**：

- 频率：每 50--100 步（5--10 分钟训练）。
- 方法：异步（非阻塞）。后台线程在下一步进行时写入。使用 FSDP 的分布式保存（每个 rank 并行保存自己的分片）。
- 内容：模型权重、优化器状态（Adam m/v）、LR scheduler、RNG 状态、KL 自适应系数、全局 step 计数器、回放缓冲指针。
- 存储：本地 NVMe（快，70B 仅 30s）+ 异步拷贝到 S3 / 共享 FS（持久化）。
- 保留：保留最近 3 个 checkpoint。自动删除更早的。

**检测与恢复流程**：

1. NCCL 集合通信超时（60s）或心跳丢失（10s）$\rightarrow$ 检测到故障。
2. 通过 NVML 健康检查定位故障节点。
3. 方案 A（快，$<$2 min）：Torch Elastic 缩减 world size、重分配分片，以 $N-1$ 节点继续。后台请求替换。
4. 方案 B（干净，$\sim$5 min）：拉起替换节点、重建进程组、加载最近 checkpoint、恢复。
5. 经验缓冲已持久化——无需重新生成。

**预防**：上线前的压力测试（GEMM、显存、NVLink）。ECC 错误监控（错误激增时抢占式迁移）。热备节点（环境预先加载）。双轨 InfiniBand 提供网络冗余。

*复习：第 11 章（系统架构与大规模基础设施）。*

### Q10：如何从 7B 扩展到 70B 再到 405B？每个量级要变什么？

**答**：

**7B（单 8-GPU 节点，数小时）**：

- 架构：单体（TRL 默认）。所有模型在同一组 GPU 上。
- 显存：LoRA + INT8 的 ref/RM 在 8$\times$80GB 内轻松容纳。
- 并行：节点内 DP=8 或 FSDP。无网络通信。
- 超参：LR=$5\times10^{-6}$，激进的 $\beta$=0.02，50K--100K 步。
- 时间：每次运行 4--12 小时。快速迭代。

**70B（32--64 GPU，2--5 天）**：

- 架构：半解耦。vLLM 生成 + FSDP 训练。
- 显存：ZeRO-3 必不可少。梯度检查点。INT8 ref/RM。
- 并行：节点内 TP=8（生成），节点间 FSDP（训练）。
- 超参：LR=$1.5\times10^{-6}$，温和的 $\beta$=0.05，10K--30K 步。
- 容错：异步 checkpoint、监控，但仍可人工管理。

**405B（256--512 GPU，1--3 周）**：

- 架构：完全解耦。独立集群。权重存储 + 队列。
- 显存：训练用 ZeRO-3 + TP=8 + PP=2。生成用 INT4。
- 并行：3D 并行（TP$\times$PP$\times$DP = 8$\times$2$\times$16 = 256 GPU 训练）。
- 超参：LR=$5\times10^{-7}$，非常保守的 $\beta$=0.1，2K--5K 步。
- 容错：必备。弹性训练、冗余 checkpoint、热备节点。
- 关键变化：所需 RL 训练量大幅减少（模型预训练后已非常强）。但每步贵 50$\times$，因此不稳定性是灾难性的。

**悖论**：更大的模型实际上*逐步 RL 训练更容易*（更稳定，loss 地形更平滑）。但不稳定性的代价随模型规模放大——405B 上一次失败的训练浪费 \$100K+ 的算力。

*复习：第 11 章（系统架构与大规模基础设施）。*

## 实战与调试题

### Q11：reward 分数上升但模型质量下降。诊断并修复。

**答**：典型的**reward hacking / 古德哈特定律（Goodhart's Law）**。

**诊断流程**：

1. **检查回答长度**：绘制训练过程中的平均长度。单调增长？= 长度作弊（RM 给更长的回答更高分）。
2. **检查 KL 散度**：$>$15？= policy 偏离参考模型过远，丧失了能力。
3. **检查多样性**：每条回答的 unique trigrams。下降？= mode collapse（重复同一高 reward 模式）。
4. **人工检查**：阅读 20 个最高 reward 的回答。它们共享什么模式？（例如：都以“Great question!”开头、都用项目符号、过度模棱两可）。
5. **胜率**：在留出 prompt 上与 SFT baseline 比较。若 RM 上升而胜率持平/下降 = 确认存在作弊。

**即时修复**：

- 回滚到胜率还在提升的最近 checkpoint。
- 将 $\beta$ 提升 2--3$\times$（更强的 KL 惩罚）。
- 添加显式长度惩罚：$r' = r - 0.001 \cdot \max(0, \text{len} - 500)$。

**结构性修复（防止复发）**：

- RM 集成：在不同数据切分上训练 3--5 个 RM。取 min 或 mean。作弊往往针对特定模型。
- RM 刷新：每 2000 步，从当前 policy 生成、获取人工标签、重训 RM。
- 多目标 reward：用独立 RM 组合“有用性 + 无害性 + 简洁性”。
- 基于胜率（而非 RM 分数）早停。你优化的指标应与训练信号不同。

*复习：第 9 与 11 章（Reward 模型训练；系统架构）。*

### Q12：如何决定 RL 训练的 prompt 分布？

**答**：prompt 质量是 RLHF 中最被低估的因素。差的 prompt = 没有学习信号。

**组成（我默认的配比）**：

- 40\% 真实用户流量（代表实际用例）。
- 30\% 合成（LLM 生成，填补覆盖空白——稀有主题、边角情况）。
- 20\% 课程式（渐进难度——先易，随模型变强提升复杂度）。
- 10\% 对抗式（red-team prompt、越狱尝试、含糊指令）。

**关键：Goldilocks 过滤器**：

1. 对每个候选 prompt，用当前模型生成 4--8 个回答。
2. 用 RM 打分。计算通过率（超过阈值的比例）。
3. 仅保留通过率在 20--80\% 的 prompt：

   - $<$20\%：太难。模型几乎总失败 $\rightarrow$ 全是负 advantage $\rightarrow$ 无有效梯度。
   - $>$80\%：太易。模型几乎总成功 $\rightarrow$ 全是正 advantage $\rightarrow$ 无对比。
   - 20--80\%：完美。成功与失败兼有 $\rightarrow$ 关于“什么有效”的清晰信号。

4. 每 500 训练步重新过滤（模型在进步，难度分布在漂移）。

**主题多样性**：确保没有单一主题主导（每类 $<$10\%）。用 Embedding 聚类验证覆盖度。否则模型会过度优化主导主题。

*复习：第 7 与 9 章（GRPO；Reward 模型训练）。*

### Q13：RLHF 中 LoRA vs 全量微调。各自何时使用？

**答**：

**LoRA**（Low-Rank Adaptation，$r$=64，$\alpha$=16）：

- 可训练参数：模型的 $\sim$0.2\%（70B 对应 200M）。
- 显存节省：无需独立参考模型（base 模型 = 参考）！节省 140GB。
- 稳定性：天然更稳定（低秩约束限制了 policy 的漂移幅度）。
- 速度：每步更快（更新参数更少），但可能需要更多步。
- 质量天花板：通常达到全量微调的 90--95\%。

**全量微调**：

- 所有参数都更新。表达力最强。
- 需要独立的参考模型副本（70B 为 140GB）。或者通过非常频繁的 checkpoint 作为“锚”。
- 灾难性遗忘的风险更高。需要更低的 LR（比 LoRA 低 $3\times$）和更强的 $\beta$。
- 更适合：需要大幅分布漂移（新语言、迥异风格），LoRA 触及容量上限的情况。

**我的决策框架**：

1. 先用 LoRA（$r$=64）。便宜 3$\times$ 且更稳定。
2. 监控 LoRA 矩阵的梯度范数。如果持续 $>$1.0（相对于参数量很大）：LoRA 已到容量上限。
3. 仅当胜率停滞 *且* 梯度分析指示容量受限时，才切换到全量微调。
4. 对全量微调：使用 $\text{LR}/3$、$\beta \times 2$、更频繁的 checkpoint，以及基于胜率的早停。

**混合**：LoRA 用于对齐/安全（小行为漂移）+ 全量微调用于能力/推理（需要大漂移）。

*复习：第 1 与 10 章（LLM 架构；SFT 最佳实践）。*

### Q14：过程奖励模型（Process Reward Model, PRM）vs 结果奖励模型（Outcome Reward Model, ORM）。设计一个 PRM 系统。

**答**：

**ORM**：只给最终答案打分。“整体回答好不好？”简单但无法定位推理*在何处*出错。

**PRM**：为每个中间步骤打分。“这个推导的第 3 步对吗？”信息量大得多，但更难训练。

**PRM 对推理任务的优势**：

- 准确定位推理失败的位置（步骤级信用分配）。
- 支持树搜索：只展开步骤分数高的分支。
- reward hacking 更难：错误步骤 + 侥幸正确答案无法拿到高分。
- 在 MATH 基准上，PRM + Best-of-N 比 ORM + Best-of-N 高 10--20\%。

**训练 PRM**：

1. **数据收集**（蒙特卡洛方法）：

   - 对每道题，逐步生成推理轨迹。
   - 在每一步 $k$，从该步出发把解完整补全 $M$ 次（$M=32$）。
   - 步骤分数 = 抵达正确答案的补全比例。
   - 分数明显下降的步骤 = “出错”的步骤。

2. **标注**：补全率 $>$ 50\% 的步骤标为“正确”，$<$ 20\% 的标为“错误”。
3. **模型**：与 base 模型同架构 + 每个 Token 位置上的分类头。用步骤标签做二元交叉熵训练。
4. **推理**：为每步打分。若任一步骤分数 $<$0.3，则将该轨迹标记为有缺陷。

**在 RLHF 中使用 PRM**：PRM 给出的逐 Token reward 直接喂入 GAE。每个 Token 都获得即时反馈，而非仅在序列末端。这极大改善了长推理链上的信用分配。

*复习：第 9 与 13 章（Reward 模型训练；大型推理模型的 RL）。*

### Q15：如何评估 RL 是否真的改进了模型？

**答**：多面向评估（单一指标无法捕捉“质量”）：

**1. 胜率**（最重要、最可靠）：

- 500+ 多样 prompt。用 LLM judge（GPT-4 或 Claude）在与 SFT baseline 的盲对比 A/B 中选胜者。
- 目标：$>$55\% 胜率 = 有意义的改进；$>$65\% = 强改进。
- 使用位置去偏（互换 A/B 顺序、取平均）。报告置信区间。

**2. 能力基准**（回退检测）：

- MMLU（知识）、HumanEval（代码）、MATH（推理）、MT-Bench（多轮）。
- 任意 $>$2\% 的下降 = 值得警惕的对齐税。深入排查哪些类别退化。

**3. 类别专项评测**：

- 安全：对有害 prompt 的拒绝率（应上升）。
- 真实性：TruthfulQA 分数（应上升或持平）。
- 有用性：指令跟随基准上的任务完成率。

**4. 分布性指标**：

- 回答长度分布（不应发生剧烈漂移）。
- 词汇多样性（每条回答的 unique tokens）。
- 格式合规（若训练目标包含特定格式）。

**5. 人工评测**（金标准，昂贵）：

- 每个样本由 3+ 名熟练标注者做盲 A/B。标注者间一致性 $>$ 70\%。
- 仅用于最终模型选择，训练过程中不用（太慢/太贵）。

**红旗信号**：RM 分数上升 + 胜率持平 = reward hacking，不是真改进。胜率上升 + 基准下降 = 对齐税过高，需降低 RL 强度。

*复习：第 14 章（LLM 评估）。*

### Q16：端到端描述 reward 模型训练流水线。

**答**：

**阶段 1——数据生成**：

- 收集 50K--100K 多样 prompt（真实流量 + 合成）。
- 对每个 prompt 用不同温度（0.3、0.7、1.0）和多个模型生成 4--8 个回答（多样性是关键——如果所有回答都相似，偏好就毫无信息量）。
- 总计：200K--800K 个候选回答。

**阶段 2——偏好收集**：

- 方案 A（昂贵，质量最佳）：人工标注者两两比较。每对 3 名标注者。成本：每次比较 \$2--5。
- 方案 B（便宜，与人类一致率 85--90\%）：LLM judge（GPT-4/Claude）。便宜 10$\times$。适合规模化。
- 格式：(prompt, chosen response, rejected response)。标注者不一致（一致率 $<$70\%）的对被丢弃。
- 最终数据集：100K--500K 对。

**阶段 3——训练**：

- 架构：与 base LLM 相同 + 标量头（每个序列一个回归输出）。
- Loss：$\mathcal{L} = -\mathbb{E}[\log\sigma(r(x,y_w) - r(x,y_l))]$（Bradley-Terry）。
- 训练：**仅 1 个 epoch！** RM 过拟合极快。验证准确率 68--75\% 即为良好（更高往往意味着过拟合到标注 artifact）。
- 技巧：将 reward 中心化到 0（减去滑动均值）。检查长度偏差（若长度与分数相关 $>$ 0.3，在训练中加长度惩罚）。

**阶段 4——验证**：

- 留出偏好对：准确率应在 68--75\%。
- 在新数据上与人类的一致率：$>$ 80\%。
- 长度偏差检查：回答长度与 RM 分数的相关 $<$ 0.2。
- 一致性检查：同一 prompt 下，改写后的回答应得到相近分数。

*复习：第 9 章（Reward 模型训练）。*

### Q17：KL 散度爆炸时会发生什么？根因与修复。

**答**：

**KL 衡量什么**：当前 policy 与参考之间的平均对数比：$D_\text{KL} = \mathbb{E}_{y\sim\pi_\theta}[\log(\pi_\theta(y|x)/\pi_\text{ref}(y|x))]$。KL=0 意味着与参考完全相同。KL=10 意味着 policy 对其偏好的输出多投放了 10 nats 的概率。

**健康区间**：训练中 3--10。缓慢增长无妨。突发飙升 = 出问题。

**KL 爆炸的根因**：

1. **学习率过高**：policy 步幅巨大、偏离参考。修复：将 LR 降低 2--5$\times$。
2. **reward hacking**：找到了远离参考行为的高 reward 漏洞。修复：增大 $\beta$、加入 RM 集成。
3. **Mode collapse**：policy 集中到一个回答模板上。在该模板处 KL 很高，其他地方都低。修复：加大熵奖励、提高温度。
4. **坏 batch**：一个具有极端 advantage 的不幸 batch 推动了 policy。修复：梯度裁剪、减小 mini-batch 大小。
5. **Value function 偏离**：错误的 advantage 估计导致错误的更新。修复：降低 value function 的 LR，或切换到 GRPO（无 value function）。

**恢复流程**：

1. 检测：KL $>$ 15 持续 50+ 步，或单步内 KL 跳 $>$5。
2. 立即：加载最近一个干净的 checkpoint（KL $<$ 10）。
3. 调整：LR 降低 50\%。$\beta$ 增大 2$\times$。cliprange 降至 0.1。
4. 恢复：在头 200 步密切监控。

*复习：第 5 与 7 章（PPO；GRPO）。*

### Q18：比较单体 vs 解耦的 RLHF 架构。各自在什么场景合理？

**答**：

**单体**（TRL 默认：单进程，所有模型同 GPU）：

- 优点：代码简单。无分布式系统复杂度。易于调试。
- 缺点：GPU 闲置 60\% 时间（生成时计算闲、训练时带宽闲）。无法高效扩展到 $\sim$16 GPU 以上。所有模型争抢同一显存。
- 适用：模型 $\leq$ 13B，单节点，研究/原型。

**半解耦**（vLLM 生成 + FSDP 训练，同一集群）：

- 优点：利用率更好（生成与训练可部分重叠）。可扩展到 64 GPU。
- 缺点：仍共享硬件，无法独立优化。比单体更复杂。
- 适用：13B--70B，2--8 节点，生产实验。

**完全解耦**（独立集群通过队列连接）：

- 优点：每个集群针对其工作负载优化。生成与训练独立扩展。生成集群无状态（容错极简）。可扩展到数百 GPU。
- 缺点：分布式系统复杂度。权重陈旧度。队列管理。网络开销。
- 适用：$\geq$ 70B 生产训练。需要规模、容错、高利用率。

**关键洞见**：生成受带宽约束，训练受算力约束。同一硬件无法同时为二者优化。解耦让生成节点拥有：更多显存带宽、INT8 权重、大 batch。训练节点拥有：完整 BF16 精度、FlashAttention、FSDP 分片。

*复习：第 11 章（系统架构与大规模基础设施）。*

### Q19：如何为 RL 训练设置课程学习？

**答**：课程学习 = 逐渐提升难度，让模型循序渐进地学习。

**为什么重要**：若把最难的 prompt 丢给弱模型，它会得到全负的 reward $\rightarrow$ 无学习信号（一切都同样糟）。若从易入手，模型先发展基础能力，再在其上构建。

**实现**：

1. **难度评分**：根据当前模型的通过率（来自 Goldilocks 过滤）对每个 prompt 打分。易 = 通过 $>$80\%，中 = 30--80\%，难 = $<$30\%。
2. **排程**：步 0--1000：70\% 易，20\% 中，10\% 难。步 1000--5000：30\% 易，50\% 中，20\% 难。步 5000+：10\% 易，40\% 中，50\% 难。
3. **动态调整**：每 500 步重新评估难度分布。模型已“掌握”（通过率 $>$ 95\%）的 prompt 退役。引入更难的新 prompt。
4. **对 GRPO**：课程保证 group 总是混合成功与失败。没有课程，难 prompt 会产生全零 group（无用）。

**证据**：DeepSeek-R1 使用隐式课程——从简单的数学/代码问题起步，模型发展出基本推理，再逐步解出更难的问题，无需显式排程。

*复习：第 7 与 12 章（GRPO；LLM 智能体训练）。*

### Q20：你有 64 张 A100-80GB GPU 的预算，需要 RL 训练一个 70B 模型。设计资源分配。

**答**：8 节点 $\times$ 8 GPU = 64 张总。需在生成、打分、训练之间分配。

**我的分配**：

- **生成**：24 GPU（3 节点）。6 个 vLLM 实例，TP=4。INT8 权重 = 70GB/模型，留下空间给 KV cache。连续批处理，总 batch $\approx$ 128。
- **打分**：8 GPU（1 节点）。RM（INT8，TP=4）+ 参考模型（INT8，TP=4）位于同一节点。或共享 4 GPU，TP=4 在 RM 与 ref 间轮换。
- **训练**：32 GPU（4 节点）。在全部 32 张上做 FSDP。每张 GPU 持有 $\sim$70B/32 = 2.2GB 参数 + 优化器片段。激活留有充足余量。开启梯度检查点以保险。

**预期吞吐**：

- 生成：6 实例 $\times$ $\sim$80 回答/分钟 = 480 回答/分钟。
- 训练：batch=128，每 $\sim$15s 一步（仅训练时间，不含等待生成）。
- 重叠：当训练第 $N$ 步进行（15s）时，生成已为第 $N+1$ 步产出 $\sim$120 个回答。完美流水。

**瓶颈分析**：128 个回答的生成需 $\sim$45s。训练 $\sim$15s。打分 $\sim$5s。生成是瓶颈。可将 8 GPU 从训练移到生成（40 生成、24 训练）以平衡，但那样训练变成瓶颈。当前分配接近最优。

**显存吃紧时的替代方案**：把打分放到训练节点（分时复用：生成时打分，训练时训练）。省下 8 GPU。流水稍差但可用。

*复习：第 2 与 11 章（系统基础；系统架构）。*

## GRPO 变体与高级 RL 题

### Q21：什么是 DAPO？它相对标准 GRPO 有哪些改进？

**答**：DAPO（动态自适应策略优化，Dynamic Adaptive Policy Optimization）引入了 5 项关键改动：

**1. Clip-Higher（非对称裁剪）**：标准 PPO/GRPO 两侧都按 $\epsilon=0.2$ 等量裁剪。DAPO 使用 $\epsilon_\text{low}=0.2$ 但 $\epsilon_\text{high}=0.28$。这允许模型*更激进地提升*好动作的概率，同时仍限制对坏动作的抑制幅度。直觉：探索比利用需要更大的空间。

**2. 超长过滤（Overlong Filtering）**：若回答达到最大长度上限（被截断，没有 EOS Token），就将其完全从 loss 中屏蔽。理由：截断的回答不包含自然停止信号——在其上训练会教会模型“句子中间停下”是可接受的。

**3. Token 级 Loss**：loss 按所有序列的总 Token 数归一，而非按序列数。这防止更长的序列主导梯度。

**4. 软超长惩罚（Soft Overlong Punishment）**：不再是二元截断过滤，而是在回答接近最大长度时施加渐进惩罚。$r_\text{soft} = -c \cdot \max(0, \text{len} - L_\text{soft})/(L_\text{max} - L_\text{soft})$。

**5. 动态采样（Dynamic Sampling）**：训练中重采样 prompt，确保每个 batch 都有成功/失败的混合（TRL 中尚未实现）。

**适用场景**：需要极大探索与超长补全（32K+ Token）的大规模推理 RL。非对称裁剪尤其有价值。

*复习：第 7 与 8 章（GRPO；偏好优化变体）。*

### Q22：解释 vLLM 的训练-推理不一致。为何发生，TIS/MIS 如何修复？

**答**：**问题**：当用 vLLM 做生成、用训练框架（DeepSpeed/FSDP）做更新时，同一模型、同一权重会产生*不同*的 Token 概率。原因：

- 数值 kernel 不同（vLLM 使用针对吞吐优化的自定义 CUDA kernel）。
- attention 实现不同（训练用 FlashAttention，vLLM 用 PagedAttention）。
- 精度处理不同（vLLM 用 FP8/INT8，训练用 BF16）。
- 批处理差异影响 layer normalization 的数值。

这会悄无声息地破坏 PPO 的 on-policy 假设：我们计算 $\pi_\theta/\pi_\text{old}$ 时，$\pi_\text{old}$ 来自 vLLM 而 $\pi_\theta$ 来自训练框架。从第零步起比率就是错的！

**TIS（截断重要性采样，Truncated Importance Sampling）**：通过乘以 $\min(\pi_\text{train}/\pi_\text{inference}, C)$ 修正梯度。$\min$ 与上限 $C$ 防止极端修正破坏训练稳定性。典型 $C=2.0$。

**MIS（掩码重要性采样，Masked Importance Sampling）**：更激进——直接丢弃任何 $\pi_\text{train}/\pi_\text{inference} > C$ 的 Token，对梯度贡献为零。防止任何估计糟糕的 Token 影响更新。

**序列级 vs Token 级**：序列级 IS 理论上正确（无偏）；Token 级 IS 有偏但方差更低。实践中，序列级带截断效果最好。

*复习：第 7 与 11 章（GRPO；系统架构）。*

### Q23：GSPO vs GRPO——根本区别在哪？什么时候重要？

**答**：**GRPO**：*逐 Token*计算重要性比 $w_{i,t} = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_\text{old}(o_{i,t}|q, o_{i,<t})$，然后独立裁剪每个 Token。

**GSPO**：在*序列级*计算重要性比：$s_i(\theta) = (\pi_\theta(o_i|q)/\pi_\text{old}(o_i|q))^{1/|o_i|}$——Token 概率的几何均值。裁剪这单一的序列级比率。

**为何重要**：GRPO 的逐 Token 裁剪把每个 Token 视为独立，但语言中它们高度相关。序列前段的微小逐 Token 变化在多个 Token 上指数级放大。GSPO 通过审视完整序列概率来捕捉这一点。

**长度归一化**：$1/|o_i|$ 指数保证不同长度序列间的公平比较。否则更长的序列总是有更低的概率比。

**何时用 GSPO**：当训练变为 off-policy 时（`steps_per_generation > 1` 或 `num_iterations > 1`）。如果完全 on-policy（比率 $\approx 1$），GRPO 与 GSPO 等价。

*复习：第 7 与 8 章（GRPO；偏好优化变体）。*

### Q24：论文 “It Takes Two” 表明 G=2 与 G=16 效果相当。这怎么可能？

**答**：关键洞见是 GRPO 的有效性并非来自精确的 advantage 估计（那需要大 $G$），而来自一个**隐式的对比目标**。

$G=2$ 加二元 reward（一个对一个错）时：归一化后 $\hat{A}_\text{correct} = +1$，$\hat{A}_\text{wrong} = -1$。loss 变成：提升正确回答的概率、降低错误回答的概率。这本质上就是一个 DPO 风格的对比 loss！

**为什么大 $G$ 帮助不大**：归一化 advantage $\hat{A}_i = (r_i - \mu)/\sigma$ 本身已制造了好坏之间的对比。更多样本能更准确估计 $\mu$，但梯度方向由最好与最差之间的*对比*主导，而非 $\mu$ 的精度。

**算力节省**：$G=2$ 意味着生成算力比 $G=16$ 少 8$\times$。由于生成占训练时间 60\%，整体训练加速 $\sim$4$\times$。

**注意**：通过率在 30--70\% 时效果最佳。如果通过率极低（$<$10\%），$G=2$ 常常给出两个失败（无信号）。难题需要更大的 $G$。

*复习：第 7 章（GRPO）。*

### Q25：什么是 SAPO？它的软门控与硬裁剪有何不同？

**答**：标准 PPO/GRPO 使用硬裁剪：$\text{clip}(r, 1-\epsilon, 1+\epsilon)$。在边界处梯度突然降为零。这制造了一个“死区”，模型在其中收不到任何学习信号。

**SAPO** 用平滑的 sigmoid 门控取代它：随着比率偏离 1，梯度被逐渐衰减，绝不会突兀归零。它使用非对称温度：

- 对正 advantage 用 $\tau_+ = 1.0$（标准衰减）。
- 对负 advantage 用 $\tau_- = 1.05$（对抑制略激进一些的衰减）。

**收益**：(1) 梯度地形上没有“悬崖”。(2) 略超出裁剪范围的 Token 仍有贡献（衰减而非归零）。(3) 优化轨迹更稳定。(4) 序列连贯——考虑了完整序列上下文。

**权衡**：信任域比硬裁剪略宽松，因此需要仔细调温度。但整体对超参更鲁棒。

*复习：第 7 与 8 章（GRPO；偏好优化变体）。*

## DPO 扩展题

### Q26：比较 f-DPO 的散度选择。前向 KL、JS 与反向 KL 各自何时使用？

**答**：标准 DPO 隐式使用反向 KL（$D_\text{KL}[\pi_\theta \| \pi_\text{ref}]$）：

- **反向 KL**（默认）：寻峰（mode-seeking）。policy 把概率集中在参考概率高的位置。避免生成参考不会生成的文本。利于安全（保守）。
- **前向 KL**：覆盖（mass-covering）。policy 试图覆盖参考的所有峰，甚至低概率的峰。利于多样性但可能生成低质量输出。
- **Jensen-Shannon**：前向与反向之间的对称折中。在覆盖与寻峰间取得平衡。对通用对齐通常最佳。
- **Alpha 散度**（$\alpha=0.5$）：在前向（$\alpha=0$）与反向（$\alpha=1$）之间插值。可调。

**实践建议**：先用反向 KL（标准 DPO）。若模型过于保守（不愿尝试创造性方案），切换到 JS 散度。若多样性至关重要（创意写作、头脑风暴），试试前向 KL。

*复习：第 6 与 8 章（DPO；偏好优化变体）。*

### Q27：你的 DPO 偏好数据有 15\% 的标签噪声。怎么办？

**答**：按复杂度递增的三个方案：

**1. Robust DPO**（已知噪声率时最佳）：解析地去偏 loss：$\mathcal{L}_\text{robust} = \frac{(1-\varepsilon)\mathcal{L}_\text{DPO}(y_w, y_l) - \varepsilon \mathcal{L}_\text{DPO}(y_l, y_w)}{1 - 2\varepsilon}$。设 $\varepsilon = 0.15$。在期望意义上可证明恢复干净的 DPO 目标。TRL：`loss_type="robust", label_smoothing=0.15`。

**2. IPO（Identity Preference Optimization, IPO）**（噪声率未知时最佳）：带目标 margin 的平方 loss。被误标的对影响有界（平方 loss 不发散）。对任意噪声模式更鲁棒，且无需知道 $\varepsilon$。TRL：`loss_type="ipo"`。

**3. TR-DPO**（分布漂移时最佳）：训练中通过 EMA 更新参考模型。即便早期数据有噪声，演化的参考也能帮助模型自我修正。TRL：`sync_ref_model=True, ref_model_mixup_alpha=0.6`。

**数据侧修复**：(1) 过滤标注者一致率 $<$70\% 的对。(2) 用 RM 给对打分；丢弃 RM 与标签不一致的对。(3) 主动学习：重新标注最不确定的对。

*复习：第 6 与 8 章（DPO；偏好优化变体）。*

### Q28：什么是 SimPO？为什么“无参考模型”是优势？

**答**：SimPO 用回答的平均对数概率作为隐式 reward 信号：$r(x,y) = \frac{1}{|y|}\sum_t \log \pi_\theta(y_t|x, y_{<t})$——无需参考模型。

loss 中加入目标 margin $\gamma$：chosen 回答的平均 log-prob 应至少比 rejected 高 $\gamma$。

**为什么“无参考”重要**：

1. **显存**：无参考模型 = 70B 节省 70--140GB。可在同样硬件上训练更大的模型。
2. **简洁性**：无需管理/加载/服务第二份模型副本。
3. **无陈旧参考**：DPO 的参考随着训练推进越来越不相关。SimPO 没有这个问题。
4. **内置长度归一化**：$1/|y|$ 天然防止长度偏差（DPO 需显式处理）。

**权衡**：没有参考锚点，模型有更多自由去塌缩或漂移。$\gamma$ margin 和长度归一化部分缓解了这一点，但在激进训练时 SimPO 可能不如 DPO 稳定。

*复习：第 8 章（偏好优化变体）。*

### Q29：解释 Iterative RPO。为何在推理任务中把 DPO 与 NLL loss 结合？

**答**：用于推理的标准 DPO 有一种微妙的失效模式：它学会*判别*（给正确轨迹更高的隐式 reward），但不一定学会*生成*它们。

**为什么**：DPO 的梯度推高 chosen 的概率、压低 rejected 的概率。但 chosen 回答可能与模型自身会生成的东西差异巨大，以至于提升其概率并不能教会模型产出类似的推理模式。

**RPO 的修复**：在 chosen 回答上添加负对数似然（NLL/SFT）loss：$\mathcal{L} = \mathcal{L}_\text{DPO} + \alpha \cdot \mathcal{L}_\text{NLL}(y_w)$。

NLL 项显式训练模型逐步生成获胜回答。DPO 项确保模型同时学会避免落败回答。结合起来：模型同时学到“如何正确推理”（NLL）和“该避免什么”（DPO）。

**迭代**：生成回答 $\rightarrow$ 检查正确性 $\rightarrow$ 构造对 $\rightarrow$ 用 RPO 训练 $\rightarrow$ 重复。每次迭代，模型更擅长生成正确推理，为下一轮提供更高质量的训练数据。

TRL：`loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0]`

*复习：第 8 与 13 章（偏好优化变体；大型推理模型的 RL）。*

## GPU 架构与硬件题

### Q30：解释 GPU 内存层次结构。它对 LLM 推理为何重要？

**答**：从最快到最慢：

1. **寄存器（Registers）**：线程私有，$\sim$256 KB/SM。即时访问（0 周期延迟）。
2. **SRAM（共享内存）**：每 SM 私有，$\sim$192--228 KB/SM（A100：164 KB 可配置）。聚合带宽：$\sim$19 TB/s。延迟：$\sim$20 周期。
3. **L2 缓存**：跨整张 GPU 共享，40--60 MB（H100：50 MB）。带宽：$\sim$5 TB/s。延迟：$\sim$200 周期。
4. **HBM**：GPU 主显存，80 GB（A100）。带宽：2--3.35 TB/s。延迟：$\sim$400 周期。
5. **CPU DRAM**：通过 PCIe，512 GB+。带宽：32--64 GB/s。延迟：$\sim$10K 周期。

**对 LLM 为何重要**：自回归生成在每个 Token 都要读取完整模型权重（70B 约 $\sim$140 GB）。以 2 TB/s 的 HBM 带宽，仅流式读权重就要 70ms。实际计算（一次矩阵-向量乘法）只要 0.5ms。GPU 有 99\% 的时间在等数据。

**FlashAttention 利用了这一点**：通过把中间结果（QK 分数、softmax）保留在 SRAM（19 TB/s）而非写到 HBM（2 TB/s），消除了 attention 的 90\% 内存流量。计算量未变，但 HBM 的读写降低 10$\times$。

*复习：第 2 章（LLM 的系统基础）。*

### Q31：FlashAttention 如何工作？什么是 online softmax 技巧？

**答**：**问题**：标准 attention 在 HBM 中物化 $n \times n$ 的 attention 矩阵。$n=8192$ 时：每个 head $8192^2 \times 2 = 134$ MB，32 头每层 4.3 GB。必须写到 HBM 再读回做 softmax 与乘法——3 次完整的 HBM 往返。

**FlashAttention 方案**：从不存储完整的 $n \times n$ 矩阵。以可放进 SRAM 的分块（tile）处理。

**算法**：

1. 将 $Q$ 拆为 $B_r$ 行的块，$K/V$ 拆为 $B_c$ 行的块。
2. 对每个 $Q$ 块：遍历所有 $K$ 块，计算部分 attention 分数。
3. **Online softmax 技巧**：维护滑动 max $m$ 与滑动 sum $\ell$ 以做 softmax 归一。处理新的 $K$ 块时更新：$m_\text{new} = \max(m_\text{old}, \max(\text{scores}))$，将上一累加器按 $e^{m_\text{old} - m_\text{new}}$ 重新缩放，再加上新的贡献。
4. 输出以增量方式累加——从不需要完整的 $n \times n$ 矩阵。

**关键洞见**：softmax 本来是全局操作（对所有元素取 $\max$ 和 $\sum$）。online 技巧将其分解为带修正因子的局部更新。数学上精确——并非近似。

**结果**：显存 $O(n)$ 而非 $O(n^2)$。速度提升 2--4$\times$（更少 HBM 访问，更多时间在 SRAM）。

**FlashAttention 2**：在 warp 之间做更好的工作划分，将非矩阵乘的 FLOPs 减半。

**FlashAttention 3**（H100/Hopper）：使用张量内存加速器（Tensor Memory Accelerator, TMA）做异步加载、warp 专门化（生产者/消费者 warp）、支持 FP8。

*复习：第 1 与 2 章（LLM 架构；系统基础）。*

### Q32：解释 PagedAttention。它如何解决 KV cache 问题？

**答**：**问题**：生成期间，每条序列都需要 KV cache（存储所有先前 Token 的 K、V 张量）。对 70B 模型：每个 Token 需要 $2 \times n_\text{layers} \times d_\text{model} \times 2$ 字节 = $2 \times 80 \times 8192 \times 2 \approx 2.5$ MB。2048 Token 的序列：$\sim$5 GB 的 KV cache。

**传统分配**：为每条活跃序列预分配 max\_sequence\_length。若 max=2048 而平均=500，就浪费 75\% 已分配显存。50 条并发序列就浪费数百 GB。

**PagedAttention**：受操作系统虚拟内存启发：

1. KV cache 被切分为固定大小的*块*（页），每块保存 16 个 Token 的 KV。
2. 一张*块表*（类似页表）将逻辑 Token 位置映射到物理显存块。
3. 块按需随序列增长分配。无预分配浪费。
4. 序列结束时，释放的块立即归还到池中。

**额外收益**：

- **前缀共享**：拥有相同系统 prompt 的多条序列共享 KV cache 块（写时复制）。聊天应用可省 30--50\% 显存。
- **抢占**：可将低优先级序列的块“换出”到 CPU，为高优先级请求腾出 GPU 显存。
- **接近零碎片**：内部碎片仅限末块（$<$16 Token）。外部碎片被消除（任意空闲块可用于任意位置）。

**结果**：同等显存下并发序列数提升 3--5$\times$ $\rightarrow$ 服务吞吐提升 3--5$\times$。

*复习：第 2 章（LLM 的系统基础）。*

### Q33：比较 NVLink 与 InfiniBand。在 RLHF 训练中各用于何处？

**答**：

**NVLink**（节点内，GPU 到 GPU）：

- 带宽：600 GB/s（A100）、900 GB/s（H100）——双向总和。
- 延迟：$\sim$1 $\mu$s。
- 范围：单个物理节点内（8 GPU 通过 NVSwitch 互联）。
- 用例：**张量并行**（TP=8）。每层矩阵乘法在 GPU 间切分，每层后都要 AllReduce。需要超高带宽 + 低延迟。

**InfiniBand NDR**（跨节点，节点到节点）：

- 带宽：每端口 400 Gb/s = 50 GB/s。8 端口（GPUDirect RDMA）：每节点聚合 400 GB/s。
- 延迟：$\sim$1--5 $\mu$s（RDMA）。
- 范围：集群中跨节点。需要交换机（fat-tree 拓扑）。
- 用例：**数据并行 / FSDP** 梯度同步。梯度 AllReduce 每训练步发生一次（而非每层），因此延迟容忍度更高。

**在 RLHF 中具体而言**：

- *生成*：节点内 NVLink 上 TP=8。跨节点的多个 vLLM 实例无需通信（embarrassingly parallel）。
- *训练*：节点内 NVLink 上 TP=8 + 节点间 InfiniBand 上 FSDP。梯度在完整反向传播后同步。
- *权重同步*：训练 $\to$ 生成使用 InfiniBand（140 GB 传输，异步，50 GB/s 下需 $\sim$3s）。

*复习：第 2 与 11 章（系统基础；系统架构）。*

## 优化与训练题

### Q34：解释 Adam 与 AdamW。这一差异对 LLM 为何重要？

**答**：**Adam 加 L2 正则化**：$\theta_{t+1} = \theta_t - \alpha \cdot (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda\theta_t)$。权重衰减项 $\lambda\theta_t$ *位于*自适应缩放*内部*。梯度大（$v_t$ 大）的参数*衰减更少*（除以 $\sqrt{v_t}$）。这不是真正的权重衰减——它与尺度相关。

**AdamW（解耦权重衰减）**：$\theta_{t+1} = (1 - \alpha\lambda)\theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$。权重衰减在自适应更新*之外*且*之前*施加。无论梯度历史如何，每个参数都获得相同比例的衰减。

**对 LLM 为何重要**：

1. LLM 的参数跨越多个数量级（embedding 层 vs attention vs FFN）。Adam 的耦合 L2 实际上对小梯度参数惩罚得比大梯度参数更多——错误行为。
2. 解耦 WD 在所有层提供统一正则化，防止某些层无界增长而另一些层过度收缩。
3. 经验：在长预训练上，AdamW 相比相同有效正则强度的 Adam+L2 给出 2--5\% 更好的 perplexity。

**对 RL 而言**：通常使用 $\lambda = 0$（无权重衰减）。由 KL 惩罚提供正则化。但对 SFT：AdamW 配合 $\lambda = 0.01$--$0.1$ 是标准做法。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q35：为什么学习率 warmup 是必要的？不做 warmup 会发生什么？

**答**：**问题**：Adam 的二阶矩估计 $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ 从 $v_0 = 0$ 起步。偏置修正 $\hat{v}_t = v_t/(1-\beta_2^t)$ 在数学上做了补偿，但在实践中：

- 头几步：$v_t$ 基于 1--5 个梯度样本。对真实方差的估计极不准确。
- 若某参数初期恰好得到小梯度，$v_t$ 微小 $\rightarrow$ 有效 LR 巨大 $\rightarrow$ 灾难性更新。
- 偏置修正放大早期更新：第 1 步时 $\hat{v}_1 = v_1/(1-0.999) = 1000 \cdot v_1$。

**无 warmup 时**：头 10--100 步经常出现梯度尖峰，永久性损坏模型。在优化器稳定之前，早期表征就被打乱。

**Warmup 的修复**：从 LR $\approx 0$ 起步，在 $W$ 步（通常为训练量的 3--10\%）内线性升至目标值。当 LR 到达满值时，$v_t$ 已积累了足够样本以变得准确。

**典型设置**：

- 预训练：2000 步 warmup（占 200K 步的 $\sim$1\%）。
- SFT：100 步 warmup（占 2000 步的 $\sim$5\%）。
- RL（PPO/GRPO）：20--50 步 warmup（短，模型经 SFT 已稳定）。

*复习：第 1 与 10 章（LLM 架构；SFT 最佳实践）。*

### Q36：比较学习率调度策略。RL 微调你会选哪种？

**答**：

**余弦衰减（Cosine decay）**：$\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos(\pi t/T))$。预训练与 SFT 的标准选择。衰减平滑，大部分时间停留在中等 LR。

**线性衰减**：$\eta_t = \eta_\text{max}(1 - t/T)$。更简单，短训练下与余弦衰减效果相近。

**预热-稳定-衰减（Warmup-Stable-Decay, WSD）**：warmup $\rightarrow$ 80\% 时间常数 LR $\rightarrow$ 最后 20\% 快速衰减。预训练的新标准。“稳定”阶段提供一致学习；最终衰减挤出剩余收益。

**常数**：无衰减。warmup 之后 $\eta_t = \eta_\text{max}$。

**对 RL 微调（PPO/GRPO），我会选择：短 warmup + 常数**。理由：

1. RL 训练长度高度不可预测（按胜率停训，而非 epoch）。
2. 余弦/线性衰减假设你提前知道总步数。
3. LR 已经非常低（$10^{-6}$），再衰减会让更新几乎不可见。
4. PPO 的自适应 KL 控制器已在调控有效步长。
5. 如必须衰减：在宽裕预算上做线性衰减，指标停滞时早停。

*复习：第 1 与 5 章（LLM 架构；PPO）。*

### Q37：为何梯度裁剪（Gradient Clipping）对 RL 训练至关重要、对 SFT 却不那么重要？

**答**：**SFT**：监督 loss 平滑且行为良好。梯度范数在 batch 间一致（通常 0.1--1.0）。在 1.0 处裁剪很少触发——只是安全网。

**RL（PPO/GRPO）**：梯度范数高度多变，因为：

1. **Reward 方差**：一个 batch 可能全是高 reward 回答，下一个全是低的。advantage $\hat{A}$ 剧烈摆动。
2. **比率爆炸**：若某个稀有 Token 的概率变化很大，$r_t = \pi_\text{new}/\pi_\text{old}$ 可能极大 $\rightarrow$ 在裁剪生效前就产生大梯度。
3. **稀疏 reward**：在二元 reward 的 GRPO 中，有些 prompt 全部正确（advantage $\approx 0$），突然一道难题给出极端 advantage。
4. **KL 项**：当 policy 偏离时，KL 惩罚的梯度可能尖峰。

**无裁剪时**：一个坏 batch 就能产生比正常大 100$\times$ 的梯度 $\rightarrow$ 一步毁掉模型。无法恢复（所有预训练遭灾难性遗忘）。

**典型设置**：`max_grad_norm=1.0`。一些人在 RL 训练早期用 0.5 以更安全。范数在所有参数上全局计算（非逐层）。

**监控**：若裁剪在超过 20\% 步触发，你的 LR 可能过高，或 batch size 过小。

*复习：第 5 与 7 章（PPO；GRPO）。*

### Q38：训练中 BF16 vs FP16。这个选择何时重要？

**答**：

**FP16**：1 符号 + 5 指数 + 10 尾数位。范围：$\pm 65504$。精度：$\sim$3.3 十进制位。

**BF16**：1 符号 + 8 指数 + 7 尾数位。范围：$\pm 3.4 \times 10^{38}$（与 FP32 相同！）。精度：$\sim$2.4 十进制位。

**为什么 BF16 在 LLM 上胜出**：

1. **无需 loss scaling**：FP16 范围窄（$\pm$65K）导致梯度和激活频繁溢出/下溢。需要动态 loss scaling（loss 乘 1024，梯度再除回去）。BF16 拥有 FP32 的范围——溢出基本不可能。
2. **代码更简单**：无 loss scaler，无 inf/nan 检查，无动态 scaling 调整。
3. **对 RL 关键**：RL 梯度比 SFT 更嘈杂、更易尖峰。FP16 的 loss scaling 经常失败（选错尺度，产生 NaN）。BF16 “开箱即用”。

**FP16 可能更好的场景**：若你需要极致精度（某些科学计算任务）并能管理好 loss scaling。FP16 多 3 个尾数位 = 结果略更精确。

**FP32 主权重**：即便前向/反向使用 BF16，也要在 FP32 中累加梯度更新，以防舍入误差在数千次小步中累积。所有 LLM 训练的标准做法。

*复习：第 2 章（LLM 的系统基础）。*

## Reward 模型与 SFT 题

### Q39：推导 Bradley-Terry reward 模型的 loss。它有哪些局限？

**答**：**Bradley-Terry 模型（Bradley-Terry Model）**：给定两个回答，更好的那个（$y_w$）被偏好的概率：$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$，其中 $\sigma$ 是 sigmoid。

**MLE 推导**：给定 $N$ 个偏好对，最大化似然：$\prod_i P(y_w^i \succ y_l^i)$。取负对数：$\mathcal{L} = -\sum_i \log\sigma(r(x_i, y_w^i) - r(x_i, y_l^i))$。

**局限**：

1. **无平局**：BT 无法建模“同样好”——强制严格偏好。
2. **可传递性**：假设若 A$>$B 且 B$>$C 则 A$>$C。人类并不可传递。
3. **上下文无关**：无论备选项是什么，reward 都相同。
4. **标量塌缩**：把所有质量维度压成一个数。一个回答可能既安全又无用——RM 必须权衡。
5. **长度偏差**：更长的回答得到更高分（信息更多 = 更可能包含标注者想要的内容）。必须显式去相关。

**缓解措施**：margin loss（要求最小间隔 $\delta$）、reward 中心化（减去滑动均值）、训练时长度惩罚、多头 RM（为有用性/安全性/准确性提供独立分数）。

*复习：第 9 章（Reward 模型训练）。*

### Q40：SFT 中的序列打包（Sequence Packing）是什么？为何重要？

**答**：**问题**：训练样本长度可变。标准批处理把所有样本填充到 max\_length。若 max=4096 而平均=500，就有 88\% 的算力浪费在填充 Token 上（梯度贡献为零）。

**打包方案**：把多个短样本拼接成一条 max\_length 序列，用 EOS Token 分隔。同时对所有样本训练。

例子：不再用 4 条填充到 4096 的序列（16K Token，14K 填充），而是打包成 1 条 4096 的序列，首尾相连地容纳 4 个样本（4096 真实 Token，0 填充）。**效率提升 4$\times$。**

**关键细节——块对角 attention 掩码**：若不做特殊处理，样本 2 会关注样本 1 的 Token（交叉污染）。必须使用块对角 attention 掩码，限制每个样本只能关注自身的 Token。

**在 TRL 中**：`SFTConfig(packing=True, max_seq_length=4096)`。自动处理掩码。

**注意**：(1) 更长的样本仍需自己的 batch 条目（不能在序列中间切分）。(2) 位置 Embedding 实现略复杂（每个样本重置）。(3) 有人认为打包改变了有效 batch size（每步样本更多）——相应调整 LR。

*复习：第 10 章（SFT 最佳实践与技巧）。*

### Q41：解释 SFT 的 completion-only masking。不使用它会怎样？

**答**：在 chat 格式的 SFT 数据中：`[system] + [user message] + [assistant response]`。标准 NLL loss 会在所有 Token 上计算 loss，包括 system prompt 和用户消息。

**不掩码的问题**：模型浪费容量去预测用户消息（推理时永远不需要生成）。更糟：若训练数据中用户消息多样，模型会困惑“现在轮到谁说话？”

**Completion-only masking**：将 prompt（system + user）中所有 Token 的 loss 权重设为 0。仅在 assistant 回答 Token 上计算 loss。

TRL：`DataCollatorForCompletionOnlyLM(response_template="<|assistant|>")`

**影响**：在指令跟随基准上通常提升 5--15\%。收敛更快（梯度信号集中于有用 Token）。算力成本不变。

**细节**：response 模板 Token 必须包含在 loss 内（教模型开始回答），但之前的所有内容都要排除。

*复习：第 10 章（SFT 最佳实践与技巧）。*

### Q42：SFT 质量如何影响 RL 的天花板？pass@k 诊断是什么？

**答**：**天花板定理（非正式表述）**：RL 只能强化模型已能以非可忽略概率产出的行为。若 SFT 模型生成正确解的概率为 0\%，RL 永远找不到它。

**为什么**：GRPO/PPO 从当前 policy 采样并强化好样本。若好样本不在分布内，就没什么可强化。RL 的探索受 base policy 支撑集（support）的限制。

**pass@k 诊断**：对每个 prompt 生成 $k$ 个回答，检查*是否有任一个*正确：

- pass@1：模型的典型表现（greedy/低温）。
- pass@8：$G=8$ 的 GRPO 所能达到的上界。
- pass@64：激进 Best-of-N 采样（Rejection Sampling）的上界。
- pass@256：RL 改进的近似天花板。

**解读**：

- pass@1=20\%、pass@64=80\%：好！RL 有 4$\times$ 上行空间。预期强劲收益。
- pass@1=20\%、pass@64=25\%：几乎没有上行空间。RL 帮不上忙。需先做更好的 SFT。
- pass@1=5\%、pass@64=60\%：模型*能*解但很少解出来。RL 的完美场景（强化稀有的成功）。

**规则**：若 pass@64 $<$ 1.5$\times$ pass@1，在开始 RL 之前先投入更好的 SFT 数据。

*复习：第 7 与 10 章（GRPO；SFT 最佳实践）。*

### Q43：为聊天模型设计一个多目标 reward 系统。如何平衡有用性与安全性？

**答**：**架构**：每个目标使用独立 reward 模型：

- $r_\text{helpful}$：在有用性偏好上训练（质量、准确性、完整性）。
- $r_\text{safe}$：在安全偏好上训练（拒答、无害、无幻觉）。
- $r_\text{format}$：基于规则（遵循指令、合适格式、合理长度）。

**组合策略**：

1. **加权求和**（最简）：$r = w_1 r_\text{helpful} + w_2 r_\text{safe} + w_3 r_\text{format}$。问题：安全性可能被有用性压过。
2. **约束式**（更安全）：在 $r_\text{safe} > \tau$ 约束下最大化 $r_\text{helpful}$。通过 $r = r_\text{helpful} - \lambda \cdot \max(0, \tau - r_\text{safe})$ 实现，$\lambda$ 取较大值。
3. **GDPO 归一化**（对 GRPO 最佳）：在 group 内独立归一化每个 reward，再组合：$\hat{A} = w_1 \hat{A}_\text{helpful} + w_2 \hat{A}_\text{safe}$。防止某个 reward 因尺度差异而主导。
4. **字典序（Lexicographic）**：安全是硬约束（必须通过），再优化有用性。分阶段训练：先做安全对齐，再做有用性。

**实用权重**：从 $w_\text{safe}=2.0, w_\text{helpful}=1.0, w_\text{format}=0.5$ 起步。安全权重 2$\times$，因为它的失败模式（有害内容）远比有用性失败（平庸答案）更糟。

*复习：第 9 与 12 章（Reward 模型训练；LLM 智能体训练）。*

## 系统架构扩展题

### Q44：投机解码（Speculative Decoding）如何工作？它在 RLHF 中何时有帮助？

**答**：**问题**：大模型每次前向只产出一个 Token（70B 约 $\sim$70ms）。慢。

**投机解码**：

1. **草稿**：小模型（1--7B）快速生成 $k$ 个候选 Token（全部 $k$ 个约 $\sim$5ms）。
2. **验证**：大模型做一次前向，并行给所有 $k$ 个 Token 打分。$p_\text{large}(t_i) \geq p_\text{draft}(t_i)$ 的 Token 总是接受。其他按概率接受。
3. **结果**：每次验证步平均接受 3--4 个 Token。加速：2--3$\times$。

**关键性质**：输出分布与单独从大模型采样*完全相同*。无质量损失。草稿模型只影响速度，不影响输出。

**对 RLHF 而言**：生成占算力 60\%。生成加速 2--3$\times$ = 端到端加速 1.5--2$\times$。结合 vLLM + INT8：生成从瓶颈变为与训练持平。

**局限**：(1) 草稿模型必须共享 tokenizer。(2) 高温下效果较差（草稿模型准确率下降）。(3) 草稿模型需要额外 GPU 显存。(4) 超过 $k=5$ 后边际收益递减（接受率下降）。

*复习：第 2 与 11 章（系统基础；系统架构）。*

### Q45：解释 roofline 模型。如何判断一个 kernel 是计算受限还是内存受限？

**答**：roofline 模型将可达性能（FLOPS）刻画为算术强度（每字节内存流量的 FLOPS）的函数。

**两个区域**：

- **内存受限**（拐点左侧）：性能受限于向计算单元喂数据的速度。实际 FLOPS = 带宽 $\times$ 算术强度。GPU 利用率 $<$ 100\%。
- **计算受限**（拐点右侧）：性能受限于峰值 FLOPS。内存够快。GPU 达到最大利用率。

**拐点**：峰值 FLOPS / 峰值带宽。A100：312 TF / 2 TB/s = 156 FLOP/byte。

**LLM 操作**：

- **自回归生成**（batch=1）：读取 140GB 权重，做 140G FLOPs = 1 FLOP/byte。*极度*内存受限（比拐点低 156$\times$）。GPU 利用率仅 0.6\%。
- **训练前向**（batch=128，seq=2048）：算术强度 $\approx 200$+ FLOP/byte。计算受限。接近峰值利用率。
- **Attention**（长序列）：$O(n^2 d)$ FLOPs / $O(n^2 + nd)$ 字节。$n$ 长：计算受限；$n$ 短：内存受限。FlashAttention 无论如何都让其留在 SRAM。

**实用法**：若 kernel 内存受限，减少内存流量（量化、缓存、分块）。若计算受限，减少 FLOPs（剪枝、蒸馏、降低精度）。

*复习：第 2 章（LLM 的系统基础）。*

### Q46：连续批处理（Continuous Batching）如何工作？为何它对 RLHF 生成至关重要？

**答**：**静态批处理**：启动 $B$ 条序列。等*所有*序列结束。若一条生成 500 Token、另一条只生成 50 Token，那 50 Token 序列的 GPU 槽位会闲置 450 Token 的时间。

**连续批处理**（迭代级调度）：每个生成步后检查哪些序列已完成。立即在腾出的槽位插入新序列。GPU 槽位从不闲置。

**为何对 RLHF 至关重要**：

1. RLHF 生成多样输出（高温度）。长度方差巨大——有些回答 50 Token，有些 2000+。
2. 无连续批处理：平均利用率 $\sim$40--50\%（等最慢的序列）。
3. 有连续批处理：利用率 $>$90\%。吞吐高 2--3$\times$。
4. RLHF 需要大 batch（每步 128+ 回答）。用静态批处理生成 128 个回答需要 max\_tokens $\times$ 128 个串行步。连续批处理摊销了这一开销。

**实现**：vLLM 的调度器在每个解码步后检查。抢占：若有新的高优先级请求到达且显存已满，可将低优先级序列的 KV cache 换出到 CPU，稍后恢复。

*复习：第 2 与 11 章（系统基础；系统架构）。*

## Transformer 架构问题

### Q：为什么在现代 LLM 中 RoPE 主导地位胜过可学习的绝对位置嵌入？

**答**：RoPE 通过旋转矩阵将*相对*位置直接编码到 Q/K 点积中。关键优势：

1. Attention 分数只依赖于相对距离 $i-j$，而不依赖绝对位置——这能更好地泛化到未见过的序列长度。
2. 可通过频率缩放（NTK-aware、YaRN）扩展到超出训练长度，无需重新训练。
3. 无需额外参数（旋转由位置索引确定性地给出）。
4. 可学习的绝对嵌入固定到训练长度，无法外推——在 4K 上下文训练的模型在 8K 时会失败。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q：解释 SwiGLU，以及它为何在现代 Transformer 中取代了 ReLU。

**答**：SwiGLU：$\text{FFN}(x) = W_2 (\text{Swish}(W_1 x) \odot W_3 x)$，其中 $\text{Swish}(x) = x \cdot \sigma(x)$。

**为什么它更好**：

- *门控*机制（$\odot W_3 x$）让网络可以有选择地抑制或放大维度——比逐点 ReLU 更具表达力。
- Swish 是平滑的（不存在 ReLU 零梯度区那样的死神经元）。
- 经验上：在相同 FLOP 预算下，语言建模基准上提升 1\%--2\%。
- 权衡：需要 3 个权重矩阵而非 2 个（通过将隐藏维度从 $4d$ 降到 $8d/3$ 来解决）。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q：什么是分组查询注意力（Grouped Query Attention, GQA），Llama-3 为何采用它？

**答**：标准 MHA：$H$ 个 query 头、$H$ 个 key 头、$H$ 个 value 头。GQA：$H$ 个 query 头但只有 $G < H$ 个 key/value 头（在 query 组间共享）。

Llama-3 70B：64 个 query 头、8 个 KV 头（每个 KV 头被 8 个 query 头共享）。

**优势**：

- KV 缓存大小缩小 $H/G = 8\times$——对推理至关重要（长序列下 KV 缓存是主导的显存开销）。
- 质量损失极小（基准上 $<$0.5\%），因为 KV 模式在不同头之间高度相关。
- 推理吞吐与 KV 缓存的减少成比例增长（更多序列能装进显存 = 更大的 Batch）。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q：为何 decoder-only 架构在 LLM 中胜过了 encoder-decoder？

**答**：

1. **统一目标**：预训练 = 微调 = 推理，都使用下一个 token 预测。不存在架构不匹配。
2. **参数效率**：所有参数都对生成有贡献。在 encoder-decoder 中，纯生成任务下 encoder 参数被“浪费”。
3. **扩展更简单**：一个模型、一个损失函数、一套超参数需要调。
4. **KV 缓存效率**：decoder-only 只有一份 KV 缓存；encoder-decoder 有两份（encoder + decoder 的交叉注意力）。
5. **涌现的少样本能力**：decoder-only 天然支持上下文学习（把示例前置到 Prompt 中）。

对于固定输入长度的 seq2seq 任务（如翻译），encoder-decoder 仍然占优，但这部分在 LLM 用例中占比正在缩小。

*复习：第 1 章（LLM 架构与优化方法）。*

## FlashAttention 问题

### Q：FlashAttention 计算的结果与标准 attention 相同，但快 2--4 倍。如果 FLOPs 数相同，这怎么可能？

**答**：FlashAttention 之所以更快，是因为它减少了*HBM 显存流量*，而非 FLOPs。标准 attention 在 HBM（慢）中物化 $n \times n$ 的 attention 矩阵，再读回来做 Softmax，再读一次做 $PV$ 乘法——总共在 $O(n^2)$ 数据上有 4 次 HBM 往返。

FlashAttention 对计算进行分块，使 $n \times n$ 矩阵完全在 SRAM（快，19 TB/s）中计算并消费，从不写入 HBM（2 TB/s）。“在线 Softmax”技巧通过维护滚动统计量来实现这一点。

结果：HBM 流量从 $O(n^2 d)$ 降到 $O(n^2 d / M)$，其中 $M$ 为 SRAM 大小。相同 FLOPs，显存流量减少 10--50$\times$ $\to$ 实际墙钟加速 2--4$\times$。

*复习：第 1 章与第 2 章（LLM 架构；系统基础）。*

### Q：为什么 FlashAttention 对 FFN 层没帮助？

**答**：FFN 层是*计算瓶颈*，而非显存瓶颈。其算术强度（大 Batch GEMM 下 $I \approx 300$ FLOP/byte）已经高于 roofline 拐点（A100 上为 156 FLOP/byte）。

FlashAttention 对 attention 有帮助是因为 attention 严重受限于显存（$I \approx 1$--$60$ FLOP/byte）。通过将数据保留在 SRAM 中，它消除了显存瓶颈。

对 FFN 而言：瓶颈已经是 Tensor Core（而非显存带宽），所以减少显存流量没用。FFN 的收益来自量化（减小权重大小 $\to$ 提高算术强度）和更大的 Batch。

*复习：第 1 章与第 2 章（LLM 架构；系统基础）。*

### Q：解释在线 Softmax 技巧，以及它对 FlashAttention 为何不可或缺。

**答**：标准 Softmax 在计算任何输出之前都需要全局最大值 $m = \max_j x_j$——这要求先看到所有 $n$ 个 attention 分数，迫使物化完整的 $n \times n$ 矩阵。

在线 Softmax 技巧按顺序处理块，维护一个滚动的 $(m, \ell, O)$ 状态：

1. 处理新块 $\to$ 更新滚动最大值：$m_{\text{new}} = \max(m_{\text{old}}, \max(s_{\text{new}}))$
2. 重新缩放旧的求和：$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \text{new terms}$
3. 重新缩放输出：$O_{\text{new}} = \text{rescaled}(O_{\text{old}}) + \text{new contribution}$

这在数学上是完全精确的——没有任何近似。它使得逐块处理成为可能，每块都能放进 SRAM，永远不需要在显存中保留完整的 $n \times n$ 矩阵。

*复习：第 1 章与第 2 章（LLM 架构；系统基础）。*

## LoRA 与 PEFT 问题

### Q：LoRA 为什么有效？什么理论洞见支持低秩更新？

**答**：Aghajanyan 等人 [aghajanyan2020intrinsic] 表明，微调是在一个非常低的*内在维度（intrinsic dimensionality）*上进行的——某个微调任务的有效参数空间远远小于模型的总参数量。175B 模型在给定任务上的内在维度可能 $<$10,000。

LoRA 直接利用了这一点：通过将更新约束到秩 $r$（$W' = W + BA$，$B \in \mathbb{R}^{d \times r}$），它将每个权重矩阵的学习限制在一个 $r$ 维子空间内。由于真实的任务子空间是低维的，这几乎没有任何损失，同时将可训练参数减少 100--1000$\times$。

**直觉**：微调并不改变模型“知道什么”（满秩 $W$ 保持冻结）；它只调整*如何*为新任务组合现有知识——一个低秩扰动。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q：对 70B 模型，比较 QLoRA、全 LoRA、全量微调。何时选择每种？

**答**：

| 方法 | 显存 | GPU 数 | 质量 | 适用场景 |
| --- | --- | --- | --- | --- |
| 全量微调 | 560+ GB | 8+ A100 | 最佳 | 预算无上限；持续预训练 |
| LoRA ($r=16$) | 145 GB | 2 A100 | 95--98\% | 预算充足；一般微调 |
| QLoRA ($r=16$) | 44 GB | 1$\times$48GB | 93--96\% | 单 GPU；原型；资源受限 |

**决策树**：(1) 若任务需要深层知识变化 $\to$ 全量微调。(2) 若适配新风格/格式 $\to$ LoRA。(3) 若显存受限或快速迭代 $\to$ QLoRA。(4) 若秩很关键：从 $r=16$ 开始；如果训练 Loss 在全量微调水平之上停滞则增大。

*复习：第 1 章与第 10 章（LLM 架构；SFT 最佳实践）。*

### Q：什么是 DoRA，为何它优于标准 LoRA？

**答**：DoRA（权重分解低秩自适应，Weight-Decomposed Low-Rank Adaptation）将 $W$ 分解为幅度 $\|W\|$ 和方向 $W/\|W\|$，然后仅对方向分量应用 LoRA：

$$
W' = m \odot \frac{W + BA}{\|W + BA\|}
$$

其中 $m$（幅度）也是可训练的，但仅作为每个输出神经元上的简单标量。

**为何有效**：全量微调自然地独立更新幅度和方向。标准 LoRA 将二者耦合（低秩更新以受约束的方式同时改变两者）。DoRA 将它们解耦，让 LoRA 拥有与全量微调相同的“自由度”结构。结果：推理任务上提升 1\%--3\%，推理阶段无额外计算开销（合并适配器即可）。

*复习：第 1 章（LLM 架构与优化方法）。*

## 模型压缩问题

### Q：解释 AWQ。为什么保护 1\% 的权重就能保留 99\% 的质量？

**答**：AWQ（激活感知权重量化，Activation-Aware Weight Quantization）观察到权重的重要性高度不均匀：乘以大激活值的权重对输出的贡献不成比例地大。

关键洞见：$\|W \cdot X\|$ 同时取决于 $W$ 和 $X$。一个小权重乘以一个大激活，比一个大权重乘以一个接近零的激活更重要。

AWQ 识别出 top 1\% 的“显著”通道（在校准数据上始终具有较大激活幅度的通道），并通过缩放对其加以保护：量化前将显著通道乘以因子 $s > 1$（然后在激活中除以 $s$）。这降低了重要通道的相对量化误差。

结果：在 70B 模型上 4-bit 量化的质量损失 $<$1\%，因为 99\% 的非显著权重能容忍激进的量化。

*复习：第 1 章与第 2 章（LLM 架构；系统基础）。*

### Q：什么时候用 FP8、4-bit 量化、BF16？

**答**：

- **BF16**：训练（RLHF 中的 Policy 模型），当精度重要时。任何被 Gradient 更新的模型的默认选择。
- **FP8 (E4M3)**：H100 上使用 Transformer Engine 训练（2$\times$ 吞吐，$<$0.5\% 质量损失）。也用于 H100 上需要最大吞吐的推理。
- **INT8/FP8 推理**：RLHF 中的冻结模型（reference 模型、Reward 模型）——不被训练，所以降低精度是安全的。
- **4-bit (AWQ/GPTQ)**：大规模推理服务。部署中显存/质量权衡最佳。也用于 QLoRA 的基模。
- **2-bit**：实验性；显存极度受限的边缘部署。质量损失 5\%--10\%。

**规则**：推理时尽可能激进地量化，训练时保持 BF16（H100 上可用 FP8）。

*复习：第 2 章（LLM 的系统基础）。*

### Q：解释 NVIDIA 2:4 结构化稀疏。加速比和约束是什么？

**答**：2:4 稀疏意味着：每 4 个连续元素一组中，恰好有 2 个必须为零。这在权重层面强制执行。

**硬件支持**：A100/H100 的 Tensor Core 有专门的 2:4 稀疏 GEMM 指令，可以跳过零元素，无软件开销地实现恰好 **2$\times$ 吞吐**。

**约束**：你必须在这一特定模式下实现*恰好*50\% 的稀疏度。不能是 30\% 或 70\% 的稀疏度；不能是任意稀疏模式。剪枝必须遵守 4 元素组结构。

**如何实现**：训练之后（或微调期间），对每组 4 个权重，将幅度最小的 2 个置零。然后微调数百步以恢复质量。质量损失：大模型（70B+）通常 $<$1\%。

*复习：第 2 章（LLM 的系统基础）。*

## 专家混合（Mixture of Experts）问题

### Q：Mixtral 8x7B 总共有 47B 参数但每个 token 只激活 13B。请解释它如何工作，以及为何高效。

**答**：Mixtral 将每个 FFN 层替换为 8 个并行的专家 FFN（每个专家的 FFN 部分约 $\sim$7B 参数）。一个 router 网络为每个 token 选择 Top-2 专家。

**为什么总共 47B**：Attention 层是共享的（不复制）= $\sim$5B。FFN 专家：8 $\times$ $\sim$5.25B = 42B。总计：$\sim$47B。

**为什么 13B 激活**：每个 token 只有 2 个专家被激活。激活参数 = attention（$\sim$5B）+ 2 个 FFN 专家（$\sim$2 $\times$ 5.25B）$\approx$ 13B。

**为什么高效**：计算成本随*激活*参数（13B）扩展，与 13B 稠密模型相当。但容量（存储的知识）随*总*参数（47B）扩展，与更大的模型相当。结果：Mixtral 以 13B 的计算成本匹配 Llama-2 70B 的质量。

**显存成本**：仍需把全部 47B 参数装进显存（所有专家都加载），所以显存 = 47B 模型，但计算 = 13B 模型。

*复习：第 1 章（LLM 架构与优化方法）。*

### Q：MoE 中的负载均衡问题是什么，如何解决？

**答**：不加约束时，router 倾向于将大部分 token 路由到 1--2 个“热门”专家（强者愈强的正反馈）。这导致：

- 容量浪费：8 个专家中有 6 个未被使用，模型实际上缩小到 2 个专家的规模。
- 计算不均衡：如果每个专家在不同 GPU 上，热门专家会成为瓶颈，其他专家闲置。

**解决方案**：辅助负载均衡 Loss：$\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot p_i$，其中 $f_i$ = 路由到专家 $i$ 的 token 比例，$p_i$ = 专家 $i$ 的平均 router 概率。这会惩罚不均匀的分布。

**替代方案**：专家容量因子——对每个 Batch 中每个专家的最大 token 数做硬上限。溢出的 token 被丢弃或重新路由。

典型 $\alpha$：0.01--0.1（足够小以不损害主 Loss，足够大以防止坍缩）。

*复习：第 1 章（LLM 架构与优化方法）。*

## 训练中的多样性问题

### Q：如果一个 GRPO 组中 N 个响应全部相同会怎样？

**答**：如果全部 $N$ 个响应相同：所有 Reward $r_i$ 相等，因此 $\sigma_G = 0$，优势 $\hat{A}_i = (r_i - \mu_G)/\sigma_G$ 未定义（除以零）。实践中，实现把所有 $\hat{A}_i = 0$，意味着**零学习信号**——这一步被浪费。

**预防**：

1. **温度**：生成时使用 $\tau = 0.7$--$1.0$（不要贪心解码）。
2. **大 $N$**：$N=8$--$16$ 增加多样化响应的概率。
3. **重复拒绝**：DAPO 的做法——拒绝重复响应并重采样。
4. **频率惩罚**：生成时对重复的 n-gram 施加惩罚。
5. **监控**：跟踪每组唯一响应比例。若 $<$50\%，调高温度。

*复习：第 7 章（GRPO）。*

### Q：解释 RLHF 中的多样性—质量权衡。如何检测模式坍缩？

**答**：**权衡**：高多样性（高熵/高温度）= 多样但可能随机/低质量的响应。低多样性 = 一致但重复、被 Reward hack 的响应。

**检测模式坍缩**（训练期间均应监控）：

1. **响应熵**：计算每个 token 的熵 $H = -\sum p_i \log p_i$。若快速下降 $\to$ 坍缩。
2. **唯一 n-gram 比例**：同一 Prompt 不同响应间唯一 4-gram 的比例。健康：$>$0.6。
3. **Reward 分布宽度**：若 $\sigma(\text{rewards})$ 收缩到接近零 $\to$ 所有响应质量相同 $\to$ 可能完全一致。
4. **KL 散度**：若 $D_\text{KL}[\pi_\theta \| \pi_\text{ref}]$ 快速增长，Policy 正远离参考 $\to$ 通常朝向某个狭窄模式。
5. **长度直方图**：若所有响应收敛到相同长度 $\to$ 模板化行为。

**修复**：提高 KL 系数 $\beta$、增大熵奖励、提高采样温度，或回滚到更早的 checkpoint。

*复习：第 7 章与第 9 章（GRPO；Reward 模型训练）。*

## 推测解码（Speculative Decoding）问题

### Q：推测解码声称“无质量损失”。以不同方式生成 token 怎么能产生完全相同的输出分布？

**答**：接受/拒绝方案保证了分布等价性：

对每个 draft token $\hat{x}$，给定 draft 概率 $q(\hat{x})$ 与目标概率 $p(\hat{x})$：

- 以概率 $\min(1, p(\hat{x})/q(\hat{x}))$ 接受
- 拒绝时：从*残差分布* $\propto \max(0, p(x) - q(x))$ 中采样

这在数学上等价于直接从 $p$（目标）采样。证明草图：输出 token $x$ 的概率为 $q(x) \cdot \min(1, p(x)/q(x)) + P(\text{reject}) \cdot \frac{\max(0, p(x)-q(x))}{\sum_y \max(0, p(y)-q(y))} = p(x)$。

加速来自摊销：当 draft 质量好（接受率高）时，一次目标模型前向就能确认多个 token。无论 draft 质量如何，这一保证始终成立——糟糕的 draft 只是带来更低的加速比（更多拒绝），而不是更差的质量。

*复习：第 2 章（LLM 的系统基础）。*

### Q：比较 Medusa 和 Eagle 在推测解码上的差异。何时选择哪一个？

**答**：

**Medusa**：在目标模型上增加 $k$ 个并行预测头。每个头独立预测位置 $t+i$ 的 token。*优点*：无需独立模型，$<$1\% 显存开销。*缺点*：各头独立预测——无法让位置 $t+2$ 以 $t+1$ 的预测为条件。接受率：60\%--80\%。

**Eagle**：在目标模型隐状态上的轻量自回归解码器。Draft token 自回归生成（每个以前一个为条件）。*优点*：捕捉 token 间依赖 $\to$ 85\%--95\% 接受率。*缺点*：稍多显存（小型解码器）以及顺序的 draft 生成。

**选 Medusa 当**：显存极度紧张；集成简单；中等加速即可（2--2.5$\times$）。

**选 Eagle 当**：需要最大加速（3--4$\times$）；可承受额外的小模型；延迟敏感的单流生成。

**选 N-gram 当**：输出重复性强（代码、结构化数据）；零成本；无需训练。

*复习：第 2 章（LLM 的系统基础）。*

### Q：为什么推测解码在大 Batch 下*帮不上忙*？

**答**：在大 Batch（$\geq$64）下，自回归生成已经*计算高效*：权重读取成本在多个序列间摊销。算术强度接近 roofline 拐点。

推测解码会带来开销：

1. Draft 生成成本（即便是小模型，大 Batch 下也不是免费的）
2. 验证前向每个序列处理 $k$ 个额外 token（Batch $\times$ $k$ 个 token）
3. Draft 模型或 Medusa 头的显存
4. 被拒绝的 token 浪费算力

在 batch=1（延迟瓶颈、显存瓶颈）：推测把 1 token/步变成 3--4 token/步——巨大胜利。

在 batch=128（已经计算高效）：推测带来的额外 token 几乎不能提升吞吐，因为 GPU 已接近饱和。开销甚至可能*降低*吞吐。

**规则**：推测解码用于延迟（小 Batch）；批处理用于吞吐（大 Batch）。别把它们组合在一起。

*复习：第 2 章（LLM 的系统基础）。*

## Agent 强化学习问题

### Q：为什么标准 RLHF（单轮 PPO/DPO）对多步 Agent 失败？

**答**：标准 RLHF 优化单轮质量：给定 Prompt，生成一个好响应。多步 Agent 面临根本不同的挑战：

1. **信用分配**：在 50 步轨迹中，哪一步导致失败？单轮 Reward 把信用均匀地分配给整个响应；多步需要*每一步*的信用。
2. **稀疏 Reward**：仅在轨迹末尾才有成功/失败。PPO 的 GAE 假设有中间 Reward；缺失时，优势估计会有噪声。
3. **动作空间**：动作是结构化的工具调用（JSON），不仅是 token 序列。模型必须同时学习语法 + 语义 + 策略。
4. **非平稳性**：每个动作都会改变环境（工具输出修改状态）。每一步都有不同的“Prompt”，不像单轮那样输入是固定的。
5. **探索**：Agent 必须发现新颖的工具使用策略，而不仅仅是改写文本。

**解决方案**：轨迹级 GRPO（对完整轨迹排序）、过程奖励模型（Process Reward Model, PRM）（每一步反馈）、或对成功轨迹做过滤后的 SFT。

*复习：第 12 章（LLM Agent 训练）。*

### Q：解释 GRPO 如何适配到 Agent 训练。与单轮 GRPO 的关键区别是什么？

**答**：单轮 GRPO：对一个 Prompt 生成 $N$ 个响应，按 Reward 排序，计算优势。

**Agent GRPO** 的区别：

1. **生成单位**：完整的*轨迹*（10--100 步），而非单个响应。每条轨迹是组内的一个“样本”。
2. **Reward**：终止 Reward（任务成功/失败）或轨迹级 Reward（步 Reward 求和）。*不是*逐 token 的。
3. **掩码**：仅在 Agent 的输出（推理 + 工具调用）上计算 Policy Loss。把工具*输出*（环境响应）从 Gradient 计算中掩掉。
4. **组大小**：通常较小（$N=4$--8），因为轨迹昂贵（每条轨迹很多次前向）。
5. **KL 惩罚**：在每一步应用，防止每个决策点偏离 SFT Policy。
6. **长度归一化**：按 Agent *动作*数（不是 token 数）归一化，避免惩罚充分的推理。

*复习：第 7 章与第 12 章（GRPO；LLM Agent 训练）。*

### Q：比较 Agent 场景下的 STaR、Reflexion 和 ReAct。

**答**：

**STaR（自学推理器，Self-Taught Reasoner）**：生成推理链 $\to$ 按正确性筛选 $\to$ 在正确链上微调。*适用*：你有可验证任务（数学、代码），希望在不使用 RL 的情况下从基模引导出推理能力。

**Reflexion**：失败后生成自然语言反馈（“哪里出错了？”）$\to$ 把反思放进上下文重试。不更新权重。*适用*：推理时改进；训练算力有限；任务允许自我诊断。

**ReAct**：在结构化循环中交替进行 Reasoning（思考）+ Acting（工具调用）。*适用*：多步工具调用任务；需要透明性（推理轨迹可解释）；Agent 必须在思考与行动之间抉择。

**关键差异**：

|  | STaR | Reflexion | ReAct |
| --- | --- | --- | --- |
| 是否更新权重？ | 是 (SFT) | 否（上下文内） | 否（Prompt） |
| 多步？ | 否（单条推理链） | 是（重试循环） | 是（思考—行动循环） |
| 工具？ | 否 | 可选 | 是（必需） |
| 最适合 | 推理能力提升 | 错误恢复 | 工具增强任务 |

*复习：第 12 章与第 18 章（LLM Agent 训练；Agent 设计模式）。*

### Q：为什么研究型 Agent 偏好 GRPO 而非 PPO？

**答**：对于具有 20--100 步轨迹的研究型 Agent：

**PPO 需要一个 Value 模型**：$V(s_t)$ 必须预测从当前状态出发的期望总 Reward。对于研究场景（其中状态 = 128K token 的上下文，包括论文、代码和结果），训练一个精确的 Value 函数极其困难——“读过 3 篇论文并写了一部分代码”的价值难以预测。

**GRPO 完全避免了 Value 估计**：它对每个研究问题生成 $N$ 条完整轨迹，并把组内排名作为优势。无需预测中间价值——只需比较结果。

**其他原因**：

- 研究质量近似二元（好报告 vs. 差报告）——排序很自然。
- 轨迹长且昂贵；GRPO 的 $N=4$ 可承受；PPO 需要很多 Rollout 才能得到稳定的 Value 估计。
- 终止 Reward 稀疏；稀疏 Reward 下 GAE 给出的每步优势本来就很有噪声。

*复习：第 7 章与第 12 章（GRPO；LLM Agent 训练）。*

### Q：为编程 Agent 设计一个 Reward 函数。存在哪些 Reward hacking 风险？

**答**：**Reward 设计**：

$$
R = 0.5 \cdot R_{\text{tests}} + 0.2 \cdot R_{\text{quality}} + 0.2 \cdot R_{\text{efficiency}} + 0.1 \cdot R_{\text{safety}}
$$

- $R_{\text{tests}}$：单元测试通过比例（0--1）。可基于真值验证。
- $R_{\text{quality}}$：LLM 评判代码风格、文档、可维护性。
- $R_{\text{efficiency}}$：$\max(0, 1 - \text{steps}/30)$——快速完成给奖励。
- $R_{\text{safety}}$：无危险操作（rm -rf、沙箱外网络访问）。

**Reward hacking 风险**：

1. **硬编码输出**：Agent 学会直接打印预期测试输出而不真正计算。*修复*：随机化测试输入；在保留测试用例上测试。
2. **删除测试**：Agent 修改/删除失败的测试。*修复*：测试沙箱设为只读。
3. **平凡解**：Agent 写出能通过测试但不泛化的最小代码。*修复*：大型、多样化的测试集；基于属性的测试。
4. **效率作弊**：Agent 跳过推理步骤以最大化效率奖励。*修复*：先满足最低质量阈值，再给效率奖励。

*复习：第 9、12、19 章（Reward 模型训练；LLM Agent 训练；Agent 环境）。*

## 列表式 Reward 与高级 RM 问题

### Q：解释 Plackett-Luce 模型。它如何推广 Bradley-Terry？

**答**：Bradley-Terry 建模*成对*偏好：$P(y_1 \succ y_2) = \sigma(r(y_1) - r(y_2))$。

Plackett-Luce 把 $K$ 项的*完整排序*建模为顺序选择：

$$
P(\pi) = \prod_{i=1}^K \frac{e^{r(y_{\pi(i)})}}{\sum_{j=i}^K e^{r(y_{\pi(j)})}}
$$

解释：依次挑选剩余项中最好的。位置 1 = 对全部 $K$ 做 Softmax；位置 2 = 对剩余 $K-1$ 做 Softmax；以此类推。

**推广关系**：当 $K=2$ 时，PL 恰好退化为 BT：$P(y_1 \succ y_2) = \frac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))$。

**优势**：$K=8$ 项的排序提供 $\binom{8}{2} = 28$ 个隐式成对比较，外加相对差距信息——比单一成对样本丰富得多。

*复习：第 9 章（Reward 模型训练）。*

### Q：什么是过程奖励模型（PRM），它在什么情况下优于结果奖励模型（Outcome Reward Model, ORM）？

**答**：**ORM**：仅对最终输出打分。$r(x, y_{\text{final}})$ = 整个响应一个标量。

**PRM**：对每一*步*推理打分。$r(x, y_{\text{step } t})$ = 每一中间步一个标量。

**PRM 更好的场景**：

1. **长推理链**：10+ 步的数学题。ORM 无法判断哪一步错了；PRM 提供逐步信用分配。
2. **搜索/验证**：PRM 支持树搜索（对推理步做 beam search，剪掉步 Reward 低的分支）。
3. **训练信号密度**：PRM 每条轨迹给 $T$ 个 Reward（每步一个），而 ORM 只有一个 $\to$ 更低方差的优势估计。

**ORM 更好的场景**：任务短（单轮）；步骤边界不清晰；每步标注成本过高。

**PRM 标注**：可通过“Math-Shepherd”方法自动化：对每一步，从该点多次补全整个解。如果从步骤 $t$ 出发的补全成功而从步骤 $t+1$ 出发的补全失败，则步骤 $t+1$ 很可能是错的。

*复习：第 9 章与第 13 章（Reward 模型训练；大型推理模型的 RL）。*

## 大型推理模型的 RL 问题

### Q：DeepSeek-R1 在长推理链上训练，却为何不使用过程奖励模型？

**答**：DeepSeek-R1 仅使用**基于结果的 Reward**（准确率 + 格式），原因如下：

1. **可验证任务**：数学和代码有确定的真值答案。即便是长链，二元的准确率 Reward 也能提供足够信号。
2. **PRM 失效模式**：步骤级 Reward 模型自身会引入 Reward hacking——模型可学到产出“在 PRM 看来正确”但实际不正确的步骤。
3. **GRPO 的组归一化**：通过每个 Prompt 采样 $G$ 个补全并在组内归一化优势，GRPO 即使没有每步 Reward 也能自然提供关于哪些推理*策略*有效的相对信号。
4. **涌现的自我纠正**：在仅有结果 Reward 时，模型学会在链内自我纠正（“aha 时刻”），如果每步 Reward 对推理过程微观管理，这种行为将无法涌现。

**关键洞见**：任务领域的可验证性使 PRM 变得不必要——对于主观任务（创意写作），仅有结果 Reward 可能不够。

*复习：第 13 章（大型推理模型的 RL）。*

### Q：解释测试时计算扩展律（Test-Time Compute Scaling Law）及其对模型部署的启示

**答**：测试时计算扩展律表述为：

$$
\text{Accuracy}(C_{\text{train}}, C_{\text{test}}) \approx f(\alpha \log C_{\text{train}} + \beta \log C_{\text{test}})
$$

**启示**：

1. **算力等价**：在推理任务上，使用 64$\times$ 推理 token 的 7B 模型可以与使用 1$\times$ token 的 70B 模型相匹敌。
2. **自适应分配**：简单问题用短链（便宜）；困难问题用长链（贵）。平均成本低于始终使用大模型。
3. **部署灵活性**：不再部署一个大模型，而是部署一个较小的推理模型，并按查询难度对推理算力进行扩展。
4. **收益递减**：对数关系意味着测试时计算翻倍带来的准确率提升递减——训练与推理算力之间存在一个最优分配。

**“过度思考”失败模式**：非常长的链可能因错误累积和 Attention 稀释而*降低*准确率。最优链长取决于问题难度。

*复习：第 13 章（大型推理模型的 RL）。*

### Q：MCTS（蒙特卡洛树搜索，Monte Carlo Tree Search）如何应用于 LLM 推理？

**答**：用于推理的 MCTS 将每个部分解视为一个树节点：

**每次迭代四个阶段**：

1. **选择**：用 UCB 从根开始导航：$\text{UCB}(s) = Q(s) + c\sqrt{\frac{\ln N(\text{parent})}{N(s)}}$
2. **扩展**：从 LLM 生成新的推理步（子节点）
3. **模拟**：从新节点完成解（Rollout）
4. **反向传播**：根据最终正确性更新路径上的 Q 值

**与博弈 MCTS 的关键差异**：

- **分支因子**：推理具有巨大的分支因子（任何下一句都可能）。实际实现使用 LLM 的 top-k 输出来限制分支。
- **Value 函数**：训练好的 PRM 估计部分解质量，替代随机 Rollout。
- **步骤粒度**：每“步”可能是一句话、一个方程或一次逻辑推断——粒度选择很关键。

**使用案例**：AlphaProof（数学奥林匹克），以及据推测 OpenAI o1/o3 的隐藏推理。

*复习：第 13 章（大型推理模型的 RL）。*

### Q：比较通过蒸馏 vs 直接 RL 来打造小型推理模型

**答**：

**蒸馏**（DeepSeek-R1-Distill 路线）：

- 从大模型（R1-671B）生成推理链
- 在这些链上对小模型做 SFT
- 结果：小模型模仿大模型的推理*格式*
- 优点：便宜（仅 SFT）。缺点：可能学到表层模式而非真正推理。

**直接对小模型做 RL**：

- 用 GRPO/PPO 在可验证 Reward 上训练小模型
- 模型发现自己的推理策略
- 优点：真实能力。缺点：算力消耗大得多；对于非常小的模型可能不收敛。

**经验发现**：R1-Distill-7B（蒸馏）在大多数基准上优于 direct-RL-7B。大模型的推理链提供了非常强的监督信号，单 SFT 就具有竞争力。然而，蒸馏模型对真正新颖的问题类型泛化较差。

**最佳实践**：先蒸馏（便宜的基线），再可选地在蒸馏模型上跑 RL 以获得进一步收益（Qwen 使用的“蒸馏 + RL”组合）。

*复习：第 13 章（大型推理模型的 RL）。*

## LLM 评估问题

### Q：推导 ELO 评分更新规则，并解释 Chatbot Arena 为何使用它

**答**：**ELO 推导**：

玩家 A 对 B 的期望得分：$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$（逻辑斯蒂模型）。

对局后，实际得分 $S_A \in \{0, 0.5, 1\}$：$R_A' = R_A + K(S_A - E_A)$

$K$ 因子控制更新幅度（$K$ 越大对近期结果反应越敏感）。

**为什么 Chatbot Arena 使用 ELO**：

1. **传递性**：若 A 胜 B 且 B 胜 C，ELO 预测 A 胜 C。这从成对比较给出全序。
2. **在线更新**：新增模型无需重新评估所有对。每次新比较都增量更新评分。
3. **置信度**：在 $N$ 次比较后，评分不确定性以 $O(1/\sqrt{N})$ 收缩。标准误：$\text{SE} \approx \frac{400}{\sqrt{N}}$。
4. **捕捉人类偏好**：真实用户无需阐明标准即可提供诚实偏好。聚合揭示了真实模型质量。

**Chatbot Arena 细节**：使用 Bradley-Terry MLE（收敛时等价于 ELO），并配合 Bootstrap 置信区间。风格控制 ELO 去除长度/格式偏倚。

*复习：第 14 章（LLM 评估）。*

### Q：代码生成的 pass@k 指标是什么，为什么无偏估计很重要？

**答**：**pass@k** = 生成的 $k$ 个样本中至少有一个通过全部测试用例的概率。

**朴素（有偏）估计器**：生成 $k$ 个样本，看是否有任意一个通过。问题：高方差、昂贵（每题需要很多次试验）。

**无偏估计器**（Chen 等，2021）：生成 $n \geq k$ 个样本，统计其中通过的数量 $c$：

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

**无偏为何重要**：

1. 一次性生成 $n$ 个样本（例如 $n=200$），从同一批样本算出 pass@1、pass@10、pass@100
2. 无需将整个评估重复 $k$ 次
3. 统计上精确（组合论证：不含任何正确样本的 $k$-子集占比）
4. 通过对数空间数值稳定计算：$\text{pass@}k = 1 - \exp\left(\sum_{i=0}^{k-1} \log(n-c-i) - \log(n-i)\right)$

**直觉**：若 50/200 个样本通过（$c=50$，$n=200$），pass@1 $\approx 0.25$，pass@10 $\approx 0.94$。该估计器统计大小为 $k$ 的抽取中至少包含一个成功的比例。

*复习：第 14 章与第 19 章（LLM 评估；Agent 环境）。*

### Q：如何检测和缓解基准污染（Benchmark Contamination）？

**答**：**污染**：训练数据中包含基准测试样例（或其近似改写），使分数虚高。

**检测方法**：

1. **N-gram 重叠**：检查训练数据是否与测试条目精确或近似匹配。8-gram 重叠且覆盖率 $>$80\% 即可疑。
2. **Canary 字符串**：在测试集中插入唯一标识符；检查模型是否能复现它们。
3. **改写基准**：创建语义等价但文本不同的基准版本。准确率大幅下降表明存在记忆。
4. **时序分析**：比较模型在训练截止前 vs 截止后测试条目上的表现。在旧条目上异常高表现暗示污染。
5. **成员推断**：统计检验判断特定样例是否出现在训练数据中。

**缓解**：

- **动态基准**：定期生成新测试条目（LiveCodeBench、Chatbot Arena）
- **私有测试集**：保持测试条目保密（LMSYS）
- **训练期去污染**：从训练数据中删除检测到的重叠
- **报告污染分析**：在基准分数旁披露重叠度指标

*复习：第 14 章（LLM 评估）。*

### Q：解释 LLM-as-Judge 中的位置偏倚（Position Bias），以及如何缓解

**答**：**位置偏倚**：当用 LLM 评判两个响应（A vs B）时，模型系统性地偏好特定位置的响应（通常是第一个或最后一个），与质量无关。

**经验幅度**：GPT-4 表现出 10\%--15\% 的位置偏倚；Claude 为 5\%--10\%。较小的模型偏倚更大。

**缓解策略**：

1. **位置交换**：每对评判两次（A-B 与 B-A）。最终决定 = 多数。若不一致，则标为“平局”。这能消除系统性位置偏倚，但成本翻倍。
2. **多评委合议**：使用 3 个及以上不同模型作为评委。多数投票减少单一模型的偏倚。
3. **参考引导**：提供评分量表或参考答案。评委按量表独立为每个响应打分，再比较分数（完全消除成对比较）。
4. **校准 Prompt**：加入明确指令：“呈现顺序是随机的，不应影响你的判断。”

**其他偏倚**：冗长偏倚（偏好更长响应）、自我增强偏倚（模型偏好自身输出）、权威偏倚（顺从于引用来源的响应）。

*复习：第 14 章（LLM 评估）。*

## Agent 记忆问题

### Q：比较 Agent 的四种记忆类型，以及各自在何时至关重要

**答**：

| 类型 | 存储内容 | 访问方式 | 何时关键 |
| --- | --- | --- | --- |
| 工作记忆 | 当前上下文/草稿板 | 始终在上下文中 | 复杂多步推理 |
| 情景记忆 | 过往经验 | 按相似度检索 | 从过去错误中学习 |
| 语义记忆 | 事实与知识 | 按概念检索 | 领域特定任务 |
| 程序性记忆 | 技能与模式 | 按任务类型触发 | 重复性工具调用 |

**关键洞见**：它们并非独立——彼此交互。情景记忆喂养语义记忆（从事件泛化为事实）。程序性记忆被情景反馈细化（学习哪些工具序列有效）。工作记忆协调对所有其他类型的检索。

**MemGPT 类比**：工作记忆 = 热（在上下文），情景/语义记忆 = 温（向量库），程序性记忆 = 冷（归档的 Policy）。Agent 自身决定何时换入/换出信息。

*复习：第 16 章（Agent 记忆系统）。*

### Q：记忆检索中的时间衰减如何工作，为何重要？

**答**：**时间衰减**在检索时给较旧记忆降权：

$$
\text{score}(m) = \alpha \cdot \text{similarity}(q, m) + (1 - \alpha) \cdot \text{recency}(m)
$$

其中 $\text{recency}(m) = e^{-\lambda \cdot \Delta t}$，$\Delta t$ = 自上次访问以来的时间。

**为什么重要**：

1. **相关性衰减**：用户偏好会变化。6 个月前的偏好可能已过时。
2. **矛盾消解**：当新旧信息冲突时，近期偏好自然倾向当前真相。
3. **检索效率**：没有衰减时记忆无界增长，检索会返回越来越不相关的远古条目。
4. **认知合理性**：人类也会遗忘——近期事件更易获取。这与间隔效应（spacing effect）相符。

**基于访问的刷新**：当记忆被检索并使用时，其时间戳更新（类似 LRU 缓存）。频繁访问的记忆无论创建日期如何都保持“新鲜”。

**衰减率调优**：$\lambda$ 取决于领域。客户服务：高衰减（偏好变化快）。法律/医疗：低衰减（事实持久）。可通过 RL 学习。

*复习：第 16 章（Agent 记忆系统）。*

### Q：如何用 RL 训练记忆操作？

**答**：记忆操作（写入/读取/更新/删除）可作为 Agent MDP 中的动作：

**建模**：

- **状态**：当前上下文 + 记忆状态
- **动作**：标准动作 + `memory_write(key/value)`、`memory_read(query)`、`memory_delete(key)`
- **Reward**：任务成功（记忆是否有帮助？）+ 记忆效率惩罚（读取越少越好）

**RL 学到什么**：

1. **存什么**：重要信息（API key / 用户偏好）vs 临时细节
2. **何时检索**：回答领域问题前 vs 一般聊天中
3. **压缩策略**：何时对旧记忆做摘要 vs 原样保留
4. **遗忘**：何时旧信息陈旧应被删除

**训练信号**：反事实——“如果 Agent 没有存储/检索这条记忆，它还会成功吗？”通过轨迹比较实现：记忆使用得当的轨迹获得更高 Reward。

**挑战**：延迟 Reward——现在存储的信息可能 100 步后才有用。需要长期信用分配（高 $\lambda$ 的 GAE）。

*复习：第 12 章与第 16 章（LLM Agent 训练；Agent 记忆系统）。*

## Agent 编排问题

### Q：解释上下文预算问题，以及如何用动态分配解决

**答**：**问题**：Agent 有上下文窗口 $L$ 个 token，但需要为以下部分留空间：

$$
C = S + M + T + H + R \leq L
$$

其中 $S$ = 系统 Prompt，$M$ = 记忆/检索上下文，$T$ = 工具描述，$H$ = 对话历史，$R$ = 为响应预留。

随着对话增长，$H$ 增大并挤掉其他组件。

**动态分配策略**：

1. **固定下限**：$S_{\min}$、$R_{\min}$ 不可妥协
2. **自适应历史**：当 $H > H_{\max}$ 时摘要旧轮次。保留最近 $k$ 轮原样；其余做摘要。
3. **按需工具**：仅包含与当前查询相关的工具描述（而非全部 50 个）。用分类器或 Embedding 相似度选 top-$k$ 工具。
4. **惰性记忆**：仅在需要时（分析查询之后）检索记忆，而非预先加载。

**溢出处理**：当压缩后总量仍超过 $L$：

- 丢弃最不重要的工具描述
- 激进地把历史摘要到每轮一句
- 减少记忆槽
- 若仍超：截断并向用户提示

**飞行前检查**：调用 LLM 之前始终先统计 token 数。绝不要在推理时才发现溢出。

*复习：第 17 章（Agent Harness——上下文管理与编排）。*

### Q：比较 ReAct 与 Plan-and-Execute 两种编排模式

**答**：

**ReAct**（Reason + Act）：

- 循环：Thought $\to$ Action $\to$ Observation $\to$ Thought $\to$ $\ldots$
- 每一步基于此前所有观察决定下一动作
- **优点**：自适应——可根据工具输出改变方向
- **缺点**：短视——无前置规划；可能陷入循环；每次 LLM 调用都看到完整历史（昂贵）

**Plan-and-Execute**：

- 阶段 1：生成完整计划（步骤列表）
- 阶段 2：顺序执行步骤（更简单的执行器；可能用更便宜的模型）
- 阶段 3：执行失败则重新规划
- **优点**：高效（一次规划比每步推理便宜）；独立步骤可并行
- **缺点**：计划脆弱——若早期步骤失败，整个计划可能失效。重规划增加延迟。

**何时选择**：

- ReAct：探索性任务；未知环境；每步结果决定下一步的任务
- Plan-and-Execute：定义清晰的任务；已知工具集；可并行子任务；成本敏感的部署
- **混合**：高层做规划，在每个规划步内用 ReAct（LangGraph 推荐模式）

*复习：第 17 章与第 18 章（Agent Harness；Agent 设计模式）。*

### Q：如何检测并防止 Agent 执行中的无限循环？

**答**：当 Agent 重复同一动作期待不同结果时，会进入无限循环。

**检测方法**：

1. **最大迭代守卫**：硬上限（例如 25 步）。简单但会在真正长任务上丢失进展。
2. **动作哈希窗口**：对最近 $k$ 个（动作/观察）对求哈希。若当前哈希与最近 $w$ 步内的哈希匹配则检测到循环。
3. **语义相似度**：对近期动作做 Embedding。若相邻动作余弦相似度超过阈值（$>$0.95）则可能卡住。
4. **进度监控**：定义任务特定的进度指标。若 $N$ 步内无进展则干预。

**恢复策略**：

1. **注入提示**：加入系统消息：“你似乎在重复动作。尝试不同方法。”
2. **强制不同动作**：下一步在动作空间中屏蔽重复动作。
3. **升级**：返回部分结果给用户并请求指导。
4. **回溯**：重置到循环开始前的 checkpoint 并尝试备选路径。

**最佳实践**：组合最大迭代（安全网）+ 基于哈希的检测（早期干预）+ 优雅升级（维护用户信任）。

*复习：第 17 章与第 18 章（Agent Harness；Agent 设计模式）。*

## MCP 协议问题

### Q：解释 MCP 的 N+M 架构，以及它对 Agent 生态系统为何重要

**答**：**N$\times$M 问题**：没有 MCP 时，$N$ 个 Agent 框架必须各自实现与 $M$ 个工具的集成 = 总共 $N \times M$ 次集成。新增一个工具需要 $N$ 次实现。

**MCP 的 N+M 方案**：标准化接口。每个 Agent 实现一个 MCP 客户端（共 $N$ 个）。每个工具实现一个 MCP 服务器（共 $M$ 个）。总集成数 = $N + M$。

**具体示例**：5 个 Agent 框架（LangChain/AutoGen/CrewAI/Claude/custom）$\times$ 20 个工具（GitHub/Slack/DB/filesystem/$\ldots$）= 没有 MCP 时 100 次集成。有 MCP 时：5 个客户端 + 20 个服务器 = 25 次实现。

**为何重要**：

1. **工具复用**：构建一次工具服务器；可被任何兼容 MCP 的 Agent 使用
2. **Agent 可移植性**：从 Claude 切换到自定义 Agent 而无需重写工具集成
3. **生态成长**：降低新增工具的门槛激励社区构建更多
4. **可组合性**：运行时将多个服务器动态连接到一个 Agent

**类比**：USB 标准化了外设连接。USB 之前：每种设备都有专有接口。USB 之后：一个口适配所有。MCP 对 Agent–工具连接做了同样的事。

*复习：第 20 章（模型上下文协议）。*

### Q：MCP 的四个核心原语是什么，何时分别使用？

**答**：

| 原语 | 方向 | 用途 | 示例 |
| --- | --- | --- | --- |
| Tools | 客户端 $\to$ 服务器 | 执行动作 | `create_issue`；`query_db` |
| Resources | 客户端 $\to$ 服务器 | 读取数据 | 文件内容；数据库记录 |
| Prompts | 客户端 $\to$ 服务器 | 获取模板 | “总结这个 PR”模板 |
| Sampling | 服务器 $\to$ 客户端 | 请求 LLM 生成 | 服务器请求 LLM 分类 |

**关键区分**：

- **Tools vs Resources**：Tools 有*副作用*（创建/修改/删除）。Resources 是*只读*的。这对安全很重要——Agent 可自由读取 Resources，但调用 Tools 必须获得批准。
- **Sampling** 方向反转：通常是客户端（Agent）调用服务器（工具）。在 Sampling 中，服务器请求客户端的 LLM 帮忙。用例：代码分析服务器需要 LLM 解释一段代码。
- **Prompts** 是元数据（可复用模板），不是执行。它们帮助 Agent 构造更好的工具调用。

*复习：第 20 章（模型上下文协议）。*

## Agent 通信（A2A）问题

### Q：Google 的 A2A 协议与 MCP 有何不同，何时两者都需要？

**答**：**核心区分**：

- **MCP**：Agent $\leftrightarrow$ 工具（具有定义 schema 的结构化函数调用）
- **A2A**：Agent $\leftrightarrow$ Agent（不透明任务委派——你不知道对方 Agent 如何工作）

**A2A 关键概念**：

- **Agent Cards**：描述 Agent 能力的 JSON（类似简历）。发现机制。
- **不透明执行**：请求者看不到受托方的内部推理。只是发送任务并获取结果。
- **任务生命周期**：submitted $\to$ working $\to$ completed/failed（通过 SSE 提供流式更新）

**何时两者都需要**：

1. 编排 Agent 用 **A2A** 将“研究这个话题”委派给一个研究 Agent
2. 研究 Agent 用 **MCP** 调用网页搜索、文件读取和数据库工具
3. 结果经 A2A 流回编排 Agent

**架构**：A2A 位于*Agent 间*层；MCP 位于*Agent–工具*层。完整系统两者并用：A2A 用于 Agent 间协调，MCP 用于每个 Agent 的工具访问。

*复习：第 20 章与第 22 章（MCP；Agent 到 Agent 通信）。*

### Q：什么是合同网协议（Contract Net Protocol），它如何应用于 LLM Agent？

**答**：**合同网协议**（Contract Net Protocol，CNP）是来自分布式 AI 的任务分配机制：

**步骤**：

1. **宣告**：管理者向所有可用 Agent 广播任务描述
2. **投标**：Agent 评估自身能力并提交投标（置信度；预估成本；预估时间）
3. **授标**：管理者按标准（能力/成本/可用性）选出最佳投标
4. **执行**：中标 Agent 执行任务
5. **回报**：Agent 将结果报告给管理者

**对 LLM Agent 而言**：

- **投标 = 自我评估**：每个 Agent LLM 评估“我能把这个任务做好吗？”并给出置信度分数。这需要校准过的自我认知。
- **专业化涌现**：代码 Agent 在代码任务上出高价；研究 Agent 在研究任务上出高价。无需中心路由逻辑。
- **负载均衡**：若一个 Agent 忙（预估时间长），其他 Agent 胜出。
- **失败处理**：若中标 Agent 失败，则向剩余 Agent 重新宣告（自动故障切换）。

**LLM 的局限**：LLM 常常高估自身能力（幻觉置信度）。投标应纳入历史记录（相似任务的历史成功率），而不仅是自报置信度。

*复习：第 22 章与第 23 章（A2A；多 Agent 系统）。*

## 多 Agent 系统问题

### Q：比较 LLM 多 Agent 系统的中心化 vs 去中心化架构

**答**：

**中心化（Supervisor）**：

- 一个编排 LLM 将任务路由给专家 worker
- 控制流清晰；易于调试（检查 supervisor 决策）
- 单点故障；supervisor 成为 token 瓶颈
- 最适合：定义清晰的工作流；小型 Agent 团队（3--5 个 Agent）

**去中心化（点对点）**：

- Agent 直接通信；无中央协调者
- 弹性（无单点故障）；可水平扩展
- 难以调试（涌现行为）；可能出现冲突和死锁
- 通信无结构时按 $O(n^2)$ 扩展
- 最适合：弹性系统；大量 Agent 群体；期望涌现行为的创造性任务

**混合（层次化）**：带子管理者的树形结构。结合优点：组内本地自治 + 顶层全局协调。通信按 $O(n \log n)$ 扩展。

**决策框架**：需要可预测性和可审计性时用中心化。需要弹性和创造性时用去中心化。大型（$>$10 个 Agent）系统用层次化。

*复习：第 23 章（多 Agent 系统）。*

### Q：什么是 CTDE，为什么它对训练多 Agent LLM 系统重要？

**答**：**CTDE** = 中心化训练；去中心化执行（Centralized Training; Decentralized Execution）。

**问题**：多 Agent RL 中，每个 Agent 的环境都是非平稳的（其他 Agent 同时在改变各自的 Policy）。这使独立训练不稳定。

**CTDE 方案**：

- **训练时**：一个中心化 critic 可访问所有 Agent 的观察和动作：$V(s_1, s_2, \ldots, s_n, a_1, a_2, \ldots, a_n)$。这通过把非平稳性从 Value 函数中剔除来稳定训练。
- **执行时**：每个 Agent 仅基于自身观察行动：$a_i = \pi_i(o_i)$。推理时没有通信开销。

**对 LLM Agent 而言**：中心化 critic 可以是一个 Reward 模型，评估所有 Agent 的*联合*输出（例如，Agent 团队是否产出了一个正确的软件系统？），而每个 Agent 通过反事实信用分配被训练以最大化其对团队 Reward 的贡献。

**实际挑战**：完整 CTDE 要求所有 Agent 在共享状态下同时训练——对 LLM 而言代价昂贵。近似方法：分轮训练 Agent（冻结其余 Agent 训练一个），或使用具有周期同步的种群训练。

*复习：第 23 章（多 Agent 系统）。*

## Agent 开发框架问题

### Q：比较 LangGraph、AutoGen、CrewAI 用于构建多 Agent 系统

**答**：

| 维度 | LangGraph | AutoGen / CrewAI |
| --- | --- | --- |
| 编排 | 显式状态图（节点 + 边） | 隐式（基于对话/角色） |
| 状态管理 | TypedDict schema；checkpoint | 以对话历史为状态 |
| 多 Agent | 带条件路由的图 | GroupChat / Crew |
| 调试 | 图可视化；步骤回放 | 聊天日志 |
| 人机协同（Human-in-the-Loop） | 一等公民（中断节点） | 通过审批工具 |
| 生产部署 | LangGraph Cloud；持久化 | 有限（AutoGen）；成长中（CrewAI） |
| 学习曲线 | 高（图概念） | 低（AutoGen）；极低（CrewAI） |

**选 LangGraph 当**：需要细粒度控制；复杂条件流；带持久化和人机协同的生产部署。

**选 AutoGen 当**：多 Agent 对话的快速原型；代码执行 Agent；研究实验。

**选 CrewAI 当**：简单的基于角色的团队；顺序任务执行；快速演示；代码量最小。

**都不选（自定义）当**：需要最大性能/控制；不希望被框架锁定；或有非标准的编排模式。

*复习：第 24 章（Agent 开发框架）。*

### Q：如何在生产中测试和评估一个 Agent 系统？

**答**：Agent 测试遵循一个**测试金字塔**：

**第 1 层——单元测试**（快；多）：

- 独立测试单个工具（mock LLM；验证工具逻辑）
- 测试 Prompt 模板（给定上下文；验证正确的 Prompt 构造）
- 测试解析器（给定 LLM 输出；验证正确提取）

**第 2 层——集成测试**（中等速度）：

- 用确定性输入测试完整的 Agent 循环
- “黄金轨迹”测试：必须能复现的已知良好执行轨迹
- 工具链测试：验证多工具序列端到端可工作

**第 3 层——行为测试**（慢；少）：

- Agent 是否遵守安全约束？（对抗性输入）
- 它是否在合适时请求澄清？
- 它是否保持在 token/成本预算内？

**生产评估**：

- **A/B 测试**：将 5\% 流量路由到新 Agent 版本
- **影子模式**：与旧 Agent 并行运行新 Agent；比较输出但不对外服务
- **LLM-as-judge**：自动化打分评估 Agent 响应质量
- **用户满意度**：点赞/点踩；任务完成率；解决时间

**关键指标**：**任务成功率**（Task Success Rate, TSR）——Agent 在无人干预下正确完成任务的比例。

*复习：第 14 章与第 24 章（LLM 评估；Agent 开发框架）。*

## Agent 环境问题

### Q：为网页浏览 Agent 环境设计一个 Reward 函数

**答**：对于 WebArena 风格任务（例如“找到 12 月 15 日从 NYC 到 SF 的最便宜航班”）：

**稀疏 Reward**（简单但难以学习）：

$$
r = \begin{cases} 1 & \text{if final page/state matches ground truth} \\ 0 & \text{otherwise} \end{cases}
$$

**密集 Reward**（更利于训练；设计更难）：

1. **进度 Reward**：每个让 Agent 更接近目标的页面 $+0.1$（用与目标状态的文本相似度衡量）
2. **效率惩罚**：每个动作 $-0.01$（鼓励更短轨迹）
3. **里程碑 Reward**：到达中间目标（如导航到航班搜索页）$+0.3$
4. **无效动作惩罚**：产生错误的动作（404；表单校验失败）$-0.05$

**基于势能的 reward shaping**（保留最优 Policy）：

$$
r_{\text{shaped}}(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s)
$$

其中 $\Phi(s) = -\text{min\_steps\_to\_goal}(s)$（由启发式或学习到的 Value 函数估计）。

**挑战**：部分可观察（无法总是判断是否更接近目标）；随机环境（页面内容会变）；Reward hacking（Agent 找到满足 Reward 但不满足用户意图的捷径）。

*复习：第 12 章与第 19 章（LLM Agent 训练；Agent 环境）。*

### Q：是什么让 SWE-bench 成为一个特别具有挑战性的 Agent 基准？

**答**：SWE-bench 在来自流行 Python 仓库的真实 GitHub issue 上测试 Agent：

**为什么难**：

1. **仓库级上下文**：Agent 必须理解 10 万+ 行的代码库。无法装入上下文窗口——必须探索、搜索和导航。
2. **规范不充分的任务**：Issue 由人类带着隐含上下文写就。Agent 必须推断真正需要什么。
3. **跨文件编辑**：解决方案通常跨越多个文件，并有级联依赖。
4. **测试验证**：必须通过现有测试 *以及*验证修复的新测试。
5. **无人辅助**：不像 HumanEval（单函数），SWE-bench 需要完整的软件工程工作流：读 issue $\to$ 探索代码 $\to$ 定位 bug $\to$ 实现修复 $\to$ 验证。

**当前 SOTA**（2024--2025）：最佳 Agent 解决 SWE-bench Verified（精选子集）的约 $\sim$50\%。完整 SWE-bench：约 $\sim$30\%。

**对训练的关键洞见**：SWE-bench 揭示了“编程能力”（编写正确函数）与“软件工程能力”（理解系统；导航代码库；做最小改动）之间的差距。在 SWE-bench 风格环境上的 RL 训练教会 Agent 探索和规划策略，而不仅是代码生成。

*复习：第 19 章（Agent 环境与基准）。*

## Agent UI 框架问题

### Q：比较 Agent 的聊天式 vs 画布式 UI 范式

**答**：

**聊天式**（ChatGPT；Claude 默认）：

- 线性消息流：用户 $\to$ 助手 $\to$ 用户 $\to$ $\ldots$
- **优点**：UX 熟悉；适合探索和问答；易于实现
- **缺点**：生成的产物（代码/文档）埋没在对话中。难以针对特定产物迭代。长对话中上下文易丢失。

**画布/Artifact 式**（Claude Artifacts；ChatGPT Canvas；Cursor）：

- 侧边栏展示生成内容；聊天栏发送指令
- Agent 可以对持久化的产物创建、编辑和迭代
- **优点**：产物独立于聊天持久化。用户可直接编辑。具有版本历史。
- **缺点**：UI 更复杂；需要 artifact 类型检测；对两个面板的流式输出实现更难。

**何时选择**：

- 聊天：头脑风暴；问答；快速任务；移动端界面
- 画布：代码生成；文档写作；数据分析——任何需要迭代、有持久化输出的任务
- **混合**（多数现代 UI）：默认聊天；检测到代码/文档/可视化输出时自动提升到画布

**对 Agent 训练而言**：UI 范式影响 Reward 信号。画布式 UI 提供显式的编辑反馈（用户修改 artifact），可用于在线学习。

*复习：第 25 章（Agent UI 框架）。*

### Q：如何为人机协同（Human-in-the-Loop）的 Agent 系统设计审批关卡？

**答**：审批关卡在关键点暂停 Agent 执行以供人工审核。

**三层模型**：

1. **自动批准**（无关卡）：安全可逆动作。读取操作；搜索；计算。
2. **通知**（软关卡）：可能有影响但可恢复。发送邮件；创建草稿；修改文件。Agent 继续，但用户被通知，可撤销。
3. **阻断**（硬关卡）：不可逆或高风险。删数据；汇款；发布内容；执行有副作用的代码。Agent 必须等待显式批准。

**设计原则**：

- **最小化打断**：关卡过多 = 用户放弃 Agent。三层模型让大多数动作通行，同时拦截危险动作。
- **显示上下文**：审批关卡展示：什么动作；为什么（Agent 的推理）；将改变什么；如何撤销。
- **批量审批**：如果 Agent 需要 5 次文件写入，一起呈现而不是逐个。
- **超时处理**：如果用户在 $T$ 分钟内未响应，则重试通知、按安全默认继续或优雅中止。
- **从审批中学习**：跟踪审批/拒绝模式。若用户总是批准某类动作，考虑自动提升其级别。

**实现**：工具注解（MCP 的 `destructiveHint` 和 `readOnlyHint`）驱动关卡的自动分配。自定义规则可根据上下文覆盖。

*复习：第 17 章与第 25 章（Agent Harness；Agent UI 框架）。*

## RAG 与 Agent 化 RAG 问题

### Q：解释互逆排名融合（Reciprocal Rank Fusion, RRF），以及它为何对混合检索有效

**答**：RRF 合并多个检索系统的排名，无需对分数做校准：

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
$$

其中 $r(d)$ 是文档 $d$ 在检索器 $r$ 中的排名，$k=60$ 是防止高排名文档主导的常数。

**为何有效**：

1. **无需分数归一化**：BM25 分数无界；稠密相似度在 $[-1, 1]$。RRF 只使用排名，使二者可直接比较。
2. **对离群值鲁棒**：单个检索器给出异常高分也不会主导，因为即便是排名 1，$1/(k+1) \approx 0.016$。
3. **互补信号**：BM25 捕捉精确关键词匹配；稠密检索捕捉语义相似度。两者都排名高的文档得到加成。

**示例**：文档 $d$ 在 BM25 中排名 3，在稠密中排名 7。RRF 分数 $= 1/(60+3) + 1/(60+7) = 0.0159 + 0.0149 = 0.0308$。在某一边排名 1 但在另一边排名 100 的文档得到 $1/61 + 1/160 = 0.0226$——尽管有排名 1，但更低。

**实践中**：混合（BM25 + 稠密 + RRF）在 85\%+ 的基准上优于单独使用任一种。

*复习：第 15 章（检索增强生成）。*

### Q：什么是 Agent 化 RAG，它与标准 RAG 有何不同？

**答**：**标准 RAG** 遵循固定流水线：查询 $\to$ 检索 $\to$ 生成。它没有能力：

- 判断是否根本需要检索
- 评估检索到的文档是否足够
- 在检索失败时重新表述查询
- 整合多步检索的信息

**Agent 化 RAG** 将检索视为 Agent MDP 中的一个*动作*：

- **是否检索决策**：Agent 评估它是否已经知道答案（对训练数据内的事实性问题跳过检索）
- **查询规划**：把复杂问题分解为子查询（“X 发生在哪一年？” + “那时谁是总统？”）
- **自我评估**：检索后评估相关性。若不足，则重新表述查询或尝试不同来源。
- **多跳推理**：检索 $\to$ 推理 $\to$ 识别知识缺口 $\to$ 再次检索
- **来源路由**：将查询路由到合适的知识库（时事查网络；公司信息查内部文档；编程查代码搜索）

**关键架构差异**：标准 RAG = 确定性流水线。Agent 化 RAG = 带条件转移的状态机（LangGraph 的 retrieve/grade/rewrite/generate 节点模式）。

**权衡**：Agent 化 RAG 在复杂查询上更准确，但增加延迟（多次 LLM 调用用于路由/评分）。简单事实查询用标准 RAG；多跳或歧义查询用 Agent 化 RAG。

*复习：第 15 章与第 17 章（RAG；Agent Harness）。*

### Q：比较 Self-RAG 与 CRAG 两种改进检索质量的方法

**答**：

**Self-RAG**（Asai 等，2023）：

- 将特殊的*反思 token*训练进 LLM 词表
- 推理时，模型输出 [Retrieve]、[IsRel]、[IsSup]、[IsUse] 等 token
- 模型决定*何时*检索（并非每个查询都需要）
- 检索后，模型自我评分：检索段落是否相关？我的答案是否从中得出？
- **训练**：在用 GPT-4 反思标签增强的数据上做 SFT
- **优点**：单一模型处理一切。**缺点**：需要定制训练。

**CRAG**（纠正式 RAG，Corrective RAG，Yan 等，2024）：

- 使用一个轻量的*检索评估器*（独立模型）来给检索到的文档评分
- 根据置信度有三种动作：`Correct`（按原样使用）、`Ambiguous`（用网络搜索增强）、`Incorrect`（丢弃；回退到网络）
- 增加*知识精炼*步骤：仅从检索文档中抽取相关句子
- **优点**：可与任何冻结 LLM 配合。**缺点**：需要额外的评估模型；增加延迟。

**关键差异**：Self-RAG 把检索决策嵌入 LLM 自身（需要训练）。CRAG 是包裹在任意 LLM 周围的流水线方法（无需训练）。Self-RAG 更优雅；CRAG 在现有模型的生产环境中更实用。

*复习：第 15 章（检索增强生成）。*

### Q：什么是“中间迷失”（lost-in-the-middle）问题，如何缓解？

**答**：**问题**：当检索到的上下文很长（很多段落）时，LLM 不成比例地关注上下文的*开头*和*结尾*，而忽略中间的信息。如果答案在 10 段中的第 5 段，模型可能漏掉。

**经验证据**：Liu 等（2023）表明，对于 20 文档检索，相关文档位于位置 5--15 时，相比位置 1--3，准确率下降 15\%--20\%。

**缓解策略**：

1. **重排序并截断**：用 cross-encoder 重排序，然后只保留 top-3 最相关段落（更少 = 更少中间迷失）。
2. **策略性排序**：把最相关的段落同时放在上下文的开头和结尾，低相关的放中间。
3. **上下文压缩**：插入前将每段摘要到 1--2 句。文本越少 = 位置偏倚越小。
4. **Map-Reduce**：独立处理每段（map），再合并答案（reduce）。完全消除位置效应。
5. **引用 Prompt**：要求模型引用其使用的段落。这迫使 Attention 关注所有段落。
6. **减小块大小**：更小的块意味着覆盖答案所需的总块数更少。

**最佳实践**：检索很多（20+），重排序到 top 3--5，按相关性排序（最佳在前）。这在大多数用例中完全绕开此问题。

*复习：第 15 章（检索增强生成）。*
