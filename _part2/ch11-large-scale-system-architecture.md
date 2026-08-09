---
layout: home
title: 大规模系统架构与基础设施
permalink: /part2/ch11-large-scale-system-architecture.html
---

用基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）训练 LLM，既是一道算法题，也是一道系统工程题。与标准 SFT 不同——后者只涉及单一模型、单次前向-反向传播、扩展规律也已被充分理解——RLHF 需要*同时*加载*多个模型*（policy、reference、reward model、value head），通过复杂的 rollout-打分-训练循环协调起来，并分布到数十乃至数百块 GPU 上。本章覆盖让大规模 RLHF 训练成为可能的系统级细节：显存预算、并行策略（Data、Tensor、Pipeline、Sequence 及其组合）、生成瓶颈、解耦式架构、权重同步、容错以及生产监控。

## 4 模型显存挑战


![70B PPO 显存预算：RLHF 所需的四个模型及其显存占用。合计 1470--1560GB。朴素方案最少需要 19--20 块 A100-80GB。使用 ZeRO-3 时：8 个节点即可装下。]({{ site.baseurl }}/figures/fig_031_fig31.png)

> **显存预算现实核对——70B BF16**
>
> | Policy 权重（BF16） | 140 GB |
> | --- | --- |
> | FP32 master 权重 | 280 GB |
> | Adam 优化器（m + v，FP32） | 560 GB |
> | 梯度（BF16） | 140 GB |
> | Reference model | 140 GB（INT8 时 70 GB） |
> | Reward model | 140 GB（INT8 时 70 GB） |
> | 激活值（batch 128，seq 2048） | 50--100 GB |
> | 生成用 KV cache | 20--60 GB |
> | **合计** | **1470--1560 GB** |
>
> $\div$ 80 GB/GPU = **至少 19--20 块 A100**（尚未计入任何并行开销）。

## 并行策略详解

训练大语言模型必须把计算分布到许多 GPU 上。可以沿*多个本质不同的维度*并行，每种都有各自的取舍。本节用数学公式、示意图和实践指引详细介绍每种策略。


![四种并行策略概览。生产系统通常会同时组合其中 2--3 种。]({{ site.baseurl }}/figures/fig_032_fig32.png)

### 数据并行（Data Parallelism, DP）与分布式数据并行（Distributed Data Parallelism, DDP）

数据并行是最简单、也最常见的分布式训练形式 [[193]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-li2020pytorch)]。每块 GPU 持有模型的*完整副本*，处理不同的 mini-batch，并同步梯度。

**原始 DP（PyTorch `DataParallel`）。**

单进程方案：一块 “主” GPU 负责分发输入、汇集输出、广播梯度。受 GIL 以及到主 GPU 的 PCIe 带宽限制。

**分布式数据并行（DDP，`DistributedDataParallel`）。**

多进程方案：每块 GPU 各自运行一个进程。在反向计算继续进行的同时，通过 ring-AllReduce [[194]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-sergeev2018horovod)] 在后台同步梯度。


![DDP：每块 GPU 持有完整的模型副本并处理不同的 batch。梯度通过 ring AllReduce 取平均，与反向计算重叠。]({{ site.baseurl }}/figures/fig_033_ddp.png)

**DDP 的关键特性**：

- **显存**：每块 GPU 都要存完整的模型 + 优化器 + 梯度。70B BF16 模型约 $\sim$560 GB/GPU——如不做显存优化根本无法承受。
- **通信**：每步对梯度张量做一次 AllReduce。大小 = 模型参数量 $\times$ 2 字节（BF16）。Ring AllReduce 开销：每 GPU 传输 $2 \cdot \frac{N-1}{N} \cdot M$ 字节。
- **扩展性**：在 $\sim$64 块 GPU 以内接近线性。超过之后通信开始占主导。
- **梯度分桶**：DDP 把参数按桶分组（默认 25 MB），只要某个桶的梯度就绪就启动 AllReduce——让通信与反向计算重叠。

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")  # NCCL 用于 GPU 间通信
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank],
            gradient_as_bucket_view=True,    # 显存优化
            static_graph=True)               # 开启通信优化
```

> **DP vs DDP —— 总是用 DDP**
>
> PyTorch 旧版的 `DataParallel`（DP）**绝不应**用于 LLM 训练：
>
> - 单进程，受 Python GIL 限制
> - 所有梯度都汇集到 GPU 0（瓶颈）
> - 即便在单节点上也比 DDP 慢 2--3$\times$
> - 无法跨机扩展
>
> DDP 是*底线*并行策略。对于 $>$7B 的 LLM，应优先使用 FSDP/ZeRO。

### 张量并行（Tensor Parallelism, TP）

张量并行（Megatron-LM 风格 [[195]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shoeybi2019megatron)]）把*单个权重矩阵*切到多块 GPU 上。每块 GPU 计算部分结果，再通过 AllReduce 合并。

**列并行（Column-Parallel）线性层。**

权重矩阵 $W \in \mathbb{R}^{d \times h}$ 按列切到 $T$ 块 GPU 上：
$$
W = [W_0 \;\mid\; W_1 \;\mid\; \cdots \;\mid\; W_{T-1}], \quad W_i \in \mathbb{R}^{d \times h/T}
$$
每块 GPU $i$ 独立计算 $Y_i = XW_i$（无需通信）。输出沿 hidden 维度被切开。

**行并行（Row-Parallel）线性层。**

权重矩阵按行切：$W = [W_0; W_1; \ldots; W_{T-1}]$，其中 $W_i \in \mathbb{R}^{d/T \times h}$。输入 $X$ 也必须切开。每块 GPU 计算一个部分和，再通过 **AllReduce** 得到最终输出。


![列并行线性层（TP=2）。权重按列切开；每块 GPU 独立计算 $XW_i$。MLP 将其与行并行层配对，从而避免多余的 AllReduce。]({{ site.baseurl }}/figures/fig_034_tp-column.png)

**TP 下的 Transformer Block。**

在 Transformer 层中，Megatron-LM 按以下方式应用 TP：

1. **MLP**：第一层线性（$h \to 4h$）列并行，第二层线性（$4h \to h$）行并行。行并行层之后做一次 AllReduce。
2. **Attention**：$Q$、$K$、$V$ 投影做列并行（按 head 切到多块 GPU）。输出投影做行并行。输出投影之后做一次 AllReduce。
3. **合计**：每个 Transformer 层 2 次 AllReduce（attention 一次，MLP 一次）。


![单个 Transformer block 中的张量并行通信模式。每层需要两次 AllReduce 操作（红色标注）——attention 之后一次，MLP 之后一次。]({{ site.baseurl }}/figures/fig_035_tp-transformer.png)

> **为什么 TP 必须限制在节点内**
>
> 每个 Transformer 层需要 2 次 AllReduce 操作（上文标注为 $f$ 和 $g$）。对于 80 层的 70B 模型，每次前向就是 160 次 AllReduce（算上反向共 320 次）。在 NVLink 速率（600 GB/s）下，每次 AllReduce 耗时 $<$0.5 ms。但在 InfiniBand（50 GB/s）上，同一操作要 $\sim$4 ms，总开销 160 $\times$ 4 = 640 ms——*比计算本身还长*。
>
> **规则**：TP 度 $\leq$ 节点内 GPU 数（通常 TP $\leq$ 8）。跨节点扩展请用 DP/FSDP。

> **TP 度的选取**
>
> - **TP=1**：不做张量并行。模型可放进单块 GPU（BF16 下通常 $\leq$ 13B）。
> - **TP=2**：最小切分。适合在 2 块 GPU 上做 13--34B 推理。开销低（$<$5%）。
> - **TP=4**：34--70B 推理的标准配置。开销 8--12%。
> - **TP=8**：占满整节点。70B+ 训练的必备配置。开销 12--18%。
> - **TP$>$8**：跨节点 TP。极少使用——仅在 PP 不足以承载的 200B+ 模型上才会用。开销 30--50%。
>
> **重要**：attention head 数必须能被 TP 度整除。例如 LLaMA-70B（64 个 head），合法的 TP 取值为 1、2、4、8、16、32、64。

### 序列并行（Sequence Parallelism, SP）

序列并行（Sequence Parallelism, SP） [[196]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-korthikanti2023reducing)] 解决的是张量并行单独无法消除的一个显存瓶颈：LayerNorm 和 Dropout 层中的**激活值显存**。

**问题。**

TP 把权重显存切到了多块 GPU 上。但 LayerNorm 和 Dropout 作用在*完整*的 hidden 维上，并在每块 GPU 上都重复存在。它们的激活值（反向传播需要）占用与 $b \times s \times d$ 成正比的显存——每块 GPU 上一样多，TP 并不会把它降下来。

**解法。**

对那些不需要跨 GPU 通信的操作（LayerNorm、Dropout、残差连接），按*序列维度*切分。每块 GPU 对这些操作只处理 $s/T$ 长的序列切片，仅在真正需要的地方（attention、线性层）再 gather 出完整序列。


![序列并行通过沿序列维度切分，降低 LayerNorm/Dropout 的激活显存。通信（AllGather/ReduceScatter）取代了标准 TP 中的 AllReduce——总传输字节数不变，但显存得以节省。]({{ site.baseurl }}/figures/fig_036_seq-parallel.png)

> **SP 通信是 “免费” 的**
>
> 标准 TP 在每个子层后做 AllReduce，等价于 ReduceScatter + AllGather。SP 只是*重排*了这些原语：
>
> - 不带 SP 的 TP：AllReduce（= ReduceScatter + AllGather）$\rightarrow$ 所有 GPU 上数据相同 $\rightarrow$ 在完整张量上算 LayerNorm（浪费）。
> - 带 SP 的 TP：ReduceScatter $\rightarrow$ 每块 GPU 拿到 $1/T$ 的序列 $\rightarrow$ 在部分张量上算 LayerNorm $\rightarrow$ 进入下一个 TP 层前再 AllGather。
>
> 总通信量完全相同！SP 是**零额外通信开销**的纯显存优化，使用 TP 时应当始终开启。

**SP 的显存节省（70B 模型，TP=8，batch=4，seq=2048）**：
$$
\text{激活值节省} = (T-1) \times b \times s \times d \times n_\text{layers} \times 2\text{ bytes} = 7 \times 4 \times 2048 \times 8192 \times 80 \times 2 \approx \textbf{59 GB/GPU}
$$

### 流水线并行（Pipeline Parallelism, PP）

流水线并行把模型按层*纵向*切分，把连续的层组指派给不同的设备（stage）。激活值在 stage 间向前流动，梯度向后流动。

**气泡（Bubble）问题。**

朴素流水线执行会产生 “气泡”——某个 stage 等待前一 stage 的输入或后一 stage 的梯度时的空闲时间：


![流水线气泡对比。左：朴素流水线只有一个 micro-batch，75% 时间空闲。右：$M=4$ 个 micro-batch 的 GPipe 大幅缩小气泡。当 $M \gg P$ 时，气泡占比趋近于 0。]({{ site.baseurl }}/figures/fig_037_pipeline-bubble.png)

**气泡占比公式。**

对于 $P$ 个流水线 stage、每步 $M$ 个 micro-batch：
$$
\text{气泡占比} = \frac{P - 1}{P + M - 1} \approx \frac{P-1}{M} \quad \text{（当 } M \gg P\text{ 时）}
$$

要让气泡开销 $<$10%，需要 $M \geq 10 \cdot (P-1)$。PP=4 时：至少需要 30 个 micro-batch。

**流水线调度。**


**流水线调度策略**
| **调度** | **气泡** | **显存** | **特性** |
| --- | --- | --- | --- |
| GPipe | $\frac{P-1}{M+P-1}$ | $M \times$ 激活值 | 简单；先全部前向再全部反向 [[197]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-huang2019gpipe)] |
| 1F1B | $\frac{P-1}{M+P-1}$ | $P \times$ 激活值 | 交错；稳态显存有界 [[198]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-narayanan2019pipedream)] |
| Interleaved 1F1B | $\frac{P-1}{M \cdot V + P - 1}$ | $P \times$ 激活值 | 虚拟 stage（$V$）；进一步减小气泡 [[199]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-narayanan2021efficient)] |
| Zero-Bubble（ZB-H1） | $\approx 0$ | $P \times$ 激活值 | 把反向拆成 B 和 W 两个阶段 [[200]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-qi2023zerobubble)] |

> **1F1B：生产标准**
>
> **1F1B**（one-forward-one-backward）调度 [[198]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-narayanan2019pipedream)] 被大多数生产系统采用（Megatron-LM [[199]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-narayanan2021efficient)]、DeepSpeed [[201]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rajbhandari2020zero)]）：
>
> **预热**：前向先填满流水线（P-1 个 micro-batch）。
>
> **稳态**：每个时间槽里交替执行一次前向和一次反向。这把激活值峰值显存限制在 $P$ 个 micro-batch（GPipe 是 $M$ 个）。
>
> **收尾**：剩余的反向把流水线排空。
>
> **显存优势**：GPipe 必须同时保存*全部* $M$ 个 micro-batch 的激活值。1F1B 在稳态下只保存 $P$ 组激活值——当 $M = 32$ 而 $P = 4$ 时至关重要。

**PP 中的通信。**

与 TP（AllReduce）不同，PP 只需要在相邻 stage 之间做激活值的**点对点（point-to-point）**通信：
$$
\text{每次传输的数据} = b_\text{micro} \times s \times d \times 2\text{ bytes (BF16)}
$$
若 micro-batch=4、seq=2048、$d$=8192：每次传输 $4 \times 2048 \times 8192 \times 2 = 128$ MB。InfiniBand 50 GB/s 时：每次 2.6 ms——相对每个 stage 的计算量来说很小。

**负载均衡。**

并非所有层的计算量都相等：

- **Embedding 层**：非常便宜（查表）。
- **Transformer block**：计算量均匀。
- **最末的 LM head**：中等（词表投影是一次大型矩阵乘法）。

把更多 Transformer 层分配给中间 stage，第一/最后 stage 少放一些，以平衡计算。

### 完全分片数据并行（Fully Sharded Data Parallelism, FSDP / ZeRO-3）

FSDP [[202]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-zhao2023pytorch)]（PyTorch）和 ZeRO-3 [[201]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rajbhandari2020zero)]（DeepSpeed）解决了 DDP 固有的显存重复问题：不再让每块 GPU 都保存完整的参数、梯度和优化器状态，而是每块 GPU 只拥有 $1/N$ 的切片，需要时即时重建完整张量。


![FSDP 把所有模型状态切片到多块 GPU 上。每块 GPU 拥有 $1/N$ 的参数、优化器状态和梯度。在每层计算前，通过 AllGather 按需重建完整参数。]({{ site.baseurl }}/figures/fig_038_fsdp.png)

**FSDP 每层的执行流程：**

1. **前向**：AllGather 参数 $\rightarrow$ 计算 $\rightarrow$ 丢弃非自有的分片。
2. **反向**：再次 AllGather 参数 $\rightarrow$ 计算梯度 $\rightarrow$ ReduceScatter 梯度（每块 GPU 拿到自己的梯度分片）$\rightarrow$ 丢弃非自有的参数分片。
3. **优化器更新**：每块 GPU 仅用自己持有的梯度分片和优化器状态更新自己拥有的分片。


**显存对比：DDP vs FSDP/ZeRO 各阶段（70B 模型，8 GPU）。基线：BF16 参数（140 GB）+ BF16 梯度（140 GB）+ FP32 master+m+v（840 GB）= 每 GPU 1120 GB。**
| **策略** | **分片对象** | **每 GPU 显存** | **通信** |
| --- | --- | --- | --- |
| DDP（不分片） | 无 | 1120 GB $\times$ | AllReduce（仅梯度） |
| ZeRO-1 | 优化器状态 | 385 GB $\times$ | AllReduce（梯度） |
| ZeRO-2 | 优化器 + 梯度 | 368 GB $\times$ | AllReduce（梯度） |
| ZeRO-3 / FSDP | 全部 | **140 GB** ✓ | AllGather + ReduceScatter（每层） |

```python
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 用 FSDP 包装模型
auto_wrap = partial(transformer_auto_wrap_policy,
                    transformer_layer_cls={LlamaDecoderLayer})
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap,  # 对每个 transformer 层都做包装
    use_orig_params=True,        # torch.compile 兼容所需
    limit_all_gathers=True,      # 限制显存峰值（同一时刻只允许 1 个 AllGather）
    forward_prefetch=True,       # 当前层计算时预取下一层参数
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 反向时预取
)
```

> **FSDP 的通信量**
>
> FSDP 每步的通信量是 DDP 的 **3$\times$**：
>
> - DDP：1 次梯度 AllReduce = 环上合计 $2M$ 字节（其中 $M$ = 模型字节数）。
> - FSDP：2 次 AllGather（前向 + 反向）+ 1 次 ReduceScatter = $3M$ 字节。
>
> 这就是显存与通信的取舍。FSDP 值得使用的场景是：(a) 模型在 DDP 下无法装进 GPU 显存；或 (b) 通信与计算重叠得很好（现代框架可达到 70--90% 重叠）。

### 3D 并行：组合多种策略

大规模生产系统（70B+）会同时组合 TP、PP 和 DP/FSDP：


![16 块 GPU 上的 3D 并行布局：TP=4（每个方框内，走 NVLink）、PP=2（橙色箭头，stage 之间）、DP=2（红色箭头，梯度同步）。每个维度利用通信层级中不同的一层。]({{ site.baseurl }}/figures/fig_039_3d-parallel.png)

> **生产配方：64 块 A100-80GB（8 节点）上的 70B**
>
> **节点内**（NVLink 600GB/s）：生成阶段用 TP=8；训练阶段在节点内使用 FSDP。\
>
> **节点间**（InfiniBand 400Gb/s）：跨节点 FSDP（8 路数据并行）。\
>
> **结果**：每块 GPU 持有 $\sim$70GB。Policy 权重在前向/反向时按层 gather。\
>
> **流水线并行**：仅当模型超过 100B+ 且 TP+ZeRO 装不下时才用。会引入复杂度（气泡开销 10--20%）和调度困难。
>
> **决策流程**：
>
> 1. 模型能装进 1 块 GPU 吗？$\rightarrow$ 用 DDP。
> 2. 用 FSDP 能装进 1 个节点吗？$\rightarrow$ 用 FSDP（ZeRO-3）。
> 3. 用 TP+FSDP 能装进 1 个节点吗？$\rightarrow$ 节点内用 TP + 节点间用 FSDP。
> 4. 仍然装不下？$\rightarrow$ 跨节点再加 PP。这是最后的手段。


**并行策略对比小结**
| **策略** | **切分对象** | **通信** | **扩展上限** | **开销** | **使用时机** |
| --- | --- | --- | --- | --- | --- |
| DP/DDP | Batch | AllReduce（梯度） | $\sim$64 GPU | 5--10% | 模型可装进 1 GPU |
| FSDP | 参数+优化器+梯度 | AllGather+RS | 数百 GPU | 10--20% | $>$13B 时的默认 |
| TP | 权重矩阵 | AllReduce（每层 2 次） | 8 GPU（1 节点） | 12--18% | 大模型推理+训练 |
| SP | 激活（序列维） | 复用 TP 通信 | 同 TP | $\approx$0% 额外 | 与 TP 一起常开 |
| PP | 按层切 stage | 点对点 | $\sim$16 stage | 15--30% | 仅 100B+ 模型 |

## 生成瓶颈：定量分析

> **Roofline 分析：为什么生成是显存带宽瓶颈**
>
> **A100 参数**：312 TFLOPS（BF16 tensor core），2 TB/s HBM 带宽。
>
> **Roofline 拐点**：$312\text{T} / 2\text{T} = 156$ FLOP/byte。低于 156 FLOP/byte 的操作受*显存带宽*约束。
>
> **自回归生成**：每生成一个 token，需要读完所有权重（70B 模型为 140GB），并执行 $2 \times 70\text{B} = 140\text{G}$ FLOPs（batch=1 时）。
>
> **算术强度**：$140\text{G FLOP} / 140\text{GB} = 1$ FLOP/byte。比 roofline 低了 $156\times$！
>
> **利用率**：$1/156 = 0.6\%$ 的峰值 FLOPS。GPU 有 99.4% 的时间在等显存读。
>
> **Token 速率**：$2\text{TB/s} / 140\text{GB} = 14.3$ tokens/秒（单流，batch=1）。
>
> **生成 512 token**：$512 / 14.3 = 35.8$ 秒每条响应（batch=1，TP=1）。
>
> **Batch 化能改善**：Batch=64、TP=4 $\rightarrow$ 只读一次权重，并行生成 64 个 token。算术强度：$64 \times 1 = 64$ FLOP/byte。好一些，但仍在 roofline 之下！


**70B 模型在各种配置下的生成吞吐（512 token）**
| **配置** | **Batch** | **每 batch 用时** | **Tok/s/GPU** | **说明** |
| --- | --- | --- | --- | --- |
| TP=1, batch=1 | 1 | 36s | 14 | 基线，最差情况 |
| TP=4, batch=1 | 1 | 9s | 57 | 生成上 TP 线性扩展 |
| TP=4, batch=32 | 32 | 15s | 1092 | 接近最佳 batch |
| TP=4, batch=128, vLLM | 128 | 45s | 1456 | 连续 batching |
| TP=4, batch=128, INT8 | 128 | 25s | 2621 | 带宽减半 |

**优化栈**（累乘加速比）：

1. **vLLM + PagedAttention** [[138]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-kwon2023efficient)]（2--4$\times$）：消除 KV cache 碎片，使更大 batch 成为可能
2. **连续 batching（continuous batching）** [[203]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yu2022orca)]（1.5--2$\times$）：不必等最长序列结束；一旦有序列结束就插入新的
3. **推测解码（Speculative decoding）** [[124]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-leviathan2023fast)]（2--3$\times$）：小 draft 模型一次猜 5 个 token，大模型一次前向就能验证。平均接受 3--4 个。
4. **生成用 INT8/FP8 权重**（2$\times$）：带宽需求减半。因为我们在做采样（不是为训练算精确 logits），质量损失极小。
5. **CUDA graphs**（1.1--1.3$\times$）：消除固定形状操作的 kernel 启动开销
6. **Prefix caching**（共享前缀 prompt 下 1.5$\times$）：不重新计算 system prompt 的 KV cache

```python
# 生产环境的 vLLM 生成配置
from vllm import LLM, SamplingParams

engine = LLM(
    model="./policy_checkpoint",
    tensor_parallel_size=4,           # 每个实例 TP=4
    gpu_memory_utilization=0.92,      # 给 KV cache 留余量
    max_num_batched_tokens=16384,     # 在飞 token 上限
    max_num_seqs=256,                 # 并发序列上限
    dtype="bfloat16",
    enable_prefix_caching=True,       # 缓存 system prompt 的 KV
    speculative_model="./draft_1B",   # 推测解码
    num_speculative_tokens=5,
    block_size=16,                    # PagedAttention block 大小
    swap_space=4,                     # 用于抢占的 swap 空间（GB）
)

# 为 RLHF batch 生成响应
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=512,
    logprobs=1,  # PPO 的 ratio 计算需要 log-prob
)
outputs = engine.generate(prompts, sampling_params)
# 提取：responses 以及每个 token 的 log_probs（PPO/GRPO 所需）
```

## 解耦式架构：生产级设计

诸如 DeepSpeed-Chat [[204]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yao2023deepspeedchat)] 与 OpenRLHF [[205]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hu2024openrlhf)] 这样的生产级 RLHF 系统采用**解耦式架构**，将生成、打分、训练拆成三个可独立扩展的集群。


![解耦式 RLHF 架构。每个集群针对自己的负载做优化。打分后的 rollout 先在经验缓冲（experience buffer）中累积，再被训练消费。]({{ site.baseurl }}/figures/fig_040_fig40.png)

> **为什么要解耦？**
>
> **生成**受显存带宽约束（需要高速 HBM，浪费算力）。\
>
> **训练**受算力约束（需要 tensor core，反向时浪费带宽）。\
>
> **同一硬件无法同时优化两者**：若把所有事情放在一起，要么生成时浪费算力，要么训练时浪费带宽。解耦让每个集群用最优的硬件/配置。
>
> **实际好处**：
>
> - 生成和训练可以独立扩展
> - 生成集群无状态 $\rightarrow$ 容错极简
> - gen(step $N+1$) 与 train(step $N$) 可重叠 $\rightarrow$ 30--40% 加速
> - 不同精度：生成用 INT8（带宽优先），训练用 BF16（精度优先）

## 权重同步策略

| **策略** | **滞后** | **带宽** | **质量影响** |
| --- | --- | --- | --- |
| 同步（每步） | 0 步 | 140 GB/步 | 完美但太慢 |
| 周期性（每 50 步） | 平均 25 步 | 摊销后 2.8 GB/步 | 质量损失 $<$2% |
| Delta 压缩（INT8） | 平均 25 步 | 0.4 GB/步 | 质量损失 $<$3% |
| 异步流式 | 5--10 步 | 14 GB/步（后台） | 质量损失 $<$1% |

> **为什么 PPO/GRPO 能容忍滞后**
>
> PPO 的裁剪目标本来就是*为*离策略数据设计的！裁剪区间 $[1-\epsilon, 1+\epsilon]$ 限制了陈旧数据的影响。当滞后在 10--50 步时：
>
> - 每步 policy 大约变化 $\sim$0.1--1%（合理学习率下）
> - 50 步累计：$\sim$5% 的 policy 漂移
> - PPO 裁剪本身就能处理高达 20% 的漂移
> - 经验上：50 步滞后的质量损失 $<$2%
>
> **带宽算账**：70B BF16 = 140GB。InfiniBand 400Gb/s = 50GB/s $\rightarrow$ 完整同步 2.8s。配合 delta 压缩：$<$0.5s。异步同步 = 免费（在后台跑）。

## 显存优化技术

| **ZeRO 阶段** | **切分对象** | **每 GPU 显存（70B，8 GPU）** |
| --- | --- | --- |
| 无（数据并行） | 无（完整副本） | 每 GPU 560GB（不可行） |
| ZeRO-1 | 仅优化器状态 | 175GB |
| ZeRO-2 | 优化器状态 + 梯度 | 105GB |
| ZeRO-3（FSDP） | 优化器 + 梯度 + 参数 | **70GB（A100-80GB 装得下！）** |

**其他技术**：

- **梯度检查点（Gradient checkpointing）** [[206]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chen2016training)]：不存全部激活，反向时重算。节省约 60% 激活显存，代价是约 33% 的额外计算。可选择性：只对 attention 层（显存重）做 checkpoint，保留 FFN 激活（重算成本高）。
- **混合精度（Mixed precision）** [[207]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-micikevicius2018mixed)]：前向用 BF16（2 字节/参数），优化器状态用 FP32（m、v 各 4 字节）。master 权重用 FP32 累积。
- **CPU offloading**（ZeRO-Infinity [[208]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rajbhandari2021zeroinfinity)]）：把优化器状态放到 CPU RAM。显存省一半，但慢 2--3$\times$（PCIe 64GB/s 瓶颈）。
- **激活 offloading**：前向时把激活搬到 CPU，反向时再搬回来。仅在显存确实紧张时使用。
- **FlashAttention** [[17]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-dao2022flashattention), [61]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-dao2023flashattention2)]：attention 显存从 O($n^2$) 降到 O($n$)。快 2--4$\times$，长序列时显存节省巨大。

### FlashAttention 对 RLHF 的影响

> **为什么 FlashAttention 对 RLHF 很重要**
>
> RLHF 涉及生成长序列（rollout）并对其训练。没有 FlashAttention 时：
>
> - 一条 4K token、32 个 head 的序列，仅 attention 矩阵就要 $\sim$4 GB
> - 这严重限制了 PPO/GRPO 训练时的 batch 大小
> - 对 attention 激活做梯度检查点代价昂贵
>
> 有了 FlashAttention：
>
> - attention 显存为 $O(n)$——由 $Q, K, V, O$ 张量主导
> - 同等 GPU 显存下可承载更长的 rollout（8K--32K token）
> - 反向从 $Q, K, V$ 重算 attention 分块（不存 $n^2$ 矩阵）
> - 这是长上下文 RLHF（如推理模型）得以成立的关键

> **FlashAttention 与梯度检查点**
>
> FlashAttention 的反向在原地从 $Q, K, V$（已存）重算 attention 分块。这意味着 FlashAttention 已经*隐式实现*了 $O(n^2)$ attention 矩阵的激活重算。无需再单独对 attention 层做 checkpoint——这样做反而会多余地重算 $Q, K, V$。

```python
# 70B RLHF 训练的 DeepSpeed ZeRO-3 配置
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,                    # 通信与计算重叠
        "contiguous_gradients": True,            # 更好的显存布局
        "reduce_scatter": True,                  # 比 allreduce 更高效
        "reduce_bucket_size": 5e7,               # 每桶 5000 万参数
        "prefetch_bucket_size": 5e7,             # 预取下一个桶
        "param_persistence_threshold": 1e5,      # 小参数在所有 GPU 上常驻
        "offload_optimizer": {"device": "cpu", "pin_memory": True},  # CPU offload
        "sub_group_size": 1e9,                   # 减少碎片
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": True,
}
```

## 大规模容错

> **硬件故障现实**
>
> **单卡 MTBF**：约 10,000 小时。\
> **512 卡集群 MTBF**：$10000/512 \approx 20$ 小时。算上软件/网络故障，**实际 4--8 小时**。\
> **多日训练**：会遇到 5--15 次故障。没有容错机制，一次故障就会让整次训练报废。

**生产级容错栈**：

1. **探测**：NCCL 超时（60s）、GPU 心跳（10s）、NVML 健康监控、ECC 错误计数。
2. **Checkpoint**：每 50--100 步异步保存。非阻塞（后台线程）。保存内容：模型权重、优化器状态（Adam m/v）、调度器状态、RNG 状态、KL 系数、replay buffer。保留最近 3 个 checkpoint。70B 用时约 30s（并行写入 NVMe）。
3. **恢复**：(a) 生成集群无状态，直接重启并加载最新权重。(b) 训练集群：加载 checkpoint，剔除故障节点重建 NCCL 进程组，重新分配 FSDP 分片，从最近 checkpoint 续训。
4. **弹性训练**：Torch Elastic / Kubernetes 自动扩缩。几分钟内替换故障节点。训练以 $N-1$ 块 GPU 暂时继续。
5. **预防**：训练前做 GPU 健康预筛（跑 GEMM 压测）。热备机待命。双轨 InfiniBand 等冗余网络通路。

## 端到端延迟拆解


![不重叠（单体式）。解耦后：生成与训练重叠，实际加速 1.4$\times$。]({{ site.baseurl }}/figures/fig_041_fig41.png)

| **阶段** | **用时（70B）** | **受限于** | **优化手段** |
| --- | --- | --- | --- |
| 生成（128$\times$512 tok） | 30--45s | 显存带宽 | vLLM、推测解码、INT8 |
| Reward 打分 | 5--8s | 算力（批量前向） | INT8 RM、batch=128 |
| Reference log-prob | 4--6s | 算力（批量前向） | INT8 ref，或 LoRA（免费） |
| PPO 更新（4 epoch） | 8--12s | 算力（反向） | FSDP、FlashAttention |
| 权重同步 | 0--3s | 网络（异步） | Delta 压缩、异步 |
| **合计（单体式）** | **50--75s** |  |  |
| **合计（解耦+重叠）** | **35--50s** |  | 生成与上一步训练重叠 |

## 监控与可观测性

> **RLHF 训练中要追踪的关键指标**
>
> **质量指标**（每 10 步记录）：
>
> - 平均 reward（应当先上升后趋稳）
> - 与 reference 的 KL 散度（应当保持在 3--10）
> - 响应长度分布（警惕 length hacking）
> - Entropy（应缓慢下降，不能崩塌）
>
> **系统指标**（每步记录）：
>
> - GPU 利用率（目标：训练 $>$80%，生成 $>$60%）
> - 每 GPU 的显存水位（在 OOM 发生前发现）
> - 生成吞吐（tokens/秒，应稳定）
> - Gradient norm（尖峰 = 即将出现不稳定）
> - NCCL 通信时间（侦测网络劣化）

## 网络拓扑与通信模式

高效的分布式训练要求理解连接 GPU 的分层通信网络。现代集群采用两级架构：超高速的节点内链路，以及较慢但可扩展的节点间网络。

### 节点内：NVLink 与 NVSwitch


**NVLink 各代及其对 LLM 训练的影响**
| **代际** | **每链路带宽** | **每 GPU 链路数** | **总带宽** | **平台** |
| --- | --- | --- | --- | --- |
| NVLink 3.0 | 50 GB/s | 12 | 600 GB/s | A100（DGX A100） |
| NVLink 4.0 | 50 GB/s | 18 | 900 GB/s | H100（DGX H100） |
| NVLink 5.0 | 100 GB/s | 18 | 1800 GB/s | B200（DGX B200） |

在单个节点内（通常 8 块 GPU），**NVSwitch** 在任意 GPU 对之间提供全二分带宽（full-bisection bandwidth）。这意味着任意 GPU 可以同时以全 NVLink 速率与任意其他 GPU 通信——这对张量并行至关重要，因为每层都需要在 8 块 GPU 间做 AllReduce。

> **NVSwitch vs PCIe 拓扑**
>
> **有 NVSwitch**（DGX/HGX）：8 块 GPU 两两全互联，速率 600--1800 GB/s。TP 的 AllReduce 每层 $\sim$0.2ms。
>
> **无 NVSwitch**（仅 PCIe 的服务器）：GPU 经由 CPU 的 PCIe root complex 通信，带宽仅 32--64 GB/s。8 卡 TP 慢上 10--30$\times$。**在仅 PCIe 的系统上绝不要用 TP$>$2。**

### 节点间：InfiniBand 与 RoCE

跨节点做 FSDP/ZeRO-3 的 AllGather 与 ReduceScatter 时，节点间网络成为主要瓶颈。


**LLM 训练集群的节点间网络选项**
| **技术** | **带宽** | **延迟** | **说明** |
| --- | --- | --- | --- |
| InfiniBand NDR | 400 Gb/s（50 GB/s） | 1--2 $\mu$s | 黄金标准，RDMA，无损 |
| InfiniBand NDR（双轨） | 800 Gb/s（100 GB/s） | 1--2 $\mu$s | 用于 H100 集群 |
| RoCE v2 | 100--400 Gb/s | 2--5 $\mu$s | 更便宜，需要调优 PFC/ECN |
| Ethernet（TCP） | 100--400 Gb/s | 10--50 $\mu$s | 不适合 $>$16 GPU 训练 |

### 通信原语及其代价

理解每种集合通信原语的使用场景有助于诊断瓶颈：


**分布式 LLM 训练中的 NCCL 集合通信操作**
| **集合通信** | **搬运数据量** | **被谁使用** | **使用时机** |
| --- | --- | --- | --- |
| AllReduce | $2 \cdot \frac{N-1}{N} \cdot M$ | TP、DP | 跨 GPU 求和梯度或激活 |
| AllGather | $\frac{N-1}{N} \cdot M$ | FSDP 前向 | 矩阵乘前重建完整参数张量 |
| ReduceScatter | $\frac{N-1}{N} \cdot M$ | FSDP 反向 | 反向后分发梯度分片 |
| Broadcast | $M$ | PP | 把激活发到下一个流水线 stage |
| Send/Recv | $M$ | PP | 相邻 stage 间的点对点 |

其中 $M$ 是消息大小（字节），$N$ 是参与者数量。


> **通信-计算重叠**
>
> 现代框架（FSDP、DeepSpeed）会积极地让通信与计算重叠：
>
> **前向传播**：当第 $i$ 层在计算时，AllGather 预取第 $i+1$ 层的参数。第 $i$ 层一结束，其参数立刻被丢弃（“free-after-forward”）。
>
> **反向传播**：当第 $i$ 层在算梯度时，ReduceScatter 发送第 $i+1$ 层的梯度。调优得当时，这种重叠可隐藏 70--90% 的通信延迟。
>
> **调优旋钮**：`prefetch_factor`（提前预取多少层）、`reduce_bucket_size`（梯度归约的粒度）、`backward_prefetch`（反向预取的 “pre” 还是 “post” 策略）。

### 网络拓扑设计

生产集群采用 **fat-tree** 或 **rail-optimized**（按轨优化）拓扑：

- **Fat-tree**：每一级都具备全二分带宽。任一节点都能以满速与任一其他节点通信。代价昂贵（需要大量交换机），但灵活性最高。
- **Rail-optimized**：每个节点的 GPU $i$ 连接到同一台叶子交换机（“轨 $i$”）。轨内 AllReduce 便宜，跨轨流量昂贵。Meta 的 RSC 和 Google 的 TPU pod 都采用此拓扑。
- **3D torus / Dragonfly**：用于 HPC 集群（Frontier、Aurora）。拓扑感知的作业放置至关重要。

> **作业放置很重要**
>
> 在 512 卡集群上，随机分配节点会因网络拥塞导致 2--3$\times$ 减速。**始终请求连续的节点块。**生产调度器（Slurm、Kubernetes）应当强制本地性：一个训练作业的所有节点应处于同一叶子交换机下，或彼此距离不超过一跳。

## 训练吞吐与模型 FLOPs 利用率

### 衡量训练效率：MFU

**模型 FLOPs 利用率（Model FLOPs Utilization, MFU）** [[209]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-chowdhery2022palm)] 是衡量训练效率的标准指标：

$$
\text{MFU} = \frac{\text{实际吞吐（tokens/秒）} \times \text{每 token 的 FLOPs}}{\text{硬件峰值 FLOPS}}
$$

对于参数量为 $P$、序列长度为 $s$、batch 大小为 $b$ 的 transformer：
$$
\text{每 token 的 FLOPs} \approx 6P + 12 \cdot n_\text{layers} \cdot d_\text{model} \cdot s
$$

系数 6 来自：2（乘加）$\times$ 3（前向 + 反向，反向约为前向的 $2\times$）。第二项对应 attention 的 $O(s^2)$ 代价。


**各规模与硬件下的 MFU 基准**
| **模型** | **硬件** | **MFU** | **Tokens/秒/GPU** | **配置** |
| --- | --- | --- | --- | --- |
| LLaMA-7B | 8$\times$A100 | 57% | 3,200 | FSDP, FlashAttn, BF16 |
| LLaMA-13B | 16$\times$A100 | 52% | 1,750 | FSDP, FlashAttn, BF16 |
| LLaMA-70B | 64$\times$A100 | 45% | 380 | FSDP+TP=8, FlashAttn |
| GPT-4（估计） | 10,000+ H100 | 40--50% | --- | 3D 并行 |
| PaLM-540B | 6144 TPUv4 | 46% | --- | DP+TP+PP |

> **为什么 MFU 会随规模下降**
>
> 更大的模型需要更多并行，会引入：
>
> 1. **通信开销**：FSDP 的 AllGather/ReduceScatter（64 GPU 时 $\sim$10--15%）
> 2. **流水线气泡**：PP 在 micro-batch 的首尾引入空闲（PP=4 时 $\sim$15--25%）
> 3. **辅助模型占用显存**：reference/RM 占用了本可用于更大 batch 的显存
> 4. **负载不均**：并非所有层计算量相等（embedding vs transformer block）
>
> **经验法则**：训练目标 MFU $>$ 40%。低于 30% 时应使用 profiling 诊断。

### 算力最优的 batch 设计

有效 batch 大小与硬件利用率之间存在并非显而易见的相互作用：

$$
\text{有效 batch} = \text{micro\_batch} \times \text{grad\_accum} \times \text{DP 度}
$$

- **过小**：GPU 利用率低（算术强度低），通信占主导。
- **过大**：每 token 的学习收益递减（超过 critical batch size），算力被浪费。
- **最佳点**：*临界 batch size*（critical batch size）$B_\text{crit}$，即梯度噪声等于梯度信号之处。对 LLM，$B_\text{crit} \sim 1$--$4$M token [[210]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-mccandlish2018empirical)]。

对 RLHF 而言，batch 包含的是*rollout*（不仅仅是 token）：
$$
\text{RLHF batch} = N_\text{prompts} \times K_\text{generations} \times L_\text{平均响应长度}
$$

生产常见取值：$N=128$ 条 prompt，$K=1$--$4$ 个生成，$L=256$--$512$ token $\rightarrow$ 每步 32K--256K token。

### Profiling 与瓶颈诊断

关键的 profiling 工具及其揭示的内容：

| **工具** | **捕获信息** | **擅长** |
| --- | --- | --- |
| `torch.profiler` | Kernel 计时、显存 | 找慢算子、显存泄漏 |
| NVIDIA Nsight Systems | 完整的 GPU 时间线 | 可视化重叠、kernel 间空隙 |
| `nccl_debug=INFO` | 集合通信大小/耗时 | 诊断通信瓶颈 |
| `torch.cuda.memory_stats` | 分配模式 | 找碎片、峰值占用 |
| DeepSpeed Flops Profiler | 每层 FLOPs | 识别负载不均 |
| `py-spy` / `scalene` | CPU profiling | 数据加载、分词瓶颈 |

> **诊断 MFU 偏低：一份清单**
>
> 1. **GPU 利用率 $<$ 80%？** $\rightarrow$ 数据加载瓶颈（检查 CPU、I/O）。
> 2. **Kernel 之间有大空隙？** $\rightarrow$ Python 开销、同步点。使用 CUDA graphs。
> 3. **通信占步时间 $>$ 20%？** $\rightarrow$ 降低 TP 度、加大 batch、检查网络健康。
> 4. **显存 99%？** $\rightarrow$ batch 无法继续增大。尝试梯度检查点、offloading。
> 5. **生成时 OOM？** $\rightarrow$ KV cache 太大。降低 max_seq_len 或生成 batch。

## 成本分析与云端部署

理解 RLHF 训练的经济账，对规划至关重要。

### 硬件成本对比


**RLHF 训练大致云端 GPU 成本（2024--2025 价格）**
| **GPU** | **按需/小时** | **Spot/小时** | **显存** | **适用场景** |
| --- | --- | --- | --- | --- |
| A100 80GB | 2.50--3.50 美元 | 1.00--1.50 美元 | 80 GB HBM2e | 经济型训练、生成集群 |
| H100 80GB | 4.00--6.00 美元 | 2.00--3.00 美元 | 80 GB HBM3 | 生产级训练 |
| H200 141GB | 6.00--8.00 美元 | --- | 141 GB HBM3e | 大上下文、少卡配置 |
| MI300X 192GB | 3.50--5.00 美元 | 1.50--2.50 美元 | 192 GB HBM3 | 性价比替代方案 |

### RLHF 训练成本估算

$$
\text{成本} = \frac{N_\text{steps} \times T_\text{step}}{3600} \times N_\text{GPUs} \times C_\text{GPU/hr}
$$

> **成本示例：70B 模型 RLHF（10K 步）**
>
> | 步数 | 10,000 |
> | --- | --- |
> | 每步用时（解耦） | 45 秒 |
> | 总训练时间 | $10000 \times 45 / 3600 = 125$ 小时 |
> | GPU 数（生成 + 训练） | 64 块 A100-80GB |
> | 每 GPU-小时成本（spot） | 1.20 美元 |
> | **总成本** | 125 × 64 × 1.20 美元 = **9,600 美元** |
>
> **按阶段拆分**：
>
> - 生成集群（32 GPU）：4,800 美元（占 60% 时间）
> - 训练集群（32 GPU）：4,800 美元（可重叠 $\rightarrow$ 实际 3,400 美元）
> - 打分（与生成共享 GPU）：已含在上文中
>
> **重叠后**：完整对齐 70B 模型的实际成本约为 **7,500 美元**。

### 成本优化策略

- **Spot/可抢占实例**：节省 50--70%。要求 checkpoint 机制健壮（每 5 分钟保存一次）。
- **合理配型**：不要用 H100 做生成（显存带宽瓶颈）；推理时 A100 的 tokens/$ 相近。
- **量化推理**：生成与打分用 INT8/FP8 可让相应集群的 GPU 数减半。
- **渐进式训练**：先用 8B 代理模型做 reward 工程/调试（约 200 美元），再扩到 70B。
- **用 LoRA 取消 reference**：彻底移除 reference 模型（显存减少 50%）。
- **先短后长**：按 256$\rightarrow$512$\rightarrow$1024 token 的 curriculum 生成可节省 40% 算力。

## 分布式 Checkpointing

在大规模下，朴素 checkpointing 会成为瓶颈。一个带优化器状态的 70B 模型，每次 checkpoint 需要保存 $\sim$840 GB（FP32 master 权重 + Adam m + v）。

### Checkpointing 策略


**大规模 RLHF 的 checkpointing 方案**
| **策略** | **保存耗时（70B）** | **每 ckpt 存储** | **特性** |
| --- | --- | --- | --- |
| 同步（所有 rank） | 30--60s（阻塞） | 420 GB | 简单，会让训练停顿 |
| 异步（后台复制） | $<$1s（非阻塞） | 420 GB | 与下一步重叠 |
| 增量（delta） | $<$1s | 5--20 GB | 只保存变化的参数 |
| 分片（FSDP 原生） | 5--10s | 420 GB 分片 | 每个 rank 各存自己的分片 |

### 用 torch.distributed.checkpoint 做生产级 Checkpointing

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

# 保存：每个 rank 并行写自己的分片
state_dict = {"model": get_state_dict(model, options=StateDictOptions(full_state_dict=False))}
dcp.save(
    state_dict=state_dict,
    storage_writer=dcp.FileSystemWriter("/mnt/checkpoints/step_5000"),
    planner=dcp.DefaultSavePlanner(),  # 自动处理 FSDP 分片
)

# 异步保存：非阻塞，运行于后台线程
future = dcp.async_save(
    state_dict=state_dict,
    storage_writer=dcp.FileSystemWriter("/mnt/checkpoints/step_5000"),
)
# 训练立即继续；只在需要时才用 future.result() 阻塞
```

> **RLHF 的 Checkpoint 卫生**
>
> RLHF 的 checkpoint 需要捕获*比*标准预训练*更多*的内容：
>
> - Policy 模型权重 + 优化器状态（标准）
> - KL 系数（$\beta$）及其调度状态
> - Replay buffer 内容（用于离策略修正）
> - 所有 GPU 的 RNG 状态（可复现性）
> - Prompt 迭代器位置（避免重复处理 prompt）
> - Reward 模型版本标签（用于审计）
> - Wandb/指标 run ID（用于连续日志）

## 硬件选型指南

合适的硬件选择取决于模型规模、预算和训练阶段。


**按模型规模与训练阶段给出的硬件建议**
| **模型规模** | **训练阶段** | **推荐** | **配置** |
| --- | --- | --- | --- |
| $\leq$7B | SFT + RLHF | 1--2$\times$ A100 | 单节点，无需并行 |
| 7--13B | SFT + RLHF | 4--8$\times$ A100 | FSDP，可选生成 TP=2 |
| 13--34B | SFT + RLHF | 8--16$\times$ A100/H100 | FSDP + 生成 TP=4 |
| 70B | RLHF（完整） | 32--64$\times$ A100/H100 | 解耦，FSDP + TP=8 |
| 70B | RLHF（LoRA） | 8--16$\times$ A100/H100 | 无 reference，LoRA adapter |
| $>$100B | RLHF | 128+$\times$ H100 | 3D 并行（TP+PP+DP） |

> **H100 vs A100：何时升级才值得？**
>
> H100 提供：
>
> - 峰值 FLOPS 约 $\sim$1.6$\times$（带稀疏 BF16：989 vs 624 TFLOPS；不带稀疏：495 vs 312）
> - 显存带宽约 $\sim$2$\times$（3.35 vs 2.0 TB/s）
> - 支持 FP8（推理再快 2$\times$）
> - NVLink 4.0（900 vs 600 GB/s）
>
> **训练上**：端到端约快 $\sim$1.8--2.2$\times$（FP8 支持与更高带宽放大了原始 FLOPS 优势）。
>
> **生成上**：约快 $\sim$1.7$\times$（带宽受限，2$\times$ BW 加上开销 $\approx$ 1.7$\times$ 吞吐）。
>
> **性价比**：在 1.5$\times$ 价格下，H100 在训练上几乎总是更划算。仅推理（生成集群）时，A100 在 spot 价格下可能更划算。

## RL 训练的优化器配置

RL 训练（PPO、GRPO、DPO）相比预训练或 SFT 对优化器有独特要求。Loss 地形非平稳（policy 变化会改变生成的数据）、梯度更嘈杂（reward 信号方差大）、且更容易出现灾难性遗忘（Catastrophic Forgetting）或 reward hacking。本节以 AdamW [[59]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-loshchilov2019adamw)] 作为默认优化器，整理 RL 特有的优化器实践指南。

### 为什么 RL 需要不同的优化器设置

> **RL vs.\ SFT 优化——关键差异**
>
> - **数据分布非平稳**：SFT 数据集固定，而 RL 每轮迭代都会生成新的 rollout——数据分布随 policy 一起漂移。
> - **高梯度方差**：reward 信号稀疏且嘈杂；梯度方差远高于在精心策划数据上的交叉熵。
> - **需要更小的更新**：policy 必须贴近 reference 模型（KL 约束），所以学习率比 SFT 小 10--100$\times$。
> - **不用 weight decay**：正则化由 KL 惩罚提供，而不是 weight decay。再叠加 WD 可能与 KL 约束相互对抗。
> - **更短的 warmup**：RL 从已收敛的 SFT checkpoint 起步——优化器状态只需极短的 warmup。

### 各 RL 方法的推荐超参


**RL 训练各阶段的优化器设置。均使用 $\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=10^{-8}$、`max_grad_norm`=1.0、BF16。**
| **方法** | **优化器** | **LR** | **WD** | **Warmup** | **调度** |
| --- | --- | --- | --- | --- | --- |
| DPO | AdamW | $5\text{e-}7$ | 0.0 | 50 步 | Constant 或 Linear |
| PPO（policy） | AdamW | $1\text{e-}6$ | 0.0 | 20 步 | Constant |
| PPO（critic） | AdamW | $1\text{e-}6$ | 0.0 | 20 步 | Constant |
| GRPO | AdamW | $1\text{e-}6$ | 0.0 | 20 步 | Constant |

> **为什么 RL 用恒定（Constant）调度？**
>
> Cosine 与线性衰减调度都假设训练 horizon 固定、loss 单调下降。RL 训练两者都不具备：reward 可能停滞、尖峰，或不可预测地震荡。短暂 warmup 之后保持恒定 LR，可让优化器在整个训练中始终保持响应能力。若必须衰减，请使用非常温和的线性调度，最小 LR 比例保持较高（$\geq 0.5$）。

### RL 用 Beta-2 = 0.95：更快的自适应

Adam 默认的 $\beta_2 = 0.999$ 给二阶矩留下了非常长的记忆（有效窗口 $\sim$1000 步）。在 RL 训练中，loss 地形随 policy 演化迅速变化——1000 步前的梯度方差已无关紧要。使用 $\beta_2 = 0.95$ 把窗口缩短到 $\sim$20 步，让自适应学习率能够迅速响应变化中的梯度统计。

> **beta2 = 0.95 反受其害的场景**
>
> 对于非常小的 batch（例如在线 RL 中 batch=1），$\beta_2 = 0.95$ 会让二阶矩估计噪声过大。这种情况下，使用 $\beta_2 = 0.99$ 作为折中，或通过梯度累积加大有效 batch。

### RL 的混合精度：FP32 Master 权重至关重要

RL 训练对数值精度尤其敏感：

- 梯度更嘈杂——小幅更新必须在多步中精确累积
- 学习率极小（$10^{-6}$--$10^{-7}$），使得 $\Delta\theta \ll \theta$
- BF16 尾数（7 位 $\approx$ 0.8% 相对精度）无法表示相对于 $10^{0}$ 量级权重的 $10^{-6}$ 量级更新

**RL 训练务必使用 FP32 master 权重。**纯 BF16 训练（不留 FP32 副本）几乎必然会让 PPO/GRPO 在 100--500 步后出现 reward 崩塌。

### 梯度裁剪对 RL 至关重要

在 PPO 和 GRPO 中，reward 信号变异性可能极高，尤其训练早期。一个坏 batch 就能产生 norm $>100$ 的梯度，足以彻底破坏模型权重。`max_grad_norm=1.0` 是标准设置。SFT 下裁剪不那么关键，但仍建议使用。

> **RL 训练绝不要关闭梯度裁剪**
>
> SFT 的梯度 norm 通常稳定在 0.1--1.0；而 RL 的梯度是尖刺型的，原因有三：(1) reward 方差经 policy gradient 传播；(2) 稀有的高 reward 轨迹会产生异常大的更新；(3) policy 漂移时 KL 惩罚项会产生大梯度。哪怕只有一步 $\|\nabla\| > 50$ 的未裁剪更新，都可能让数百步训练前功尽弃。

### 诊断 RL 训练的不稳定性

> **RL 优化的红旗与修复**
>
> | **症状** | **可能原因与修复** |
> | --- | --- |
> | Reward 先升后崩 | LR 过高或 KL 系数过低。将 LR 降 2--5$\times$，或加大 $\beta_\text{KL}$。 |
> | 梯度 norm 一直顶在裁剪阈值 | 更新过于激进。降低 LR（持续裁剪意味着每一步都丢失梯度方向信息）。 |
> | KL 散度爆炸（$>$15 nat） | LR 过高。降低 10$\times$ 或加上自适应 KL 惩罚。 |
> | Reward 卡在基线 | LR 过低，或 reward 模型信号过弱。试着把 LR 提 2--5$\times$。检查 reward 模型校准。 |
> | 100+ 步后 loss NaN | 缺少 FP32 master 权重，或梯度 norm 溢出。开启 FP32 master 权重；验证 BF16 模式。 |

### HuggingFace TRL 的 RL 配置

TRL 库 [[160]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-vonwerra2022trl)] 为 LLM 的 PPO、DPO 等 RL 方法提供了生产级实现。

```python
from trl import PPOConfig, PPOTrainer, DPOConfig, DPOTrainer

# --- PPO 配置 ---
ppo_config = PPOConfig(
    # 优化器（AdamW，RL 特化设置）
    learning_rate=1e-6,           # 比 SFT 小 10-100x

    # PPO 特有
    ppo_epochs=4,                 # 每个 rollout 上做的 mini-batch 更新次数
    mini_batch_size=16,
    batch_size=64,                # rollout 的 batch 大小

    # 梯度控制
    max_grad_norm=1.0,

    # KL 惩罚（取代 weight decay 作为正则化）
    init_kl_coef=0.2,            # 初始 KL 惩罚系数
    adap_kl_ctrl=True,           # 自适应 KL 目标
    target_kl=6.0,               # 目标 KL 散度

    # 混合精度
    bf16=True,                   # BF16 计算，FP32 master 权重
)

ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    config=ppo_config,
    tokenizer=tokenizer,
    dataset=dataset,
)

# --- DPO 配置 ---
dpo_config = DPOConfig(
    output_dir="./dpo_output",

    # 优化器
    learning_rate=5e-7,           # 比 PPO 还小
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.95,              # RL 用更短的记忆
    weight_decay=0.0,            # 不用 WD —— KL 提供正则化

    # 调度
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,

    # 梯度控制
    max_grad_norm=1.0,

    # DPO 特有
    beta=0.1,                    # KL 约束强度
    loss_type="sigmoid",         # 标准 DPO loss

    # 混合精度
    bf16=True,

    # 训练
    num_train_epochs=1,          # DPO 通常 1 个 epoch
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

### MoE 在 RL 训练中的考虑

> **MoE 用于 RLHF**
>
> 混合专家（Mixture-of-Experts, MoE）模型 [[89]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-fedus2022switch)] 在 RLHF 中越来越常见：
>
> - **优势**：相同算力下容量提升 3--4$\times$。对 reward 模型尤其有利（更多容量去打分）。
> - **挑战**：专家并行需要 all-to-all 通信（token 跨 GPU 路由），与流水线并行存在冲突。
> - **GRPO + MoE**：效果不错，因为生成成本由激活参数主导（而非总参数）。
> - **MoE 用 LoRA**：可以只对路由器 + 共享层加 LoRA，或对所有专家都加（昂贵）。

> **RL 优化器口诀**
>
> 做 RL 微调请记住：**小 LR、不用 weight decay、恒定调度、FP32 master 权重、激进裁剪**。把正则化交给 KL 惩罚——优化器只管沿 policy gradient 前进，且不要冲过头。

