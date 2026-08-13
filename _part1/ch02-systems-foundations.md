---
layout: home
title: 面向 LLM 的系统基础
permalink: /part1/ch02-systems-foundations.html
---

## GPU 架构——从硅片到 LLM 训练

现代大型语言模型几乎完全依赖 GPU（图形处理器，Graphics Processing Unit）进行训练和服务。理解 GPU 架构对于在并行策略、内存管理、内核优化和基础设施规模等方面做出明智决策至关重要。本节针对 LLM 工作负载，系统地介绍 GPU 硬件相关知识。

### 为什么深度学习要用 GPU？

GPU 和 CPU 代表了根本不同的硬件设计哲学。理解这种差异能解释为何 LLM 训练在 GPU 上能快 100--1000 倍。

> **CPU 与 GPU——基本设计哲学**
>
> - **CPU** 针对*延迟*进行优化——它们以尽可能快的速度执行少量线程，配备大型缓存、分支预测器和乱序执行单元。现代 CPU 拥有 8--96 个核心。
> - **GPU** 针对*吞吐量*进行优化——它们并行执行数千个线程，每个线程只做简单工作。现代 GPU 拥有数千个"核心"（执行单元），它们被组织成流式多处理器（Streaming Multiprocessor，SM）。
>
> 深度学习工作负载以矩阵乘法（在 $O(n^2)$ 数据上执行 $O(n^3)$ 操作）为主，这是天然并行的。70B 模型的单次 transformer 前向传播每个 token 需要约 140 TFLOP 计算量——非常适合 GPU 的吞吐能力。

### NVIDIA GPU 微架构代际演进

NVIDIA 已经发布了一系列 GPU 架构，每一代都为深度学习带来关键创新：

面向深度学习的 NVIDIA GPU 微架构时间线。

| 架构 | 年份 | 旗舰 | 深度学习关键创新 |
| --- | --- | --- | --- |
| Pascal | 2016 | P100 | 首款 HBM GPU；FP16 支持；NVLink 1 |
| Volta | 2017 | V100 | **Tensor Cores**（第一代）；混合精度训练 |
| Turing | 2018 | T4 | INT8 推理；RT cores（非 ML 用途） |
| Ampere | 2020 | A100 | BF16 Tensor Cores；TF32；第 3 代 NVLink；MIG |
| Hopper | 2022 | H100 | FP8 Tensor Cores；TMA；Transformer Engine；NVLink 4 |
| Blackwell | 2024 | B200 | 第 2 代 Transformer Engine；NVLink 5（1.8 TB/s）；FP4 |

### LLM 训练与推理常用 GPU

与 LLM 工作负载相关的 GPU 规格。所有带宽数字均为双向。

| GPU | 架构 | HBM | BF16 TF | HBM 带宽 | NVLink | LLM 角色 |
| --- | --- | --- | --- | --- | --- | --- |
| V100-32GB | Volta | 32 GB | 125 TF* | 900 GB/s | 300 GB/s | 旧型号；小模型微调 |
| A100-40GB | Ampere | 40 GB | 312 TF | 1.5 TB/s | 600 GB/s | 经济型训练/推理 |
| A100-80GB | Ampere | 80 GB | 312 TF | 2.0 TB/s | 600 GB/s | 标准 RLHF（70B 模型用 8--64 卡） |
| H100 SXM | Hopper | 80 GB | 990 TF | 3.35 TB/s | 900 GB/s | 训练快 3$\times$ |
| H200 SXM | Hopper | 141 GB | 990 TF | 4.8 TB/s | 900 GB/s | 用更少 GPU 装下 70B 策略+参考模型 |
| B200 SXM | Blackwell | 192 GB | 2250 TF | 8.0 TB/s | 1800 GB/s | 新一代；比 H100 快 2$\times$ |
| *AMD 和 Google 替代方案：* | | | | | | |
| MI300X | CDNA3 | 192 GB | 1300 TF | 5.3 TB/s | N/A | 显存最大；ROCm |
| TPU v5e | Google | 16 GB | 197 TF | 1.6 TB/s | ICI 1.6 TB/s | 仅云端；JAX/XLA |

> **警告：如何选择 GPU？**
>
> - **训练 70B+ 模型**：带 NVLink 的 H100/B200 节点（张量并行需要快速互联）。每个实例至少 8$\times$H100。
> - **推理（延迟敏感）**：H100/H200 适合高带宽场景；MI300X 适合内存密集（Memory-Bound）场景（巨大的键值缓存（KV Cache））。
> - **微调 7B--13B 模型**：A100-80GB 性价比高。用 LoRA 单卡即可。
> - **预算紧张**：A100-40GB，甚至用 A10（24GB）在 7B 模型上跑 LoRA。

### GPU 内部架构——流式多处理器（SM）

GPU 由一组**流式多处理器（Streaming Multiprocessor，SM）**阵列组成，每个 SM 都是独立的处理器，拥有自己的寄存器文件、共享内存和执行单元。理解 SM 是理解 GPU 性能的关键。

![左：A100 单个流式多处理器（SM）的内部结构——64 个 FP32 CUDA cores、4 个 Tensor Cores、4 个 warp 调度器、256 KB 寄存器文件以及 192 KB 共享内存/L1 缓存。右：完整 A100 芯片包含 108 个 SM，共享 40 MB L2 缓存和 80 GB HBM2e。左侧边栏的带宽标注显示了从寄存器到 HBM 的急剧下降。]({{ site.baseurl }}/figures/fig_016_fig16.png)

> **SM 的关键组件**
>
> - **CUDA Cores：**用于 FP32/INT32 运算的标量 ALU。A100 上每个 SM 有 64 个。用于逐元素运算、归约和非矩阵运算。
> - **Tensor Cores：**专用的矩阵乘加（matrix-multiply-accumulate，MMA）单元。每个单元每周期执行一次 $4{\times}4{\times}4$ 融合乘加运算。A100 每个 SM 有 4 个，在支持的精度下提供相对 CUDA cores $16\times$ 的吞吐量。
> - **寄存器文件：**最快的存储（延迟 1 周期）。所有活动线程共享。溢出到 L1 会显著降速。
> - **共享内存 / L1：**由程序员显式管理的片上 SRAM。这是 FlashAttention 性能的关键（数据块完全装入共享内存）。
> - **Warp 调度器：**每个 SM 有 4 个 warp 调度器（A100）。一个 *warp* = 32 个步调一致执行的线程（SIMT 模型）。调度器通过在 warp 之间切换来隐藏内存延迟。

> **SIMT 执行模型**
>
> GPU 采用单指令多线程（Single Instruction, Multiple Threads，SIMT）执行模型。在一个 warp（32 个线程）内，所有线程执行相同指令但作用于不同数据。当线程出现分支（例如 `if/else`）时，两条路径都会被串行执行——称为 *warp 分支（warp divergence）*。这就是为什么 GPU 内核必须最小化分支的原因。
>
> 对于 LLM 工作负载，主要运算（GEMM、attention、softmax）在线程间具有统一的控制流，非常适合 SIMT 执行。

### 各代际 GPU 芯片的扩展

NVIDIA GPU 架构的演进显示出在计算密度、片上内存和深度学习专用单元方面持续扩展：

各代 NVIDIA 架构在 SM 层面的扩展。

| 架构 | SM 数 | TCs/SM | SRAM/SM | L2 | 关键变化 |
| --- | --- | --- | --- | --- | --- |
| Volta (V100) | 80 | 8 | 128 KB | 6 MB | 引入 Tensor Cores |
| Ampere (A100) | 108 | 4 | 192 KB | 40 MB | BF16/TF32；更大的 L2 |
| Hopper (H100) | 132 | 4 | 256 KB | 50 MB | TMA；FP8；Thread Block Clusters |
| Blackwell (B200) | 148 | 4 | 256 KB | 128 MB | 2$\times$ 裸片；FP4；TMEM；NVLink 5 |

### GPU 内存层级与带宽

现代 GPU 训练和推理性能几乎完全取决于*你如何管理数据在内存层级之间的搬运*。理解内存层级不是可选项——它是后续章节讨论每一种优化技术的基础。

> **GPU 内存层级——A100 80GB 参考数字**
>
> | 层级 | 容量 | 带宽 | 延迟 | 位置 |
> | --- | --- | --- | --- | --- |
> | 寄存器 | $\sim$256 KB/SM | $>$100 TB/s | 1 周期 | 片上，每线程 |
> | SRAM (shared) | 164 KB/SM | $\sim$19 TB/s | $\sim$20 周期 | 片上，每 SM |
> | L2 缓存 | 共 40 MB | $\sim$5 TB/s | $\sim$200 周期 | 片上，共享 |
> | HBM2e (VRAM) | 80 GB | 2 TB/s | $\sim$200 ns | 封装上（5 个堆栈） |
> | CPU DRAM | 512 GB+ | $\sim$25 GB/s | $\sim$10 $\mu$s | 主机（PCIe 4） |
> | NVMe SSD | TB 级 | 7 GB/s | $\sim$100 $\mu$s | 主机存储 |

> **为何各层级差距如此巨大**
>
> 内存层级中每一级大约比上一级**慢 10$\times$，大 100--1000$\times$**。A100 有 312 TFLOP/s 的 BF16 tensor-core 吞吐，但 HBM 带宽只有 2 TB/s。这意味着从 HBM 每加载一字节，你可以在下一字节到达之前完成 $312 \times 10^{12} / (2 \times 10^{12}) \approx 156$ 次浮点运算。如果你的内核每字节运算少于 156 FLOPs，它就是*内存密集（Memory-Bound）*的——计算单元在空等数据。

**寄存器。**

每个 CUDA 线程都有自己的私有寄存器文件。寄存器是芯片上最快的存储——读写在一个时钟周期内完成且无需仲裁。A100 每个 SM 有 65,536 个 32 位寄存器。寄存器溢出到本地内存（L1/L2）是主要的性能隐患。

**SRAM——共享内存 / L1。**

A100 上每个 SM 拥有 192 KB 的 L1/共享内存池（H100 为 256 KB），其中最多 164 KB 可配置为共享内存。共享内存由程序员（或较新版本 CUDA 中由编译器）显式管理。例如，FlashAttention 的整个设计都基于一个洞察：注意力分块计算可以装入 SRAM。

**L2 缓存。**

A100 上的 40 MB L2 由所有 108 个 SM 共享。它充当 SRAM 与 HBM 之间的中转区。对于具有良好空间局部性的工作负载（例如在一个 batch 内被反复访问的权重矩阵），L2 命中率可以大幅降低实际 HBM 流量。

**HBM——高带宽显存。**

高带宽显存（High Bandwidth Memory，HBM）是直接安装在 GPU 封装上的堆叠 DRAM，通过宽带中介层连接。A100 SXM 拥有 80 GB HBM2e，带宽 2 TB/s；H100 SXM5 拥有 80 GB HBM3，带宽 3.35 TB/s。它是模型权重、KV cache、激活和优化器状态的主要工作内存。

**通过 PCIe 访问 CPU DRAM。**

GPU HBM 与 CPU DRAM 之间的数据传输需经过 PCIe 总线。PCIe Gen4 $\times$16 每方向提供约 32 GB/s（双向 64 GB/s）；Gen5 翻倍。这相对于 HBM（每方向）是约 60$\times$ 的带宽下降。CPU 卸载（ZeRO-Infinity、DeepSpeed）利用这条链路，但必须谨慎使用以免成为瓶颈。

**NVMe。**

NVMe SSD（例如 Samsung 990 Pro）顺序读取可达约 7 GB/s。ZeRO-Infinity 可以将优化器状态卸载到 NVMe，但只有在计算与 I/O 比率非常高（大 batch 尺寸、训练步骤较慢）时才可行。

### 算术强度与 Roofline 模型

> **算术强度（Arithmetic Intensity）**
>
> $$
> I = \frac{\text{FLOPs}}{\text{从 HBM 访问的字节数}}
>   \quad \text{(FLOPs / Byte)}
> $$
>
> 当 $$I < I_{\text{ridge}}$$ 时内核是**内存密集（Memory-Bound）**的，当 $$I > I_{\text{ridge}}$$ 时则是**计算密集（Compute-Bound）**的，其中
>
> $$
> I_{\text{ridge}} = \frac{\text{峰值 FLOP/s}}{\text{峰值带宽}}
>   = \frac{312 \times 10^{12}}{2 \times 10^{12}} = 156 \text{ FLOP/Byte (A100 BF16)}
> $$

![A100 BF16 的 Roofline 模型（Roofline Model）。Attention 深陷于内存密集区间；大型 GEMM（FFN 层）则是计算密集的。]({{ site.baseurl }}/figures/fig_017_fig17.png)

> **示例：Attention 的算术强度**
>
> 对于序列长度 $n=4096$、头维度 $d=128$ 的单个注意力头：
>
> **FLOPs**：$QK^T$ 耗费 $2n^2d$，softmax 为 $O(n^2)$，$\text{Attn} \times V$ 耗费 $2n^2d$。总计：$\approx 4n^2 d = 4 \times 4096^2 \times 128 \approx 8.6$ GFLOP。
>
> **内存流量**（标准实现，非 Flash 版）：
>
> - 读取 $Q, K$：$2 \times n \times d \times 2 = 2$ MB
> - 写入注意力分数 $S = QK^T$：$n^2 \times 2 = 33.5$ MB
> - 为 softmax 读取 $S$：$n^2 \times 2 = 33.5$ MB
> - 写入 softmax 输出 $P$：$n^2 \times 2 = 33.5$ MB
> - 为最终矩阵乘读取 $P$ 和 $V$：$n^2 \times 2 + n \times d \times 2 = 34.5$ MB
> - 写入输出 $O$：$n \times d \times 2 = 1$ MB
>
> **总内存**：$\approx 138$ MB（由对 $n^2$ 注意力矩阵的 4 次遍历主导）。
>
> **算术强度**：
>
> $$
> I = \frac{8.6 \times 10^9}{138 \times 10^6} \approx 62 \text{ FLOP/Byte}
> $$
>
> 这是 A100 脊点的 $62/156 = 40\%$——**明确属于内存密集**。GPU 有 60% 的时间在等内存空转。
>
> **FlashAttention 解决方案**：通过永远不物化 $n \times n$ 矩阵（在 SRAM 中对 $Q, K, V$ 分块），FlashAttention 把 HBM 流量降至仅读取 $Q, K, V$ 并写入 $O$：$4 \times n \times d \times 2 = 4$ MB。每个加载的字节被复用于 $O(n)$ 次计算（每个 query 都要 attend 所有 key），因此：
>
> $$
> I = \frac{4n^2 d}{4 \cdot n \cdot d \cdot 2} = \frac{n}{2} = \frac{4096}{2} = 2048 \text{ FLOP/Byte}
> $$
>
> 这比脊点（156）高出 $13\times$——**深度计算密集**。GPU 跑满峰值 312 TFLOPS，只需要 $312\text{T}/2048 \approx 152$ GB/s 带宽（HBM 容量的 7.6%）。内存不再是瓶颈。

### Attention 是内存密集的；FFN 是计算密集的

> **Transformer 中的两种区间**
>
> 一个 transformer 块有两个主要组件，它们的算术强度截然不同：
>
> - **Attention：**作用于 $n \times d$ 张量。$QK^T$ 乘积是 $O(n^2 d)$ FLOPs，但注意力分数需要 $O(n^2)$ 内存。在长序列下，内存流量占主导地位——attention 是*内存密集（Memory-Bound）*的。
> - **FFN（MLP）：**两个大型线性层，权重矩阵形状为 $$[d_{\text{model}}, 4d_{\text{model}}]$$。这些是高算术强度的大型 GEMM——FFN 是*计算密集（Compute-Bound）*的。
>
> 这就是为什么 FlashAttention（内存优化）对 attention 有效但对 FFN 无效，而量化（缩小权重体积）对 FFN 的帮助比对 attention 更大。

### Tensor Cores

> **什么是 Tensor Cores？**
>
> 张量核心（Tensor Core）是 Volta 架构（2017）引入的专用矩阵乘加（MMA）单元。每个 Tensor Core 在单个时钟周期内执行一次 $4\times4\times4$ 矩阵乘法：
>
> $$
> D = A \times B + C \quad (4\times4 \text{ matrices})
> $$
>
> A100 在 108 个 SM 中共有 **432 个 Tensor Cores**（每个 SM 4 个，每个子分区 1 个）。在 BF16 精度下提供 312 TFLOP/s 吞吐——约是 FP32 CUDA cores 吞吐的 $16\times$。

- **支持的精度：**FP64、TF32、BF16、FP16、INT8、FP8（H100+）。
- **累加：**内部始终使用 FP32，即使输入为 BF16。这能防止点积过程中的灾难性抵消。
- **要求：**当矩阵维度是 8（BF16）或 16（FP8）的倍数时，Tensor Cores 效率最高。为此进行填充通常是值得的。
- **WGMMA（H100）：**Hopper 引入了 warpgroup 级别的 MMA 指令，作用于更大的分块（64$\times$256$\times$16），并可与 TMA（Tensor Memory Accelerator）数据搬运形成流水线。

> **警告：Tensor Core 陷阱**
>
> 只有当你的内核是*计算密集（Compute-Bound）*时，Tensor Cores 才有用。如果你跑的是小 batch（batch size 1 的推理），GEMM 分块很小，Tensor Core 利用率很低，你又回到内存密集区间。这就是为什么推理引擎要激进地对请求进行 batch。

### 通信架构——NVLink、InfiniBand 与 PCIe

分布式 LLM 训练和推理需要在 GPU、节点和存储之间搬运海量数据。通信结构往往是大规模训练的瓶颈。

**PCIe——主机与设备的链路。**

> **PCIe 各代**
>
> | 代际 | x16 带宽（每方向） | 双向 | 备注 |
> | --- | --- | --- | --- |
> | PCIe Gen3 | 16 GB/s | 32 GB/s | 常见于较旧服务器 |
> | PCIe Gen4 | 32 GB/s | 64 GB/s | A100 PCIe、当前主流服务器 |
> | PCIe Gen5 | 64 GB/s | 128 GB/s | H100 PCIe，正在普及 |

PCIe 用于：

- CPU $\leftrightarrow$ GPU 数据传输（模型加载、CPU 卸载）
- 当 NVLink 不可用时的跨节点 GPU 通信（罕见，且非常慢）
- NVMe 存储访问（经由 CPU）

> **警告：PCIe 不应用于 GPU 间通信**
>
> 如果有 NVLink 可用，绝不要让 GPU-GPU 通信走 PCIe。PCIe 带宽（32 GB/s）比 NVLink 4（900 GB/s）低 28$\times$。在没有 NVLink 的多 GPU 服务器中（例如消费级 GPU），GPU 间带宽被限制在 PCIe 水平，使得张量并行极其缓慢。

**NVLink——节点内高速互联。**

> **NVLink 各代**
>
> | 代际 | 链路数 | 总带宽 | GPU |
> | --- | --- | --- | --- |
> | NVLink 2 | 6 | 300 GB/s | V100 |
> | NVLink 3 | 12 | 600 GB/s | A100 |
> | NVLink 4 | 18 | 900 GB/s | H100 |
> | NVLink 5 | 18 | 1800 GB/s | B200 (Blackwell) |

NVLink 是同一节点内 GPU 之间的点对点互联。每条链路都是双向的。H100 SXM5 拥有 18 条 NVLink 4 链路，每条提供 50 GB/s 双向带宽，总计 900 GB/s。

**NVSwitch。**

在 DGX H100 系统中，所有 8 块 GPU 都通过 NVSwitch 互联——这是一种提供*全二分带宽（full bisection bandwidth）*的专用交换芯片。这意味着任意 GPU 都能以完整 NVLink 速度同时与任何其他 GPU 通信，而不仅仅是环上邻居。

> **环形拓扑 vs. 全二分**
>
> 在环形拓扑（8 GPU）中，AllReduce 需要数据绕环传输。每条链路必须承载总数据的 $\frac{2(N-1)}{N}$，因此算法带宽为 $$B_{\text{link}} \times \frac{N}{2(N-1)}$$（$N=8$ 时约为 $$0.57 \times B_{\text{link}}$$）。借助 NVSwitch 的全二分带宽，AllReduce 可以使用基于树的算法同时使用所有链路，达到接近峰值的带宽。在 DGX H100 实际测试中：ring 达到约 700 GB/s 总线带宽，NVSwitch 达到约 900 GB/s。

**InfiniBand——节点间通信。**

对于节点（服务器）之间的通信，InfiniBand 提供高带宽、低延迟的网络，并支持直接 GPU 内存访问。

> **InfiniBand NDR**
>
> - **NDR 400Gb/s** = 每端口 50 GB/s（单向）
> - **HDR 200Gb/s** = 每端口 25 GB/s（上一代）
> - **RDMA：**远程直接内存访问（Remote Direct Memory Access）——GPU 可读写远端 GPU 内存而无需远端 CPU 参与
> - **GPUDirect RDMA：**数据直接经由 HBM $\to$ NIC $\to$ 网络 $\to$ NIC $\to$ HBM，完全绕过 CPU 和系统 DRAM
> - **延迟：**小消息约 1--2 $\mu$s（对比 TCP/IP 的约 100 $\mu$s）

**胖树拓扑（Fat-Tree Topology）。**

大型 GPU 集群使用胖树（Fat-Tree Topology）网络拓扑。使用 $k$ 端口交换机的 3 层胖树可支持 $k^3/4$ 个节点并保持全二分带宽。对于 $k=64$ 端口的 400Gb/s NDR 交换机：$64^3/4 = 65{,}536$ 个节点。

**轨道优化拓扑（Rail-Optimized Topology）。**

实践中，集群使用*轨道优化拓扑（Rail-Optimized Topology）*，节点中的每块 GPU 连接到不同的机架顶端（top-of-rack）交换机。这样可以保证 AllReduce 操作（涉及所有 GPU）同时使用所有网络链路，从而最大化带宽。

**分布式 LLM 训练中的通信模式。**

分布式训练依赖集体通信原语。原语的选择决定了带宽需求和扩展行为。

> **通信原语**
>
> | 原语 | 使用场景 | 通信量 |
> | --- | --- | --- |
> | AllReduce | 梯度同步（DDP、FSDP） | $2(N-1)/N \times$ 参数大小 |
> | AllGather | 收集分片权重（FSDP） | $(N-1)/N \times$ 参数大小 |
> | ReduceScatter | 分散梯度（FSDP） | $(N-1)/N \times$ 参数大小 |
> | AllGather | 张量并行激活 | 激活大小 |
> | Point-to-Point | 流水线并行（send/recv） | micro-batch 激活 |
> | Broadcast | 权重同步（新 worker） | 完整模型大小 |

> **示例：带宽计算——70B 模型的梯度 AllReduce**
>
> **设定：**70B 参数模型，BF16 梯度，8 节点 $\times$ 8 GPU = 64 GPU。数据并行度 = 64。
>
> **梯度大小：**$70 \times 10^9 \times 2$ 字节 $= 140$ GB。
>
> **每块 GPU 的 AllReduce 量**（环形）：$2 \times (64-1)/64 \times 140 \approx 275$ GB。
>
> **可用节点间带宽：**8 GPU/节点 $\times$ 50 GB/s/GPU $= 400$ GB/s（采用轨道优化拓扑，8 块网卡全部启用）。
>
> **AllReduce 耗时：**每步 $275 / 400 \approx 0.69$ 秒。
>
> **含义：**对于一个 1 秒的计算步骤，通信增加 0.69 秒（占总步骤时间 41%）。这就是为什么梯度压缩、混合精度、以及把通信与计算重叠的 FSDP 至关重要。

**网络拓扑示意图。**

下图展示了典型的双节点 GPU 集群拓扑，同时显示节点内（NVLink）和节点间（InfiniBand）通信路径。

![两节点 8-GPU 拓扑。节点内：通过 NVSwitch 的 NVLink 4（总计 900 GB/s）。节点间：通过机架顶端交换机的 InfiniBand NDR 400Gb/s。每个节点配有 8 块 IB 网卡（每块 GPU 一块），用于轨道优化 AllReduce。]({{ site.baseurl }}/figures/fig_018_fig18.png)

> **根据带宽选择并行策略**
>
> - **张量并行（TP）：**每层都需要 all-reduce——仅在节点内通过 NVLink 使用。TP=8 是 H100 DGX 节点的标准。
> - **流水线并行（PP）：**阶段之间的点对点通信——可以跨节点，但会引入流水线气泡开销。当模型大到单靠 TP 装不下时使用。
> - **数据并行（DP）：**梯度的 AllReduce——可以通过 IB 跨节点。在快速 IB 下扩展性良好。
> - **FSDP/ZeRO：**AllGather + ReduceScatter——类似于 DP 但会分片优化器状态。对于大模型优于 DP。

## vLLM——分页注意力与高吞吐推理

vLLM[[138]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-kwon2023efficient)] 引入了分页注意力（PagedAttention），它借鉴操作系统用于管理 RAM 的分页抽象，并将其应用于 GPU 的键值缓存（KV Cache）。在 LLM 推理过程中，*KV cache*——存储了所有先前 token 的 key 和 value 张量——是最大的内存消耗者。高效管理 KV cache 是高吞吐推理的核心挑战。

### KV cache 碎片化问题

> **KV cache 显存计算公式**
>
> 对于具有 $L$ 层、$H$ 个头、头维度 $d$、序列长度为 $n$ 个 token 的模型：
>
> $$
> \text{KV cache 大小} = 2 \times L \times H \times d \times n \times \text{每元素字节数}
> $$
>
> 对于 Llama-3 70B（BF16）：$L=80$、$H=8$（GQA）、$d=128$：
>
> $$
> = 2 \times 80 \times 8 \times 128 \times n \times 2 = 327{,}680 \times n \text{ bytes}
> $$
>
> 当 $n=4096$ token 时：每条序列约 1.3 GB。

> **内部碎片与外部碎片**
>
> 传统推理系统会为每条序列的 KV cache 预先分配一块连续内存，大小为*可能的最大序列长度*。这会造成两类浪费：
>
> **内部碎片：**一条只生成 500 token 的序列仍然占据为 4096 token 预留的块。未使用的 3596 个 token 槽位被浪费。
>
> **外部碎片：**许多序列完成后，剩余空闲内存由许多小的不连续空隙组成。即使总空闲内存足够，新的长序列也无法被分配，因为没有任何一块连续区域足够大。
>
> 实际中，朴素分配下的 GPU 显存利用率通常只有 20--40%。

### 分页注意力——为 KV cache 引入虚拟内存

分页注意力（PagedAttention，Kwon 等，2023）借鉴了操作系统的*分页*抽象。它不再为每条序列分配一块连续内存，而是把 KV cache 切分成固定大小的**页（page）**（即块），并通过一个间接表——类似 CPU 页表——将每条序列的逻辑 token 位置翻译为分散的物理 GPU 内存地址。

> **PagedAttention 核心概念**
>
> - **块大小：**通常每块 16 个 token（可调）。每个块存储 $16 \times 2 \times L \times H \times d$ 个元素。
> - **块表（Block table）：**每条序列一份的映射，把逻辑块索引映射到 GPU 内存池中的物理块索引。
> - **物理块池：**预先分配好的固定大小块池。分配为 $O(1)$——只需从空闲列表弹出。
> - **注意力内核：**做了修改，在注意力计算时使用块表从不连续的物理位置 gather KV 块。

> **示例：块表示例**
>
> 假设块大小 = 4 个 token，我们有两条序列：
>
> - 序列 A（7 个 token）：逻辑块 [0,1] $\to$ 物理块 [3, 7]
> - 序列 B（5 个 token）：逻辑块 [0,1] $\to$ 物理块 [1, 5]
>
> 物理块 3 持有序列 A 的 token 0--3。物理块 7 持有序列 A 的 token 4--6（部分填充）。序列 A 的注意力内核按顺序从物理块 3 和 7 读取，并以块表作为间接层。

### PagedAttention 的收益

**近零浪费。**

内部碎片至多为每条序列一个部分填充块（最后一个块）。当块大小为 16 时，最差浪费为每条序列 15 个 token——可忽略不计。外部碎片被消除，因为块是固定大小、可互换的。

**动态分配。**

块在序列增长时按需分配。无需预先知道最终序列长度。这对生成任务至关重要，因为输出长度未知。

**前缀共享（写时复制）。**

多条共享同一前缀（例如同一系统提示）的序列可以*共享相同的物理块*来表示该前缀。块表只需让多条序列指向相同的物理块。当某条序列需要写入共享块（与前缀分叉）时，会触发写时复制（Copy-on-Write）。

> **前缀共享带来的节约**
>
> 在一个拥有 1000-token 系统提示、服务 128 位并发用户的聊天机器人中：
>
> - 无前缀共享：仅系统提示 KV cache 就需 $128 \times 1000 \times 327{,}680 / 10^9 \approx 42$ GB
> - 有前缀共享：$1 \times 1000 \times 327{,}680 / 10^9 \approx 0.33$ GB
> - 节约：共享前缀部分约 128$\times$

**通过 swap 抢占。**

当 GPU 显存耗尽时，vLLM 可以通过把某条序列的 KV 块换出到 CPU DRAM（或者直接丢弃，之后重新计算）来*抢占*该序列。这之所以可行，是因为块是自包含且非连续的——若是连续分配则需要复制整个缓冲区。

### 连续批处理

传统的批处理（"静态批处理"）会等到一批中*所有*序列都完成后才开始新批次。如果一条序列生成 500 个 token、另一条生成 10 个，那么 GPU 在短序列上有 490 步是空闲的。这极其浪费。

> **连续批处理（Continuous Batching）**
>
> 连续批处理（Continuous Batching，也叫迭代级调度）一次处理一个*解码步*。每一步之后：
>
> 1. 检查哪些序列已完成（生成了 EOS token）
> 2. 把已完成的序列从批中移除，释放其 KV 块
> 3. 加入新的等待序列以填补空出的槽位
> 4. 在更新后的批上运行下一解码步
>
> 批次组成每一步都在变化——序列动态地加入和离开。这让 GPU 利用率接近 100%，并大幅提升吞吐（相对静态批处理提升 1.5--3$\times$）。PagedAttention 在这里至关重要：在批次中动态添加/移除序列需要动态分配/释放 KV 块，只有在分页内存下才高效。

### vLLM 中的投机解码

投机解码（Speculative Decoding）使用一个小型*草稿模型*（draft model，例如 1B 参数）快速提出 $k$ 个候选 token，再由大型*目标模型*（target model）在一次前向传播中验证。从第一个被拒绝处之前的所有 token 都会被接受（预期接受数：每次验证 3--5 个 token）。这为延迟敏感的单序列生成带来 2--3$\times$ 提速，且无质量损失。

vLLM 将投机解码与 PagedAttention 集成：

- 草稿 token 被分配投机性 KV 块
- 拒绝时，投机块被释放（在分页分配下成本极低）
- 接受时，投机块被提升为主序列的一部分
- 块表更新为 $O(k)$——只需更新少量表项

### 具体显存节约——大规模 70B 模型

> **示例：显存预算——70B BF16 推理**
>
> **设定：**Llama-3 70B，BF16，单个 A100 80GB 节点（8 GPU，张量并行）。
>
> **模型权重：**$70 \times 10^9 \times 2$ 字节 $= 140$ GB $\div$ 8 GPU $= 17.5$ GB/GPU。
>
> **KV cache 可用部分：**$80 - 17.5 - 3$（开销）$= 59.5$ GB/GPU。
>
> **每块 GPU 每 token 的 KV cache**（TP=8 时每块 GPU 持有 $1/8$ 的头）：$2 \times 80 \times 1 \times 128 \times 2 = 40{,}960$ 字节 $\approx 40$ KB/token。
>
> **KV cache 最大 token 数：**$59.5 \times 10^9 / 40{,}960 \approx 145$ 万 token。
>
> **当 128 条并发序列每条 4096 个 token 时：**$128 \times 4096 = 524{,}288$ 个 token——完全在预算之内。
>
> **若不使用 PagedAttention**（为每条序列预分配最大长度 4096）：同样的计算，但碎片化平均浪费约 50% $\to$ 只能容纳 64 条序列。

> **警告：块大小的权衡**
>
> 更大的块大小可减少块表开销，并提高内存访问局部性（更少的分散读取）。更小的块大小可减少内部碎片，并支持更细粒度的前缀共享。vLLM 默认每块 16 个 token，是一个不错的平衡。对于非常长的序列（10 万+ token），更大的块（32--64）可能更合适。

### vLLM：端到端系统

vLLM 将 PagedAttention 封装在一整套服务栈中：连续批处理（Continuous Batching）、前缀缓存（Prefix Caching）、投机解码（Speculative Decoding）以及张量并行的模型分片协同工作，最大化每美元 GPU 的吞吐。

#### 架构总览

![vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读取数据并执行批量推理。]({{ site.baseurl }}/figures/fig_019_fig19.png)

#### 核心组件

- **API Server**：接收 OpenAI 兼容的请求（completions、chat）。对输入进行 tokenize，并创建“序列组（sequence groups）”（用于 beam search 或多样本）。
- **Scheduler**：vLLM 的大脑。维护三个队列：
  - `waiting`：尚未开始的新请求（等待 prefill）
  - `running`：正在生成 token（decode 阶段）
  - `swapped`：被抢占、KV cache 已卸载到 CPU 的请求

  每次迭代，调度器根据可用 GPU 内存块决定运行哪些请求。
- **Block Manager**：为 KV cache 实现虚拟内存抽象。将逻辑块（按序列）映射到物理块（GPU 内存池中）。负责处理：
  - 分配（生成新 token $\rightarrow$ 需要新块）
  - 写时复制（Copy-on-Write）（用于 beam search：多 beam 共享前缀块，仅在分叉时复制）
  - Swap（抢占/恢复时在 GPU $\leftrightarrow$ CPU 之间迁移）
  - 前缀缓存（Prefix Caching）（当 prompt 共享相同前缀时复用已缓存块）
- **Model Executor**：执行实际的 LLM 前向传播。管理跨 GPU 的张量并行，调度从分页 KV cache 块读取的注意力内核。
- **KV Cache Pool**：预分配的 GPU 内存，划分为固定大小的块（默认每块 16 token $\times$ num\_heads $\times$ head\_dim $\times$ 2 字节）。运行时无动态分配 $\rightarrow$ 零碎片。

#### 请求生命周期（端到端流程）

1. **到达**：客户端发送 prompt。API server 将其 tokenize，创建一个 `SequenceGroup`，放入 `waiting` 队列。
2. **调度**：每一步，调度器执行：
   1. 检查是否有 `swapped` 序列可以恢复（有足够空闲块）。
   2. 检查是否有 `waiting` 序列可以开始 prefill（有足够块容纳完整 prompt）。
   3. 在 `running` 序列之间预算剩余块（若当前块已满，每条序列每步需要 1 个新块）。
   4. 如果超出预算：**抢占**优先级最低的运行中序列（把 KV 换出到 CPU 或之后重新计算）。
3. **Prefill**（请求的第一次迭代）：整个 prompt 在一次前向传播中被处理。所有 prompt token 的 KV cache 被计算并存入已分配的块。这是计算密集（Compute-Bound）的（大批 token）。
4. **Decode**（后续迭代）：每条序列每步生成一个新 token。所有运行中的序列被一起 batch（连续批处理）。这是内存密集（Memory-Bound）的（读取完整 KV cache，生成 1 个 token）。
5. **块分配**：每个 decode 步后，如果某条序列的最后一个块已满，Block Manager 分配一个新的物理块并将其映射到下一逻辑块。
6. **完成**：当一条序列遇到 EOS 或达到最大长度时，它会从 `running` 中移除。其物理块立即被释放 $\rightarrow$ 可供其他序列使用。响应被流式回传给客户端。

#### 前缀缓存（自动 Prompt 缓存）

当多个请求共享相同前缀（系统 prompt、few-shot 示例）时：

1. 对每个逻辑块的 token 内容计算哈希。
2. 新请求到达时，检查是否有前缀块已在缓存中。
3. 命中时：跳过这些 token 的 prefill，直接复用物理 KV 块。首 token 延迟（Time-to-first-token）大幅下降。
4. 驱逐：LRU 策略。缓存块仅在内存压力需要时才被释放。

**影响**：对于具有长系统提示（2K+ token、所有用户共享）的聊天应用，前缀缓存可降低 TTFT 60--80%。

#### vLLM 中的引导（受约束）解码

vLLM 通过可插拔的后端原生支持受约束解码（参见本章前面“受约束解码”一节），可以在服务期间以极小的性能开销*保证*结构化输出。

**支持的约束类型。**

OpenAI 兼容 API 通过 `guided_*` 参数或 `response_format` 字段接受约束：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")

# --- JSON Schema 约束 ---
response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[{"role": "user",
               "content": "Extract: name, age, city from: "
                          "'John is 30 and lives in NYC'"}],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }
    }
)
# 输出保证是符合该 schema 的合法 JSON

# --- 正则表达式约束 ---
response = client.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    prompt="Generate an IPv4 address: ",
    extra_body={
        "guided_regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    }
)

# --- 选项约束 ---
response = client.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    prompt="Sentiment: ",
    extra_body={"guided_choice": ["positive", "negative", "neutral"]}
)
```

**后端架构。**

vLLM 将掩码计算委托给后端引擎：

- **XGrammar**（自 v0.7 起为默认）：基于下推自动机的引擎，支持 JSON schema、正则和任意 EBNF 语法。得益于高效的 C++ 核心，对复杂 schema 而言最快。
- **Outlines**[[95]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-willard2023outlines)]：基于 FSM；支持 JSON 和正则。在 XGrammar 不可用时作为回退方案。

掩码在模型前向传播产生 logits *之后*、采样*之前*施加——实际中每步只增加不到 1 ms，因为 FSM/PDA 状态转移和预先计算的索引查找均为 $O(1)$。

**性能影响。**

由于约束仅对 logits 进行掩码（不重新计算 attention 或 FFN），吞吐损失可忽略（基准测试中 $<$2%）。主要成本来自把 schema *编译*成 FSM/PDA 索引，耗时 0.5--5 秒，取决于 schema 复杂度。vLLM 会跨请求缓存已编译的 schema，因此每个独特 schema 只需支付一次成本。

> **警告：结构化输出 $\neq$ 正确输出**
>
> 受约束解码保证输出*语法上*有效（能解析为 JSON、符合 schema 的类型）。但它*不*保证*语义上*的正确性——模型仍可能产生能成功解析但事实错误的幻觉值。务必在下游验证业务逻辑。

**vLLM 与其他方案的性能对比（70B 模型，A100 × 4，TP=4）。**

| 指标 | vLLM | HF Generate | 原因 |
| --- | --- | --- | --- |
| 吞吐（tok/s） | 2,500--4,000 | 300--600 | 连续批处理 + PagedAttention |
| 显存利用率 | 90--95% | 50--60% | 零碎片，动态块分配 |
| 最大并发序列数 | 200--500 | 16--32 | 分页 KV 消除每序列预留 |
| 首 token 延迟 | 100--300ms | 500--2000ms | 对重复系统提示启用前缀缓存 |
