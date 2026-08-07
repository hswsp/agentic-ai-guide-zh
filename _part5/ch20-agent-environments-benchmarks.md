---
layout: home
title: Agent 环境与基准
permalink: /part5/ch20-agent-environments-benchmarks.html
---

## 动机：为什么 Agent 需要环境

对话式语言模型的评估在原则上是直截了当的：给出一个 Prompt、收集一次回复，并依据参考答案或人工评判进行打分。Agent 评估则根本不同：Agent 必须在世界中*行动*、观察后果，并在一连串步骤中调整其行为。任何单次回复都无法刻画这一点；只有结构化的*环境*才能做到。

**范围说明。** 我们在强化学习意义上使用*环境*一词：Agent 为训练或评估而与之交互的世界——而非在线服务时承载 Agent 的生产基础设施（harness、编排器）。执行沙箱出现在此处，是因为它们*使*这种环境成为可能，但 Agent harness 本身在第 18 章讨论。

> **Chatbot 与 Agent 评估的鸿沟**
>
> **Chatbot 评估**衡量单次生成的质量：流畅度、事实性、有用性。**Agent 评估**衡量 *Policy* 的质量：Agent 能否在多样化的长时程任务上稳定达成目标？这一鸿沟不仅是量上的——它需要完全不同的基础设施。

三股力量驱动了对专用 Agent 环境的需求：

**安全探索。**

真实世界系统——生产数据库、在线网站、金融 API——无法承受训练中 Agent 的探索行为。沙箱环境提供一个忠实副本，Agent 可在其中失败、恢复、学习，且不造成不可逆的损害。安全隔离（如 Docker 容器、受限网络的虚拟机）不是可选项，而是一等设计要求。

**可复现的评估。**

基准测试要求每个 Agent 在相同条件下面对相同任务。环境必须按需具备确定性、受版本控制并可分发，以便一个实验室报告的结果能在另一个实验室复现。该性质的缺失历来使 Agent 基准难以比较。

**课程学习。**

从零开始在困难任务上训练 Agent 在样本效率上很低。提供*难度课程*的环境——随着 Agent 的进步逐步增加任务复杂度——可显著减少达到目标性能所需的环境交互次数。这与人类学习的方式相似：子技能的掌握先于整体的掌握。

> **把环境视为 LLM 的 RL "Gym"**
>
> 正如 OpenAI Gym [brockman2016openai] 标准化了 RL 算法与模拟控制任务之间的接口，Agent 环境标准化了基于 LLM 的 Agent 与其必须解决的各类任务之间的接口。这一类比相当贴切：`reset()` 初始化新的 Episode，`step(action)` 推进世界状态并返回观察和奖励，`render()` 产生当前状态的人类可读视图。

## 环境设计原则

一个设计良好的 Agent 环境暴露出四个正交的设计轴：*观察空间*、*动作空间*、*奖励信号*和*Episode 结构*。让每一项都正确是必要的；让四者同时正确则是环境工程的手艺所在。

### 观察空间设计

观察是 Agent 在每一步*看到*的内容。对于基于 LLM 的 Agent，观察几乎总以文本形式呈现，但其来源材料差异极大：

- **纯文本**：终端输出、文件内容、API 响应、错误信息。与任意 LLM 兼容度最高，但丢失空间和视觉结构。
- **结构化（JSON/XML）**：机器可读的状态表示。可实现精确的指代落地，但要求 Agent 解析结构而非阅读散文。
- **多模态**：截图、可访问性树（accessibility tree）、渲染后的 HTML。GUI 和 Web 任务必备；需要具视觉能力的模型或独立的感知模块。
- **混合**：截图配合可访问性树（OSWorld 和 VisualWebArena 中使用），同时提供视觉上下文和结构化元素标识符，融合两种模态的优势。

> **观察泄露（Observation Leakage）**
>
> 一个常见的设计错误是在观察中包含 Agent 本不应访问的信息——例如真值答案、奖励值或未来的任务步骤。观察泄露会虚高基准分数，并产出在真实环境（缺乏这些信息）中部署时灾难性失败的 Agent。

### 动作空间设计

动作空间定义了 Agent 可以*做*什么。对 LLM Agent 而言，动作通常是一段文本字符串，由环境解析并执行。常见动作类型包括：

- **工具调用**：对外部函数（搜索、计算器、日历）的结构化调用。常以 JSON 或 XML 函数调用语法格式化。
- **代码执行**：Agent 写出的代码在沙箱中运行；stdout/stderr 作为下一次观察返回。这是最具表达力的动作类型。
- **API 交互**：对 Web 服务的 HTTP 请求、数据库查询、Shell 命令。
- **GUI 动作**：`click(x,y)`、`type("text")`、`scroll(direction)`、`key("Enter")`。用于计算机使用类环境。
- **自然语言**：发送给另一个 Agent、人类或子任务规划器的自由文本。

### 奖励信号设计

奖励设计是环境工程中最难的部分。奖励必须：

1. **对齐**：高奖励应对应真正的任务完成，而非表面代理指标。
2. **可学习**：信号必须足够稠密，以使 Agent 能够取得进展；长时程任务上的纯稀疏奖励若无额外塑形（shaping）通常无法学习。
3. **抗作弊**：Agent 不应能够在未真正完成任务的情况下获得高奖励（reward hacking，奖励欺骗）。

| 奖励类型 | 优点 | 缺点 |
| --- | --- | --- |
| 稀疏（结束时 0/1） | 对齐性好，难以作弊 | 难以学习 |
| 稠密（步级） | 易于学习 | 易产生塑形伪影 |
| 内在（好奇心） | 驱动探索 | 可能偏离任务 |
| LLM-as-judge | 灵活、细腻 | 昂贵、不一致 |
| 基于执行 | 真值依据 | 仅适用于可验证任务 |

### Episode 结构

Episode 可以采用若干结构：

- **定长**：Agent 恰好走 $T$ 步。实现简单；在已解决的任务上浪费算力。
- **提前终止**：当 Agent 标示完成或达到终止状态时 Episode 结束。更高效，但需要可靠的终止检测器。
- **开放式**：无固定时程；Agent 持续运行直到资源预算（Token、API 调用次数、墙钟时间）耗尽。最接近真实部署，但最难评估。

**自适应 Episode 长度与提前终止。**

近期工作挑战了"Episode 长度必须在训练前固定"的假设：

- **时程课程。** AELA [yoo2025aela] 从短 Episode 开始，并随 Agent 能力的提升（以 Policy 熵的收敛来度量）逐渐扩展时程。前期的短 Episode 每个训练样本能暴露更多样化的初始状态。
- **截断作为 RL 惩罚。** DLER [liu2025dler] 表明，对于推理模型，最简单的长度控制——硬截断——只要配合批级别的奖励归一化和动态采样，就能很好地工作，从而避免被截断的 rollout 丢失奖励信号。
- **学习停止。** 模型本身可以学习何时停止推理，而非依赖固定预算。[liu2025answerstop] 提出三种策略：当连续推理步骤收敛到同一答案时停止；提升"思考结束" Token 的概率；或在隐状态激活上训练一个轻量分类器以预测最优停止点。
- **部分 rollout 回收。** APRIL [april2025] 超额发起 rollout 请求，一旦达到目标 Batch 数即终止；未完成的回复被回收作为后续步骤的热启动前缀，从而消除了少数慢样本阻塞整个 Batch 的长尾停顿（吞吐提升 20--35\%）。TLT [hu2025tlt] 针对同一瓶颈，通过即时训练一个自适应草稿模型来对掉队样本进行投机解码（端到端加速 1.7$\times$，无损）。

### 难度课程与自适应环境

静态基准测量 Agent 能力的一个固定快照。自适应环境更进一步：在线监控 Agent 表现并调整任务难度，使 Agent 保持在"最近发展区"——难到足以从中学习，易到偶尔能成功。相关技术包括：

- **程序化生成**：从参数化分布中采样任务；根据近期成功率调整难度参数。Prioritized Level Replay [jiang2021plr] 通过估计的学习潜力（如 GAE 幅值）对每个生成关卡打分，并更频繁地回放高价值关卡。
- **自博弈 / 对抗式环境设计**：PAIRED [dennis2020paired] 训练一个对手提出能最大化"主角"与"反派" Agent 之间*悔值（regret）*的环境，从而无需手工设计难度时间表即可产生复杂度逐步上升的自然课程。
- **后见之明重标注**：用 Agent *实际*达到的目标重新标注失败轨迹，使失败也能提供学习信号（Hindsight Experience Replay, HER）[andrychowicz2017hindsight]。
- **面向 LLM 的难度定向数据筛选**：在 RLVR 训练中，并非所有问题都提供等量信号。近期工作优先选择中等难度的问题——即模型成功率大致在 30--70\% 之间的题目——因为它们能提供最高的梯度信息量 [wang2025dataefficiency]。ADCL [liu2025adcl] 随着模型提升而周期性地重新估计难度，避免课程过时。

## Agent 环境的类型

### 代码执行沙箱

对 LLM 而言，最基础的 Agent 环境就是代码执行沙箱：Agent 写代码，沙箱运行，返回输出。这一简单循环支撑了出人意料比例的真实 Agent 部署。

**基于 Docker 的隔离**是最常见的做法。每个 Episode 从已知镜像启动一个全新容器，在其中执行 Agent 代码，并在 Episode 结束时销毁该容器。网络访问、文件系统写入、进程派生均可在容器级别加以控制。

**E2B**（Environments to Benchmarks）提供一个托管云沙箱 API：Agent 通过 HTTP 发送代码，E2B 在隔离的 Firecracker microVM 中执行（启动时间不到 200 毫秒）并返回 stdout/stderr。E2B 处理容器生命周期管理的基础设施复杂性，便于集成进 Agent 训练循环。

**Modal** 提供类似的托管执行模型，但 GPU 支持更强，适合需要在任务中运行 ML 工作负载的 Agent。

> **沙箱逃逸与安全**
>
> 代码执行沙箱是主要的攻击面。一个能力足够强的 Agent（或被 Prompt 注入的恶意载荷）可能尝试通过内核漏洞、网络外渗或资源耗尽来逃逸沙箱。纵深防御不可或缺：结合使用容器隔离、seccomp 配置、只读根文件系统、网络出站过滤以及 CPU/内存 cgroups。绝不要以宿主级别权限运行 Agent 生成的代码。

### Web 环境

Web 环境向 Agent 提供一个浏览器，并要求其在真实或模拟网站上完成任务。

**WebArena** [zhou2024webarena] 提供一个自托管测试平台，包含四个功能性 Web 应用——电商商城、社交论坛、GitLab 实例和 CMS——外加一个地图服务，共计 812 个长时程任务。Agent 通过浏览器自动化 API 进行交互；任务涉及多步导航、表单填写和信息检索。人类性能约 78\%；最先进的 LLM Agent 仅在 35--45\% 左右。

**VisualWebArena** [koh2024visualwebarena] 在 WebArena 基础上扩展，加入需要解读网页图像的视觉落地任务。观察为截图配合可访问性树；Agent 必须在两种模态中落地其动作。

**Mind2Web** [deng2024mind2web] 是一个大规模数据集，涵盖 137 个真实网站上的 2,000 个任务，通过人类示范收集。与 WebArena 不同，Mind2Web 聚焦于对未见网站的泛化，是更困难的分布外测试。

> **WebArena 任务示例**
>
> **任务**："在电商网站上找到价格低于 50 美元的最便宜的红色连衣裙并将其加入购物车。"
>
> **Agent 轨迹**：
>
> 1. 导航至服装类目。
> 2. 应用颜色筛选：红色。
> 3. 按价格升序排序。
> 4. 识别第一个低于 50 美元的商品。
> 5. 点击"加入购物车"。
> 6. 核实购物车内容。
>
> 环境对照真值商品检查最终购物车状态；若正确奖励为 1，否则为 0。

### 计算机使用环境

计算机使用环境（Computer Use）让 Agent 控制一个完整的桌面操作系统，通过截图和/或可访问性 API 进行观察。

**OSWorld** [xie2024osworld] 跨三种操作系统（Ubuntu、Windows、macOS）测试桌面自动化，涵盖 369 个任务，覆盖各类生产力应用（LibreOffice、VS Code、Chrome、GIMP 等）。Agent 通过截图观察，并以 `pyautogui` 风格的鼠标键盘命令行动。人-Agent 差距相当悬殊：标注者在约 72\% 的任务上成功，而最强 LLM Agent 仅能达到 $\sim$18\%，凸显了像素级 GUI 控制的难度。

**WindowsAgentArena** [bonatti2024windows] 专门聚焦于 Windows 11，包含 19 个应用上的 154 个任务，强调企业工作流：Excel 公式、PowerPoint 编辑、Outlook 邮件管理。

> **截图瓶颈**
>
> 计算机使用类 Agent 面临一个根本挑战：截图维度极高（通常 $1920 \times 1080 \times 3$ 像素），但其中大部分信息与当前动作无关。高效的 Agent 学会关注屏幕的小区域、使用可访问性树通过名称（而非像素坐标）识别可交互元素，并对此前访问过的 UI 状态维护一个紧凑的工作记忆。

### 软件工程环境

软件工程（Software Engineering, SWE）环境要求 Agent 解决真实世界的编程任务：修 bug、实现功能、写测试。

**SWE-bench** [jimenez2024swebench] 取材自 12 个广泛使用的 Python 项目（Django、Flask、scikit-learn 等）的 2,294 个真实 pull request。每个实例将一段 issue 描述与一份预留的测试套件配对，仅在应用正确补丁后测试才通过。Agent 必须理解仓库结构、定位相关代码、实施修复并用测试套件加以验证。**SWE-bench Verified** 子集（500 个 issue）经过人工正确性校验，是标准评估目标。

**SWE-agent** [yang2024sweagent] 既是基准环境也是 Agent 框架。它引入了*Agent-计算机接口（Agent-Computer Interface, ACI）*：一组为 LLM Agent 优化的 Shell 命令（如 `search_file`、`open`、`edit`），相比原始 bash 降低了动作空间复杂度。

> **SWE-bench 工作流**
>
> **输入**：一段 GitHub issue 描述，以及该 issue 提交时刻的完整仓库快照。
>
> **Agent 动作**：`find_file`、`view`、`edit`、`python -m pytest tests/`。
>
> **奖励**：Agent 补丁应用后若所有目标测试通过则为 1，否则为 0。不计部分得分。

### 科研环境

科研环境推动 Agent 走向自主知识生成：阅读论文、形成假设、设计实验、解读结果。

**PaperQA2** [lala2023paperqa] 是一个检索增强 Agent，通过搜索 PDF 语料库、抽取相关段落、综合出带引用的答案来回答科学问题。它既是文献依据型推理的工具，也是其基准。

**AI Scientist** [lu2024aiscientist] 是一个端到端的科研自动化系统：给定一个研究方向，Agent 即生成假设、撰写并运行实验、解读结果、产出论文初稿。该环境包含一个 Python 执行沙箱、文献搜索 API 和 LaTeX 编译器。

**MLAgentBench** [huang2024mlagentbench] 在机器学习工程任务上评估 Agent：在算力预算内提升给定数据集上的模型精度。Agent 可读取数据、编写训练脚本、运行实验并迭代。

### 游戏与仿真环境

游戏提供了丰富的长时程环境，具有定义良好的奖励信号且无真实世界后果。

**NetHack** [kuttler2020nethack] 是一款程序化生成的 roguelike 游戏，状态空间巨大，要求长期规划、物品管理以及对意外事件的适应。NetHack Learning Environment（NLE）提供了 Gym 兼容接口。

**Voyager / Minecraft** [wang2023voyager] 把 Minecraft 引擎用作一个开放式环境。Voyager 引入了难度逐级上升的任务课程（采集木材 $\to$ 制造工具 $\to$ 建造庇护所 $\to$ 探索下界）以及一个跨 Episode 累积可复用代码片段的技能库。

**GAIA** [mialon2023gaia] 提出 466 个问题，要求链式工具使用——Web 搜索、代码执行、文件解析——并按所需推理步数划分为三个难度等级。该基准鲜明地暴露出人类能力（约 92\% 准确率）与当前 LLM Agent（GPT-4 配合插件发布初期约 15\%，后续系统约 30\%）之间的鸿沟。

### 多 Agent 环境

多 Agent 环境涉及两个或更多 LLM Agent 之间以及与共享世界的交互。

- **协商（Negotiation）**：具有私有效用函数的 Agent 必须通过对话达成交易。经典环境包括 DealOrNoDeal [lewis2017dealornodeal] 和 CaSiNo [chawla2021casino]。
- **辩论（Debate）**：两个 Agent 持相对立场进行辩论；由裁判 Agent（或人类）评判论证质量。用于通过对抗压力引出真实的推理。
- **协作式任务完成**：具有互补能力（规划者、执行者、评论者）的 Agent 必须协作完成任一单独 Agent 无法解决的任务。相关框架包括 AutoGen [wu2023autogen]、CrewAI [moura2023crewai] 和 MetaGPT [hong2023metagpt]。
- **竞争性游戏**：Agent 在零和博弈（国际象棋、围棋、扑克）中对弈，对手本身就是另一个 LLM Agent。在这类环境中的自博弈已经在狭窄领域内产生了超人类的表现。

## OpenEnv：标准化的 Agent 环境接口

Agent 环境的激增带来了碎片化问题：每个环境都暴露不同的 API、使用不同的观察格式、需要不同的脚手架。**OpenEnv** [huggingface2025openenv] 是 Hugging Face 最近推出的开源框架，直接针对这一问题：它为 Agent 执行环境提供 Gymnasium 风格 [towers2024gymnasium] 的接口（`step()`、`reset()`、`state()`），以基于 Docker 的隔离部署通过 WebSocket 通信。OpenEnv 与更广泛的标准化努力互补，如 AgentGym [xi2024agentgym]（为 LLM Agent 跨多种环境提供统一格式平台）和 BrowserGym [drouin2024browsergym]（标准化 Web Agent 基准的观察和动作空间）。下文的设计原则概括了这些项目共同收敛出的最佳实践。

![OpenEnv 架构与一个 LLM Agent。Agent 通过 harness 循环进行推理，并调用类型化的 EnvClient。客户端通过 WebSocket 与运行在 Docker 容器内的 HTTPEnvServer 通信。RL 训练器（虚线）可选地包裹该循环，以采集 rollout 和奖励信号用于 Policy 优化。]({{ site.baseurl }}/figures/fig_064_openenv-arch.png)

### 标准化的 Agent--环境接口

OpenEnv 为 Agent 执行环境定义了一个类型化接口。其设计沿袭了 Gymnasium 的简洁，但面向通过 HTTP/WebSocket 与工具交互的 LLM Agent：

- `env.reset()` $\to$ `StepResult`：开始一个新 Episode；返回初始观察。
- `env.step(action)` $\to$ `StepResult(observation, reward, done)`：执行一个动作并返回结果观察、标量奖励和终止标志。
- `env.state()` $\to$ 当前环境状态（Episode ID、步数、特定环境的字段）。
- `env.close()`：释放资源（停止容器、关闭连接）。

动作和观察是强类型的 Python dataclass，针对每个环境特定。例如，编码环境定义 `CodeAction(code=...)` 并返回带 `stdout`、`stderr` 和 `exit_code` 的观察；游戏环境则定义其自己的动作/观察类型。这种逐环境的类型化使 Agent 获得结构化、可预测的接口，同时保持三个核心方法（`reset`、`step`、`state`）的通用性。

**架构。**

每个环境是一个继承自 `Environment` 的 Python 类（实现 `reset()` 和 `step()`）。它在 Docker 容器内通过 `HTTPEnvServer` 提供服务，后者暴露一个 FastAPI/WebSocket 端点。客户端使用 `EnvClient` 的特定环境子类，处理序列化和连接生命周期。容器可通过 `from_docker_image()` 本地启动，或通过 base URL 远程连接：

```python
from coding_env import CodeAction, CodingEnv

# 选项 1：启动本地 Docker 容器
client = CodingEnv.from_docker_image("coding-env:latest")

# 选项 2：连接到远端部署
# client = CodingEnv(base_url="http://localhost:8000")

# 与环境交互
result = client.reset()
print(result.observation.stdout)
print(result.observation.stderr)
print(result.observation.exit_code)

result = client.step(CodeAction(code="print(2 + 2)"))
print(result.observation.stdout)       # "4\n"
print(result.observation.exit_code)    # 0
print(result.reward, result.done)

# 查看状态
state = client.state()
print(state.episode_id, state.step_count)

client.close()
```

**环境即服务器。**

创建新环境只需实现 `Environment` 基类：

```python
from openenv.core.env_server import Environment, create_app
from dataclasses import dataclass

@dataclass
class MyAction:
    text: str

@dataclass
class MyObservation:
    response: str
    reward: float = 0.0
    done: bool = False

class MyEnvironment(Environment):
    def reset(self) -> MyObservation:
        return MyObservation(response="Ready")

    def step(self, action: MyAction) -> MyObservation:
        return MyObservation(response=f"Echo: {action.text}",
                             reward=1.0, done=False)

app = create_app(MyEnvironment(), MyAction, MyObservation)
# 运行：uvicorn module:app --host 0.0.0.0 --port 8000
```

**harness 集成（实验性）。**

RFC 0054 引入了一个面向 harness 的层，RL 训练框架通过 MCP 风格的工具调用与环境交互。`build_harness_rollout_func()` 辅助函数可产生一个 TRL 兼容的 rollout 函数，将 OpenEnv 直接桥接进 TorchForge [meta2025torchforge] 等现有训练流水线。

**治理。**

OpenEnv 由一个包含 Meta-PyTorch、NVIDIA、Unsloth、Modal、Prime Intellect、Reflection 和 Hugging Face 的技术委员会公开治理——确保该标准在广泛的产业输入下演进，而非任由单一厂商的议程主导。

### 环境注册与发现

OpenEnv 环境可部署为 Hugging Face Spaces 或本地 Docker 镜像，无需手动安装即可发现和使用。无论部署目标如何，客户端接口都保持一致：

```python
from echo_env import EchoAction, EchoEnv

# 连接到远端 HF Space 部署
client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
result = client.reset()
print(result.observation.echoed_message)  # "Echo environment ready!"

result = client.step(EchoAction(message="Hello!"))
print(result.observation.echoed_message)  # "Hello!"
print(result.reward)
client.close()
```

OpenEnv 生态系统已覆盖 70 多个环境（OpenSpiel 游戏、Atari、BrowserGym、编码沙箱、金融 RL、交通仿真等）。RFC 0025 提出一个正式的*工具发现*协议，使 Agent 能在运行时查询陌生环境接受哪些动作。

### 组合式环境

真实的 Agent 部署很少只使用单一工具。OpenEnv 支持通过类型化动作暴露多种能力的丰富环境。例如，编码环境在单一沙箱会话内支持代码执行、文件 I/O 和 Shell 命令：

```python
from coding_env import CodeAction, CodingEnv

client = CodingEnv.from_docker_image("coding-env:latest")
result = client.reset()

# 执行代码
result = client.step(CodeAction(code="x = 42\nprint(x)"))
print(result.observation.stdout)   # "42"
print(result.observation.exit_code)  # 0

# 状态在同一 Episode 内跨步骤保持
result = client.step(CodeAction(code="print(x + 1)"))
print(result.observation.stdout)   # "43"

state = client.state()
print(state.step_count)  # 2

client.close()
```

对于需要多样化工具访问（代码 + Web + 文件）的 Agent，OpenEnv 的 RFC 0036 提出 MCP（模型上下文协议，Model Context Protocol）集成，允许任何兼容 MCP 的工具服务器被封装为一个 OpenEnv 环境。此外，`openenv` CLI 可通过单条命令脚手架化、构建并部署新环境到 Hugging Face Spaces。

### 环境版本控制与可复现性

基准的可信度要求环境行为在评估时被冻结。最佳实践包括：

- **语义化版本**：`WebArena-v1.2.0` 保证在同一次要版本号内向后兼容。
- **Docker 镜像固定**：环境运行时被打包为带内容寻址哈希的 Docker 镜像。
- **基于种子的确定性**：所有随机元素（程序化生成、网络响应）都被设种子并记录，使任意轨迹都能精确重放。
- **排行榜快照**：公开排行榜在分数旁记录环境版本，防止基准悄然漂移。

## 构建自定义环境

### 面向 LLM Agent 的 Gymnasium 风格 API

Gymnasium API [towers2024gymnasium]（OpenAI Gym 的继任者）是 RL 环境的事实标准。将其适配到 LLM Agent 需要两处修改：(1) 观察和动作是字符串（或包含字符串的字典）而非数值数组；(2) `step` 方法必须处理异步工具执行。

### 奖励函数工程

LLM Agent 环境的奖励函数通常是*基于执行*的：环境在每个 Episode 后运行一个验证器，若任务被解决则返回 1，否则返回 0。对于没有明确验证器的任务，选项包括：

- **LLM-as-judge**：由一个独立的 LLM 根据任务描述对 Agent 的最终状态打分。
- **基于评分量表（Rubric）**：结构化的评分量表将任务分解为若干子标准，每项独立打分。
- **人工标注**：由人类评估者对一组随机抽样的轨迹打分；这些分数用于校准一个自动化代理指标。

### 状态管理与检查点

长时程任务可能需要数小时的墙钟时间。环境应支持：

- **状态序列化**：完整的环境状态（文件系统、浏览器 cookies、数据库内容）可序列化到磁盘并恢复。
- **Episode 中检查点**：Agent 可在任意步骤保存检查点并从中恢复，从而支持树搜索式探索。
- **轨迹日志**：每个观察、动作和奖励都被记录到结构化文件中，供离线分析和奖励模型训练使用。

### 用于训练数据采集的并行化

通过 RL 训练 LLM Agent 需要数百万次环境交互。并行化策略包括：

- **进程级并行**：派生 $N$ 个独立的环境进程；并行采集轨迹。
- **异步 rollout worker**：使用异步事件循环（如 `asyncio`）将 LLM 推理延迟与环境执行重叠。
- **向量化环境**：将多个环境批处理到单次 `step` 调用中，摊销 Python 开销。
- **云原生扩展**：使用作业调度器（Ray、SLURM）在集群中分发环境 worker，由中心化的回放缓冲（replay buffer）聚合轨迹。

## 环境--Agent 接口模式

图 展示了实践中使用的四种主要接口模式。

![四种 Agent--环境接口模式。(a) 基于文本是 LLM 最常用的方式；(b) 结构化 JSON 可实现精确解析；(c) 多模态结合截图与可访问性树用于 GUI 任务；(d) 流式接口支持没有离散回合边界的实时交互。]({{ site.baseurl }}/figures/fig_065_env-agent-interface.png)

**基于文本的观察/动作。**

Agent 接收字符串观察并产出字符串动作。环境解析该动作（如从 `<tool>...</tool>` 块中抽取工具调用）并以字符串形式返回结果。这是兼容性最佳的模式：任何 LLM 无需特殊架构即可参与。

**结构化 JSON 观察/动作。**

观察和动作是符合既定 Schema 的 JSON 对象。这支持严格校验（在执行前拒绝畸形动作）、结构化日志，以及更便利的程序化轨迹分析。代价是 Agent 必须可靠地产出有效 JSON，这要求微调或受约束解码。

**多模态（截图 + 可访问性树）。**

用于计算机使用类和 Web 环境。观察为元组 `(screenshot: PIL.Image, a11y_tree: dict)`。截图提供视觉上下文；可访问性树提供元素标识符，使得动作可不依赖像素级坐标。该混合方式比纯截图控制更鲁棒。

**流式 vs. 回合式交互。**

当前大多数环境采用回合式模型：Agent 产出完整动作，环境执行后返回下一个观察。流式环境允许 Agent 在观察到达时即接收部分观察（如长时间运行命令的输出），并在流过程中中断或重定向执行。这更接近人类与计算机交互的方式，但对 Agent 架构要求更复杂。

## 评估 harness 设计

评估 harness 是在一组基准上运行 Agent、采集结果、产出汇总统计的基础设施。优秀的 harness 设计与优秀的环境设计同等重要。

### 确定性 vs. 随机性环境

- **确定性环境**对相同动作序列产出相同观察序列。易于调试和复现，但可能无法反映真实世界的变异性。
- **随机性环境**引入随机性（程序化生成、网络延迟、用户模拟）。每个任务需要多次运行以估计平均性能和置信区间。

> **多少次运行才够？**
>
> 对于含 $N$ 个任务、二元奖励的基准，平均成功率的标准误差为 $\sqrt{p(1-p)/N}$。当 $N=500$ 且 $p=0.4$ 时，95\% 置信区间约为 $\pm 4.3\%$。对随机性环境，需乘以 $\sqrt{k}$，其中 $k$ 是每个任务的独立运行次数。随机性基准的常见做法是每任务 3--5 次运行。

### 留出测试环境

基准的可信度要求在*环境*级别（而不仅是任务级别）做严格的训练/测试划分。一个在 WebArena 任务上训练过的 Agent 应在训练时未使用的留出任务集合上评估。理想情况下，留出集合应覆盖与训练集不同的网站、任务类型和难度等级。

### 跨环境泛化

对 Agent 的终极考验是其在某一环境中学到的技能能否迁移到另一环境。跨环境评估协议测量：

- **零样本迁移**：在环境 A 上训练，无微调地在环境 B 上测试。
- **少样本适配**：评估前提供来自环境 B 的 $k$ 个示范。
- **持续学习**：依次在环境 A、B、C 上训练；在 C 上训练完成后测量三者上的性能。

### 人类基线采集

每个基准都应包含人类性能作为参考点。人类基线有三个用途：

1. 它们为任务难度建立上界。
2. 它们揭示任务是否根本可解（某些基准任务被证实是模糊的或不可解的）。
3. 它们为解读 Agent 分数提供校准点（"该 Agent 达到了人类性能的 40\%"）。

人类基线应从具备领域专长的工作者采集（如 SWE-bench 应使用软件工程师而非众包工人），并应包含任务用时测量以便进行效率比较。

## 代码示例：最小的自定义 LLM Agent 环境

> **用于 LLM Agent 训练的最小自定义环境**
>
> 以下 Python 类实现了一个文件编辑环境，Agent 必须修改一个 Python 文件以使一个失败的测试通过。它遵循通过为 LLM Agent 适配过的 Gymnasium API。

```python
"""
minimal_env.py  --  用于 LLM Agent 的最小文件编辑环境。

Agent 接收一个含 bug 的 Python 文件以及一个失败的测试。
它必须不断编辑该文件直到测试通过。
奖励：所有测试通过则为 1.0，否则为 0.0。
"""

from __future__ import annotations
import subprocess, shutil, tempfile, textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    observation: str          # 输入给 LLM 的文本
    reward: float             # 0.0 或 1.0
    terminated: bool          # Episode 结束（任务解决或达到最大步数）
    truncated: bool           # Episode 被截断（预算超出）
    info: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------

class FileEditEnv:
    """
    一个 Gymnasium 风格的 LLM 代码修复环境。

    观察空间 : str  （文件内容 + 测试输出）
    动作空间 : str  （以下之一：view、edit、run_tests、submit）
    奖励     : 所有测试通过为 1.0，否则为 0.0
    """

    MAX_STEPS = 20          # Episode 硬上限
    TIMEOUT   = 30          # 每次测试运行的秒数

    def __init__(self, buggy_code: str, test_code: str,
                 task_description: str):
        self.buggy_code       = buggy_code
        self.test_code        = test_code
        self.task_description = task_description
        self._workdir: Path | None = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        """初始化一个新 Episode；返回 (observation, info)。"""
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir)

        self._workdir    = Path(tempfile.mkdtemp(prefix="fileenv_"))
        self._step_count = 0

        # 写入初始文件
        (self._workdir / "solution.py").write_text(self.buggy_code)
        (self._workdir / "test_solution.py").write_text(self.test_code)

        obs = self._build_observation(
            action_taken="[Episode start]",
            test_output=self._run_tests()
        )
        return obs, {"step": 0}

    def step(self, action: str) -> StepResult:
        """执行一个 Agent 动作；返回 StepResult。"""
        self._step_count += 1
        action = action.strip()

        # --- 解析并分派动作 ---
        if action.startswith("view"):
            result_text = self._action_view()
        elif action.startswith("edit"):
            result_text = self._action_edit(action)
        elif action.startswith("run_tests"):
            result_text = self._run_tests()
        elif action.startswith("submit"):
            result_text = self._run_tests()
        else:
            result_text = (
                f"Unknown action: {action!r}\n"
                "Valid actions: view | edit <new_content> | "
                "run_tests | submit"
            )

        test_output = self._run_tests()
        passed      = "passed" in test_output and "failed" not in test_output
        reward      = 1.0 if passed else 0.0
        terminated  = passed or action.startswith("submit")
        truncated   = self._step_count >= self.MAX_STEPS

        obs = self._build_observation(action, test_output)
        return StepResult(obs, reward, terminated, truncated,
                          {"step": self._step_count,
                           "passed": passed})

    def render(self) -> str:
        """返回当前状态的人类可读摘要。"""
        if self._workdir is None:
            return "[Environment not initialised]"
        code = (self._workdir / "solution.py").read_text()
        return f"=== solution.py ===\n{code}\n"

    def close(self) -> None:
        """释放资源。"""
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir)
            self._workdir = None

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _action_view(self) -> str:
        code = (self._workdir / "solution.py").read_text()
        return f"Current solution.py:\n```python\n{code}\n```"

    def _action_edit(self, action: str) -> str:
        # 期望格式：edit\n```python\n<code>\n```
        try:
            new_code = action.split("```python")[1].split("```")[0]
            (self._workdir / "solution.py").write_text(new_code)
            return "File updated successfully."
        except IndexError:
            return "Edit failed: wrap new code in ```python ... ```"

    def _run_tests(self) -> str:
        result = subprocess.run(
            ["python", "-m", "pytest", "test_solution.py",
             "-v", "--tb=short", "--no-header"],
            cwd=self._workdir,
            capture_output=True, text=True,
            timeout=self.TIMEOUT
        )
        return result.stdout + result.stderr

    def _build_observation(self, action_taken: str,
                           test_output: str) -> str:
        code = (self._workdir / "solution.py").read_text()
        return textwrap.dedent(f"""
            TASK: {self.task_description}
            STEP: {self._step_count}/{self.MAX_STEPS}

            --- Last action ---
            {action_taken}

            --- Current solution.py ---
            {code}

            --- Test output ---
            {test_output}

            --- Available actions ---
            view                          # show current file
            edit\n```python\n<code>\n```  # replace file contents
            run_tests                     # run pytest
            submit                        # finalise and end episode
        """).strip()

# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BUGGY = "def add(a, b):\n    return a - b\n"   # bug：本应是加号
    TESTS = (
        "from solution import add\n"
        "def test_add(): assert add(2, 3) == 5\n"
    )

    env = FileEditEnv(BUGGY, TESTS, "Fix the add() function.")
    obs, _ = env.reset(seed=0)
    print(obs)

    # 模拟一次正确的编辑
    fix = "edit\n```python\ndef add(a, b):\n    return a + b\n```"
    result = env.step(fix)
    print(f"\nReward: {result.reward}  |  Terminated: {result.terminated}")
    env.close()
```

> **示例环境中的设计决策**
>
> - **纯文本接口**：观察和动作均为纯字符串，兼容任意 LLM。
> - **基于执行的奖励**：奖励来自实际运行测试套件，而非 LLM 裁判。这使其抗作弊且完美对齐。
> - **隔离子进程**：测试在带超时的独立进程中运行，防止死循环拖垮训练循环。
> - **兼容 Gymnasium**：`reset`/`step`/`render`/`close` 遵循标准 API，可直接用于 RL 训练框架。

## 主要 Agent 环境对比

下表总结了本节讨论的主要 Agent 环境的关键属性。

| 环境 | 观察类型 | 动作空间 | 领域 | 任务数 | 人类 | SoTA LLM |
| --- | --- | --- | --- | --- | --- | --- |
| WebArena | 文本 + DOM | 浏览器 API | Web 导航 | 812 | 78\% | $\sim$45\% |
| VisualWebArena | 截图 + DOM | 浏览器 API | 视觉 Web | 910 | 88\% | $\sim$35\% |
| Mind2Web | 截图 + DOM | 浏览器 API | 真实网站 | 2,000 | --- | $\sim$30\% |
| OSWorld | 截图 | 鼠标 + 键盘 | 桌面 OS | 369 | 72\% | $\sim$18\% |
| WindowsAgentArena | 截图 | 鼠标 + 键盘 | Windows 应用 | 154 | 75\% | $\sim$20\% |
| SWE-bench Verified | 文本（仓库） | Shell + 编辑器 | 代码修复 | 500 | 100\% | $\sim$50\% |
| GAIA (Level 1) | 文本 + 文件 | 工具调用 | 通用 QA | 165 | 92\% | $\sim$55\% |
| GAIA (Level 3) | 文本 + 文件 | 工具调用 | 困难 QA | 42 | 92\% | $\sim$10\% |
| NetHack (NLE) | 文本 + 字符图 | 离散动作 | Roguelike 游戏 | --- | $>$10k 分 | $\sim$5k 分 |
| Voyager (Minecraft) | 文本 + 代码 | 代码执行 | 开放世界游戏 | 课程 | --- | 15+ 科技树 |
| MLAgentBench | 文本 + 代码 | Shell + 编辑器 | ML 工程 | 13 | --- | $\sim$40\% |

> **解读对比表**
>
> 人类性能与 SoTA LLM 性能之间的差距，在*计算机使用*类任务上最大（OSWorld：72\% vs. 18\%），在*代码修复*上最小（SWE-bench：100\% vs. 50\%）。这一模式反映了动作空间的成熟度：LLM 已在海量代码上训练，但基于截图的交互数据相对较少。随着计算机使用类训练数据的积累，差距有望缩小。

## 小结

Agent 环境是训练和评估 LLM Agent 的基底。本节的关键要点是：

1. **环境不是可选项。** 安全探索、可复现评估和课程学习都需要结构化环境。若没有环境，Chatbot 评估与 Agent 评估之间的鸿沟无法跨越。
2. **四个维度都需谨慎设计。** 观察空间、动作空间、奖励信号和 Episode 结构各有可能让整个基准失效的失败模式。
3. **生态丰富但碎片化。** 代码沙箱、Web 环境、计算机使用类环境、SWE 环境、科研环境、游戏和多 Agent 竞技场各测试不同能力。任何单一环境都不充分。
4. **标准化很重要。** OpenEnv [huggingface2025openenv] 提供 Gymnasium 风格的 API，配合 Docker 隔离以及把 Hugging Face Spaces 作为注册中心——降低了构建新环境以及在不同环境间比较 Agent 的成本。
5. **人类差距真实存在但正在收窄。** 当前 LLM Agent 在大多数基准上达到人类性能的 20--50\%。进展最快的是训练数据充裕的领域（代码），最慢的是需要细粒度感知的领域（GUI 控制）。

> **Agent 环境中的开放研究问题**
>
> - 如何为那些正确性带有主观性或上下文依赖的任务设计奖励函数？
> - 单一 Agent 架构能否在无需任务特定微调的前提下跨基于文本和多模态环境泛化？
> - 训练所需的环境保真度应当达到何种程度？在简化的仿真器上训练能迁移到真实部署吗？
> - 当 LLM 在可能包含基准答案的越来越大的 Web 语料上训练时，如何防止基准污染？
