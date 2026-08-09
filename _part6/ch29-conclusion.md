---
layout: home
title: 总结与未来方向
permalink: /part6/ch29-conclusion.html
---

## 总结

本书完整勾勒了从 Transformer 基础、面向对齐的强化学习，到自主智能体系统构建的全过程。跨越各章节，浮现出以下关键主题：

1. **对齐是一个系统工程问题。** 仅有一个好的损失函数并不够。生产级 RLHF 需要同时管理 4 个以上的模型、在数百块 GPU 上分布计算、处理容错，并监控奖励黑客行为。
2. **不存在单一最优方法。** PPO 仍是追求极致质量的金标准，但需要巨大的工程投入。DPO 及其变体则为基础设施有限的团队提供了极具吸引力的折中。GRPO 弥合了可验证奖励领域的鸿沟。正确的选择取决于你的数据、算力预算与质量门槛。
3. **推理能力可以从奖励中涌现。** DeepSeek-R1 证明，思维链、自我验证和回溯能力可以从简单的二值奖励信号与组相对优化中涌现，无需对推理过程进行显式示范。测试时算力扩展意味着“更小的模型 + 更多思考”可以匹敌更大模型。
4. **标准开启生态。** MCP 将工具集成问题从 $N \times M$ 降到 $N + M$。A2A 让不同团队构建的 Agent 在不共享内部实现的前提下协作。这些协议之于智能体 AI，正如 HTTP 之于 Web——是开放生态的基础设施。
5. **Agent 是自然的下一步。** 一旦模型被对齐，前沿就从“单次回复有多好？”转向“模型能否自主解决多步问题？”这需要新的训练范式（带环境奖励的智能体 RL）、新的基础设施（运行框架、工具协议、记忆系统）以及新的评估方法（轨迹级基准）。
6. **评估驱动一切。** 没有严格的评估——从奖励模型验证到 Agent 任务成功率，从污染检测到 LLM-as-Judge 校准——进步无法度量、回退也不可见。你所选择的基准塑造了你所构建的系统。
7. **简单才能扩展。** 最可靠的生产级 Agent 采用满足需求的最简单架构——优先使用 prompt chaining 与 routing，而非自主循环；优先使用单 Agent，而非多 Agent 集群。复杂度只能通过被证实的需要去赢得。

## 前路：开放挑战

### 从交互中学习

当前的 RLHF 流水线 [[99]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-ouyang2022training)] 将对齐视为一次性的训练阶段。未来则指向**从部署中持续学习**：Agent 能够从每一次用户交互、工具失败与环境观察中改进——同时避免灾难性遗忘（Catastrophic Forgetting） [[192]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-kirkpatrick2017overcoming)] 与奖励漂移。关键开放问题：

- 非平稳奖励分布下的在线学习。
- 生产环境中的安全探索 [[395]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-garcia2015comprehensive)]（在学习过程中避免有害动作）。
- 在长 Agent 轨迹（数百次工具调用）上的高效信用分配。

### 可扩展监督

当 Agent 能力越来越强时，人类监督本身成为瓶颈。当前方法（RLHF [[99]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-ouyang2022training)]、Constitutional AI [[110]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-bai2022constitutional)]）依赖人类评估模型输出——但当模型输出超出人类理解能力时会发生什么？

- **递归奖励建模（Recursive Reward Modeling）** [[158]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-christiano2017deep)]：用 AI 协助人类评估 AI。
- **辩论与放大（Debate and Amplification）** [[396]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-irving2018debate)]：两个模型相互辩论，人类裁判哪一方论证更令人信服。
- **基于过程的监督（Process-based Supervision）** [[231]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lightman2023lets)]：奖励正确的推理步骤，而不仅是最终答案。
- **机制可解释性（Mechanistic Interpretability）** [[46]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-olsson2022context)]：理解模型内部*在做什么*，而不仅是它输出了什么。

### 世界模型与规划

当前 Agent 是反应式的——它们一次只观察并响应一步。未来 Agent 将需要**内部世界模型（World Models）** [[153]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hafner2020dream)] 以支持前瞻式规划：

- 在执行动作之前预测其后果。
- 对可能动作序列进行树搜索（类似 AlphaGo [[154]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-silver2016mastering)] 与 MuZero [[152]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schrittwieser2020mastering)]，但面向开放任务）。
- 从交互轨迹中学习环境的状态转移。

### 多智能体生态

A2A 协议 [[360]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-google-a2a-2025)] 与多智能体框架预示着这样一种未来：成百上千的专用 Agent 协作、协商并相互委派——形成所谓的“智能体经济（Economy of Agents）” [[385]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-nisan2007algorithmic)]。开放挑战包括：

- 服务于不同委托方的 Agent 之间的信任与验证。
- 竞争性场景中涌现的合作与涌现的欺骗的对比 [[397]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hubinger2024sleeper)]。
- 资源分配（算力、工具访问、优先级）的市场机制。
- 治理：当一条由 10 个 Agent 组成的链路产生有害结果时，由谁负责？ [[398]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-amodei2016concrete)]

### 智能体安全与信任

自主 Agent 继承了其底层 LLM 的全部安全漏洞，再加上工具访问、多智能体委派与持久化记忆所带来的新攻击面（第 19--21 章）。关键的未解问题：

- **规模化的 Prompt 注入** [[399]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-greshake2023indirect)]：随着 Agent 消费不可信内容（网页、邮件、API 响应），间接 prompt 注入成为系统性问题。目前尚无稳健的防御手段。
- **混淆代理（Confused Deputy）攻击**：持有合法凭证的 Agent 可能被骗，代表潜伏在数据流中的攻击者滥用其凭证 [[323]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic-mcp-2024)]。
- **不削弱能力的沙箱化**：最小权限执行限制了 Agent 能做的事，但过度严格的沙箱会抹杀智能体的价值。如何找到合适边界仍是开放设计问题。
- **审计与归因**：当 Agent 链跨越多个组织（通过 A2A [[360]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-google-a2a-2025)]）时，追溯*谁*授权了*什么*动作在架构上仍未解决。
- **信任校准**：Agent 必须学会何时*不要*信任——工具响应是否真实、其他 Agent 的声明是否经过验证。

### 超越基准的评估

第 14 章揭示了基准塑造了我们所构建的系统——然而当前评估仍存在关键缺口：

- **真实部署指标**：SWE-bench [[254]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-jimenez2024swebench)] 与 GAIA [[350]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-mialon2023gaia)] 等基准衡量的是孤立任务；而生产 Agent 面对的是模糊目标、变动的需求与多轮恢复。
- **奖励模型有效性**：RLHF 假设 RM 能捕捉人类偏好，但奖励黑客 [[400]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-skalse2022defining)] 与分布漂移会在规模化时侵蚀这一假设。
- **成本—质量前沿**：两个 Agent 可能取得相同准确率，但其中一个的 token 成本高出 10$\times$。评估必须开始关注成本。
- **分布漂移下的安全性**：在测试中安全的 Agent 在新输入下可能行为不安全。智能体规模上的对抗评估 [[137]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-perez2022red)] 与红队测试仍不成熟。

### 效率与可获得性

对 70B 模型做 RLHF 训练成本在 \$10K--\$100K。运行一个自主 Agent 在每个复杂任务上的成本是 \$1--\$50。要让智能体 AI 产生广泛影响，需要：

- 将智能体能力从大模型蒸馏到小模型 [[123]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hinton2015distilling), [401]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-kim2024distillation)]。
- 更高效的 RL 算法（更少样本、更低方差） [[149]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schulman2017proximal)]。
- 无需云端往返即可运行的端侧 Agent。
- 在智能体任务上达到闭源质量的开放权重模型 [[156]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-deepseek2025r1)]。

## 延伸阅读

### 基础论文

- **Attention Is All You Need** [[8]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-vaswani2017attention)] --- Transformer 架构。
- **RLHF / InstructGPT** [[99]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-ouyang2022training)] --- 首个大规模 RLHF 部署。
- **PPO** [[149]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schulman2017proximal)] --- 近端策略优化（Proximal Policy Optimization）。
- **DPO** [[159]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rafailov2023direct)] --- 直接偏好优化（Direct Preference Optimization）。
- **GRPO / DeepSeek-R1** [[168]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shao2024deepseekmath), [156]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-deepseek2025r1)] --- 组相对策略优化与涌现推理。
- **ReAct** [[108]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-yao2023react)] --- LLM Agent 的 Reasoning + Acting 框架。
- **Toolformer** [[320]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-schick2023toolformer)] --- 教 LLM 学会使用工具。
- **RAG** [[109]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-lewis2020retrieval)] --- 检索增强生成（Retrieval-Augmented Generation）。

### 系统与扩展

- **Megatron-LM** [[195]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shoeybi2019megatron)] --- 张量并行与流水线并行。
- **DeepSpeed ZeRO** [[201]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-rajbhandari2020zero)] --- 显存高效的分布式训练。
- **vLLM** [[138]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-kwon2023efficient)] --- 为高效 LLM 服务而生的 PagedAttention。
- **FlashAttention** [[17]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-dao2022flashattention)] --- IO 感知的精确 attention。

### 智能体 AI

- **Building Effective Agents** [[330]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic2024buildingagents)] --- 设计模式与原则。
- **Voyager** [[216]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-wang2023voyager)] --- Minecraft 中带有技能库的开放式 Agent。
- **SWE-bench** [[254]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-jimenez2024swebench)] --- 自主软件工程基准。
- **OSWorld** [[344]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-xie2024osworld)] --- 完整的计算机使用基准。
- **GAIA** [[350]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-mialon2023gaia)] --- 面向真实任务的通用 AI 助手基准。
- **MemGPT** [[304]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-packer2023memgpt)] --- 受操作系统启发、面向无界上下文的记忆管理。
- **Model Context Protocol** [[323]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-anthropic-mcp-2024)] --- 工具集成的开放标准。
- **Agent-to-Agent Protocol** [[360]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-google-a2a-2025)] --- 智能体间通信标准。

### 对齐与安全

- **Constitutional AI** [[110]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-bai2022constitutional)] --- 自监督对齐。
- **Sleeper Agents** [[397]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-hubinger2024sleeper)] --- 关于欺骗性对齐的担忧。
- **Reflexion** [[212]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-shinn2023reflexion)] --- 从口头自我反思中学习。
- **Indirect Prompt Injection** [[399]({{ site.baseurl }}/part6/ch29-conclusion.html#ref-greshake2023indirect)] --- LLM 集成应用的安全风险。

### 在线资源

- **HuggingFace TRL**：https://github.com/huggingface/trl --- 生产级 RL 库。
- **LangGraph**：https://github.com/langchain-ai/langgraph --- Agent 工作流图。
- **OpenAI Agents SDK**：https://github.com/openai/openai-agents-python --- 官方 Agent 框架。
- **DeepSpeed-Chat**：https://github.com/microsoft/DeepSpeedExamples --- 端到端 RLHF 流水线。
- **DSPy**：https://github.com/stanfordnlp/dspy --- 声明式 prompt 优化。
- **AutoGen**：https://github.com/microsoft/autogen --- 多智能体对话框架。

*“预测未来最好的方式，就是把它创造出来。”*

--- Alan Kay

---

## Bibliography（参考文献）

1. <a id="ref-sennrich2016bpe"></a>Sennrich, R., Haddow, B., Birch, A. "Neural Machine Translation of Rare Words with Subword Units." Proceedings of the 54th Annual Meeting of the ACL, 2016. <https://arxiv.org/abs/1508.07909>
2. <a id="ref-openai2023gpt4"></a>OpenAI. "GPT-4 Technical Report." arXiv Preprint arXiv:2303.08774, 2023.
3. <a id="ref-grattafiori2024llama3"></a>Grattafiori, A., Dubey, A., Jauhri, A., et al. "The Llama 3 Herd of Models." arXiv Preprint arXiv:2407.21783, 2024. <https://arxiv.org/abs/2407.21783>
4. <a id="ref-jiang2023mistral"></a>Jiang, A., Sablayrolles, A., Mensch, A., et al. "Mistral 7B." arXiv Preprint arXiv:2310.06825, 2023. <https://arxiv.org/abs/2310.06825>
5. <a id="ref-devlin2019bert"></a>Devlin, J., Chang, M., Lee, K., Toutanova, K. "BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding." Proceedings of NAACL-HLT, 2019. <https://arxiv.org/abs/1810.04805>
6. <a id="ref-sanh2019distilbert"></a>Sanh, V., Debut, L., Chaumond, J., Wolf, T. "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." arXiv Preprint arXiv:1910.01108, 2019.
7. <a id="ref-radford2019gpt2"></a>Radford, A., Wu, J., Child, R., Luen, D., Amodei, D., Sutskever, I. "Language Models Are Unsupervised Multitask Learners." OpenAI Blog, 2019. <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
8. <a id="ref-vaswani2017attention"></a>Vaswani, A., Shazeer, N., Parmar, N., et al. "Attention Is All You Need." Advances in Neural Information Processing Systems (NeurIPS), 2017. <https://arxiv.org/abs/1706.03762>
9. <a id="ref-qwen2024qwen25"></a>Team, Q. "Qwen2.5: A Party of Foundation Models." arXiv Preprint arXiv:2412.15115, 2024. <https://arxiv.org/abs/2412.15115>
10. <a id="ref-raffel2020t5"></a>Raffel, C., Shazeer, N., Roberts, A., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." Journal of Machine Learning Research, 2020. <https://arxiv.org/abs/1910.10683>
11. <a id="ref-lewis2020bart"></a>Lewis, M., Liu, Y., Goyal, N., et al. "BART: Denoising Sequence-to-Sequence Pre-Training for Natural Language Generation, Translation, and Comprehension." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020. <https://arxiv.org/abs/1910.13461>
12. <a id="ref-chung2022flan"></a>Chung, H., Hou, L., Longpre, S., et al. "Scaling Instruction-Finetuned Language Models." Journal of Machine Learning Research, 2024. <https://arxiv.org/abs/2210.11416>
13. <a id="ref-liu2019roberta"></a>Liu, Y., Ott, M., Goyal, N., et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv Preprint arXiv:1907.11692, 2019. <https://arxiv.org/abs/1907.11692>
14. <a id="ref-firth1957synopsis"></a>Firth, J. "A Synopsis of Linguistic Theory, 1930–1955." Studies in Linguistic Analysis, 1957.
15. <a id="ref-ethayarajh2019contextual"></a>Ethayarajh, K. "How Contextual Are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019. <https://arxiv.org/abs/1909.00512>
16. <a id="ref-su2021whitening"></a>Su, J., Cao, J., Liu, W., Ou, Y. "Whitening Sentence Representations for Better Semantics and Faster Retrieval." arXiv Preprint arXiv:2103.15316, 2021. <https://arxiv.org/abs/2103.15316>
17. <a id="ref-dao2022flashattention"></a>Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2205.14135>
18. <a id="ref-beltagy2020longformer"></a>Beltagy, I., Peters, M., Cohan, A. "Longformer: The Long-Document Transformer." arXiv Preprint arXiv:2004.05150, 2020. <https://arxiv.org/abs/2004.05150>
19. <a id="ref-zaheer2020bigbird"></a>Zaheer, M., Guruganesh, G., Dubey, K., et al. "Big Bird: Transformers for Longer Sequences." Advances in Neural Information Processing Systems (NeurIPS), 2020. <https://arxiv.org/abs/2007.14062>
20. <a id="ref-guo2022longt5"></a>Guo, M., Ainslie, J., Uthus, D., et al. "LongT5: Efficient Text-to-Text Transformer for Long Sequences." Findings of the Association for Computational Linguistics: NAACL 2022, 2022. <https://arxiv.org/abs/2112.07916>
21. <a id="ref-gu2023mamba"></a>Gu, A., Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv Preprint arXiv:2312.00752, 2023. <https://arxiv.org/abs/2312.00752>
22. <a id="ref-peng2023rwkv"></a>Peng, B., Alcaide, E., Anthony, Q., et al. "RWKV: Reinventing RNNs for the Transformer Era." Findings of the Association for Computational Linguistics: EMNLP 2023, 2023. <https://arxiv.org/abs/2305.13048>
23. <a id="ref-zhang2023h2o"></a>Zhang, Z., Sheng, Y., Zhou, T., et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2306.14048>
24. <a id="ref-xiao2024streamingllm"></a>Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. "Efficient Streaming Language Models with Attention Sinks." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2309.17453>
25. <a id="ref-liu2024kivi"></a>Liu, Z., Yuan, J., Jin, H., et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." International Conference on Machine Learning (ICML), 2024. <https://arxiv.org/abs/2402.02750>
26. <a id="ref-liu2023ringattention"></a>Liu, H., Zaharia, M., Abbeel, P. "Ring Attention with Blockwise Transformers for Near-Infinite Context." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2310.01889>
27. <a id="ref-workshop2023bloom"></a>Workshop, B. "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model." arXiv Preprint arXiv:2211.05100, 2023. <https://arxiv.org/abs/2211.05100>
28. <a id="ref-mosaicml2023mpt"></a>MosaicML. "MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs." MosaicML Blog, 2023. <https://www.mosaicml.com/blog/mpt-7b>
29. <a id="ref-su2024roformer"></a>Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing, 2024.
30. <a id="ref-peng2023yarn"></a>Peng, B., Quesnelle, J., Fan, H., Shao, E. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv Preprint arXiv:2309.00071, 2023.
31. <a id="ref-press2022train"></a>Press, O., Smith, N., Lewis, M. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization." ICLR, 2022.
32. <a id="ref-anthropic2024claude3"></a>Anthropic. "The Claude 3 Model Family: Opus, Sonnet, Haiku." Anthropic Technical Report, 2024. <https://www.anthropic.com/news/claude-3-family>
33. <a id="ref-geminiteam2024gemini15"></a>Gemini, G. "Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context." arXiv Preprint arXiv:2403.05530, 2024. <https://arxiv.org/abs/2403.05530>
34. <a id="ref-chen2023extending"></a>Chen, S., Wong, S., Chen, L., Tian, Y. "Extending Context Window of Large Language Models via Positional Interpolation." arXiv Preprint arXiv:2306.15595, 2023. <https://arxiv.org/abs/2306.15595>
35. <a id="ref-liu2024lost"></a>Liu, N., Lin, K., Hewitt, J., et al. "Lost in the Middle: How Language Models Use Long Contexts." Transactions of the Association for Computational Linguistics, 2024. <https://arxiv.org/abs/2307.03172>
36. <a id="ref-geva2021transformer"></a>Geva, M., Schuster, R., Berant, J., Levy, O. "Transformer Feed-Forward Layers Are Key-Value Memories." Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.
37. <a id="ref-ba2016layernorm"></a>Ba, J., Kiros, J., Hinton, G. "Layer Normalization." arXiv Preprint arXiv:1607.06450, 2016. <https://arxiv.org/abs/1607.06450>
38. <a id="ref-zhang2019rmsnorm"></a>Zhang, B., Sennrich, R. "Root Mean Square Layer Normalization." Advances in Neural Information Processing Systems (NeurIPS), 2019. <https://arxiv.org/abs/1910.07467>
39. <a id="ref-meta2025llama4"></a>AI, M. "The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI." Meta AI Blog, 2025. <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>
40. <a id="ref-jiang2024mistrallarge2"></a>AI, M. "Mistral Large 2." Mistral AI Blog, 2024. <https://mistral.ai/news/mistral-large-2407/>
41. <a id="ref-deepseekv3"></a>DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv Preprint arXiv:2412.19437, 2024. <https://arxiv.org/abs/2412.19437>
42. <a id="ref-xiao2024efficient"></a>Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. "Efficient Streaming Language Models with Attention Sinks." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2309.17453>
43. <a id="ref-fu2024data"></a>Fu, Y., Panda, R., Niu, X., et al. "Data Engineering for Scaling Language Models to 128K Context." arXiv Preprint arXiv:2402.10171, 2024. <https://arxiv.org/abs/2402.10171>
44. <a id="ref-gu2024mamba"></a>Gu, A., Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. <https://arxiv.org/abs/2312.00752>
45. <a id="ref-voita2019analyzing"></a>Voita, E., Talbot, D., Moiseev, F., Sennrich, R., Titov, I. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. <https://arxiv.org/abs/1905.09418>
46. <a id="ref-olsson2022context"></a>Olsson, C., Elhage, N., Nanda, N., et al. "In-Context Learning and Induction Heads." Transformer Circuits Thread, 2022. <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>
47. <a id="ref-wu2024retrieval"></a>Wu, Z., Arora, A., Wang, Z., Kim, B., Huang, T. "Retrieval Head Mechanistically Explains Long-Context Factuality." arXiv Preprint arXiv:2404.15574, 2024. <https://arxiv.org/abs/2404.15574>
48. <a id="ref-vig2019bertviz"></a>Vig, J. "A Multiscale Visualization of Attention in the Transformer Model." Proceedings of the 57th ACL: System Demonstrations, 2019. <https://arxiv.org/abs/1906.05714>
49. <a id="ref-abnar2020quantifying"></a>Abnar, S., Zuidema, W. "Quantifying Attention Flow in Transformers." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020. <https://arxiv.org/abs/2005.00928>
50. <a id="ref-barkan2021grad"></a>Barkan, O., Hauon, E., Caciularu, A., Dagan, I., Koenigstein, N. "Grad-SAM: Explaining Transformers via Gradient Self-Attention Maps." Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM), 2021. <https://arxiv.org/abs/2104.13299>
51. <a id="ref-jain2019attention"></a>Jain, S., Wallace, B. "Attention Is Not Explanation." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019. <https://arxiv.org/abs/1902.10186>
52. <a id="ref-cunningham2023sparse"></a>Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2309.08600>
53. <a id="ref-bricken2023monosemanticity"></a>Bricken, T., Templeton, A., Batson, J., et al. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Transformer Circuits Thread, 2023. <https://transformer-circuits.pub/2023/monosemantic-features/index.html>
54. <a id="ref-templeton2024scaling"></a>Templeton, A., Conerly, T., Marcus, J., et al. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Transformer Circuits Thread, 2024. <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>
55. <a id="ref-anthropic2026nla"></a>Anthropic. "Natural Language Autoencoders: Interpreting Neural Networks with Natural Language Descriptions." Anthropic Research Blog, 2026. <https://www.anthropic.com/research/natural-language-autoencoders>
56. <a id="ref-rumelhart1986learning"></a>Rumelhart, D., Hinton, G., Williams, R. "Learning Representations by Back-Propagating Errors." Nature, 1986. <https://doi.org/10.1038/323533a0>
57. <a id="ref-robbins1951stochastic"></a>Robbins, H., Monro, S. "A Stochastic Approximation Method." The Annals of Mathematical Statistics, 1951.
58. <a id="ref-kingma2015adam"></a>Kingma, D., Ba, J. "Adam: A Method for Stochastic Optimization." International Conference on Learning Representations (ICLR), 2015. <https://arxiv.org/abs/1412.6980>
59. <a id="ref-loshchilov2019adamw"></a>Loshchilov, I., Hutter, F. "Decoupled Weight Decay Regularization." arXiv Preprint arXiv:1711.05101, 2019. <https://arxiv.org/abs/1711.05101>
60. <a id="ref-hu2024minicpm"></a>Hu, S., Tu, Y., Han, X., et al. "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies." arXiv Preprint arXiv:2404.06395, 2024. <https://arxiv.org/abs/2404.06395>
61. <a id="ref-dao2023flashattention2"></a>Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2307.08691>
62. <a id="ref-shah2024flashattention3"></a>Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision." arXiv Preprint arXiv:2407.08691, 2024. <https://arxiv.org/abs/2407.08691>
63. <a id="ref-zadouri2026flashattention4"></a>Zadouri, T., Shah, J., Bikshandi, G., Dao, T. "FlashAttention-4: Hardware-Efficient Attention on Blackwell GPUs with Minimal Software Design." arXiv Preprint arXiv:2603.05451, 2026. <https://arxiv.org/abs/2603.05451>
64. <a id="ref-hoffmann2022chinchilla"></a>Hoffmann, J., Borgeaud, S., Mensch, A., et al. "Training Compute-Optimal Large Language Models." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2203.15556>
65. <a id="ref-brown2020language"></a>Brown, T., Mann, B., Ryder, N., et al. "Language Models Are Few-Shot Learners." NeurIPS, 2020.
66. <a id="ref-lee2022deduplicating"></a>Lee, K., Ippolito, D., Nystrom, A., et al. "Deduplicating Training Data Makes Language Models Better." Proceedings of the 60th Annual Meeting of the ACL, 2022. <https://arxiv.org/abs/2107.06499>
67. <a id="ref-zhou2023lima"></a>Zhou, C., Liu, P., Xu, P., et al. "LIMA: Less Is More for Alignment." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2305.11206>
68. <a id="ref-hsu2024liger"></a>Hsu, P., Dai, Y., Kothapalli, V., et al. "Liger-Kernel: Efficient Triton Kernels for LLM Training." arXiv Preprint arXiv:2410.10989, 2024. <https://arxiv.org/abs/2410.10989>
69. <a id="ref-unsloth2024"></a>Han, D., Han, M. "Unsloth: Efficient LLM Fine-Tuning." 2024. <https://github.com/unslothai/unsloth>
70. <a id="ref-torchtune2024"></a>Team, P. "Torchtune: PyTorch Native Post-Training Library." 2024. <https://github.com/pytorch/torchtune>
71. <a id="ref-jain2024neftune"></a>Jain, N., Chiang, P., Wen, Y., et al. "NEFTune: Noisy Embeddings Improve Instruction Finetuning." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2310.05914>
72. <a id="ref-hu2021lora"></a>Hu, E., Shen, Y., Wallis, P., et al. "LoRA: Low-Rank Adaptation of Large Language Models." International Conference on Learning Representations (ICLR), 2022. <https://arxiv.org/abs/2106.09685>
73. <a id="ref-aghajanyan2020intrinsic"></a>Aghajanyan, A., Gupta, S., Zettlemoyer, L. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. <https://arxiv.org/abs/2012.13255>
74. <a id="ref-kalajdzievski2023rslora"></a>Kalajdzievski, D. "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA." arXiv Preprint arXiv:2312.03732, 2023. <https://arxiv.org/abs/2312.03732>
75. <a id="ref-dettmers2023qlora"></a>Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. "QLoRA: Efficient Finetuning of Quantized Language Models." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2305.14314>
76. <a id="ref-liu2024dora"></a>Liu, S., Wang, C., Yin, H., et al. "DoRA: Weight-Decomposed Low-Rank Adaptation." arXiv Preprint arXiv:2402.09353, 2024. <https://arxiv.org/abs/2402.09353>
77. <a id="ref-hayou2024loraplus"></a>Hayou, S., Ghosh, N., Yu, B. "LoRA+: Efficient Low Rank Adaptation of Large Models." arXiv Preprint arXiv:2402.12354, 2024. <https://arxiv.org/abs/2402.12354>
78. <a id="ref-zhang2023adalora"></a>Zhang, Q., Chen, M., Bukharin, A., et al. "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning." arXiv Preprint arXiv:2303.10512, 2023. <https://arxiv.org/abs/2303.10512>
79. <a id="ref-kopiczko2024vera"></a>Kopiczko, D., Blankevoort, T., Nagel, M. "VeRA: Vector-Based Random Matrix Adaptation." arXiv Preprint arXiv:2310.11454, 2024. <https://arxiv.org/abs/2310.11454>
80. <a id="ref-houlsby2019adapters"></a>Houlsby, N., Giber, A., Jastrzebski, S., et al. "Parameter-Efficient Transfer Learning for NLP." International Conference on Machine Learning (ICML), 2019. <https://arxiv.org/abs/1902.00751>
81. <a id="ref-li2021prefix"></a>Li, X., Liang, P. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." Proceedings of the 59th Annual Meeting of the ACL, 2021. <https://arxiv.org/abs/2101.00190>
82. <a id="ref-lester2021prompt"></a>Lester, B., Al-Rfou, R., Constant, N. "The Power of Scale for Parameter-Efficient Prompt Tuning." Proceedings of the 2021 Conference on EMNLP, 2021. <https://arxiv.org/abs/2104.08691>
83. <a id="ref-liu2022ia3"></a>Liu, H., Tam, D., Muqeeth, M., et al. "Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper Than in-Context Learning." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2205.05638>
84. <a id="ref-zaken2022bitfit"></a>Zaken, E., Ravfogel, S., Goldberg, Y. "BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-Based Masked Language-Models." Proceedings of the 60th Annual Meeting of the ACL, 2022. <https://arxiv.org/abs/2106.10199>
85. <a id="ref-shazeer2017outrageously"></a>Shazeer, N., Mirhoseini, A., Maziarz, K., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." International Conference on Learning Representations (ICLR), 2017. <https://arxiv.org/abs/1701.06538>
86. <a id="ref-jiang2024mixtral"></a>Jiang, A., Sablayrolles, A., Roux, A., et al. "Mixtral of Experts." arXiv Preprint arXiv:2401.04088, 2024. <https://arxiv.org/abs/2401.04088>
87. <a id="ref-jang2017categorical"></a>Jang, E., Gu, S., Poole, B. "Categorical Reparameterization with Gumbel-Softmax." International Conference on Learning Representations (ICLR), 2017.
88. <a id="ref-deepseekv2"></a>DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv Preprint arXiv:2405.04434, 2024. <https://arxiv.org/abs/2405.04434>
89. <a id="ref-fedus2022switch"></a>Fedus, W., Zoph, B., Shazeer, N. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." Journal of Machine Learning Research, 2022.
90. <a id="ref-databricks2024dbrx"></a>Databricks. "DBRX: A New State-of-the-Art Open LLM." Databricks Blog, 2024. <https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm>
91. <a id="ref-zhang2025verbalized"></a>Zhang, J., Yu, S., Chong, D., et al. "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity." arXiv Preprint arXiv:2510.01171, 2025. <https://arxiv.org/abs/2510.01171>
92. <a id="ref-vijayakumar2018diverse"></a>Vijayakumar, A., Cogswell, M., Selvaraju, R., et al. "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models." AAAI, 2018.
93. <a id="ref-nguyen2024minp"></a>Nguyen, M. "Min-p Sampling: A Simple Baseline for Better LLM Decoding." arXiv Preprint arXiv:2310.06022, 2024.
94. <a id="ref-li2023contrastive"></a>Li, X., Holtzman, A., Fried, D., et al. "Contrastive Decoding: Open-Ended Text Generation as Optimization." ACL, 2023.
95. <a id="ref-willard2023outlines"></a>Willard, B., Louf, R. "Efficient Guided Generation for Large Language Models." arXiv Preprint arXiv:2307.09702, 2023. <https://arxiv.org/abs/2307.09702>
96. <a id="ref-dong2024xgrammar"></a>Dong, Y., Moon, C., Wang, Y., et al. "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models." arXiv Preprint arXiv:2411.15100, 2024.
97. <a id="ref-xie2022explanation"></a>Xie, S., Raghunathan, A., Liang, P., Ma, T. "An Explanation of in-Context Learning as Implicit Bayesian Inference." Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022. <https://arxiv.org/abs/2111.02080>
98. <a id="ref-todd2024function"></a>Todd, E., Li, M., Sharma, A., Mueller, A., Wallace, B., Bau, D. "Function Vectors in Large Language Models." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2310.15213>
99. <a id="ref-ouyang2022training"></a>Ouyang, L., Wu, J., Jiang, X., et al. "Training Language Models to Follow Instructions with Human Feedback." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2203.02155>
100. <a id="ref-lu2022fantastically"></a>Lu, Y., Bartolo, M., Moore, A., Riedel, S., Stenetorp, P. "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity." Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), 2022. <https://arxiv.org/abs/2104.08786>
101. <a id="ref-liu2022makes"></a>Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., Chen, W. "What Makes Good in-Context Examples for GPT-3?." Proceedings of Deep Learning Inside Out (DeeLIO), ACL Workshop, 2022. <https://arxiv.org/abs/2101.06804>
102. <a id="ref-min2022rethinking"></a>Min, S., Lyu, X., Holtzman, A., et al. "Rethinking the Role of Demonstrations: What Makes in-Context Learning Work?." Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022. <https://arxiv.org/abs/2202.12837>
103. <a id="ref-wei2022chain"></a>Wei, J., Wang, X., Schuurmans, D., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2201.11903>
104. <a id="ref-kojima2022large"></a>Kojima, T., Gu, S., Reid, M., Matsuo, Y., Iwasawa, Y. "Large Language Models Are Zero-Shot Reasoners." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2205.11916>
105. <a id="ref-wang2023selfconsistency"></a>Wang, X., Wei, J., Schuurmans, D., Le, Q., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023. <https://arxiv.org/abs/2203.11171>
106. <a id="ref-yao2023tree"></a>Yao, S., Yu, D., Zhao, J., et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2305.10601>
107. <a id="ref-wang2023planandsolve"></a>Wang, L., Xu, W., Lan, Y., et al. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023. <https://arxiv.org/abs/2305.04091>
108. <a id="ref-yao2023react"></a>Yao, S., Zhao, J., Yu, D., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." International Conference on Learning Representations (ICLR), 2023. <https://arxiv.org/abs/2210.03629>
109. <a id="ref-lewis2020retrieval"></a>Lewis, P., Perez, E., Piktus, A., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." Advances in Neural Information Processing Systems (NeurIPS), 2020. <https://arxiv.org/abs/2005.11401>
110. <a id="ref-bai2022constitutional"></a>Bai, Y., Jones, A., Ndousse, K., et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv Preprint arXiv:2212.08073, 2022. <https://arxiv.org/abs/2212.08073>
111. <a id="ref-zhou2023large"></a>Zhou, Y., Muresanu, A., Han, Z., et al. "Large Language Models Are Human-Level Prompt Engineers." Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023. <https://arxiv.org/abs/2211.01910>
112. <a id="ref-khattab2023dspy"></a>Khattab, O., Singhvi, A., Maheshwari, P., et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2310.03714>
113. <a id="ref-yang2024large"></a>Yang, C., Wang, X., Lu, Y., et al. "Large Language Models as Optimizers." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2309.03409>
114. <a id="ref-yang2025arq"></a>Yang, J., Zhu, Y., Wang, B., et al. "ARQ: Attentive Reasoning Queries for Multi-Hop Question Answering over Long Contexts." arXiv Preprint arXiv:2501.08290, 2025. <https://arxiv.org/abs/2501.08290>
115. <a id="ref-frantar2023gptq"></a>Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." arXiv Preprint arXiv:2210.17323, 2023. <https://arxiv.org/abs/2210.17323>
116. <a id="ref-lin2024awq"></a>Lin, J., Tang, J., Tang, H., et al. "AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration." Proceedings of Machine Learning and Systems (MLSys), 2024. <https://arxiv.org/abs/2306.00978>
117. <a id="ref-gerganov2023gguf"></a>Gerganov, G. "GGUF: GPT-Generated Unified Format (llama.cpp)." 2023. <https://github.com/ggerganov/llama.cpp>
118. <a id="ref-xiao2023smoothquant"></a>Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. <https://arxiv.org/abs/2211.10438>
119. <a id="ref-liu2023llmqat"></a>Liu, Z., Oguz, B., Zhao, C., et al. "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models." arXiv Preprint arXiv:2305.17888, 2023. <https://arxiv.org/abs/2305.17888>
120. <a id="ref-egiazarian2024aqlm"></a>Egiazarian, V., Panferov, A., Kuznedelev, D., Frantar, E., Babber, A., Alistarh, D. "Extreme Compression of Large Language Models via Additive Quantization." arXiv Preprint arXiv:2401.06118, 2024. <https://arxiv.org/abs/2401.06118>
121. <a id="ref-frantar2023sparsegpt"></a>Frantar, E., Alistarh, D. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. <https://arxiv.org/abs/2301.00774>
122. <a id="ref-sun2024wanda"></a>Sun, M., Liu, Z., Bair, A., Kolter, J. "A Simple and Effective Pruning Approach for Large Language Models." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2306.11695>
123. <a id="ref-hinton2015distilling"></a>Hinton, G., Vinyals, O., Dean, J. "Distilling the Knowledge in a Neural Network." arXiv Preprint arXiv:1503.02531, 2015.
124. <a id="ref-leviathan2023fast"></a>Leviathan, Y., Kalman, M., Matias, Y. "Fast Inference from Transformers via Speculative Decoding." Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. <https://arxiv.org/abs/2211.17192>
125. <a id="ref-cai2024medusa"></a>Cai, T., Li, Y., Geng, Z., et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. <https://arxiv.org/abs/2401.10774>
126. <a id="ref-li2024eagle"></a>Li, Y., Wei, F., Zhang, C., Zhang, H. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. <https://arxiv.org/abs/2401.15077>
127. <a id="ref-fu2024lookahead"></a>Fu, Y., Bailis, P., Stoica, I., Zhang, H. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." arXiv Preprint arXiv:2402.02057, 2024. <https://arxiv.org/abs/2402.02057>
128. <a id="ref-gloeckle2024multi"></a>Gloeckle, F., Idrissi, B., Rozière, B., Lopez-Paz, D., Synnaeve, G. "Better & Faster Large Language Models via Multi-Token Prediction." Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. <https://arxiv.org/abs/2404.19737>
129. <a id="ref-ji2023hallucination"></a>Ji, Z., Lee, N., Frieske, R., et al. "Survey of Hallucination in Natural Language Generation." ACM Computing Surveys, 2023. <https://arxiv.org/abs/2202.03629>
130. <a id="ref-kadavath2022language"></a>Kadavath, S., Conerly, T., Askell, A., et al. "Language Models (Mostly) Know What They Know." arXiv Preprint arXiv:2207.05221, 2022. <https://arxiv.org/abs/2207.05221>
131. <a id="ref-manakul2023selfcheckgpt"></a>Manakul, P., Liusie, A., Gales, M. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." Proceedings of EMNLP, 2023. <https://arxiv.org/abs/2303.08896>
132. <a id="ref-kuhn2023semantic"></a>Kuhn, L., Gal, Y., Farquhar, S. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." International Conference on Learning Representations (ICLR), 2023. <https://arxiv.org/abs/2302.09664>
133. <a id="ref-chuang2024dola"></a>Chuang, Y., Xie, Y., Luo, H., Kim, Y., Glass, J., He, P. "DoLA: Decoding by Contrasting Layers Improves Factuality in Large Language Models." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2309.03883>
134. <a id="ref-gallegos2024bias"></a>Gallegos, I., Rossi, R., Barber, J., Tong, S., et al. "Bias and Fairness in Large Language Models: A Survey." Computational Linguistics, 2024. <https://arxiv.org/abs/2309.00770>
135. <a id="ref-carlini2021extracting"></a>Carlini, N., Tramer, F., Wallace, E., et al. "Extracting Training Data from Large Language Models." USENIX Security Symposium, 2021. <https://arxiv.org/abs/2012.07805>
136. <a id="ref-zou2023universal"></a>Zou, A., Wang, Z., Kolter, J., Fredrikson, M. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv Preprint arXiv:2307.15043, 2023. <https://arxiv.org/abs/2307.15043>
137. <a id="ref-perez2022red"></a>Perez, E., Huang, S., Song, F., et al. "Red Teaming Language Models with Language Models." Proceedings of EMNLP, 2022. <https://arxiv.org/abs/2202.03286>
138. <a id="ref-kwon2023efficient"></a>Kwon, W., Li, Z., Zhuang, S., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP), 2023. <https://arxiv.org/abs/2309.06180>
139. <a id="ref-sutton2018reinforcement"></a>Sutton, R., Barto, A. "Reinforcement Learning: An Introduction." 2018. <http://incompleteideas.net/book/the-book-2nd.html>
140. <a id="ref-sutton1988learning"></a>Sutton, R. "Learning to Predict by the Methods of Temporal Differences." Machine Learning, 1988.
141. <a id="ref-watkins1989learning"></a>Watkins, C. "Learning from Delayed Rewards." 1989.
142. <a id="ref-rummery1994online"></a>Rummery, G., Niranjan, M. "On-Line q-Learning Using Connectionist Systems." 1994.
143. <a id="ref-mnih2015human"></a>Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-Level Control Through Deep Reinforcement Learning." Nature, 2015. <https://www.nature.com/articles/nature14236>
144. <a id="ref-lin1992self"></a>Lin, L. "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching." Machine Learning, 1992.
145. <a id="ref-schaul2016prioritized"></a>Schaul, T., Quan, J., Antonoglou, I., Silver, D. "Prioritized Experience Replay." Proceedings of the 4th International Conference on Learning Representations (ICLR), 2016. <https://arxiv.org/abs/1511.05952>
146. <a id="ref-williams1992simple"></a>Williams, R. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." Machine Learning, 1992. <https://link.springer.com/article/10.1007/BF00992696>
147. <a id="ref-mnih2016asynchronous"></a>Mnih, V., Badia, A., Mirza, M., et al. "Asynchronous Methods for Deep Reinforcement Learning." Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016. <https://arxiv.org/abs/1602.01783>
148. <a id="ref-schulman2015trust"></a>Schulman, J., Levine, S., Abbeel, P., Jordan, M., Moritz, P. "Trust Region Policy Optimization." Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015. <https://arxiv.org/abs/1502.05477>
149. <a id="ref-schulman2017proximal"></a>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. "Proximal Policy Optimization Algorithms." arXiv Preprint arXiv:1707.06347, 2017. <https://arxiv.org/abs/1707.06347>
150. <a id="ref-schulman2016high"></a>Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." Proceedings of the 4th International Conference on Learning Representations (ICLR), 2016. <https://arxiv.org/abs/1506.02438>
151. <a id="ref-haarnoja2018soft"></a>Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." Proceedings of the 35th International Conference on Machine Learning (ICML), 2018. <https://arxiv.org/abs/1801.01290>
152. <a id="ref-schrittwieser2020mastering"></a>Schrittwieser, J., Antonoglou, I., Hubert, T., et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature, 2020. <https://arxiv.org/abs/1911.08265>
153. <a id="ref-hafner2020dream"></a>Hafner, D., Lillicrap, T., Ba, J., Norouzi, M. "Dream to Control: Learning Behaviors by Latent Imagination." Proceedings of the 8th International Conference on Learning Representations (ICLR), 2020. <https://arxiv.org/abs/1912.01603>
154. <a id="ref-silver2016mastering"></a>Silver, D., Huang, A., Maddison, C., et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search." Nature, 2016. <https://www.nature.com/articles/nature16961>
155. <a id="ref-ng1999policy"></a>Ng, A., Harada, D., Russell, S. "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." Proceedings of the 16th International Conference on Machine Learning (ICML), 1999.
156. <a id="ref-deepseek2025r1"></a>DeepSeek-AI, Guo, D., Yang, D., et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv Preprint arXiv:2501.12948, 2025. <https://arxiv.org/abs/2501.12948>
157. <a id="ref-ziegler2019fine"></a>Ziegler, D., Stiennon, N., Wu, J., et al. "Fine-Tuning Language Models from Human Preferences." arXiv Preprint arXiv:1909.08593, 2019. <https://arxiv.org/abs/1909.08593>
158. <a id="ref-christiano2017deep"></a>Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D. "Deep Reinforcement Learning from Human Preferences." Advances in Neural Information Processing Systems (NeurIPS), 2017. <https://arxiv.org/abs/1706.03741>
159. <a id="ref-rafailov2023direct"></a>Rafailov, R., Sharma, A., Mitchell, E., Manning, C., Ermon, S., Finn, C. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2305.18290>
160. <a id="ref-vonwerra2022trl"></a>Werra, L., Belkada, Y., Tunstall, L., et al. "TRL: Transformer Reinforcement Learning." 2022. <https://github.com/huggingface/trl>
161. <a id="ref-wang2023fdpo"></a>Wang, J., Duan, J., Liu, Y., Yue, Z., Tong, H., Wang, J. "Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2309.16240>
162. <a id="ref-chowdhury2024robustdpo"></a>Chowdhury, S., Chakraborty, A., Natarajan, S., Agarwal, A., Sontag, D. "Provably Robust DPO: Aligning Language Models with Noisy Feedback." arXiv Preprint arXiv:2403.00409, 2024. <https://arxiv.org/abs/2403.00409>
163. <a id="ref-gorbatenko2024trdpo"></a>Gorbatenko, A. "Online DPO with Synchronised Reference Model Updates." TRL Documentation, 2024.
164. <a id="ref-ji2024exo"></a>Ji, J., Fisch, A., Weston, J., Sukhbaatar, S. "Towards Exact Optimization of Language Model Alignment." arXiv Preprint arXiv:2402.05369, 2024. <https://arxiv.org/abs/2402.05369>
165. <a id="ref-chen2024nca"></a>Chen, H., Zheng, G., Kim, Y., Chen, Y. "Noise Contrastive Alignment of Language Models with Explicit Rewards." arXiv Preprint arXiv:2402.05369, 2024. <https://arxiv.org/abs/2402.05369>
166. <a id="ref-zhao2023slichf"></a>Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., Liu, P. "SLiC-HF: Sequence Likelihood Calibration with Human Feedback." arXiv Preprint arXiv:2305.10425, 2023. <https://arxiv.org/abs/2305.10425>
167. <a id="ref-meng2024simpo"></a>Meng, Y., Xia, M., Chen, D. "SimPO: Simple Preference Optimization with a Reference-Free Reward." Advances in Neural Information Processing Systems (NeurIPS), 2024. <https://arxiv.org/abs/2405.14734>
168. <a id="ref-shao2024deepseekmath"></a>Shao, Z., Wang, P., Zhu, Q., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv Preprint arXiv:2402.03300, 2024. <https://arxiv.org/abs/2402.03300>
169. <a id="ref-yu2025dapo"></a>Yu, Q., Sun, Z., Wen, S., et al. "DAPO: An Open-Source LLM Reinforcement Learning System." arXiv Preprint arXiv:2503.14476, 2025. <https://arxiv.org/abs/2503.14476>
170. <a id="ref-chen2025gspo"></a>Chen, Z., Deng, Y., Zhang, R., Sun, H., Chen, W. "GSPO: Sequence-Level Policy Optimization for Language Model Alignment." arXiv Preprint arXiv:2502.12459, 2025. <https://arxiv.org/abs/2502.12459>
171. <a id="ref-liu2024drgrpo"></a>Liu, Y., Han, L., Tan, Y., et al. "Understanding and Mitigating the Pretraining Distribution Bias in GRPO." arXiv Preprint arXiv:2505.07888, 2025. <https://arxiv.org/abs/2505.07888>
172. <a id="ref-xu2025twograpo"></a>Xu, H., Zhao, H., Liu, Y., Wei, D. "It Takes Two: Pairwise Preference Optimization with Two Rollouts." arXiv Preprint arXiv:2505.07856, 2025. <https://arxiv.org/abs/2505.07856>
173. <a id="ref-han2025sapo"></a>Han, C., Li, M., Chen, W. "SAPO: Soft Adaptive Policy Optimization for Efficient LLM Alignment." arXiv Preprint arXiv:2503.01739, 2025. <https://arxiv.org/abs/2503.01739>
174. <a id="ref-zhong2025tismis"></a>Zhong, Y., Chen, Y., Li, Z., Chen, M. "Importance Sampling Corrections for Large Language Model Alignment with Asynchronous Generation." arXiv Preprint arXiv:2503.09057, 2025. <https://arxiv.org/abs/2503.09057>
175. <a id="ref-luo2025vespo"></a>Luo, Z., Shi, J., Yu, T., Chen, M. "VESPO: Variational Sequence-Level Soft Policy Optimization for LLM Alignment." arXiv Preprint arXiv:2505.07508, 2025. <https://arxiv.org/abs/2505.07508>
176. <a id="ref-an2025dppo"></a>An, Y., Shen, L., Xu, Y., Liu, X. "DPPO: Direct Divergence-Based Policy Optimization for Language Model Alignment." arXiv Preprint arXiv:2503.14532, 2025. <https://arxiv.org/abs/2503.14532>
177. <a id="ref-luo2025scalerl"></a>Luo, Z., Chen, Z., Jiang, Y., et al. "ScaleRL: Scaling Reinforcement Learning for LLM Reasoning." arXiv Preprint arXiv:2505.16356, 2025. <https://arxiv.org/abs/2505.16356>
178. <a id="ref-zhong2025gdpo"></a>Zhong, Y., Shi, J., Chen, Y., Chen, M. "GDPO: Learning to Directly Align Language Models with Group-Decoupled Reward." arXiv Preprint arXiv:2501.17888, 2025. <https://arxiv.org/abs/2501.17888>
179. <a id="ref-choi2025gopo"></a>Choi, S., Park, H., Moon, D., Kim, K., Choi, E. "GOPO: Group Ordinal Policy Optimization for LLM Alignment with Non-Verifiable Rewards." arXiv Preprint arXiv:2505.12948, 2025. <https://arxiv.org/abs/2505.12948>
180. <a id="ref-guo2024direct"></a>Guo, S., Zhang, B., Liu, T., et al. "Direct Language Model Alignment from Online AI Feedback." arXiv Preprint arXiv:2402.04792, 2024. <https://arxiv.org/abs/2402.04792>
181. <a id="ref-ethayarajh2024kto"></a>Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., Kiela, D. "KTO: Model Alignment as Prospect Theoretic Optimization." arXiv Preprint arXiv:2402.01306, 2024. <https://arxiv.org/abs/2402.01306>
182. <a id="ref-azar2024general"></a>Azar, M., Rowland, M., Piot, B., et al. "A General Theoretical Paradigm to Understand Learning from Human Feedback." arXiv Preprint arXiv:2310.12036, 2024. <https://arxiv.org/abs/2310.12036>
183. <a id="ref-hong2024orpo"></a>Hong, J., Lee, N., Thorne, J. "ORPO: Monolithic Preference Optimization Without Reference Model." arXiv Preprint arXiv:2403.07691, 2024. <https://arxiv.org/abs/2403.07691>
184. <a id="ref-nakano2021webgpt"></a>Nakano, R., Hilton, J., Balaji, S., et al. "WebGPT: Browser-Assisted Question-Answering with Human Feedback." arXiv Preprint arXiv:2112.09332, 2021. <https://arxiv.org/abs/2112.09332>
185. <a id="ref-gao2023scaling"></a>Gao, L., Schulman, J., Hilton, J. "Scaling Laws for Reward Model Overoptimization." Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
186. <a id="ref-bradley1952rank"></a>Bradley, R., Terry, M. "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." Biometrika, 1952. <https://www.jstor.org/stable/2334029>
187. <a id="ref-plackett1975analysis"></a>Plackett, R. "The Analysis of Permutations." Journal of the Royal Statistical Society: Series C (Applied Statistics), 1975.
188. <a id="ref-xia2008listwise"></a>Xia, F., Liu, T., Wang, J., Zhang, W., Li, H. "Listwise Approach to Learning to Rank: Theory and Algorithm." Proceedings of the 25th International Conference on Machine Learning (ICML), 2008.
189. <a id="ref-cao2007listnet"></a>Cao, Z., Qin, T., Liu, T., Tsai, M., Li, H. "Learning to Rank: From Pairwise Approach to Listwise Approach." Proceedings of the 24th International Conference on Machine Learning (ICML), 2007.
190. <a id="ref-burges2006lambdarank"></a>Burges, C., Ragno, R., Le, Q. "Learning to Rank with Nonsmooth Cost Functions." Advances in Neural Information Processing Systems (NeurIPS), 2006.
191. <a id="ref-burges2005ranknet"></a>Burges, C., Shaked, T., Renshaw, E., et al. "Learning to Rank Using Gradient Descent." Proceedings of the 22nd International Conference on Machine Learning (ICML), 2005.
192. <a id="ref-kirkpatrick2017overcoming"></a>Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. "Overcoming Catastrophic Forgetting in Neural Networks." Proceedings of the National Academy of Sciences, 2017.
193. <a id="ref-li2020pytorch"></a>Li, S., Zhao, Y., Varma, R., et al. "PyTorch Distributed: Experiences on Accelerating Data Parallel Training." Proceedings of the VLDB Endowment, 2020.
194. <a id="ref-sergeev2018horovod"></a>Sergeev, A., Balso, M. "Horovod: Fast and Easy Distributed Deep Learning in TensorFlow." arXiv Preprint arXiv:1802.05799, 2018.
195. <a id="ref-shoeybi2019megatron"></a>Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., Catanzaro, B. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv Preprint arXiv:1909.08053, 2019.
196. <a id="ref-korthikanti2023reducing"></a>Korthikanti, V., Casper, J., Lym, S., et al. "Reducing Activation Recomputation in Large Transformer Models." Proceedings of Machine Learning and Systems (MLSys), 2023.
197. <a id="ref-huang2019gpipe"></a>Huang, Y., Cheng, Y., Bapna, A., et al. "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism." Advances in Neural Information Processing Systems (NeurIPS), 2019.
198. <a id="ref-narayanan2019pipedream"></a>Narayanan, D., Harlap, A., Phanishayee, A., et al. "PipeDream: Generalized Pipeline Parallelism for DNN Training." Proceedings of the 27th ACM Symposium on Operating Systems Principles (SOSP), 2019.
199. <a id="ref-narayanan2021efficient"></a>Narayanan, D., Shoeybi, M., Casper, J., et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." arXiv Preprint arXiv:2104.04473, 2021.
200. <a id="ref-qi2023zerobubble"></a>Qi, P., Wan, X., Huang, G., Lin, M. "Zero Bubble Pipeline Parallelism." arXiv Preprint arXiv:2401.10241, 2023.
201. <a id="ref-rajbhandari2020zero"></a>Rajbhandari, S., Rasley, J., Rber, O., He, Y. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." arXiv Preprint arXiv:1910.02054, 2020.
202. <a id="ref-zhao2023pytorch"></a>Zhao, Y., Gu, A., Varma, R., et al. "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." Proceedings of the VLDB Endowment, 2023.
203. <a id="ref-yu2022orca"></a>Yu, G., Jeong, J., Kim, G., Kim, S., Chun, B. "Orca: A Distributed Serving System for Transformer-Based Generative Models." Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2022.
204. <a id="ref-yao2023deepspeedchat"></a>Yao, Z., Rajbhandari, S., Aminabadi, R., et al. "DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-Like Models at All Scales." arXiv Preprint arXiv:2308.01320, 2023.
205. <a id="ref-hu2024openrlhf"></a>Hu, J., Tao, X., Zhu, W., Yang, S., Liu, J., Li, Z. "OpenRLHF: An Easy-to-Use, Scalable and High-Performance RLHF Framework." arXiv Preprint arXiv:2405.11143, 2024.
206. <a id="ref-chen2016training"></a>Chen, T., Xu, B., Zhang, C., Guestrin, C. "Training Deep Nets with Sublinear Memory Cost." arXiv Preprint arXiv:1604.06174, 2016.
207. <a id="ref-micikevicius2018mixed"></a>Micikevicius, P., Narang, S., Alben, J., et al. "Mixed Precision Training." International Conference on Learning Representations (ICLR), 2018.
208. <a id="ref-rajbhandari2021zeroinfinity"></a>Rajbhandari, S., Ruwase, O., Rasley, J., Smith, S., He, Y. "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning." arXiv Preprint arXiv:2104.07857, 2021.
209. <a id="ref-chowdhery2022palm"></a>Chowdhery, A., Narang, S., Devlin, J., et al. "PaLM: Scaling Language Modeling with Pathways." arXiv Preprint arXiv:2204.02311, 2022.
210. <a id="ref-mccandlish2018empirical"></a>McCandlish, S., Kaplan, J., Amodei, D., {OpenAI Dota Team}. "An Empirical Model of Large-Batch Training." arXiv Preprint arXiv:1812.06162, 2018.
211. <a id="ref-zelikman2022star"></a>Zelikman, E., Wu, Y., Mu, J., Goodman, N. "STaR: Bootstrapping Reasoning with Reasoning." Advances in Neural Information Processing Systems (NeurIPS), 2022. <https://arxiv.org/abs/2203.14465>
212. <a id="ref-shinn2023reflexion"></a>Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S. "Reflexion: Language Agents with Verbal Reinforcement Learning." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2303.11366>
213. <a id="ref-zhou2024lats"></a>Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., Wang, Y. "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models." arXiv Preprint arXiv:2310.04406, 2024.
214. <a id="ref-putta2024agentq"></a>Putta, P., Mills, E., Garg, N., et al. "Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents." arXiv Preprint arXiv:2408.07199, 2024.
215. <a id="ref-wang2024openhands"></a>Wang, X., Ding, B., Hoang, Z., et al. "OpenHands: An Open Platform for AI Software Developers as Generalist Agents." arXiv Preprint arXiv:2407.16741, 2024.
216. <a id="ref-wang2023voyager"></a>Wang, G., Xie, Y., Jiang, Y., et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv Preprint arXiv:2305.16291, 2023. <https://arxiv.org/abs/2305.16291>
217. <a id="ref-le2024rlef"></a>Le, H., Wang, Y., Gotmare, A., Savarese, S., Hoi, S. "CodeRL: Mastering Code Generation Through Pretrained Models and Deep Reinforcement Learning." Advances in Neural Information Processing Systems (NeurIPS), 2023.
218. <a id="ref-zelikman2024quietstar"></a>Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., Goodman, N. "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking." arXiv Preprint arXiv:2403.09629, 2024. <https://arxiv.org/abs/2403.09629>
219. <a id="ref-hosseini2024vstar"></a>Hosseini, A., Yuan, X., Poupart, P., Trischler, A., Bengio, Y. "V-STaR: Training Verifiers for Self-Taught Reasoners." arXiv Preprint arXiv:2402.06457, 2024.
220. <a id="ref-yang2024sweagent"></a>Yang, J., Jimenez, C., Wettig, A., et al. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." Advances in Neural Information Processing Systems (NeurIPS), 2024. <https://arxiv.org/abs/2405.15793>
221. <a id="ref-yu2026rwml"></a>Yu, X., Peng, B., Xu, R., et al. "Reinforcement World Model Learning for LLM-Based Agents." arXiv Preprint arXiv:2602.05842, 2026.
222. <a id="ref-yao2024tree"></a>Yao, S., Yu, D., Zhao, J., et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." Advances in Neural Information Processing Systems (NeurIPS), 2024. <https://arxiv.org/abs/2305.10601>
223. <a id="ref-besta2024graph"></a>Besta, M., Blach, N., Kubicek, A., et al. "Graph of Thoughts: Solving Elaborate Problems with Large Language Models." Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
224. <a id="ref-stiennon2020learning"></a>Stiennon, N., Ouyang, L., Wu, J., et al. "Learning to Summarize from Human Feedback." Advances in Neural Information Processing Systems (NeurIPS), 2020. <https://arxiv.org/abs/2009.01325>
225. <a id="ref-kocsis2006bandit"></a>Kocsis, L., Szepesvári, C. "Bandit Based Monte-Carlo Planning." European Conference on Machine Learning (ECML), 2006.
226. <a id="ref-alphaproof2024"></a>DeepMind, G. "AlphaProof and AlphaGeometry 2: Solving Olympiad Geometry Without Human Demonstrations." 2024. <https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/>
227. <a id="ref-qi2024mutual"></a>Qi, Z., Wan, M., Cao, J., Lin, M. "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers." arXiv Preprint arXiv:2408.06195, 2024.
228. <a id="ref-madaan2023selfrefine"></a>Madaan, A., Tandon, N., Gupta, P., et al. "Self-Refine: Iterative Refinement with Self-Feedback." Advances in Neural Information Processing Systems (NeurIPS), 2023.
229. <a id="ref-openai2024o1"></a>OpenAI. "Learning to Reason with LLMs." 2024. <https://openai.com/index/learning-to-reason-with-llms/>
230. <a id="ref-openai2025o3"></a>OpenAI. "OpenAI o3 and o4-mini System Card." 2025. <https://openai.com/index/o3-and-o4-mini-system-card/>
231. <a id="ref-lightman2023lets"></a>Lightman, H., Kosaraju, V., Burda, Y., et al. "Let's Verify Step by Step." arXiv Preprint arXiv:2305.20050, 2023.
232. <a id="ref-qwen2024qwq"></a>Team, Q. "QwQ: Reflect Deeply on the Boundaries of the Unknown." Qwen Blog, 2024. <https://qwenlm.github.io/blog/qwq-32b-preview/>
233. <a id="ref-qwen2025qwen3"></a>Team, Q. "Qwen3 Technical Report." arXiv Preprint arXiv:2505.09388, 2025.
234. <a id="ref-wang2024mathshepherd"></a>Wang, P., Li, L., Shao, Z., et al. "Math-Shepherd: Verify and Reinforce LLMs Step-by-Step Without Human Annotations." arXiv Preprint arXiv:2312.08935, 2024. <https://arxiv.org/abs/2312.08935>
235. <a id="ref-lambert2024tulu3"></a>Lambert, N., Morrison, J., Pyatkin, V., et al. "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv Preprint arXiv:2411.15124, 2024. <https://arxiv.org/abs/2411.15124>
236. <a id="ref-qin2024o1journey"></a>Qin, Y., Li, X., Zou, H., et al. "O1 Replication Journey: A Strategic Progress Report." arXiv Preprint arXiv:2410.18982, 2024. <https://arxiv.org/abs/2410.18982>
237. <a id="ref-snell2024scaling"></a>Snell, C., Lee, J., Xu, K., Kumar, A. "Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters." arXiv Preprint arXiv:2408.03314, 2024.
238. <a id="ref-wu2024empirical"></a>Wu, Z., Hu, Q., Zhang, Y., Gao, Y., Chen, J. "An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models." arXiv Preprint arXiv:2408.00724, 2024.
239. <a id="ref-kaplan2020scaling"></a>Kaplan, J., McCandlish, S., Henighan, T., et al. "Scaling Laws for Neural Language Models." arXiv Preprint arXiv:2001.08361, 2020. <https://arxiv.org/abs/2001.08361>
240. <a id="ref-cohen1960coefficient"></a>Cohen, J. "A Coefficient of Agreement for Nominal Scales." Educational and Psychological Measurement, 1960.
241. <a id="ref-fleiss1971measuring"></a>Fleiss, J. "Measuring Nominal Scale Agreement Among Many Raters." Psychological Bulletin, 1971.
242. <a id="ref-guo2017calibration"></a>Guo, C., Pleiss, G., Sun, Y., Weinberger, K. "On Calibration of Modern Neural Networks." International Conference on Machine Learning (ICML), 2017.
243. <a id="ref-wang2022selfinstruct"></a>Wang, Y., Kordi, Y., Mishra, S., et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." arXiv Preprint arXiv:2212.10560, 2022. <https://arxiv.org/abs/2212.10560>
244. <a id="ref-xu2023wizardlm"></a>Xu, C., Sun, Q., Zheng, K., et al. "WizardLM: Empowering Large Language Models to Follow Complex Instructions." arXiv Preprint arXiv:2304.12244, 2023. <https://arxiv.org/abs/2304.12244>
245. <a id="ref-zheng2023judging"></a>Zheng, L., Chiang, W., Sheng, Y., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2306.05685>
246. <a id="ref-elo1978rating"></a>Elo, A. "The Rating of Chess Players, Past and Present." 1978.
247. <a id="ref-herbrich2006trueskill"></a>Herbrich, R., Minka, T., Graepel, T. "TrueSkill™: A Bayesian Skill Rating System." Advances in Neural Information Processing Systems (NeurIPS), 2006. <https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html>
248. <a id="ref-wilson1927probable"></a>Wilson, E. "Probable Inference, the Law of Succession, and Statistical Inference." Journal of the American Statistical Association, 1927. <https://www.jstor.org/stable/2276774>
249. <a id="ref-papineni2002bleu"></a>Papineni, K., Roukos, S., Ward, T., Zhu, W. "BLEU: A Method for Automatic Evaluation of Machine Translation." Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), 2002. <https://aclanthology.org/P02-1040/>
250. <a id="ref-lin2004rouge"></a>Lin, C. "ROUGE: A Package for Automatic Evaluation of Summaries." Text Summarization Branches Out: Proceedings of the ACL-04 Workshop, 2004. <https://aclanthology.org/W04-1013/>
251. <a id="ref-zhang2020bertscore"></a>Zhang, T., Kishore, V., Wu, F., Weinberger, K., Artzi, Y. "BERTScore: Evaluating Text Generation with BERT." International Conference on Learning Representations (ICLR), 2020. <https://arxiv.org/abs/1904.09675>
252. <a id="ref-banerjee2005meteor"></a>Banerjee, S., Lavie, A. "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments." Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 2005. <https://aclanthology.org/W05-0909/>
253. <a id="ref-chen2021evaluating"></a>Chen, M., Tworek, J., Jun, H., et al. "Evaluating Large Language Models Trained on Code." arXiv Preprint arXiv:2107.03374, 2021. <https://arxiv.org/abs/2107.03374>
254. <a id="ref-jimenez2024swebench"></a>Jimenez, C., Yang, J., Wettig, A., et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2310.06770>
255. <a id="ref-zhou2024webarena"></a>Zhou, S., Xu, F., Zhu, H., et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2307.13854>
256. <a id="ref-shridhar2021alfworld"></a>Shridhar, M., Yuan, X., Côté, M., Bisk, Y., Trischler, A., Hausknecht, M. "ALFWorld: Aligning Text and Embodied Environments for Interactive Learning." International Conference on Learning Representations (ICLR), 2021.
257. <a id="ref-liu2023agentbench"></a>Liu, X., Yu, H., Zhang, H., et al. "AgentBench: Evaluating LLMs as Agents." arXiv Preprint arXiv:2308.03688, 2023.
258. <a id="ref-liu2023geval"></a>Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., Zhu, C. "G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment." Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. <https://arxiv.org/abs/2303.16634>
259. <a id="ref-hendrycks2021measuring"></a>Hendrycks, D., Burns, C., Basart, S., et al. "Measuring Massive Multitask Language Understanding." International Conference on Learning Representations (ICLR), 2021.
260. <a id="ref-goodhart1984problems"></a>Goodhart, C. "Problems of Monetary Management: The U.K. Experience." Monetary Theory and Practice, 1984.
261. <a id="ref-robertson2009probabilistic"></a>Robertson, S., Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval, 2009.
262. <a id="ref-karpukhin2020dense"></a>Karpukhin, V., O{ğ}uz, B., Min, S., et al. "Dense Passage Retrieval for Open-Domain Question Answering." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. <https://arxiv.org/abs/2004.04906>
263. <a id="ref-johnson2019billion"></a>Johnson, J., Douze, M., Jégou, H. "Billion-Scale Similarity Search with GPUs." IEEE Transactions on Big Data, 2021.
264. <a id="ref-malkov2018efficient"></a>Malkov, Y., Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.
265. <a id="ref-cormack2009reciprocal"></a>Cormack, G., Clarke, C., Buettcher, S. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." Proceedings of the 32nd International ACM SIGIR Conference, 2009.
266. <a id="ref-formal2021splade"></a>Formal, T., Piwowarski, B., Clinchant, S. "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.
267. <a id="ref-formal2021spladev2"></a>Formal, T., Lassance, C., Piwowarski, B., Clinchant, S. "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval." arXiv Preprint arXiv:2109.10086, 2021.
268. <a id="ref-nogueira2020document"></a>Nogueira, R., Jiang, Z., Pradeep, R., Lin, J. "Document Ranking with a Pretrained Sequence-to-Sequence Model." Findings of EMNLP, 2020.
269. <a id="ref-bajaj2016msmarco"></a>Bajaj, P., Campos, D., Craswell, N., et al. "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset." arXiv Preprint arXiv:1611.09268, 2016.
270. <a id="ref-khattab2020colbert"></a>Khattab, O., Zaharia, M. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." Proceedings of the 43rd International ACM SIGIR Conference, 2020. <https://arxiv.org/abs/2004.12832>
271. <a id="ref-santhanam2022colbertv2"></a>Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2022.
272. <a id="ref-sparckjones1972idf"></a>Sparck, K. "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." Journal of Documentation, 1972.
273. <a id="ref-nogueira2019passage"></a>Nogueira, R., Cho, K. "Passage Re-Ranking with BERT." arXiv Preprint arXiv:1901.04085, 2019.
274. <a id="ref-gao2022precise"></a>Gao, L., Ma, X., Lin, J., Callan, J. "Precise Zero-Shot Dense Retrieval Without Relevance Labels." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023. <https://arxiv.org/abs/2212.10496>
275. <a id="ref-asai2023selfrag"></a>Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. "Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection." arXiv Preprint arXiv:2310.11511, 2023. <https://arxiv.org/abs/2310.11511>
276. <a id="ref-yan2024crag"></a>Yan, S., Gu, J., Zhu, Y., Ling, Z. "Corrective Retrieval Augmented Generation." arXiv Preprint arXiv:2401.15884, 2024. <https://arxiv.org/abs/2401.15884>
277. <a id="ref-jeong2024adaptive"></a>Jeong, S., Baek, J., Cho, S., Hwang, S., Park, J. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models Through Question Complexity." arXiv Preprint arXiv:2403.14403, 2024. <https://arxiv.org/abs/2403.14403>
278. <a id="ref-edge2024local"></a>Edge, D., Trinh, H., Cheng, N., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv Preprint arXiv:2404.16130, 2024. <https://arxiv.org/abs/2404.16130>
279. <a id="ref-rackauckas2023ragfusion"></a>Rackauckas, A. "RAG-Fusion: A New Take on Retrieval-Augmented Generation." 2024.
280. <a id="ref-lin2025refrag"></a>Lin, X., Ghosh, A., Low, B., Shrivastava, A., Mohan, V. "REFRAG: Rethinking RAG Based Decoding." arXiv Preprint arXiv:2509.01092, 2025.
281. <a id="ref-jin2025searchr1"></a>Jin, B., Zeng, H., et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." arXiv Preprint arXiv:2503.09516, 2025.
282. <a id="ref-kwiatkowski2019natural"></a>Kwiatkowski, T., Palomaki, J., Redfield, O., et al. "Natural Questions: A Benchmark for Question Answering Research." Transactions of the Association for Computational Linguistics, 2019.
283. <a id="ref-joshi2017triviaqa"></a>Joshi, M., Choi, E., Weld, D., Zettlemoyer, L. "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension." Proceedings of ACL, 2017.
284. <a id="ref-yang2018hotpotqa"></a>Yang, Z., Qi, P., Zhang, S., et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering." Proceedings of EMNLP, 2018.
285. <a id="ref-es2023ragas"></a>Es, S., James, J., Espinosa-Anke, L., Schockaert, S. "RAGAs: Automated Evaluation of Retrieval Augmented Generation." arXiv Preprint arXiv:2309.15217, 2023. <https://arxiv.org/abs/2309.15217>
286. <a id="ref-liu2023lost"></a>Liu, N., Lin, K., Hewitt, J., et al. "Lost in the Middle: How Language Models Use Long Contexts." Transactions of the Association for Computational Linguistics, 2024. <https://arxiv.org/abs/2307.03172>
287. <a id="ref-lee2024nvembed"></a>Lee, C., Roy, R., Xu, M., et al. "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." arXiv Preprint arXiv:2405.17428, 2024.
288. <a id="ref-li2023gte"></a>Li, Z., Zhang, X., Zhang, Y., Long, D., Xie, P., Zhang, M. "Towards General Text Embeddings with Multi-Stage Contrastive Learning." arXiv Preprint arXiv:2308.03281, 2023.
289. <a id="ref-chen2024bgem3"></a>Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." arXiv Preprint arXiv:2402.03216, 2024.
290. <a id="ref-xiao2023cpack"></a>Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. "C-Pack: Packaged Resources to Advance General Chinese Embedding." arXiv Preprint arXiv:2309.07597, 2023.
291. <a id="ref-zhang2024raft"></a>Zhang, T., Patil, S., Jain, N., et al. "RAFT: Adapting Language Model to Domain Specific RAG." arXiv Preprint arXiv:2403.10131, 2024. <https://arxiv.org/abs/2403.10131>
292. <a id="ref-guu2020realm"></a>Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M. "REALM: Retrieval-Augmented Language Model Pre-Training." Proceedings of the 37th International Conference on Machine Learning (ICML), 2020. <https://arxiv.org/abs/2002.08909>
293. <a id="ref-tulving1985memory"></a>Tulving, E. "Memory and Consciousness." Canadian Psychology / Psychologie Canadienne, 1985.
294. <a id="ref-squire1992declarative"></a>Squire, L. "Declarative and Nondeclarative Memory: Multiple Brain Systems Supporting Learning and Memory." Journal of Cognitive Neuroscience, 1992. <https://doi.org/10.1162/jocn.1992.4.3.232>
295. <a id="ref-nye2021show"></a>Nye, M., Andreassen, A., Gur-Ari, G., et al. "Show Your Work: Scratchpads for Intermediate Computation with Language Models." arXiv Preprint arXiv:2112.00114, 2021. <https://arxiv.org/abs/2112.00114>
296. <a id="ref-guo2020scann"></a>Guo, R., Sun, P., Lindgren, E., et al. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." International Conference on Machine Learning (ICML), 2020.
297. <a id="ref-chen2022hybrid"></a>Chen, J., Luo, S., Zhang, M., Liu, Z., Xiao, Y., Han, D. "Hybrid Retrieval for Open-Domain Question Answering." arXiv Preprint arXiv:2210.06029, 2022. <https://arxiv.org/abs/2210.06029>
298. <a id="ref-forte2022building"></a>Forte, T. "Building a Second Brain: A Proven Method to Organize Your Digital Life and Unlock Your Creative Potential." 2022.
299. <a id="ref-harris2013sparql"></a>Harris, S., Seaborne, A. "SPARQL 1.1 Query Language." 2013. <https://www.w3.org/TR/sparql11-query/>
300. <a id="ref-francis2018cypher"></a>Francis, N., Green, A., Guagliardo, P., et al. "Cypher: An Evolving Query Language for Property Graphs." Proceedings of the 2018 International Conference on Management of Data (SIGMOD), 2018.
301. <a id="ref-lacroix2020tensor"></a>Lacroix, T., Obozinski, G., Usunier, N. "Tensor Decompositions for Temporal Knowledge Base Completion." International Conference on Learning Representations (ICLR), 2020. <https://arxiv.org/abs/2004.04926>
302. <a id="ref-weston2014memory"></a>Weston, J., Chopra, S., Bordes, A. "Memory Networks." International Conference on Learning Representations (ICLR), 2015. <https://arxiv.org/abs/1410.3916>
303. <a id="ref-sukhbaatar2015end"></a>Sukhbaatar, S., Szlam, A., Weston, J., Fergus, R. "End-to-End Memory Networks." Advances in Neural Information Processing Systems (NeurIPS), 2015. <https://arxiv.org/abs/1503.08895>
304. <a id="ref-packer2023memgpt"></a>Packer, C., Fang, V., Patil, S., Lin, K., Wooders, S., Gonzalez, J. "MemGPT: Towards LLMs as Operating Systems." arXiv Preprint arXiv:2310.08560, 2023. <https://arxiv.org/abs/2310.08560>
305. <a id="ref-xu2025amem"></a>Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. "A-Mem: Agentic Memory for LLM Agents." Advances in Neural Information Processing Systems (NeurIPS), 2025.
306. <a id="ref-chhikara2025mem0"></a>Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." arXiv Preprint arXiv:2504.19413, 2025. <https://arxiv.org/abs/2504.19413>
307. <a id="ref-park2023generative"></a>Park, J., O'Brien, J., Cai, C., Morris, M., Liang, P., Bernstein, M. "Generative Agents: Interactive Simulacra of Human Behavior." Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST), 2023. <https://arxiv.org/abs/2304.03442>
308. <a id="ref-ebbinghaus1885memory"></a>Ebbinghaus, H. "Über Das Gedächtnis: Untersuchungen Zur Experimentellen Psychologie." 1885.
309. <a id="ref-hayes1985blackboard"></a>Hayes-Roth, B. "A Blackboard Architecture for Control." Artificial Intelligence, 1985. <https://doi.org/10.1016/0004-3702(85)90063-3>
310. <a id="ref-andrychowicz2017hindsight"></a>Andrychowicz, M., Wolski, F., Ray, A., et al. "Hindsight Experience Replay." Advances in Neural Information Processing Systems (NeurIPS), 2017.
311. <a id="ref-duan2016rl2"></a>Duan, Y., Schulman, J., Chen, X., Bartlett, P., Sutskever, I., Abbeel, P. "RL2: Fast Reinforcement Learning via Slow Reinforcement Learning." arXiv Preprint arXiv:1611.02779, 2016.
312. <a id="ref-pathak2017curiosity"></a>Pathak, D., Agrawal, P., Efros, A., Darrell, T. "Curiosity-Driven Exploration by Self-Supervised Prediction." Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.
313. <a id="ref-graves2016hybrid"></a>Graves, A., Wayne, G., Reynolds, M., et al. "Hybrid Computing Using a Neural Network with Dynamic External Memory." Nature, 2016.
314. <a id="ref-wu2024longmemeval"></a>Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K., Yu, D. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." International Conference on Learning Representations (ICLR), 2025. <https://arxiv.org/abs/2410.10813>
315. <a id="ref-maharana2024locomo"></a>Maharana, A., Lee, D., Tuber, S., Ruber, M., Barbieri, F., Bansal, M. "LoCoMo: Long-Context Conversation with Memory Operations." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
316. <a id="ref-zhang2024infinitebench"></a>Zhang, X., Chen, Y., Hu, S., et al. "InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024. <https://arxiv.org/abs/2402.13718>
317. <a id="ref-sumers2023coala"></a>Sumers, T., Yao, S., Narasimhan, K., Griffiths, T. "Cognitive Architectures for Language Agents." Transactions on Machine Learning Research (TMLR), 2024. <https://arxiv.org/abs/2309.02427>
318. <a id="ref-lin2025sleeptime"></a>Lin, K., Snell, C., Wang, Y., et al. "Sleep-Time Compute: Beyond Inference Scaling at Test-Time." arXiv Preprint arXiv:2504.13171, 2025. <https://arxiv.org/abs/2504.13171>
319. <a id="ref-zhang2025rlm"></a>Zhang, A., Mahdavi, S., Liang, P., Hashimoto, T. "Recursive Language Models." arXiv Preprint arXiv:2512.24601, 2025. <https://arxiv.org/abs/2512.24601>
320. <a id="ref-schick2023toolformer"></a>Schick, T., Dwivedi-Yu, J., Dessì, R., et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." Advances in Neural Information Processing Systems, 2023.
321. <a id="ref-patil2023gorilla"></a>Patil, S., Zhang, T., Wang, X., Gonzalez, J. "Gorilla: Large Language Model Connected with Massive APIs." Advances in Neural Information Processing Systems (NeurIPS), 2024. <https://arxiv.org/abs/2305.15334>
322. <a id="ref-qin2024toolllm"></a>Qin, Y., Liang, S., Ye, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs." Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2307.16789>
323. <a id="ref-anthropic-mcp-2024"></a>Anthropic. "Model Context Protocol." 2024. <https://modelcontextprotocol.io>
324. <a id="ref-openai2024swarm"></a>OpenAI. "Swarm: An Educational Framework for Lightweight Multi-Agent Orchestration." 2024. <https://github.com/openai/swarm>
325. <a id="ref-langchain2024langgraph"></a>Inc, L. "LangGraph: Build Stateful Multi-Actor Applications with LLMs." 2024. <https://github.com/langchain-ai/langgraph>
326. <a id="ref-wu2023autogen"></a>Wu, Q., Bansal, G., Zhang, J., et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv Preprint arXiv:2308.08155, 2023.
327. <a id="ref-chen2023frugalgpt"></a>Chen, L., Zaharia, M., Zou, J. "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." arXiv Preprint arXiv:2305.05176, 2023. <https://arxiv.org/abs/2305.05176>
328. <a id="ref-chase2022langchain"></a>Chase, H. "LangChain." 2022. <https://github.com/langchain-ai/langchain>
329. <a id="ref-moura2023crewai"></a>Moura, J. "CrewAI: Framework for Orchestrating Role-Playing Autonomous AI Agents." 2023. <https://github.com/crewAIInc/crewAI>
330. <a id="ref-anthropic2024buildingagents"></a>Anthropic. "Building Effective Agents." 2024. <https://www.anthropic.com/research/building-effective-agents>
331. <a id="ref-wang2022selfconsistency"></a>Wang, X., Wei, J., Schuurmans, D., Le, Q., et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." International Conference on Learning Representations (ICLR), 2023. <https://arxiv.org/abs/2203.11171>
332. <a id="ref-brockman2016openai"></a>Brockman, G., Cheung, V., Pettersson, L., et al. "OpenAI Gym." 2016.
333. <a id="ref-yoo2025aela"></a>Yoo, J., Shin, D. "Adaptive Episode Length Adjustment for Multi-Agent Reinforcement Learning." Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2025.
334. <a id="ref-liu2025dler"></a>Liu, Z., Dong, H., et al. "DLER: Doing Length Penalty Right—Incentivizing More Intelligence Per Token." arXiv Preprint arXiv:2510.15110, 2025. <https://arxiv.org/abs/2510.15110>
335. <a id="ref-liu2025answerstop"></a>Liu, Q., et al. "Answer Convergence as a Signal for Early Stopping in Reasoning." Proceedings of EMNLP, 2025. <https://aclanthology.org/2025.emnlp-main.904/>
336. <a id="ref-april2025"></a>Mei, J., et al. "APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-Tail Generation." arXiv Preprint arXiv:2509.18521, 2025. <https://arxiv.org/abs/2509.18521>
337. <a id="ref-hu2025tlt"></a>Hu, Q., Yang, S., et al. "Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter." arXiv Preprint arXiv:2511.16665, 2025. <https://arxiv.org/abs/2511.16665>
338. <a id="ref-jiang2021plr"></a>Jiang, M., Grefenstette, E., Rocktäschel, T. "Prioritized Level Replay." Proceedings of the 38th International Conference on Machine Learning (ICML), 2021. <https://arxiv.org/abs/2010.03934>
339. <a id="ref-dennis2020paired"></a>Dennis, M., Jaques, N., Vinitsky, E., et al. "Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design." Advances in Neural Information Processing Systems (NeurIPS), 2020.
340. <a id="ref-wang2025dataefficiency"></a>Wang, Y., et al. "Improving Data Efficiency for LLM Reinforcement Fine-Tuning via Difficulty-Targeted Online Data Selection." Advances in Neural Information Processing Systems (NeurIPS), 2025. <https://arxiv.org/abs/2506.05316>
341. <a id="ref-liu2025adcl"></a>Liu, X., et al. "Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning." arXiv Preprint arXiv:2505.08364, 2025. <https://arxiv.org/abs/2505.08364>
342. <a id="ref-koh2024visualwebarena"></a>Koh, J., Lo, R., Jang, L., et al. "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024. <https://arxiv.org/abs/2401.13649>
343. <a id="ref-deng2024mind2web"></a>Deng, X., Gu, Y., Zheng, B., et al. "Mind2Web: Towards a Generalist Agent for the Web." Advances in Neural Information Processing Systems (NeurIPS), 2023. <https://arxiv.org/abs/2306.06070>
344. <a id="ref-xie2024osworld"></a>Xie, T., Zhang, D., Chen, J., et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." Advances in Neural Information Processing Systems, 2024.
345. <a id="ref-bonatti2024windows"></a>Bonatti, R., Zhao, D., Bonacci, F., et al. "Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale." arXiv Preprint arXiv:2409.08264, 2024. <https://arxiv.org/abs/2409.08264>
346. <a id="ref-lala2023paperqa"></a>Lála, J., O'Donoghue, O., Shtedritski, A., Cox, S., Rodriques, S., White, A. "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research." arXiv Preprint arXiv:2312.07559, 2023. <https://arxiv.org/abs/2312.07559>
347. <a id="ref-lu2024aiscientist"></a>Lu, C., Lu, C., Lange, R., Foerster, J., Clune, J., Ha, D. "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." arXiv Preprint arXiv:2408.06292, 2024. <https://arxiv.org/abs/2408.06292>
348. <a id="ref-huang2024mlagentbench"></a>Huang, Q., Vora, J., Liang, P., Leskovec, J. "MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation." Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. <https://arxiv.org/abs/2310.03302>
349. <a id="ref-kuttler2020nethack"></a>Küttler, H., Nardelli, N., Miller, A., et al. "The NetHack Learning Environment." Advances in Neural Information Processing Systems (NeurIPS), 2020. <https://arxiv.org/abs/2006.13760>
350. <a id="ref-mialon2023gaia"></a>Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., Scialom, T. "GAIA: A Benchmark for General AI Assistants." International Conference on Learning Representations (ICLR), 2024. <https://arxiv.org/abs/2311.12983>
351. <a id="ref-lewis2017dealornodeal"></a>Lewis, M., Yarats, D., Dauphin, Y., Parikh, D., Batra, D. "Deal or No Deal? End-to-End Learning for Negotiation Dialogues." Proceedings of EMNLP, 2017.
352. <a id="ref-chawla2021casino"></a>Chawla, K., Ramirez, J., Clever, R., Lucas, G., May, J., Gratch, J. "CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems." Proceedings of NAACL, 2021.
353. <a id="ref-hong2023metagpt"></a>Hong, S., Zhuge, M., Chen, J., et al. "MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework." 2024.
354. <a id="ref-huggingface2025openenv"></a>Face, H. "OpenEnv: An Interface Library for RL Post-Training with Environments." 2025. <https://github.com/huggingface/OpenEnv>
355. <a id="ref-towers2024gymnasium"></a>Towers, M., Kwiatkowski, A., Terry, J., et al. "Gymnasium: A Standard Interface for Reinforcement Learning Environments." NeurIPS Datasets and Benchmarks, 2024. <https://arxiv.org/abs/2407.17032>
356. <a id="ref-xi2024agentgym"></a>Xi, Z., Ding, Y., Chen, W., et al. "AgentGym: Evolving Large Language Model-Based Agents Across Diverse Environments." arXiv Preprint arXiv:2406.04151, 2024. <https://arxiv.org/abs/2406.04151>
357. <a id="ref-drouin2024browsergym"></a>Le, T., Gasse, M., Drouin, A., Caccia, M., et al. "The BrowserGym Ecosystem for Web Agent Research." arXiv Preprint arXiv:2412.05467, 2024. <https://arxiv.org/abs/2412.05467>
358. <a id="ref-meta2025torchforge"></a>Team, M. "TorchForge: PyTorch-Native Post-Training at Scale." 2025. <https://github.com/meta-pytorch/torchforge>
359. <a id="ref-jsonrpc2010spec"></a>Group, J. "JSON-RPC 2.0 Specification." 2010. <https://www.jsonrpc.org/specification>
360. <a id="ref-google-a2a-2025"></a>Google. "Agent2Agent (A2A) Protocol." 2025. <https://developers.google.com/agent2agent>
361. <a id="ref-preston2024semver"></a>Preston-Werner, T. "Semantic Versioning 2.0.0." 2024. <https://semver.org/>
362. <a id="ref-smith1980contract"></a>Smith, R. "The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver." IEEE Transactions on Computers, 1980. <https://doi.org/10.1109/TC.1980.1675516>
363. <a id="ref-rfc7519"></a>Jones, M., Bradley, J., Sakimura, N. "JSON Web Token (JWT)." 2015. <https://datatracker.ietf.org/doc/html/rfc7519>
364. <a id="ref-rfc8705"></a>Campbell, B., Bradley, J., Sakimura, N., Lodderstedt, T. "OAuth 2.0 Mutual-TLS Client Authentication and Certificate-Bound Access Tokens." 2020. <https://datatracker.ietf.org/doc/html/rfc8705>
365. <a id="ref-w3c-did-2022"></a>Consortium, W. "Decentralized Identifiers (DIDs) v1.0." 2022. <https://www.w3.org/TR/did-core/>
366. <a id="ref-rfc8446"></a>Rescorla, E. "The Transport Layer Security (TLS) Protocol Version 1.3." 2018. <https://datatracker.ietf.org/doc/html/rfc8446>
367. <a id="ref-rfc6749"></a>Hardt, D. "The OAuth 2.0 Authorization Framework." 2012. <https://datatracker.ietf.org/doc/html/rfc6749>
368. <a id="ref-weiss1999multiagent"></a>Weiss, G. "Multiagent Systems: A Modern Approach to Distributed Artificial Intelligence." 1999.
369. <a id="ref-wooldridge2009introduction"></a>Wooldridge, M. "An Introduction to MultiAgent Systems." 2009.
370. <a id="ref-durfee1989distributed"></a>Durfee, E., Lesser, V., Corkill, D. "Trends in Cooperative Distributed Problem Solving." IEEE Transactions on Knowledge and Data Engineering, 1989. <https://doi.org/10.1109/69.43404>
371. <a id="ref-fipa2002acl"></a>Intelligent, F. "FIPA ACL Message Structure Specification." 2002. <http://www.fipa.org/specs/fipa00061/>
372. <a id="ref-grasse1959reconstruction"></a>Grassé, P. "La Reconstruction Du Nid Et Les Coordinations Interindividuelles Chez Bellicositermes Natalensis Et Cubitermes Sp. La Théorie de La Stigmergie: Essai d'interprétation Du Comportement Des Termites Constructeurs." Insectes Sociaux, 1959.
373. <a id="ref-debono1985six"></a>Bono, E. "Six Thinking Hats." 1985.
374. <a id="ref-du2023improving"></a>Du, Y., Li, S., Torralba, A., Tenenbaum, J., Mordatch, I. "Improving Factuality and Reasoning in Language Models Through Multiagent Debate." Proceedings of the 41st International Conference on Machine Learning (ICML), 2023. <https://arxiv.org/abs/2305.14325>
375. <a id="ref-shapley1953stochastic"></a>Shapley, L. "Stochastic Games." Proceedings of the National Academy of Sciences, 1953. <https://www.pnas.org/doi/10.1073/pnas.39.10.1095>
376. <a id="ref-lowe2017multi"></a>Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., Mordatch, I. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." Advances in Neural Information Processing Systems (NeurIPS), 2017. <https://arxiv.org/abs/1706.02275>
377. <a id="ref-rashid2018qmix"></a>Rashid, T., Samvelyan, M., Witt, C., Farquhar, G., Foerster, J., Whiteson, S. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." Proceedings of the 35th International Conference on Machine Learning (ICML), 2018. <https://arxiv.org/abs/1803.11605>
378. <a id="ref-sukhbaatar2016learning"></a>Sukhbaatar, S., Szlam, A., Fergus, R. "Learning Multiagent Communication with Backpropagation." Advances in Neural Information Processing Systems (NeurIPS), 2016. <https://arxiv.org/abs/1605.07736>
379. <a id="ref-das2019tarmac"></a>Das, A., Gerber, T., Gkioxari, G., Lee, S., Parikh, D., Batra, D. "TarMAC: Targeted Multi-Agent Communication." Proceedings of the 36th International Conference on Machine Learning (ICML), 2019. <https://arxiv.org/abs/1810.11187>
380. <a id="ref-lazaridou2020emergent"></a>Lazaridou, A., Baroni, M. "Emergent Multi-Agent Communication in the Deep Learning Era." arXiv Preprint arXiv:2006.02419, 2020. <https://arxiv.org/abs/2006.02419>
381. <a id="ref-silver2017mastering"></a>Silver, D., Schrittwieser, J., Simonyan, K., et al. "Mastering the Game of Go Without Human Knowledge." Nature, 2017. <https://www.nature.com/articles/nature24270>
382. <a id="ref-jaderberg2019human"></a>Jaderberg, M., Czarnecki, W., Dunning, I., et al. "Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning." Science, 2019. <https://www.science.org/doi/10.1126/science.aau6249>
383. <a id="ref-shoham2008multiagent"></a>Shoham, Y., Leyton-Brown, K. "Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations." 2008. <http://www.masfoundations.org/>
384. <a id="ref-zhang2021multiagent"></a>Zhang, K., Yang, Z., Ba{ş}ar, T. "Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms." Handbook of Reinforcement Learning and Control, 2021. <https://arxiv.org/abs/1911.10635>
385. <a id="ref-nisan2007algorithmic"></a>Nisan, N., Roughgarden, T., Tardos, É., Vazirani, V. "Algorithmic Game Theory." 2007. <https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.pdf>
386. <a id="ref-openai2024agentssdk"></a>OpenAI. "OpenAI Agents SDK." 2025. <https://github.com/openai/openai-agents-python>
387. <a id="ref-microsoft2023semantickernel"></a>Microsoft. "Semantic Kernel: SDK for Integrating AI Models into Applications." 2023. <https://github.com/microsoft/semantic-kernel>
388. <a id="ref-beurerkellner2023lmql"></a>Beurer-Kellner, L., Fischer, M., Vechev, M. "Prompting Is Programming: A Query Language for Large Language Models." Proceedings of PLDI, 2023.
389. <a id="ref-parasuraman1997humans"></a>Parasuraman, R., Riley, V. "Humans and Automation: Use, Misuse, Disuse, Abuse." Human Factors, 1997. <https://doi.org/10.1518/001872097778543886>
390. <a id="ref-vercel2024aisdk"></a>Vercel. "Vercel AI SDK." 2024. <https://sdk.vercel.ai>
391. <a id="ref-chainlit2024"></a>Chainlit. "Chainlit: Build Production-Ready Conversational AI Applications." 2024. <https://chainlit.io>
392. <a id="ref-abid2019gradio"></a>Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., Zou, J. "Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild." 2019. <https://arxiv.org/abs/1906.02569>
393. <a id="ref-streamlit2024"></a>Inc, S. "Streamlit: The Fastest Way to Build and Share Data Apps." 2024. <https://streamlit.io>
394. <a id="ref-langgraph2024studio"></a>Inc, L. "LangGraph Studio: The First Agent IDE." 2024. <https://github.com/langchain-ai/langgraph-studio>
395. <a id="ref-garcia2015comprehensive"></a>García, J., Fernández, F. "A Comprehensive Survey on Safe Reinforcement Learning." Journal of Machine Learning Research, 2015.
396. <a id="ref-irving2018debate"></a>Irving, G., Christiano, P., Amodei, D. "AI Safety via Debate." arXiv Preprint arXiv:1805.00899, 2018. <https://arxiv.org/abs/1805.00899>
397. <a id="ref-hubinger2024sleeper"></a>Hubinger, E., Denison, C., Mu, J., et al. "Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training." arXiv Preprint arXiv:2401.05566, 2024.
398. <a id="ref-amodei2016concrete"></a>Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., Mané, D. "Concrete Problems in AI Safety." arXiv Preprint arXiv:1606.06565, 2016. <https://arxiv.org/abs/1606.06565>
399. <a id="ref-greshake2023indirect"></a>Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., Fritz, M. "Not What You've Signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec), 2023.
400. <a id="ref-skalse2022defining"></a>Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. "Defining and Characterizing Reward Hacking." Advances in Neural Information Processing Systems, 2022. <https://arxiv.org/abs/2209.13085>
401. <a id="ref-kim2024distillation"></a>Kim, S., Jang, S., et al. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes." Findings of the ACL, 2023. <https://arxiv.org/abs/2305.02301>
