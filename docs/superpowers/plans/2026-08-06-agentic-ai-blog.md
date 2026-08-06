# 智能体 AI 漫游指南博客站点 —— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 `gh-pages` 分支上用 Jekyll + jekyll-gitbook 主题搭建一个 GitHub Pages 博客站点，并把 LaTeX 书籍的 30 章内容逐章转换为 Markdown 页面。

**Architecture:** 站点文件放在 `gh-pages` 分支仓库根；使用 `remote_theme: sighingnow/jekyll-gitbook` 提供 GitBook 风格书式导航；覆盖 `_includes/head.html` 加入 MathJax v3 渲染 `$...$`/`$$...$$` 公式；章节页用 `_pages/` collection 组织，按 6 大板块建目录；`figures/` 从 main 复制过来；`_data/sidebar.yml` 定义全部 30 章的侧边栏导航。

**Tech Stack:** Jekyll、jekyll-gitbook 主题、MathJax v3、GitHub Pages、Markdown、LaTeX。

## Global Constraints

- 站点构建在 `gh-pages` 分支，从仓库根由 GitHub Pages 构建部署。
- 主题通过 `remote_theme: sighingnow/jekyll-gitbook` 引入，不复制主题源码进仓库。
- MathJax v3 用 CDN（`https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js`），配置 `inlineMath: [['$','$'],['\\(','\\)']]`。
- 图片复用 `figures/`，需复制到 `gh-pages` 分支根目录；Markdown 引用路径为 `/figures/<file>`。
- 章节页放在 `_pages/` 的 collection 目录，按 `part1/`…`part6/` 组织，不使用 `_posts/`。
- 侧边栏导航顺序 = 书的章节顺序（6 大板块 + 30 章），见 `_data/sidebar.yml`。
- 许可证：CC BY-SA 4.0，在 `index.md` 标注。
- 转换规则（见 spec 2026-08-06）：`\chapter`→`#`、`\section`→`##`、`\subsection`→`###`；公式/图片/代码/表格按 spec 规则处理。
- 实际章节数为 30（part1 含一个被注释的 `\chapter`，不做转换）。

---

### Task 1: 搭建 Jekyll 站点骨架

**Files:**
- Create: `_config.yml`
- Create: `index.md`
- Create: `_includes/head.html`
- Create: `_data/sidebar.yml`
- Create: `Gemfile`
- Create: `.gitignore`

**Interfaces:**
- Produces: `_config.yml`（含 `remote_theme`、`collections`、`plugins`）、`_data/sidebar.yml`（导航数据，后续部分任务逐步填充章节条目）、`_includes/head.html`（MathJax 注入）。

- [ ] **Step 1: 创建 `_config.yml`**

```yaml
remote_theme: sighingnow/jekyll-gitbook

title: 智能体 AI 漫游指南
description: 从基础到系统 —— The Hitchhiker's Guide to Agentic AI 中文译本
lang: zh-CN

collections:
  pages:
    output: true
  posts:
    output: true

defaults:
  - scope:
      path: ""
    values:
      lang: zh-CN
  - scope:
      path: "_pages"
      type: pages
    values:
      layout: home
      permalink: /:path/

plugins:
  - jekyll-remote-theme
  - jekyll-include-cache

exclude:
  - README.md
  - book.tex
  - contents
  - book.bib
  - Makefile
  - docs
  - .github
```

- [ ] **Step 2: 创建 Gemfile**

```ruby
source "https://rubygems.org"
gem "jekyll"
gem "jekyll-remote-theme"
gem "jekyll-include-cache"
gem "webrick"
gem "kramdown-parser-gfm"
```

- [ ] **Step 3: 创建 `_includes/head.html` 覆盖，注入 MathJax v3**

```html
<meta name="generator" content="Jekyll (using style of GitBook 3.2.3)">
<link rel="stylesheet" href="{{ site.baseurl }}/assets/gitbook/style.css">
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  svg: { fontCache: 'global' }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
```

- [ ] **Step 4: 创建 `_data/sidebar.yml` 骨架（先放板块标题，章节后续任务填充）**

```yaml
- title: 首页
  url: /
- divider: true
- title: Part I — Foundations 基础
  children:
    - title: 第 1 章 LLM 架构与优化方法
      url: /part1/ch01-llm-architecture.html
- divider: true
- title: Part II — RL Methods for LLMs 面向 LLM 的 RL 方法
  children: []
- divider: true
- title: Part III — Reasoning 推理
  children: []
- divider: true
- title: Part IV — Evaluation 评估
  children: []
- divider: true
- title: Part V — Agentic AI 智能体 AI
  children: []
- divider: true
- title: Part VI — Assessment & Reference 自测与参考
  children: []
```

- [ ] **Step 5: 创建 `index.md` 首页**

```markdown
---
layout: home
title: 智能体 AI 漫游指南
---

# 智能体 AI 漫游指南：从基础到系统

> _The Hitchhiker's Guide to Agentic AI: From Foundations to Systems_ — 中文翻译版

本书系统覆盖从 Transformer 内部机制、GPU 训练系统、强化学习、对齐方法、推理模型，
到智能体编排、多智能体协作与 Agentic UI 的完整技术栈。

## 全书结构

- Part I — Foundations 基础（第 1–3 章）
- Part II — RL Methods for LLMs 面向 LLM 的 RL 方法（第 4–12 章）
- Part III — Reasoning 推理（第 13 章）
- Part IV — Evaluation 评估（第 14 章）
- Part V — Agentic AI 智能体 AI（第 15–26 章）
- Part VI — Assessment & Reference 自测与参考（第 27–29 章）

## License

本译本遵循 **CC BY-SA 4.0** 许可证。原文为 Haggai Roitman 所著。
```

- [ ] **Step 6: 创建 `.gitignore`**

```
_site/
.jekyll-cache/
.bundle/
vendor/
Gemfile.lock
```

- [ ] **Step 7: 本地验证站点可启动**

Run: `bundle install && bundle exec jekyll serve --baseurl ""`
Expected: 站点在本机启动，`http://localhost:4000/` 显示首页，无报错。

- [ ] **Step 8: 提交**

```bash
git add _config.yml Gemfile _includes/head.html _data/sidebar.yml index.md .gitignore
git commit -m "feat(blog): scaffold jekyll-gitbook site with MathJax and sidebar"
```

---

### Task 2: 复制 figures/ 到 gh-pages 分支

**Files:**
- Create: `figures/`（从 main 复制）

**Interfaces:**
- Produces: `figures/*.png|jpg` 位于 gh-pages 分支根，供所有章节 Markdown 引用。

- [ ] **Step 1: 从 main 检出 figures 目录**

Run: `git checkout main -- figures/`
Expected: `figures/`（73 PNG + 1 JPG）出现在 gh-pages 分支根。

- [ ] **Step 2: 确认文件存在**

Run: `ls figures/ | wc -l`
Expected: `74`

- [ ] **Step 3: 提交**

```bash
git add figures/
git commit -m "chore(blog): copy figures directory for blog deployment"
```

---

### Task 3: 转换 Part I（第 1–3 章）

**Files:**
- Create: `_pages/part1/ch01-llm-architecture.md`
- Create: `_pages/part1/ch02-systems-foundations.md`
- Create: `_pages/part1/ch03-rl-intro.md`
- Modify: `_data/sidebar.yml`（填充 Part I 章节）

**Interfaces:**
- Consumes: Task 1 的 `_config.yml`（`_pages` collection、permalink `/part1/...`）、MathJax。
- Consumes: Task 2 的 `figures/`。
- Produces: 3 个章节 Markdown，示范全部转换规则（公式、图片、代码、表格、引用）。

**转换规则（对每个章节相同）：**
- `\chapter{标题}` → 文件 front matter `title: 标题` + `# 标题`。
- `\section{...}` → `## ...`；`\subsection` → `###`。
- `$...$` / `$$...$$` / `\begin{equation}` / `\begin{align}` → 直接保留为行内/块级公式。
- `\includegraphics[..]{figures/fig_XXX.png}` + `\caption{...}` → `![caption](/figures/fig_XXX.png)`。
- `\begin{lstlisting}` … `\end{lstlisting}` → 围栏代码块（标注语言）。
- `\begin{table}`/`tabular`/`longtable` → Markdown 表格。
- `\cite{key}` → 保留为 `[key]` 文字引用（或脚注）。
- `\label{...}` 忽略；`\ref{...}` → 对应章节标题文本。
- tcolorbox 环境（`tipbox`/`example` 等）→ `> **注意/提示：**` 引用块。
- 删除 LaTeX 注释行（`% ...`）。

- [ ] **Step 1: 转换第 1 章 `contents/part1.tex` 第一个 `\chapter`（LLM 架构与优化方法）到 `_pages/part1/ch01-llm-architecture.md`**

按上述转换规则逐节转换，确保公式、图片路径、代码块完整。完成后继续第 2 章（面向 LLM 的系统基础）、第 3 章（强化学习导论）。

- [ ] **Step 2: 更新 `_data/sidebar.yml` 填充 Part I 三个章节**

```yaml
- title: Part I — Foundations 基础
  children:
    - title: 第 1 章 LLM 架构与优化方法
      url: /part1/ch01-llm-architecture.html
    - title: 第 2 章 面向 LLM 的系统基础
      url: /part1/ch02-systems-foundations.html
    - title: 第 3 章 强化学习导论
      url: /part1/ch03-rl-intro.html
```

- [ ] **Step 3: 本地验证渲染**

Run: `bundle exec jekyll serve --baseurl ""`
Expected: 三章页面可访问，公式渲染、图片 `/figures/...` 正常加载、侧边栏显示三章。

- [ ] **Step 4: 提交**

```bash
git add _pages/part1/ _data/sidebar.yml
git commit -m "feat(blog): convert Part I chapters 1-3 to markdown"
```

---

### Task 4: 转换 Part II（第 4–12 章）

**Files:**
- Create: `_pages/part2/ch04-*.md` … `ch12-*.md`（共 9 章）
- Modify: `_data/sidebar.yml`

**Interfaces:**
- Consumes: Task 1/2/3 的配置与规则。
- Produces: 9 个章节 Markdown（RL 方法核心，含大量 PPO/DPO/GRPO 公式）。

Part II 章节（第 4–12 章）：
- 第 4 章 大语言模型的强化学习基础
- 第 5 章 PPO——近端策略优化
- 第 6 章 DPO——直接偏好优化
- 第 7 章 GRPO——组相对策略优化
- 第 8 章 偏好优化变体
- 第 9 章 奖励模型训练
- 第 10 章 SFT 最佳实践与技巧
- 第 11 章 大规模系统架构与基础设施
- 第 12 章 LLM 智能体训练

- [ ] **Step 1: 将 `contents/part2.tex` 的 9 个 `\chapter` 分别转换为 `_pages/part2/ch04-….md` … `ch12-….md`**（按 Task 3 的转换规则逐章进行）
- [ ] **Step 2: 更新 `_data/sidebar.yml` 填充 Part II 九个章节**
- [ ] **Step 3: 本地验证渲染**（公式、图片、导航）
Run: `bundle exec jekyll serve --baseurl ""`
- [ ] **Step 4: 提交**

```bash
git add _pages/part2/ _data/sidebar.yml
git commit -m "feat(blog): convert Part II chapters 4-12 to markdown"
```

---

### Task 5: 转换 Part III–IV（第 13–14 章）

**Files:**
- Create: `_pages/part3/ch13-reasoning.md`
- Create: `_pages/part4/ch14-evaluation.md`
- Modify: `_data/sidebar.yml`

**Interfaces:**
- Consumes: Task 1/2 配置与规则。
- Produces: 第 13 章（推理）与第 14 章（评估）Markdown。

- [ ] **Step 1: 将 `contents/part3.tex` 转换为 `_pages/part3/ch13-reasoning.md`**
- [ ] **Step 2: 将 `contents/part4.tex` 转换为 `_pages/part4/ch14-evaluation.md`**
- [ ] **Step 3: 更新 `_data/sidebar.yml` 填充 Part III、Part IV**
- [ ] **Step 4: 本地验证渲染**
- [ ] **Step 5: 提交**

```bash
git add _pages/part3/ _pages/part4/ _data/sidebar.yml
git commit -m "feat(blog): convert Part III-IV chapters 13-14 to markdown"
```

---

### Task 6: 转换 Part V（第 15–26 章）

**Files:**
- Create: `_pages/part5/ch15-*.md` … `ch26-*.md`（共 12 章）
- Modify: `_data/sidebar.yml`

**Interfaces:**
- Consumes: Task 1/2 配置与规则。
- Produces: 12 个章节 Markdown（智能体 AI 核心，篇幅最大，约 13000 行 LaTeX）。

Part V 章节（第 15–26 章）：
- 第 15 章 智能体 AI 简介
- 第 16 章 检索增强生成（RAG）
- 第 17 章 智能体记忆系统
- 第 18 章 Agent Harness——上下文管理与编排
- 第 19 章 Agent 设计模式
- 第 20 章 Agent 环境与基准
- 第 21 章 模型上下文协议（Model Context Protocol, MCP）
- 第 22 章 Agent Skills
- 第 23 章 智能体到智能体通信（Agent-to-Agent, A2A）
- 第 24 章 多智能体系统（Multi-Agent Systems）
- 第 25 章 Agent 开发框架
- 第 26 章 Agentic UI 框架

- [ ] **Step 1: 将 `contents/part5.tex` 的 12 个 `\chapter` 分别转换**（按规则逐章转换，注意代码框架示例较多，确保围栏代码块配对正确）
- [ ] **Step 2: 更新 `_data/sidebar.yml` 填充 Part V 十二个章节**
- [ ] **Step 3: 本地验证渲染**（尤其检查长代码块、多级标题、图片）
- [ ] **Step 4: 提交**

```bash
git add _pages/part5/ _data/sidebar.yml
git commit -m "feat(blog): convert Part V chapters 15-26 to markdown"
```

---

### Task 7: 转换 Part VI（第 27–29 章）

**Files:**
- Create: `_pages/part6/ch27-quiz.md`
- Create: `_pages/part6/ch28-quickref.md`
- Create: `_pages/part6/ch29-conclusion.md`
- Modify: `_data/sidebar.yml`

**Interfaces:**
- Consumes: Task 1/2 配置与规则。
- Produces: 第 27 章（测验题与详细解答）、第 28 章（速查手册）、第 29 章（总结与未来方向）Markdown。

- [ ] **Step 1: 将 `contents/part6.tex` 的 3 个 `\chapter` 分别转换为 `_pages/part6/ch27-quiz.md`、`ch28-quickref.md`、`ch29-conclusion.md`**（第 27 章为 108 道测验题，转换为有序列表 + 详细解答）
- [ ] **Step 2: 更新 `_data/sidebar.yml` 填充 Part VI 三个章节**
- [ ] **Step 3: 本地验证渲染**
- [ ] **Step 4: 提交**

```bash
git add _pages/part6/ _data/sidebar.yml
git commit -m "feat(blog): convert Part VI chapters 27-29 to markdown"
```

---

### Task 8: 全站验证与最终检查

**Files:**
- 无新文件，仅验证。

**Interfaces:**
- Consumes: 全部已完成的任务。

- [ ] **Step 1: 全量本地构建**

Run: `bundle exec jekyll build`
Expected: 无报错，`_site/` 生成所有 30 章页面。

- [ ] **Step 2: 检查侧边栏完整性**

Run: `grep -c "url:" _data/sidebar.yml`
Expected: 覆盖首页 + 30 章，共 31 个 url 条目（或核对实际数字，确保无遗漏）。

- [ ] **Step 3: 检查图片引用无 404**

Run: `bundle exec jekyll serve --baseurl ""` 后抽查章节页，确认所有 `/figures/...` 图片可加载。

- [ ] **Step 4: 检查公式渲染**

抽查含公式的章节（如第 5 章 PPO），确认 `$...$`/`$$...$$` 由 MathJax 正确渲染。

- [ ] **Step 5: 提交（如有遗漏修复）**

```bash
git add -A
git commit -m "chore(blog): final verification pass"
```

---