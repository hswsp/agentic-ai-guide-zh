# 智能体 AI 漫游指南 —— 博客站点设计

日期：2026-08-06
分支：`gh-pages`

## 目标

把本仓库的 LaTeX 书籍《智能体 AI 漫游指南：从基础到系统》（The Hitchhiker's Guide to Agentic AI 中文译本）逐章转换为一个 GitHub Pages 博客站点，采用 Jekyll + jekyll-gitbook 主题，GitBook 风格的书式导航，与 transformers.run 风格一致。

## 背景与现状

- 仓库当前 `main` 分支保存 LaTeX 书籍源码：`book.tex`、`contents/part1~6.tex`、`figures/`（73 PNG + 1 JPG）、`book.bib`。
- 内容共 29 章、6 大板块（Part I–VI），涵盖 LLM 基础、RL 方法、推理、评估、智能体 AI、自测与参考。
- 全书遵循 CC BY-SA 4.0 许可证。

## 技术栈决策

| 项 | 选择 |
|----|------|
| 静态站点 | Jekyll |
| 主题 | jekyll-gitbook（`remote_theme: sighingnow/jekyll-gitbook`） |
| 数学渲染 | MathJax v3（CDN，`tex-chtml`） |
| 部署 | GitHub Pages（`gh-pages` 分支，仓库根构建） |
| 图片 | 复用 `figures/`，复制到 `gh-pages` 分支 |

## 分支与目录策略

- `main`：保留 LaTeX 书籍源码不变。
- `gh-pages`：博客站点，从仓库根由 GitHub Pages 构建部署。

```
gh-pages 分支
├── _config.yml          # Jekyll 配置：主题、MathJax、导航
├── _includes/           # 主题覆盖（head 加 MathJax）
├── _layouts/            # 布局覆盖
├── _data/               # 导航数据（6 大板块 + 29 章）
├── _pages/
│   ├── part1/ … part6/  # 29 章 Markdown 页面，按板块组织
├── index.md             # 首页
├── figures/             # 复用的图片（复制自 main）
└── assets/              # 覆盖的静态资源
```

## 站点架构

- 章节页使用 `_pages/`（collection 模式，jekyll-gitbook 支持），而非 `_posts/`——这是"书"而非按时间倒序的博客流，侧边栏按书的章节顺序排列。
- 侧边栏导航通过 `_data/` 定义 6 大板块 + 29 章，GitBook 风格左侧目录树。
- 首页 `index.md` 仿照 transformers.run 的 "Hello!" 页：书名简介、板块导航、License（CC BY-SA 4.0）。

## 主题集成与数学渲染

- 引入 `remote_theme: sighingnow/jekyll-gitbook`，GitHub Pages 构建时自动拉取，无需复制主题源码。
- 覆盖 `_includes/head.html` 加入 MathJax v3：
  - 行内公式 `$...$`、块级公式 `$$...$$`（LaTeX 原书语法直接迁移）。
  - 支持 `\(...\)` / `\[...\]`。

## 内容转换规范（LaTeX → Markdown）

1. **结构**：`\chapter{...}` → `#`；`\section` → `##`；`\subsection` → `###`，逐级映射。
2. **数学公式**：`$...$`、`$$...$$`、`\begin{equation}`、`\begin{align}` 等直接保留为 Markdown 行内/块级公式，MathJax 渲染。
3. **图片**：`\includegraphics{figures/fig_XXX.png}` → `![fig_XXX](/figures/fig_XXX.png)`，`\caption` 作为 alt 文本。
4. **代码**：`\begin{lstlisting}` / `\begin{minted}` → 代码块（标注语言）。
5. **引用/交叉引用**：`\cite{...}` 保留原文引用或转脚注/参考链接；`\ref{...}` → 中文章节标题文字。
6. **表格**：`\begin{table}` / `tabular` → Markdown 表格。
7. **保留**：环境、算法框、Blockquote 等转为对应 Markdown 元素。

## 交付顺序

按 part 逐部分批转换：
1. 配置好 Jekyll 站点、主题、MathJax、导航、首页。
2. 转换 Part I 章节 → 本地 `jekyll serve` 验证（公式、图片、导航正常）。
3. 依次完成 Part II–VI，每批验证后再继续下一批。

## 验证

- 本地 `jekyll serve` 渲染无报错。
- 每章数学公式、图片路径、导航链接正确。
- GitHub Pages 部署后站点可访问。

## 范围边界

- 本轮只搭站点 + 转换全部 29 章为 Markdown 页面。
- 不引入额外功能（评论、搜索插件若无必要）。
- 不修改 `main` 分支的 LaTeX 源码。