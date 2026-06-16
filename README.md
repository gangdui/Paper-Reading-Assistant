# 面向研究生的科研论文发现与阅读助手

## 项目简介

本项目是一个面向研究生科研阅读场景的课程 MVP。它提供“论文发现与核查”和“PDF 阅读笔记”两个核心功能，帮助用户从研究方向出发发现候选论文，再基于文献核查状态、推荐理由和阅读建议筛选值得精读的论文；也可以上传 PDF 论文，生成中文结构化文献笔记。

项目强调可运行、可验证和适合课堂 demo。系统不会让 AI 直接编造论文列表，论文标题、作者、年份、venue、URL 和 PDF 链接必须来自真实论文 API，或明确标注为 mock 示例数据。

## 核心功能

### 论文发现与核查

- 输入研究方向，例如 `AIGC 模型水印`。
- 使用 DeepSeek 扩展英文关键词和检索式。
- 优先尝试 Semantic Scholar API 检索真实论文。
- 未配置 Semantic Scholar API Key 或受限时，可切换到 arXiv 预印本数据源。
- 真实检索失败时，使用 mock fallback 示例数据保证课堂 demo 流程可展示。
- 展示 Top 3 优先关注论文和完整论文推荐列表。
- 每篇论文展示标题、年份、来源、venue、PDF 状态、推荐优先级、推荐理由、阅读建议、核查状态、论文链接和 PDF 链接。
- 英文 abstract、作者、缺失字段、命中关键词和 paperId 默认折叠。
- 明确说明 arXiv 是预印本来源，不等于正式发表论文，正式 venue 需要后续核查。

### PDF 阅读笔记

- 上传单篇 PDF 论文。
- 使用 `pdfplumber` 提取文本。
- 对文本进行基础清洗，包括空格、换行、断词、页眉页脚等问题。
- 调用 DeepSeek API 生成中文结构化文献笔记。
- 支持 Markdown 下载。
- 支持 API 错误提示和 fallback 示例输出，保证课堂 demo 稳定性。

### 文献核查 Agent

- 不允许 AI 直接生成论文列表。
- 论文元数据必须来自 Semantic Scholar、arXiv，或明确标注的 mock 示例数据。
- 检查来源是否可追溯。
- 检查元数据是否完整。
- 给出质量判断：`arXiv 预印本` / `正式发表` / `顶会顶刊` / `待进一步确认`。
- 明确提示：质量判断不等于论文结论可靠，只表示来源或 venue 层面的初步判断。

## 产品流程

```text
输入研究方向
  -> 扩展英文关键词
  -> 检索真实论文
  -> 文献核查
  -> 查看推荐理由与阅读建议
  -> 打开感兴趣论文 PDF
  -> 切换到“阅读 PDF”
  -> 上传 PDF
  -> 生成结构化文献笔记
  -> 下载 Markdown
```

## 技术栈

- Python
- Streamlit
- pdfplumber
- OpenAI Python SDK
- DeepSeek API
- Semantic Scholar API
- arXiv API
- python-dotenv
- requests
- Markdown

## 项目目录结构

```text
.
├── app.py
├── paper_discovery.py
├── requirements.txt
├── .env.example
├── README.md
├── docs
│   ├── idea.md
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md
├── tests
│   ├── test_cases.md
│   └── acceptance_checklist.md
└── prompts
    └── ai_prompts.md
```

## 环境变量说明

在项目根目录创建 `.env` 文件：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-v4-flash
SEMANTIC_SCHOLAR_API_KEY=
```

说明：

- `DEEPSEEK_API_KEY`：必需。用于关键词扩展和 PDF 文献笔记生成。
- `DEEPSEEK_MODEL`：可选。默认可使用 `deepseek-v4-flash`，也可以根据 DeepSeek 平台可用模型调整。
- `SEMANTIC_SCHOLAR_API_KEY`：可选。配置后可提高 Semantic Scholar 请求稳定性；不配置时系统会先尝试公开检索，必要时切换 arXiv。
- arXiv API 不需要 API Key。

注意：`.env` 保存真实密钥，不应提交到仓库；应提交 `.env.example`。

## 安装依赖

```powershell
cd C:\Users\28789\Documents\Paper-Reading-Assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果已经创建过虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行项目

```powershell
streamlit run app.py
```

启动后访问：

```text
http://localhost:8501
```

## Demo 步骤

### 论文发现 demo

1. 打开页面后进入“发现论文”。
2. 输入研究方向，例如 `AIGC 模型水印`。
3. 点击“扩展关键词并检索论文”。
4. 查看“助手总结”和“优先关注论文”。
5. 打开完整论文列表，检查推荐理由、阅读建议、核查摘要、论文链接和 PDF 链接。
6. 如果结果来自 arXiv，说明这些是预印本，正式发表信息需要后续核查。

### PDF 阅读笔记 demo

1. 切换到“阅读 PDF”。
2. 上传一篇可解析文本的 PDF。
3. 等待文本提取和清洗完成。
4. 点击“生成结构化文献笔记”。
5. 查看 9 个模块的中文结构化笔记。
6. 下载 Markdown 文献笔记。

### 稳定性 demo

- 如果 DeepSeek API 不可用，可在 PDF 阅读页使用示例输出 fallback。
- 如果 Semantic Scholar 未配置 Key 或公共限流，系统会尝试 arXiv 预印本数据源。
- 如果真实论文 API 都不可用，系统会使用 mock fallback 示例数据，仅用于课堂演示。

## 常见问题

### 为什么显示 arXiv 预印本？

当 Semantic Scholar 未配置 API Key、公共限流或请求失败时，系统会尝试使用 arXiv API 作为真实论文数据源。arXiv 结果适合追踪最新论文，但通常是预印本，正式发表 venue 需要后续通过 Semantic Scholar、DBLP、Crossref 或论文主页进一步核查。

### 为什么显示 mock fallback？

当 Semantic Scholar 和 arXiv 都不可用时，系统会展示 mock fallback 示例数据，保证课堂 demo 页面流程不中断。mock fallback 不代表真实推荐，也不能用于真实文献调研。

### 为什么 Semantic Scholar 未认证模式会限流？

Semantic Scholar 公开 API 在未配置 Key 时可能有更严格的公共请求限制。配置 `SEMANTIC_SCHOLAR_API_KEY` 可以提高稳定性，但本项目仍支持无 Key 运行。

### 为什么 arXiv 论文不等于高质量论文？

arXiv 是预印本平台，论文可能尚未经过正式同行评审。系统会把 arXiv 标注为“预印本”，不会直接判断为顶会或顶刊。质量判断只表示来源或 venue 层面的初步判断，不代表论文结论一定可靠。

### 为什么 PDF 无法解析？

可能原因包括：

- PDF 是扫描版图片。
- PDF 被加密。
- PDF 排版复杂，`pdfplumber` 无法稳定提取文本。

当前 MVP 暂不支持 OCR。扫描版 PDF 需要后续扩展 OCR 功能。

### 为什么 DeepSeek 调用失败？

可能原因包括：

- `.env` 中没有配置 `DEEPSEEK_API_KEY`。
- API Key 无效或余额不足。
- 模型名称填写错误。
- 网络无法访问 DeepSeek API。

可以检查 `.env` 并重启 Streamlit。

## 关于 .env

不要提交：

```text
.env
```

可以提交：

```text
.env.example
```

`.env.example` 只保留变量名和示例值，用于说明项目需要哪些环境变量。
