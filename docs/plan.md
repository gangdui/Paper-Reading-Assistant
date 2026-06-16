# 技术方案

## 技术栈

- 前端与页面框架：Streamlit
- PDF 解析：pdfplumber
- AI 调用：OpenAI Python SDK 调用 DeepSeek 兼容接口
- 论文检索：Semantic Scholar API、arXiv API
- HTTP 请求：requests
- 环境变量：python-dotenv
- 文档格式：Markdown

## 系统模块

### 1. Streamlit 页面模块

负责展示产品界面，包括：

- 顶部状态条。
- “发现论文”页面。
- “阅读 PDF”页面。
- sidebar 项目说明和当前状态。
- 结果卡片、Top 3 推荐区、调试折叠区。

### 2. 论文发现模块

负责从研究方向到候选论文列表的流程：

1. 接收用户输入的研究方向。
2. 使用 DeepSeek 扩展英文关键词和检索式。
3. 调用 Semantic Scholar API。
4. 在失败或受限时切换到 arXiv API。
5. 在真实 API 都失败时使用 mock fallback。
6. 合并、去重、排序论文结果。

### 3. 文献核查 Agent

负责检查论文结果是否可展示：

- 检查 title 是否存在。
- 检查 authors、year、venue、abstract、citationCount 等元数据。
- 检查 url 或 paperId 是否存在。
- 检查 openAccessPdf 是否存在。
- 输出来源核查、元数据完整度和质量判断。

该 Agent 的核心原则是：不让 AI 编造论文列表，论文元数据必须来自 API 或明确标注的 mock 示例。

### 4. 论文推荐展示模块

负责将论文结果转化为研究生易理解的阅读线索：

- Top 3 优先关注论文。
- 推荐优先级：高 / 中 / 低。
- 推荐理由。
- 阅读建议。
- 阅读定位：入门综述 / 方法精读 / 最新进展 / 扩展阅读。
- 折叠展示 abstract、作者、paperId 等详细元数据。

推荐理由和阅读建议只基于真实元数据生成，不补充不存在的作者、会议、年份或引用数。

### 5. PDF 解析与清洗模块

负责处理上传 PDF：

1. 读取 PDF 文件。
2. 使用 pdfplumber 提取文本。
3. 清洗换行、空格、断词、页眉页脚和水印。
4. 保存原始文本和清洗文本到 session_state。

### 6. 结构化文献笔记模块

负责调用 DeepSeek API 生成中文结构化笔记：

- 最多发送清洗后前 15000 个字符。
- 输出 9 个固定模块。
- 对原文未明确说明的部分要求输出“原文未明确说明”。
- 支持 fallback 示例输出。

### 7. 状态与 fallback 模块

负责维护页面稳定性：

- 使用 `st.session_state` 保存检索结果、当前数据源、当前模式和笔记结果。
- 使用 `st.cache_data` 缓存相同 query 的检索结果。
- 处理 Semantic Scholar 未认证限流。
- 处理 arXiv fallback。
- 处理 mock fallback。
- 处理 DeepSeek 调用失败。

## 数据流程

### 论文发现流程

```text
用户输入研究方向
  -> DeepSeek 扩展英文关键词
  -> Semantic Scholar 检索
  -> 如果失败或限流，切换 arXiv
  -> 如果仍失败，使用 mock fallback
  -> 文献核查 Agent
  -> 去重和排序
  -> Top 3 推荐
  -> 完整论文列表
```

### PDF 阅读笔记流程

```text
用户上传 PDF
  -> pdfplumber 提取文本
  -> clean_paper_text 清洗文本
  -> 截取前 15000 字符
  -> DeepSeek 生成结构化文献笔记
  -> 页面展示
  -> Markdown 下载
```

## API 调用设计

### DeepSeek API

- 使用 OpenAI Python SDK。
- `base_url="https://api.deepseek.com"`。
- 从 `.env` 读取 `DEEPSEEK_API_KEY`。
- 用于关键词扩展和 PDF 笔记生成。
- 不用于直接编造论文列表。

### Semantic Scholar API

- 优先用于真实论文检索。
- `SEMANTIC_SCHOLAR_API_KEY` 可选。
- 未配置 Key 时使用公开检索，但可能受到限流。
- 返回的论文元数据用于展示和核查。

### arXiv API

- 无需 API Key。
- 用作 Semantic Scholar 不可用时的真实预印本数据源。
- arXiv 结果必须标注为预印本。
- 正式发表 venue 需要后续核查。

### mock fallback

- 仅用于课堂 demo 稳定性。
- 必须明确标注为示例数据。
- 不代表真实推荐。

## 风险与应对

| 风险 | 应对 |
|---|---|
| DeepSeek API 不可用 | 显示错误提示；PDF 笔记支持示例输出 fallback |
| Semantic Scholar 限流 | 切换 arXiv 预印本数据源 |
| arXiv 请求失败 | 使用 mock fallback 示例数据 |
| AI 编造论文 | 禁止 AI 生成论文列表，论文元数据只来自 API |
| arXiv 被误认为正式发表 | 页面明确提示 arXiv 是预印本 |
| PDF 无法解析 | 提示可能需要 OCR，当前 MVP 暂不支持 |
| 课堂网络不稳定 | 使用 session_state、cache 和 fallback 保证 demo 流程 |

## 后续扩展

- 增加 OpenAlex、Crossref、DBLP 数据源。
- 增加 OCR 支持扫描版 PDF。
- 增加本地文献库保存。
- 增加引用格式导出。
- 增加多篇论文对比阅读。
- 增加更完整的自动化 UI 测试。
