# AI 辅助编程关键提示词记录

本文件记录“面向研究生的科研论文发现与阅读助手”开发过程中使用 AI 辅助编程的关键提示词，适合放入课程报告，用于说明 Agent-based MVP、SDD、TDD 和产品化迭代过程。

## 1. 创建 Streamlit + PDF 上传项目

- 阶段：项目初始化
- 我的提示词：
  > 请帮我创建一个 Python + Streamlit 项目的基础骨架，支持用户上传单篇 PDF 文件，使用 pdfplumber 提取 PDF 文本，页面显示提取到的前 2000 个字符，暂时不调用 AI，生成 app.py、requirements.txt、README.md。
- AI 生成的内容：
  - Streamlit 应用入口。
  - PDF 上传组件。
  - pdfplumber 文本提取逻辑。
  - requirements.txt 和 README 基础说明。
- 我人工修改或确认的内容：
  - 确认项目能运行。
  - 确认 PDF 上传和文本预览可用于课程 demo。
- 作用：
  - 快速建立最小可运行项目骨架。

## 2. 实现 PDF 文本提取

- 阶段：PDF 阅读功能
- 我的提示词：
  > 使用 pdfplumber 提取 PDF 文本，并在页面显示提取文本前 2000 个字符，用于确认 PDF 解析成功。
- AI 生成的内容：
  - PDF 读取函数。
  - 页面上传和预览逻辑。
- 我人工修改或确认的内容：
  - 用实际 PDF 测试文本提取效果。
- 作用：
  - 打通 PDF 输入到文本预览的基础链路。

## 3. 增加文本清洗

- 阶段：PDF 文本质量优化
- 我的提示词：
  > 增加 clean_paper_text(text)，去除多余空行和连续空格，修复英文断词，合并不必要换行，尽量去掉页码、重复页眉和水印，保留论文结构词。
- AI 生成的内容：
  - `clean_paper_text` 函数。
  - 原始文本和清洗文本长度显示。
  - 调试 expander。
- 我人工修改或确认的内容：
  - 确认清洗后文本更适合发送给 AI。
- 作用：
  - 降低 PDF 解析噪声，提高笔记生成质量。

## 4. 接入 DeepSeek API

- 阶段：AI 文献笔记生成
- 我的提示词：
  > 使用 OpenAI Python SDK 调用 DeepSeek API，从 .env 读取 DEEPSEEK_API_KEY，base_url 使用 https://api.deepseek.com，模型使用 deepseek-v4-flash。
- AI 生成的内容：
  - DeepSeek 客户端初始化。
  - `.env` 配置读取。
  - API 错误处理。
- 我人工修改或确认的内容：
  - 将模型名调整为当前可用模型。
  - 确认 API Key 配置方式。
- 作用：
  - 完成 PDF 文本到 AI 结构化笔记的核心能力。

## 5. 生成结构化文献笔记

- 阶段：输出格式设计
- 我的提示词：
  > AI 输出中文结构化文献笔记，必须包含一句话总结、论文摘要、研究背景与研究问题、主要内容概述、提出的方法、实验设计与结果分析、主要贡献、不足与局限、研究生阅读建议。如果原文没有明确说明，请写“原文未明确说明”。
- AI 生成的内容：
  - 固定 9 模块提示词。
  - Markdown 结果展示。
- 我人工修改或确认的内容：
  - 确认输出适合课程报告展示。
- 作用：
  - 让生成结果结构稳定，便于阅读和下载。

## 6. 增加错误处理、fallback 示例输出和 Markdown 下载

- 阶段：demo 稳定性
- 我的提示词：
  > 点击生成后显示加载状态，使用 session_state 保存结果，API 调用失败时显示错误提示，增加 Markdown 下载，增加“使用示例输出进行演示”的 fallback 选项。
- AI 生成的内容：
  - 加载进度提示。
  - session_state 保存笔记。
  - Markdown 下载按钮。
  - 示例 Markdown fallback。
- 我人工修改或确认的内容：
  - 确认断网或 API 不可用时页面不崩溃。
- 作用：
  - 提高课堂 demo 可靠性。

## 7. 生成 SDD 文档

- 阶段：课程文档
- 我的提示词：
  > 请生成 docs/idea.md、docs/spec.md、docs/plan.md、docs/tasks.md，内容适合课程报告，不要过于商业化，体现 Agent-based MVP、AI 辅助编程、SDD 和可验证 demo。
- AI 生成的内容：
  - 项目想法文档。
  - 需求规格文档。
  - 技术方案文档。
  - 任务拆解文档。
- 我人工修改或确认的内容：
  - 根据项目迭代持续更新文档。
- 作用：
  - 支撑 SDD 作业要求。

## 8. 生成 TDD / 验证方案

- 阶段：测试设计
- 我的提示词：
  > 请为当前 MVP 项目生成最小 TDD / 验证方案，包含测试用例和手动验收清单，覆盖 PDF、API Key、fallback、Markdown 下载和页面不崩溃。
- AI 生成的内容：
  - `tests/test_cases.md`
  - `tests/acceptance_checklist.md`
- 我人工修改或确认的内容：
  - 补充论文发现、arXiv 和文献核查相关测试场景。
- 作用：
  - 让课程 demo 前有可执行的验收依据。

## 9. 完善 README

- 阶段：项目交付说明
- 我的提示词：
  > 请完善 README.md，包含项目简介、功能说明、技术栈、目录结构、环境配置、DEEPSEEK_API_KEY、运行项目、demo、常见问题和 .env 注意事项。
- AI 生成的内容：
  - README 运行说明。
  - 常见问题。
  - `.env` 与 `.env.example` 说明。
- 我人工修改或确认的内容：
  - 根据 DeepSeek 和论文发现功能更新说明。
- 作用：
  - 降低运行和展示门槛。

## 10. 论文发现功能

- 阶段：功能升级
- 我的提示词：
  > 新增“论文发现与文献核查”功能，用户输入研究方向，使用 DeepSeek 扩展英文关键词，调用真实论文 API 检索论文，不允许 AI 编造论文列表。
- AI 生成的内容：
  - 新增“论文发现与核查”页面。
  - 关键词扩展流程。
  - 论文检索结果展示。
- 我人工修改或确认的内容：
  - 明确论文元数据必须来自真实 API。
- 作用：
  - 项目从单一 PDF 摘要工具升级为科研论文发现与阅读助手。

## 11. Semantic Scholar API

- 阶段：真实论文检索
- 我的提示词：
  > 第一版优先使用 Semantic Scholar API，返回 title、authors、year、venue、abstract、citationCount、url、openAccessPdf、paperId，并实现 verify_paper(paper)。
- AI 生成的内容：
  - Semantic Scholar 请求逻辑。
  - 字段映射。
  - 文献核查函数。
- 我人工修改或确认的内容：
  - 检查未配置 API Key 时仍可公开检索。
- 作用：
  - 保证论文列表来自真实数据源。

## 12. arXiv fallback

- 阶段：检索稳定性
- 我的提示词：
  > 如果没有 Semantic Scholar API Key，不要报错；如果请求失败或 429，被限流时显示友好提示，并使用 arXiv API 或 mock 示例数据作为 fallback。
- AI 生成的内容：
  - arXiv API 查询。
  - arXiv XML 解析。
  - Semantic Scholar 失败后的 fallback 流程。
  - mock fallback 示例数据。
- 我人工修改或确认的内容：
  - 明确 arXiv 是预印本，不等于正式发表论文。
- 作用：
  - 提升课堂 demo 稳定性，同时保留真实论文来源。

## 13. 文献核查 Agent

- 阶段：可信度控制
- 我的提示词：
  > 文献核查状态拆分为 metadata_verified、source_verified、quality_verified。页面不要把 arXiv 预印本直接描述成高质量论文。
- AI 生成的内容：
  - 来源核查。
  - 元数据完整度检查。
  - 质量判断。
  - 缺失字段展示。
- 我人工修改或确认的内容：
  - 增加“质量判断不等于论文结论可靠”的说明。
- 作用：
  - 避免 AI 推荐论文时造成可信度误解。

## 14. Product design 评审

- 阶段：产品体验评审
- 我的提示词：
  > 请从产品设计角度评审当前 Streamlit 页面，分析首页标题、功能入口、sidebar、API 状态、论文推荐卡片、核查状态、深色模式和课程 demo 视觉风格。
- AI 生成的内容：
  - 当前界面问题清单。
  - 推荐的信息架构。
  - 推荐页面布局。
  - 推荐视觉风格和组件设计。
- 我人工修改或确认的内容：
  - 确认先做第一批信息架构优化，再做论文卡片产品化。
- 作用：
  - 推动项目从“调试 demo”向“产品原型”过渡。

## 15. UI 产品化

- 阶段：界面结构优化
- 我的提示词：
  > 调整 tab 顺序，第一个 tab 为“发现论文”，第二个为“阅读 PDF”；增加顶部状态条；精简 sidebar；将关键词、检索式、warning 和 fallback 详情移入 expander。
- AI 生成的内容：
  - 新 tab 顺序。
  - 顶部状态条。
  - sidebar 精简。
  - 调试信息折叠。
- 我人工修改或确认的内容：
  - 修复顶部状态条和 sidebar 状态不同步问题。
- 作用：
  - 让用户优先看到论文发现流程，减少调试感。

## 16. 论文卡片优化

- 阶段：结果展示产品化
- 我的提示词：
  > 优化每篇论文卡片，默认展示标题、年份、来源、venue、引用数、推荐优先级、为什么推荐、阅读建议、核查摘要、论文链接和 PDF 链接；英文 abstract、作者、缺失字段、命中关键词、paperId 放入 expander。
- AI 生成的内容：
  - 产品化论文卡片。
  - Top 3 优先关注论文。
  - 差异化推荐理由和阅读建议。
  - arXiv 引用数字段说明。
- 我人工修改或确认的内容：
  - 要求推荐理由只能基于真实元数据。
  - 要求 arXiv 不显示为正式发表或高质量论文。
- 作用：
  - 帮助研究生快速判断论文是否值得阅读。

## 17. 文档更新

- 阶段：最终交付
- 我的提示词：
  > 请根据当前项目最新功能更新 README、docs、tests 和 prompts 文档，说明论文发现、文献核查、arXiv fallback、mock fallback、PDF 阅读笔记和课程 demo 流程。
- AI 生成的内容：
  - 最新 README。
  - 最新 SDD 文档。
  - 最新测试用例和验收清单。
  - 最新 AI prompts 记录。
- 我人工修改或确认的内容：
  - 确认不夸大系统能力。
  - 明确 arXiv 是预印本，mock fallback 只是 demo 示例。
- 作用：
  - 形成适合课程报告和产品原型展示的完整交付材料。
