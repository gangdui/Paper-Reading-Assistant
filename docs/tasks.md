# 开发任务清单

## 已完成任务

- [x] 创建 Python + Streamlit 项目骨架。
- [x] 实现 PDF 上传功能。
- [x] 使用 pdfplumber 提取 PDF 文本。
- [x] 显示原始提取文本和清洗文本调试信息。
- [x] 实现 `clean_paper_text(text)` 文本清洗函数。
- [x] 接入 DeepSeek API 生成结构化文献笔记。
- [x] 从 `.env` 读取 `DEEPSEEK_API_KEY`。
- [x] 限制 AI 输入为清洗后前 15000 个字符。
- [x] 支持结构化文献笔记 Markdown 下载。
- [x] 支持 API 错误提示。
- [x] 支持 PDF 笔记 fallback 示例输出。
- [x] 新增“发现论文”页面。
- [x] 使用 DeepSeek 扩展英文关键词和检索式。
- [x] 接入 Semantic Scholar API。
- [x] 支持 `SEMANTIC_SCHOLAR_API_KEY` 可选配置。
- [x] 支持 Semantic Scholar 未认证模式。
- [x] 支持 Semantic Scholar 失败或限流时切换 arXiv。
- [x] 支持 arXiv 预印本数据源。
- [x] 支持真实 API 都失败时 mock fallback。
- [x] 实现文献核查 Agent。
- [x] 检查论文来源、元数据完整度、PDF 链接和 venue。
- [x] 避免 AI 编造论文列表。
- [x] 实现论文去重和排序。
- [x] 实现 Top 3 优先关注论文。
- [x] 实现论文推荐卡片产品化展示。
- [x] 增加推荐理由、阅读建议和推荐优先级。
- [x] 将英文 abstract 和详细元数据默认折叠。
- [x] 增加 arXiv 预印本说明。
- [x] 增加顶部状态条和 sidebar 状态同步。
- [x] 精简 sidebar。
- [x] 生成 SDD 文档。
- [x] 生成 TDD / 验证方案。
- [x] 记录 AI 辅助编程提示词。
- [x] 完善 README。

## 后续任务

- [ ] 补充课程报告截图：论文发现页面。
- [ ] 补充课程报告截图：Top 3 优先关注论文。
- [ ] 补充课程报告截图：arXiv 预印本说明。
- [ ] 补充课程报告截图：PDF 文献笔记生成结果。
- [ ] 补充课程报告截图：Markdown 下载。
- [ ] 增加更系统的单元测试，覆盖论文类型判断和推荐优先级。
- [ ] 增加端到端手动 demo 录屏或截图说明。
- [ ] 补充 `.env.example` 使用说明截图。
- [ ] 后续接入 OpenAlex。
- [ ] 后续接入 Crossref。
- [ ] 后续接入 DBLP。
- [ ] 后续支持 OCR。
- [ ] 后续支持保存历史检索结果和文献笔记。
- [ ] 后续支持多篇论文对比阅读。

## 30-60 分钟任务拆解

- [x] 搭建 Streamlit 页面与 tab 结构。
- [x] 实现 PDF 上传与文本提取。
- [x] 实现文本清洗函数。
- [x] 接入 DeepSeek 生成结构化笔记。
- [x] 增加 Markdown 下载。
- [x] 增加论文发现 tab。
- [x] 实现关键词扩展。
- [x] 接入 Semantic Scholar。
- [x] 实现 arXiv fallback。
- [x] 实现 mock fallback。
- [x] 实现文献核查 Agent。
- [x] 优化论文推荐卡片。
- [x] 优化 sidebar 和顶部状态条。
- [x] 更新 README 和 SDD 文档。
- [x] 更新测试用例和验收清单。
- [x] 更新 AI prompts 记录。
