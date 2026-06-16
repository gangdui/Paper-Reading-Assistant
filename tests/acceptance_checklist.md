# 手动验收清单

## 启动与环境

- [ ] 已创建并激活 `.venv`。
- [ ] 已执行 `pip install -r requirements.txt`。
- [ ] 已创建 `.env`。
- [ ] `.env` 中已配置 `DEEPSEEK_API_KEY`。
- [ ] 已确认 `SEMANTIC_SCHOLAR_API_KEY` 是可选项。
- [ ] 已通过 `streamlit run app.py` 启动项目。
- [ ] README 中的运行说明可以照着执行。

## 论文发现流程

- [ ] 页面默认优先展示“发现论文”。
- [ ] 输入研究方向后，点击按钮才触发检索。
- [ ] 页面展示顶部状态条。
- [ ] 页面展示助手总结。
- [ ] 页面展示 Top 3 优先关注论文。
- [ ] Top 3 小卡片只展示标题、推荐原因和阅读定位。
- [ ] 完整论文列表展示标题、年份、数据源、venue、PDF 状态、推荐优先级、推荐理由和阅读建议。
- [ ] 英文 abstract 默认折叠。
- [ ] 论文链接和 PDF 链接可点击。

## 文献核查流程

- [ ] 每篇论文显示来源核查。
- [ ] 每篇论文显示元数据完整度。
- [ ] 每篇论文显示质量判断。
- [ ] 页面说明质量判断不等于论文结论可靠。
- [ ] arXiv 结果显示为预印本。
- [ ] arXiv 结果不被描述为顶会或顶刊。
- [ ] mock fallback 明确标注为 demo 示例数据。

## PDF 阅读笔记流程

- [ ] 可以切换到“阅读 PDF”页面。
- [ ] 可以上传单篇 PDF。
- [ ] 页面显示 PDF 读取进度。
- [ ] 页面显示文本清洗完成提示。
- [ ] 可以生成结构化文献笔记。
- [ ] 笔记包含 9 个模块。
- [ ] 原始文本和清洗文本位于折叠调试区。
- [ ] 可以下载 Markdown 文献笔记。

## fallback 流程

- [ ] 未配置 Semantic Scholar Key 时，页面仍可尝试论文检索。
- [ ] Semantic Scholar 受限时，可以切换 arXiv。
- [ ] arXiv 结果提示“正式发表信息需后续核查”。
- [ ] 真实 API 都失败时，可以展示 mock fallback。
- [ ] DeepSeek 失败时，页面显示友好错误提示。
- [ ] PDF 阅读页可使用示例输出 fallback。
- [ ] fallback 不会导致页面崩溃。

## 截图材料

- [ ] 截图：项目首页和顶部状态条。
- [ ] 截图：论文发现输入和助手总结。
- [ ] 截图：Top 3 优先关注论文。
- [ ] 截图：完整论文卡片和核查摘要。
- [ ] 截图：arXiv 预印本说明。
- [ ] 截图：PDF 上传与解析进度。
- [ ] 截图：结构化文献笔记结果。
- [ ] 截图：Markdown 下载。

## 课程报告检查

- [ ] README 已更新为最新项目名称。
- [ ] docs/idea.md 已说明项目从 PDF 摘要助手升级为科研论文发现与阅读助手。
- [ ] docs/spec.md 已覆盖 F1-F9。
- [ ] docs/plan.md 已说明 Semantic Scholar、arXiv 和 mock fallback。
- [ ] docs/tasks.md 已区分已完成任务和后续任务。
- [ ] prompts/ai_prompts.md 已补充新增功能开发提示词。
