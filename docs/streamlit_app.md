# Phase 11：Streamlit 聊天前端

> 当前后端已重构为强制 LangChain LCEL，前端仍通过相同的 `build_pipeline()` / `answer()` 接口调用，控件、授权及引用显示保持原行为。缓存身份增加框架版本，详见 [LangChain RAG 实现](langchain_rag.md)。以下验收数字是原 Phase 11 记录。

> 本文保留该阶段交付时的历史状态和实测口径；当前最终状态与完整流程见 [README](../README.md)、[复现指南](reproduction.md) 和 [最终实验汇总](results_summary.md)。后续阶段已实现的内容不回写历史实验数据。

本阶段仅增加 Streamlit 展示层，直接调用 Phase 9 的 `build_pipeline(...)` 和 `pipeline.answer(question)`。不修改核心 RAG Pipeline、Prompt、检索、重排、预算或引用规则，不添加 Gradio、React/Next.js、用户登录、Agent、MCP、插件或新的持久化系统。

## 启动

```powershell
uv sync --locked
uv run streamlit run app/streamlit_app.py
```

在项目根目录启动，打开 `http://127.0.0.1:8501`。`.streamlit/config.toml` 使用浅色、绿色强调色、简洁工具栏，关闭使用统计，默认仅监听本机回环地址。没有把服务暴露到公网，也没有为公网访问设计身份认证。

程序读取进程环境，不自动加载 `.env`，不会创建含密钥的文件。DeepSeek 使用既有 `DEEPSEEK_API_KEY`，本地 Qwen 使用既有 `OLLAMA_HOST`。环境变量应在启动 Streamlit 的进程中配置；若变量改变，释放管线资源或重启服务后再调用。不要通过命令参数、Git 或聊天消息传递 API Key。

首次页面打开不加载模型、不调用 Ollama 或 DeepSeek。真正提交问题时才初始化现有管线。需要已经准备好的 Phase 2 Chunk、Phase 4 固定向量、Phase 9 索引和 tokenizer，以及固定 BGE-M3/BGE-Reranker 权重；完整准备方法见 [RAG 管线文档](rag_pipeline.md)。UI 不偷偷下载其他模型、替换文件或更改实验配置。

本地后端需要先启动配置对应的 Ollama，并具备 Phase 6 固定的 Qwen 模型、digest 和服务版本。首次装载加 CPU 重排可能超过一分钟；loading 文案显示这一限制，而不是承诺即时响应。

## 使用与数据边界

- 左侧提供临时会话列表、新建会话、DeepSeek/本地 Qwen 选择、管线资源释放与使用说明；默认选择本地 Qwen。
- 中间使用 `st.chat_message` 区分用户/助手，正常渲染 Markdown；底部使用 `st.chat_input`，最大字符数读取既有 RAG 配置。
- 每次助手回答下面展开真实 Citation，显示标题、厂商、PDF 物理页码、章节、URL 和 Chunk ID。章节为空时显示“未标注”，不虚构章节。点击“查看原报告”打开 Metadata 中的官方来源，不使用 LLM 自己生成的 URL 替代 Citation。
- 模型标识、生成耗时、输入/输出 Token 和不含首次装载的管线耗时附在回答下方；不将这些单条耗时当新 Benchmark。
- 消息仅存于 Streamlit 服务的会话内存，按浏览器会话隔离；不会自动写入磁盘、Git、浏览器持久存储或实验结果。刷新页面、断线、关闭浏览器或服务重启可能丢失消息，不保证持久恢复。
- **聊天式界面不等于多轮记忆。** 历史只用于展示，后端仍只接收本轮原问题，每轮独立检索。请写完整问题，不依赖“它”“继续解释上一个”等省略指代。没有将聊天历史拼接进 Prompt 或偷偷改写 Query。

### DeepSeek 显式授权

选择 DeepSeek 后，必须手动勾选“本会话允许使用 DeepSeek 云端 API”。未勾选时输入禁用，并说明不会调用云端。模型工厂仍收到真实 `allow_cloud` 参数；每次生成之前还有展示层的独立授权检查，缓存命中也不能绕过。

授权意味着**当前问题与已核验的公开报告片段会发送给 DeepSeek，生成可能计费**，公开 Corpus 并不能保证用户问题不敏感。不要输入内部情报、个人隐私或真实密钥。

新建/切换会话、切换模型和手动释放管线资源会撤销授权；切换到本地不自动调用云端。授权持续于当前会话，后续主动提交的问题仍使用该授权；不是每条消息单独弹窗。加载失败、生成失败、预算/引用校验失败均不自动重试或换模型。

生成期间禁用会话切换、模型选择、授权与输入，显示 loading 和经过时间。消息由“排队 → 执行 → 已记录”状态推进；记录响应后才重绘。若执行中被页面重跑打断，UI 显示结果/计费未确认，并拒绝自动重发。仍不保证网络请求 exactly-once；手动再次提交可能产生新调用，应自行检查提供方账单。

### 资源与异常处理

用 `st.cache_resource(scope="session", max_entries=1, on_release=...)` 缓存当前后端管线，缓存键包含配置身份、后端、授权以及独立会话标识。不用全局共享 mutable Pipeline，避免跨用户共享 HTTP client、云端 fingerprint 或回答状态；普通 UI 重跑不反复装载模型。切换模型先清缓存/关闭 client，再按需装载，避免同时缓存两个大管线。

新建同模型聊天可复用模型资源，但不会发送其他会话历史。资源释放关闭前端持有的 HTTP client 并清除管线缓存；不额外停止 Ollama 进程或改变其原 `keep_alive` 策略。缓存回收与进程关闭不保证在任意崩溃情形下即时触发。该 UI 面向本机研究使用，多浏览器会话各自装载模型可能占用较多内存，不宣称已完成多用户生产部署。

后端拒绝的未验证生成原文不会被当作答案显示；只显示已有安全异常摘要。未知异常不把 traceback、请求对象或内部详情输出到页面。UI 仅保存验证后的回答与引用，不保留完整报告 Context 或请求头。既有 `[S0183]` 等命名冲突仍由 Phase 9 拒绝，本阶段不偷偷放宽 Citation 规则。

## 验证

```powershell
uv run pytest tests/test_streamlit_app.py -q
uv run pytest -q
uv run python scripts/validate_rag_results.py
uv run python scripts/validate_ablation_results.py
uv build
```

前端使用 Streamlit 官方 AppTest 测试真实控件、状态与呈现，只有模型工厂使用假后端；这些自动化测试不读取真实密钥、不加载模型、不调用付费 API。覆盖默认懒加载、Markdown、全部 Citation 字段、会话切换、云端授权/撤销（含已有缓存）、会话隔离、重复重跑、非法问题、未验证答案不展示、异常安全和中断不重试。不会把测试用的虚构报告当研究证据或真实运行结果。

本阶段实际验收结果：

- 前端自动化测试 **12 项通过**，完整测试套件 **98 项通过**；Phase 9/10 结果离线校验、Phase 8 技术栈锁定校验、Phase 6 LLM 结果校验及 `uv build` 均通过。服务健康检查返回 HTTP 200。
- 浏览器实际提交 Benchmark 的 APT44 背景问题，通过原管线调用本地 `qwen3.5:4b`，观察到 loading、控件禁用、回答完成后恢复输入和 Markdown 渲染。
- 回答映射到 `APT44: Unearthing Sandworm` 的两条真实引用：`[S1]` 为物理第 4 页、`Overview of APT44`、`mandiant-apt44-2024-p0004-c000`；`[S4]` 为物理第 6 页、`A Global Targeting Mandate`、`mandiant-apt44-2024-p0006-c000`。页面同时展示厂商和官方 URL。
- 该单次运行生成约 7.0 秒，管线约 69.7 秒（不含首次装载），输入/输出为 3057/127 tokens。这是 UI 冒烟验证，不是新增 Benchmark，也不能据此宣称准确率或性能提升。
- DeepSeek 授权门槛通过自动化假后端验证；浏览器切换到 DeepSeek 时，授权默认未勾选、输入被禁用，随后恢复本地 Qwen。本轮没有发起真实付费云端生成。实际浏览器截图保存在本机被忽略的 `data/processed/phase11-chat-local.jpg`，不作为可复现研究结果提交到 Git。

安装锁定 Streamlit `1.65.0`；仅新增 Streamlit 及必要依赖，旧模型/检索依赖版本不变。API 对照当前官方文档及安装包签名，不使用旧教程中的弃用缓存 API：

- [Chat Message](https://docs.streamlit.io/develop/api-reference/chat/st.chat_message)
- [Chat Input](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input)
- [Resource Cache](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
- [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)

## 阶段边界

本阶段只交付聊天前端与相关运行/测试说明，未进入 Phase 12。Phase 10 的独立人工评审仍未完成，引用映射通过和 UI 能正常运行都不等于研究已经证明答案正确或无幻觉。
