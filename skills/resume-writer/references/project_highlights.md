# 项目亮点库（以当前代码为准）

> 用于挑选简历 bullet。每个亮点都尽量绑定到真实模块。  
> 如果要写精确数字，先在当前仓库重新核实，不要直接套历史文档。

## 1. 课程学习 Agent 产品主链路

- 核心事实：
  这个仓库当前的主产品形态是“课程学习 Agent”，不是单纯 MCP Server
- 关键文件：
  `DEV_SPEC.md`、`run_server.py`、`src/server/app.py`、`src/agent/agent.py`
- 可写成：
  设计并实现面向课程学习场景的智能 Agent 平台，围绕问答、测验、复习和知识管理构建完整交互闭环
- 适合岗位：
  LLM 应用、Agent、后端平台

## 2. Agent 编排与流式交互

- 核心事实：
  `Agent` 负责会话管理、prompt 组装、tool loop、memory/skill 注入和流式输出
- 关键文件：
  `src/agent/agent.py`、`src/agent/conversation.py`、`src/server/chat_handler.py`
- 可写成：
  自研基于 ReAct 的对话编排链路，将工具调用、上下文增强与 SSE 流式响应统一到单次对话执行模型中
- 适合岗位：
  Agent、后端、平台架构

## 3. 自适应知识问答工具链

- 核心事实：
  `knowledge_query` 在检索前后串起 query routing、semantic cache、query rewrite、HyDE/multi-query、parent chunk 与冲突检测
- 关键文件：
  `src/agent/tools/knowledge_query.py`、`src/core/query_engine/query_router.py`、`src/core/cache/semantic_cache.py`
- 可写成：
  将查询路由、缓存、查询增强与混合检索编排为统一知识问答工具，减少直接把用户原始问题裸投到检索层的质量波动
- 适合岗位：
  RAG、搜索、LLM 应用

## 4. 混合检索主引擎

- 核心事实：
  `HybridSearch` 负责 QueryProcessor、Dense/Sparse 并行召回、RRF 融合、rerank、MMR、低分过滤和 post-dedup
- 关键文件：
  `src/core/query_engine/hybrid_search.py`、`dense_retriever.py`、`sparse_retriever.py`、`fusion.py`
- 可写成：
  设计并实现 dense + sparse 的混合检索引擎，通过 RRF 融合与重排/多样性控制提升召回和排序质量
- 适合岗位：
  RAG、检索、搜索工程

## 5. 多阶段文档摄取流水线

- 核心事实：
  `IngestionPipeline` 串联完整性检查、加载、切块、transform、编码、写库，支持 PDF/PPTX/DOCX 和题库文档
- 关键文件：
  `src/ingestion/pipeline.py`、`src/libs/loader/*.py`、`src/ingestion/transform/*.py`
- 可写成：
  构建面向课程资料的多阶段入库流水线，覆盖结构化解析、问题抽取、元数据增强、图片描述和稠密/稀疏双路索引写入
- 适合岗位：
  RAG、数据基础设施、平台

## 6. 个性化记忆与复习闭环

- 核心事实：
  memory 系统包含学生画像、错题、知识掌握度、session memory、上下文压缩和复习调度
- 关键文件：
  `src/agent/memory/enhancer.py`、`knowledge_map.py`、`error_memory.py`、`student_profile.py`
- 可写成：
  将长期画像、错题记录与知识掌握度建模接入 Agent，对问答和练习过程进行个性化增强与主动复习推荐
- 适合岗位：
  Agent、教育 AI、LLM 产品

## 7. 技能系统与课程学习工具

- 核心事实：
  Agent 已接入 skill registry/workflow，以及知识问答、资料导入、总结、出题、判题等工具
- 关键文件：
  `src/agent/skills/registry.py`、`src/agent/skills/workflow.py`、`src/agent/tools/*.py`
- 可写成：
  将课程学习场景拆解为可组合工具与技能流，支持知识问答、资料导入、习题生成、自动评测与学习路径切换
- 适合岗位：
  Agent、产品工程、工作流编排

## 8. 可插拔配置与多后端支持

- 核心事实：
  LLM、Embedding、VectorStore、Reranker 等通过 Base + Factory + settings 配置切换
- 关键文件：
  `src/core/settings.py`、`config/settings.yaml`、`src/libs/*/*factory*.py`
- 可写成：
  通过抽象接口与工厂模式实现多 Provider 可插拔架构，使模型、向量库和重排策略可以按配置替换
- 适合岗位：
  后端、平台、架构

## 9. Web Agent 与 MCP 双接口

- 核心事实：
  仓库同时提供 FastAPI Web Agent 和 MCP Server；但当前 Web 主路径更完整，`main.py` 仍是早期占位入口
- 关键文件：
  `src/server/app.py`、`src/mcp_server/server.py`、`src/mcp_server/protocol_handler.py`、`main.py`
- 可写成：
  在统一知识基础设施之上同时封装 Web 对话接口与 MCP 协议接口，兼顾用户端交互和外部 Agent 集成
- 适合岗位：
  平台、Agent、工具集成

## 10. 质量保障与可观测性

- 核心事实：
  项目包含 trace、dashboard、evaluation runner、unit/integration 测试体系
- 关键文件：
  `src/core/trace/*.py`、`src/observability/dashboard/*`、`src/observability/evaluation/*`、`tests/`
- 可写成：
  为检索与入库链路补齐可观测性和评估闭环，支持追踪分析、指标验证和回归测试
- 适合岗位：
  后端、RAG、平台工程

## 写简历时的证据规则

- 想写“当前默认课程/默认集合/启动自动入库”，先检查 `config/settings.yaml`
- 想写“当前仓库已有多少文档/多少 chunk/多少图片”，先现场核实 `data/` 与向量库
- 想写“测试规模/通过率”，先重新跑测试或至少重新统计 `tests/`
- 想写“MCP 是主入口”，必须先说明当前真实情况：Web 主链路更完整，包脚本仍有历史残留
