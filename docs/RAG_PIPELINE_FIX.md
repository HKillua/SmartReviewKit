# 阶段 N：RAG 全链路修复 — 复习文档

> 本文档总结 Phase N 中对 RAG 系统进行的全链路修复，方便面试复习和技术回顾。

---

## 一、修复背景

Phase M 实现了 QueryEnhancer、MMR、SimHash 去重、ContextualEnricher、CachedEmbedding 等模块，但它们并未接入主流程。Phase N 的目标是"接线 + 补齐"——将所有已实现组件接入生产流程，并补充缺失的 Grounding、Query Routing、反馈系统等环节。

---

## 二、核心修复项与面试话术

### 2.1 Query Enhancement 三件套 (N1)

**问题**：用户原始查询往往口语化、含糊，直接用于检索效果差。

**方案**：在 `KnowledgeQueryTool.execute()` 中串联三个策略：
1. **Query Rewriting** — LLM 改写查询，补充关键词
2. **HyDE** — 生成假设文档并用其 embedding 替代原始 query embedding
3. **Multi-Query** — 将复杂问题拆为 2-4 个子查询，分别检索后合并

**面试要点**：
- 三者是互补关系：Rewrite 优化语义表达，HyDE 弥合 query-document 的分布差异，Multi-Query 覆盖多角度
- 使用 async 实现，集成在 Tool 层而非 Search 层，保持 HybridSearch 的同步简洁性
- 每个策略都有 try/except 降级——任一失败不影响整体

### 2.2 MMR 多样性控制 (N2)

**问题**：Dense + Sparse 融合后可能返回语义高度重复的结果。

**方案**：在 HybridSearch 的 reranker 之后加入 MMR reranking，平衡 relevance 和 diversity。

**公式**：MMR = λ · sim(d_i, q) - (1-λ) · max sim(d_i, d_j ∈ S)

**面试要点**：
- λ=0.7 偏重相关性，λ=0.3 偏重多样性
- 需要额外的 embedding 计算（query + candidates），因此注入了 embedding_client
- 通过 `mmr_enabled` 配置控制

### 2.3 Ingestion Pipeline 扩展 (N3 + N4)

**修复内容**：
- **Stage 4b2 — Contextual Enrichment**：在每个 chunk 前加上文档级别的上下文前缀（来源文件、章节标题），提升模糊片段的检索命中率
- **Stage 4d — SimHash Dedup**：入库前去除近重复 chunk（Hamming distance ≤ 3），减少索引冗余

**面试要点**：
- Contextual Retrieval 参考 Anthropic 的论文，核心思想是"给每个 chunk 加 context prefix"
- SimHash 是局部敏感哈希（LSH）的一种，64-bit fingerprint，O(n²) 比较但 n 通常 < 1000
- 两者都通过 settings 控制，支持关闭

### 2.4 CachedEmbedding (N5)

**问题**：同一文本多次 embedding（如 re-ingestion、重复查询）浪费 API 调用。

**方案**：LRU cache 包装 BaseEmbedding，SHA256 hash 作 key，命中缓存时跳过 API 调用。

**面试要点**：
- 缓存 4096 个向量（可配置），淘汰策略为 LRU
- 追踪 hit/miss 率和 API 调用次数，暴露 `cache_stats` 属性
- 在 app.py 中统一包装，所有下游（检索 + ingestion）共享

### 2.5 Prompt Grounding (N6)

**问题**：LLM 可能在知识库无相关内容时编造答案（幻觉问题）。

**方案**：在 system prompt 中加入明确的 grounding 指令：
- 回答必须基于检索到的内容
- 无法回答时明确告知
- 部分命中时说明哪些有据可查

**面试要点**：
- 这是解决 RAG 幻觉的"最后一道防线"
- 与 min_score 过滤配合：低分结果被过滤后，LLM 看到的 context 更干净
- 引用 chunk 编号实现溯源

### 2.6 Min Score 过滤 (N7)

**问题**：低相关度结果混入 context 会增加噪声，导致答案质量下降。

**方案**：在 HybridSearch 中加入 `min_score` 阈值（默认 0.15），低于此分数的结果直接过滤。

### 2.7 Chunk ID 引用 (N8)

**方案**：在 KnowledgeQueryTool 输出中追加 `chunk: {id[:12]}`，方便 LLM 在回答中引用来源 chunk。

### 2.8 Query Routing (N9)

**问题**：LLM 不清楚何时该用哪个工具。

**方案**：在 system prompt 中加入 few-shot 示例，按用户意图引导 LLM 选择正确的工具。

### 2.9 Post-Retrieval Dedup (N10)

**问题**：多路检索（Dense + Sparse）+ Multi-Query 可能产生内容高度相似但 chunk_id 不同的结果。

**方案**：在 HybridSearch 的 min_score 过滤之后，用 SimHash 对结果文本做二次去重。

### 2.10 反馈系统 (N11)

**方案**：
- `FeedbackStore`：SQLite 持久化用户反馈（up/down, comment）
- `POST /api/feedback`：提交反馈
- `GET /api/feedback/stats`：聚合统计

**面试要点**：
- 这是 RAG 系统"闭环优化"的基础——有了反馈数据才能做 RLHF、主动学习、golden set 扩充
- 目前是离线分析，后续可接入在线 A/B 测试

### 2.11 成本追踪 (N12)

**方案**：
- CachedEmbedding 追踪 `api_calls` 和 `texts_embedded`
- CrossEncoderReranker 追踪 `rerank_calls` 和 `rerank_pairs_scored`

**面试要点**：
- 生产 RAG 系统必须关注成本——embedding 和 reranking 是两大调用量来源
- 通过 `cache_stats` / `rerank_stats` 属性可随时查询

---

## 三、完整检索流程（修复后）

```
用户 query
  │
  ├─ [N1] Query Rewrite → 改写 query
  ├─ [N1] HyDE → 假设文档 embedding
  ├─ [N1] Multi-Query → 子查询列表
  │
  ▼
HybridSearch.search(query, query_vector)
  │
  ├─ Dense Retrieval (query_vector 直通)
  ├─ Sparse Retrieval (BM25)
  │
  ├─ RRF 加权融合
  ├─ [M] Reranker (Cross-Encoder)
  ├─ [N2] MMR 多样性控制
  ├─ [N7] Min Score 过滤
  ├─ [N10] Post-Retrieval SimHash 去重
  │
  ▼
Top-K 结果 → LLM 生成回答
  │
  ├─ [N6] Grounding 检查
  ├─ [N8] Chunk 引用
  ├─ [N11] 用户反馈收集
```

## 四、完整入库流程（修复后）

```
文件上传
  │
  ├─ Stage 1: SHA256 完整性检查
  ├─ Stage 2: 文档加载 (PDF/PPTX/DOCX)
  ├─ Stage 3: 分块 (structure-aware / recursive / semantic)
  ├─ Stage 4a: Chunk 精炼
  ├─ Stage 4b: Metadata 标注
  ├─ [N4] Stage 4b2: Contextual Enrichment
  ├─ Stage 4c: 图片描述
  ├─ [N3] Stage 4d: SimHash 去重
  ├─ Stage 5: Embedding (Dense + Sparse)
  ├─ Stage 6: 存储 (ChromaDB + BM25)
```

---

## 五、配置速查

```yaml
retrieval:
  rerank_enabled: true      # 精排
  mmr_enabled: true          # 多样性
  query_rewrite_enabled: true # 查询改写
  hyde_enabled: false         # 假设文档嵌入（需额外 LLM 调用）
  multi_query_enabled: false  # 多查询分解（需额外 LLM 调用）
  min_score: 0.15            # 最低分数阈值
  post_dedup_enabled: true   # 检索后去重
  dedup_enabled: true         # 入库去重
  embedding_cache_size: 4096  # 嵌入缓存大小
```
