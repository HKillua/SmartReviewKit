# RAG 深度优化 II — 面试复习文档

> 本文档用于面试复习，覆盖阶段 M 全部 16 项优化的设计动机、技术细节和面试话术。

---

## 目录

1. [整体架构升级](#1-整体架构升级)
2. [Group 1: 基础修复](#2-group-1-基础修复)
3. [Group 2: 查询增强](#3-group-2-查询增强)
4. [Group 3: 检索质量](#4-group-3-检索质量)
5. [Group 4: 工程优化](#5-group-4-工程优化)
6. [Group 5: 评估体系](#6-group-5-评估体系)
7. [面试高频问题与话术](#7-面试高频问题与话术)
8. [新增/修改文件清单](#8-新增修改文件清单)

---

## 1. 整体架构升级

阶段 M 的核心目标是将 RAG 系统从"能用"升级为"能经受深度面试拷打"。优化覆盖检索管线全链路：

```
用户查询
  ↓
[Query Enhancement]  ← M5 Rewriting / M6 HyDE / M7 Multi-Query
  ↓
[QueryProcessor]     ← M1 jieba 中文分词
  ↓
┌─────────────┬────────────────┐
│ Dense       │ Sparse (BM25)  │  ← M4 BM25 mtime 缓存
│ Retriever   │ Retriever      │
└─────┬───────┴───────┬────────┘
      │               │
      ↓               ↓
[RRF Fusion]  ← M12 可配权重 (dense_weight / sparse_weight)
      ↓
[Reranker]    ← M9 CrossEncoder 集成 + M3 timeout 修复
      ↓
[MMR]         ← M8 多样性控制
      ↓
返回 Top-K
```

入库侧：
```
文档 → Loader → Chunking
  ↓
[Contextual Enricher]  ← M10 上下文注入
  ↓
[SimHash Dedup]        ← M13 语义去重
  ↓
[Embedding]            ← M11 LRU 缓存 + M2 batch fix
  ↓
Vector Store + BM25 Index
```

---

## 2. Group 1: 基础修复

### M1: jieba 中文分词

**问题**: 原 `SparseEncoder._tokenize` 使用 `re.findall(r'\b[\w-]+\b')` 只能处理英文，中文文本被整体当作一个 token。

**方案**: 引入 jieba 分词库，检测 CJK 字符范围后调用 `jieba.cut()`。添加 41 个高频中文停用词过滤。

**面试话术**:
> "BM25 的效果高度依赖分词质量。对于中文课件，我们不能用英文的空格分词，所以集成了 jieba 进行精准分词，并配合中文停用词表过滤无意义词汇。QueryProcessor 也做了同步升级，保证查询端和索引端的分词一致性。"

### M2: SemanticSplitter batch embed 修复

**问题**: `_get_default_embed_fn` 中 `[embedder.embed(t) for t in texts]` 会对每个文本单独调用，而 `embed()` 期望输入 `List[str]` 返回 `List[List[float]]`，导致返回的是嵌套的 `List[List[List[float]]]`。

**修复**: 改为 `embedder.embed(texts)` 一次批量调用。

### M3: DocxLoader 死代码 + CrossEncoder timeout

**DocxLoader**: `if False` 永远不执行的死代码分支被删除。

**CrossEncoder**: 声明了 `timeout` 参数但从未真正使用。使用 `ThreadPoolExecutor + future.result(timeout)` 实现真正的超时控制，超时时抛出 `CrossEncoderRerankError` 供上游 fallback。

### M4: BM25 索引 mtime 缓存

**问题**: 每次查询都从磁盘重新加载 BM25 JSON 索引文件。

**方案**: 使用 `Path.stat().st_mtime` 对比文件修改时间，仅在变更时重新加载，否则复用内存中的索引。

---

## 3. Group 2: 查询增强

### M5: Query Rewriting (LLM 查询改写)

**原理**: 用户的口语化提问（如"TCP 那个三次的东西怎么搞的"）不适合直接检索。LLM 将其改写为规范的检索查询。

**实现**: `QueryEnhancer.rewrite()` 调用 LLM，使用可定制的 prompt 模板 (`config/prompts/query_rewrite.txt`)。

**面试话术**:
> "我们实现了 Query Rewriting 来桥接用户口语和知识库索引之间的语义鸿沟。prompt 是可配置的，可以根据领域调整。这是 RAG 系统中常见的查询理解优化。"

### M6: HyDE (Hypothetical Document Embedding)

**原理**: 让 LLM 生成一个假设性的答案文档，用该文档的 embedding 替代原始查询的 embedding 进行 dense 检索。因为假设性文档在语义空间中更接近真实文档。

**论文**: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)

**面试话术**:
> "HyDE 解决的是 query 和 document 之间的语义不对称问题。用户的查询通常很短，而文档段落较长。让 LLM 先生成一段假设性回答，用它的 embedding 做检索，可以显著提升召回率。这是 2022 年的 SOTA 技术。"

### M7: Multi-Query Retrieval (查询分解)

**原理**: 复杂问题（如"OSI 和 TCP/IP 模型的区别以及各层的协议"）包含多个子话题。分解后分别检索，合并去重，覆盖更全面。

**实现**: LLM 将问题分解为 2-4 个子查询，每个子查询独立走 HybridSearch，结果合并。

---

## 4. Group 3: 检索质量

### M8: MMR (Maximal Marginal Relevance)

**问题**: 纯相关性排序可能返回高度相似的重复内容片段。

**算法**: `MMR = λ · sim(d, q) - (1-λ) · max_j sim(d, d_j)`

- λ = 0.7: 偏向相关性
- λ = 0.3: 偏向多样性

**实现**: `src/core/query_engine/mmr.py`，使用 numpy 向量化计算余弦相似度矩阵。

**面试话术**:
> "MMR 在保证相关性的同时引入多样性。比如用户问 TCP 拥塞控制，我们不会返回 5 个都在说慢开始的片段，而是会覆盖慢开始、拥塞避免、快重传、快恢复等不同方面。"

### M9: Reranker 集成

**设计**: HybridSearch 的构造函数新增 `reranker: BaseReranker` 参数。在 RRF 融合之后、返回之前，若 `config.rerank_enabled=True`，则调用 `reranker.rerank()` 进行精排。

**容错**: reranker 抛出异常时 fallback 到 RRF 融合的原始排序，打 warning 日志。

**面试话术**:
> "我们采用了 retrieve-then-rerank 的两阶段架构。第一阶段用 BM25 + dense embedding 做粗召回，第二阶段用 Cross-Encoder 做精排。Cross-Encoder 直接对 (query, passage) 对打分，精度更高但计算量大，所以只对 top-20 候选精排。"

### M10: Contextual Retrieval

**原理**: 参考 Anthropic 的 Contextual Retrieval 技术。在入库时为每个 chunk 添加文档级上下文前缀，解决片段脱离上下文后语义模糊的问题。

**两种模式**:
- `rule`: 拼接文档标题 + 章节标题 + 源文件路径（零 LLM 成本）
- `llm`: LLM 生成 1-2 句上下文描述（更精准但有成本）

**面试话术**:
> "这是 Anthropic 2024 年提出的技术。一个 chunk 如果脱离了原文上下文，可能丢失关键信息。比如 '该协议使用 3 次握手' 这个片段，不知道 '该协议' 是什么。我们在入库时注入上下文前缀 '[上下文：计算机网络 > 传输层 > TCP]'，让 chunk 自包含，大幅提升检索准确率。"

---

## 5. Group 4: 工程优化

### M11: Embedding LRU 缓存

**设计**: `CachedEmbedding` 是一个透明代理，包装任何 `BaseEmbedding` 实现。使用 SHA-256 哈希文本作为缓存 key，`OrderedDict` 实现 LRU 淘汰。

**效果**: 对于重复文本（如重新入库或相同查询），直接命中缓存，避免 API 调用。默认容量 4096 条。

**面试话术**:
> "Embedding API 调用有延迟和成本。我们在 embedding 层加了 LRU 缓存，用 SHA-256 做 key。重复查询或重新入库时直接命中，节省了大量 API 调用。这是一个典型的空间换时间优化。"

### M12: RRF 可配权重

**优化**: 原来的 RRF fusion 对 dense 和 sparse 等权。现在通过 `dense_weight` 和 `sparse_weight` 配置项，使用 `fuse_with_weights()` 进行加权融合。

**场景**: 对于专业术语较多的学科，可以适当提高 sparse_weight；对于语义理解要求高的场景，提高 dense_weight。

### M13: SimHash 语义去重

**算法**: 64-bit SimHash + Hamming 距离。对每个 chunk 计算指纹，两个 chunk 的 Hamming 距离 ≤ threshold (默认 3) 则认为近似重复，保留先出现的。

**集成点**: Ingestion Pipeline 在 chunking 之后、embedding 之前执行去重。

**面试话术**:
> "课件中经常有重复内容，比如每章的总结可能重复引用前面的定义。SimHash 是一种局部敏感哈希，能高效检测近似重复文本。我们在入库时去重，减少冗余存储和检索噪音。"

---

## 6. Group 5: 评估体系

### M14: 检索评估指标

**新文件**: `src/libs/evaluator/retrieval_metrics.py`

实现了 5 个标准 IR 指标：

| 指标 | 说明 |
|------|------|
| Hit Rate | 至少命中一个相关文档则为 1.0 |
| MRR | 第一个相关文档的排名倒数 |
| NDCG@k | 归一化折扣累积增益 |
| Precision@k | top-k 中相关文档的比例 |
| Recall@k | top-k 覆盖的相关文档比例 |

**面试话术**:
> "我们建立了完整的 IR 评估体系，包括 Hit Rate、MRR、NDCG 等标准指标。Hit Rate 衡量的是是否能找到，MRR 衡量的是找到的位置，NDCG 综合考虑排序质量。这些指标可以量化地评估每次优化的效果。"

### M15: Golden Test Set + 自动化评估脚本

**测试集**: `tests/fixtures/golden_test_set.json`，包含 10 条计算机网络知识点的标准查询。

**评估脚本**: `scripts/run_eval.py`
```bash
python scripts/run_eval.py --top-k 10 --output data/eval_report.json
```

输出 JSON 报告，包含每条查询的详细指标和聚合平均值。

---

## 7. 面试高频问题与话术

### Q: 你的 RAG 系统检索质量怎么保证的？

> "我们采用了 **Hybrid Search** 架构，结合 dense embedding 的语义理解和 BM25 的精确关键词匹配，通过 **加权 RRF** 融合。在查询端，有 **Query Rewriting** 桥接口语和索引语义，**HyDE** 处理 query-document 语义不对称，**Multi-Query** 处理复杂问题分解。在排序端，有 **Cross-Encoder Reranker** 精排和 **MMR** 多样性控制。在入库端，有 **Contextual Retrieval** 和 **SimHash 去重**。全链路每个环节都有对应优化。"

### Q: 你怎么评估 RAG 效果？

> "我们建立了包含 Hit Rate、MRR、NDCG@k、Precision@k、Recall@k 的完整评估体系。有一个 Golden Test Set 作为基准，配合自动化评估脚本可以快速跑出各项指标。每次优化后都会跑一遍评估，用数据说话。"

### Q: 中文处理有什么特殊考虑？

> "BM25 的效果高度依赖分词质量。对中文文本，我们使用 jieba 进行精准分词，配合自定义停用词表。查询端和索引端使用相同的分词器，保证一致性。对于专业术语，jieba 支持自定义词典扩展。"

### Q: Embedding 有什么优化？

> "我们在 embedding 层加了 **LRU 缓存**（SHA-256 作 key），避免重复文本的冗余 API 调用。同时修复了 SemanticSplitter 中逐条 embed 的 N+1 问题，改为批量调用。缓存大小可配置，有命中率监控。"

### Q: RRF 融合的权重怎么定？

> "RRF 的 k 值使用论文推荐的 60。dense 和 sparse 的权重可以通过 `dense_weight` 和 `sparse_weight` 配置，默认各 1.0。如果是专业术语较多的学科可以适当提高 sparse 权重，因为 BM25 对精确术语匹配更有优势。我们还提供了评估框架来 A/B 测试不同权重组合。"

### Q: 如果 Reranker 挂了怎么办？

> "我们采用 **graceful degradation** 设计。Reranker 有超时控制（默认 10s），超时或异常时 fallback 到 RRF 融合的原始排序，打 warning 日志。同样，如果 dense retriever 或 sparse retriever 挂了一个，会自动降级到单路检索。"

---

## 8. 新增/修改文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/core/query_engine/query_enhancer.py` | QueryRewriting + HyDE + Multi-Query |
| `src/core/query_engine/mmr.py` | MMR 多样性重排序 |
| `src/ingestion/transform/contextual_enricher.py` | Contextual Retrieval chunk 上下文注入 |
| `src/ingestion/transform/chunk_dedup.py` | SimHash 语义去重 |
| `src/libs/embedding/cached_embedding.py` | Embedding LRU 缓存 |
| `src/libs/evaluator/retrieval_metrics.py` | IR 评估指标 (Hit Rate / MRR / NDCG / P@k / R@k) |
| `tests/fixtures/golden_test_set.json` | 10 条计算机网络 Golden Test Set |
| `scripts/run_eval.py` | 自动化评估脚本 |
| `config/prompts/query_rewrite.txt` | Query Rewriting prompt 模板 |
| `config/prompts/hyde.txt` | HyDE prompt 模板 |
| `config/prompts/multi_query.txt` | Multi-Query prompt 模板 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/ingestion/embedding/sparse_encoder.py` | 接入 jieba 中文分词 + 停用词 |
| `src/core/query_engine/query_processor.py` | 接入 jieba 中文分词 |
| `src/libs/splitter/semantic_splitter.py` | 修复 batch embed 调用方式 |
| `src/libs/loader/docx_loader.py` | 删除 `if False` 死代码 |
| `src/libs/reranker/cross_encoder_reranker.py` | 实现 timeout 超时控制 |
| `src/core/query_engine/sparse_retriever.py` | BM25 索引 mtime 缓存 |
| `src/core/query_engine/hybrid_search.py` | 集成 Reranker + RRF 可配权重 + 新配置字段 |
| `config/settings.yaml` | 新增检索相关配置项 |
| `pyproject.toml` | 新增 jieba 依赖 |
| `DEV_SPEC.md` | 阶段 M 详细 spec |
