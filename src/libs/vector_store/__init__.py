"""Vector Store Module — provider registry.

Auto-registers ChromaStore and MilvusStore (if dependencies are available).
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

try:
    from src.libs.vector_store.chroma_store import ChromaStore
    VectorStoreFactory.register_provider('chroma', ChromaStore)
except ImportError:
    pass

try:
    from src.libs.vector_store.milvus_store import MilvusStore
    VectorStoreFactory.register_provider('milvus', MilvusStore)
except ImportError:
    pass

__all__ = [
    'BaseVectorStore',
    'VectorStoreFactory',
    'ChromaStore',
    'MilvusStore',
]
