"""ZhiPu (智谱) Embedding implementation.

This module provides the ZhiPu BigModel Embedding implementation using
the OpenAI-compatible API. Supports embedding-3 (256/512/1024/2048 dims)
and embedding-2 (1024 dims).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class ZhipuEmbeddingError(RuntimeError):
    """Raised when ZhiPu Embeddings API call fails."""


class ZhipuEmbedding(BaseEmbedding):
    """ZhiPu BigModel Embedding provider implementation.

    Uses the OpenAI-compatible API at https://open.bigmodel.cn/api/paas/v4/.
    Supports models: embedding-3 (default 2048d), embedding-2 (1024d).

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = ZhipuEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ZhiPu Embedding provider.

        Args:
            settings: Application settings containing Embedding configuration.
            api_key: Optional API key override.
            base_url: Optional base URL override.
            **kwargs: Additional configuration overrides.
        """
        self.model = settings.embedding.model

        # Dimensions
        self.dimensions = getattr(settings.embedding, 'dimensions', None)

        # API key: explicit > settings.embedding.api_key > env var
        self.api_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("ZHIPU_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "ZhiPu API key not provided. Set ZHIPU_API_KEY environment variable, "
                "pass api_key parameter, or set embedding.api_key in settings.yaml."
            )

        # Base URL
        if base_url:
            self.base_url = base_url
        else:
            settings_base_url = getattr(settings.embedding, 'base_url', None)
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL

        self._extra_config = kwargs

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using ZhiPu API.

        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability.
            **kwargs: Override parameters.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            ZhipuEmbeddingError: If API call fails.
        """
        self.validate_texts(texts)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package is required for ZhiPu Embedding. "
                "Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Build API params
        api_params: dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }

        # Add dimensions if specified (embedding-3 supports 256/512/1024/2048)
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None:
            api_params["dimensions"] = dimensions

        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise ZhipuEmbeddingError(
                f"ZhiPu Embeddings API call failed: {e}"
            ) from e

        # Extract embeddings
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise ZhipuEmbeddingError(
                f"Failed to parse ZhiPu Embeddings API response: {e}"
            ) from e

        if len(embeddings) != len(texts):
            raise ZhipuEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension for the configured model."""
        if self.dimensions is not None:
            return self.dimensions

        model_dimensions = {
            "embedding-3": 2048,
            "embedding-2": 1024,
        }
        return model_dimensions.get(self.model)
