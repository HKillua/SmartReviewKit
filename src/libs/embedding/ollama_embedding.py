"""Ollama Embedding implementation for local embedding models.

This module provides the Ollama Embedding implementation that works with
locally running Ollama instances. Ollama enables running embedding models like
nomic-embed-text, mxbai-embed-large, etc. on local hardware.

Improvements over original:
- Connection reuse: single httpx.Client per instance (not per-text)
- Retry with exponential backoff (3 attempts)
- Batch API (/api/embed) with fallback to single-text API (/api/embeddings)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama Embeddings API call fails.

    This exception provides clear error messages without exposing
    sensitive configuration details like internal URLs.
    """


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider implementation for local embedding.

    This class implements the BaseEmbedding interface for Ollama's embeddings API,
    enabling local embedding generation without cloud dependencies.

    Attributes:
        base_url: The base URL for the Ollama server (default: http://localhost:11434).
        model: The model identifier to use (e.g., 'nomic-embed-text', 'mxbai-embed-large').
        timeout: Request timeout in seconds.
        dimension: The dimensionality of embeddings produced by this model.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_DIMENSION = 768
    MAX_RETRIES = 3
    BACKOFF_BASE = 1.0  # seconds

    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama Embedding provider.

        Args:
            settings: Application settings containing Embedding configuration.
            base_url: Optional base URL override (falls back to env var OLLAMA_BASE_URL).
            timeout: Optional timeout override for requests.
            **kwargs: Additional configuration overrides.
        """
        self.model = settings.embedding.model

        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.dimension = getattr(settings.embedding, 'dimensions', self.DEFAULT_DIMENSION)
        self._extra_config = kwargs

        # Create a long-lived httpx client for connection reuse
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create a reusable httpx Client."""
        if self._client is None:
            import httpx
            client = httpx.Client(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30,
                ),
            )
            # Real httpx.Client.__enter__ returns self; test doubles often
            # hang their mocked ``post`` method off the entered client.
            self._client = client.__enter__() if hasattr(client, "__enter__") else client
        return self._client

    def _request_with_retry(
        self, url: str, payload: dict, max_retries: int = MAX_RETRIES
    ) -> dict:
        """Make an HTTP POST request with exponential backoff retry.

        Args:
            url: The API endpoint URL.
            payload: JSON payload to send.
            max_retries: Maximum number of retry attempts.

        Returns:
            Parsed JSON response dict.

        Raises:
            OllamaEmbeddingError: If all retries fail.
        """
        import httpx

        client = self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                try:
                    return response.json()
                except Exception as e:
                    raise OllamaEmbeddingError(
                        "Failed to parse Ollama API response"
                    ) from e
            except httpx.HTTPStatusError as e:
                last_error = OllamaEmbeddingError(
                    f"Ollama API request failed with status {e.response.status_code}"
                )
                if e.response.status_code == 404:
                    raise last_error from e
                logger.warning(
                    "Ollama API returned status %d (attempt %d/%d)",
                    e.response.status_code, attempt + 1, max_retries,
                )
            except httpx.ConnectError as e:
                last_error = OllamaEmbeddingError("Failed to connect to Ollama server")
                logger.warning(
                    "Failed to connect to Ollama at %s (attempt %d/%d): %s",
                    self.base_url, attempt + 1, max_retries, e,
                )
            except httpx.TimeoutException as e:
                last_error = OllamaEmbeddingError("Ollama API request timed out")
                logger.warning(
                    "Ollama request timed out (attempt %d/%d)", attempt + 1, max_retries,
                )
            except httpx.RequestError as e:
                last_error = OllamaEmbeddingError(f"Ollama API request failed: {e}")
                logger.warning(
                    "Ollama request error (attempt %d/%d): %s", attempt + 1, max_retries, e,
                )

            if attempt < max_retries - 1:
                sleep_time = self.BACKOFF_BASE * (2 ** attempt)
                logger.info("Retrying in %.1fs...", sleep_time)
                time.sleep(sleep_time)

        # All retries exhausted
        if isinstance(last_error, OllamaEmbeddingError):
            raise last_error
        raise OllamaEmbeddingError(
            f"Ollama API request failed after {max_retries} attempts. Last error: {last_error}"
        ) from last_error

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Try batch /api/embed endpoint (Ollama 0.4+), fallback to single /api/embeddings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        use_batch_api = bool(self._extra_config.get("use_batch_api", False))
        if use_batch_api and len(texts) > 1:
            batch_url = f"{self.base_url}/api/embed"
            try:
                result = self._request_with_retry(
                    batch_url,
                    {"model": self.model, "input": texts},
                )
                embeddings = result.get("embeddings")
                if embeddings and len(embeddings) == len(texts):
                    return embeddings
            except OllamaEmbeddingError as e:
                logger.warning("Batch /api/embed failed: %s, falling back to single-text API", e)

        # Fallback: single-text API (/api/embeddings)
        single_url = f"{self.base_url}/api/embeddings"
        embeddings: List[List[float]] = []

        for text in texts:
            result = self._request_with_retry(
                single_url,
                {"model": self.model, "prompt": text},
            )
            if "embedding" not in result:
                raise OllamaEmbeddingError(
                    f"Unexpected response format from Ollama API. "
                    f"Expected 'embedding' field but got: {list(result.keys())}"
                )
            embeddings.append(result["embedding"])

        return embeddings

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Ollama API.

        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters (currently unused).

        Returns:
            List of embedding vectors, where each vector is a list of floats.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            OllamaEmbeddingError: If API call fails after retries.
        """
        self.validate_texts(texts)

        try:
            import httpx  # noqa: F401 — ensure httpx is available
        except ImportError as e:
            raise OllamaEmbeddingError(
                "httpx library is required for Ollama Embedding. "
                "Install with: pip install httpx"
            ) from e

        return self._embed_batch(texts)

    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        return self.dimension

    def close(self) -> None:
        """Close the underlying HTTP client to release connections."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        """Ensure client is closed on garbage collection."""
        self.close()
