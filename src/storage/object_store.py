"""Object store adapters for production file and image persistence."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.storage.postgres import PostgresExecutor

try:
    from minio import Minio

    MINIO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in dev
    Minio = None
    MINIO_AVAILABLE = False


class ObjectStore(ABC):
    """Abstract object storage interface."""

    @abstractmethod
    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    @abstractmethod
    def put_file(self, key: str, file_path: Union[str, Path], content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def download_to_path(self, key: str, destination: Union[str, Path]) -> str:
        raise NotImplementedError

    @abstractmethod
    def delete_object(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def uri_for(self, key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def key_from_uri(self, uri: str) -> Optional[str]:
        raise NotImplementedError


class LocalObjectStore(ObjectStore):
    """Filesystem-backed object store for development and tests."""

    def __init__(self, root: str = "data/object_store") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        safe_key = key.strip("/").replace("..", "_")
        path = (self._root / safe_key).resolve()
        if not str(path).startswith(str(self._root.resolve())):
            raise ValueError("Invalid object key")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._resolve(key)
        path.write_bytes(content)
        return self.uri_for(key)

    def put_file(self, key: str, file_path: Union[str, Path], content_type: str = "application/octet-stream") -> str:
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Object source not found: {file_path}")
        dest = self._resolve(key)
        shutil.copy2(source, dest)
        return self.uri_for(key)

    def read_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def download_to_path(self, key: str, destination: Union[str, Path]) -> str:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._resolve(key), dest)
        return str(dest)

    def delete_object(self, key: str) -> bool:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def uri_for(self, key: str) -> str:
        return str(self._resolve(key))

    def key_from_uri(self, uri: str) -> Optional[str]:
        try:
            path = Path(uri).resolve()
        except Exception:
            return None
        root = self._root.resolve()
        if not str(path).startswith(str(root)):
            return None
        return str(path.relative_to(root)).replace("\\", "/")


class MinioObjectStore(ObjectStore):
    """MinIO-backed object store."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ) -> None:
        if not MINIO_AVAILABLE:
            raise ImportError("minio is required for MinioObjectStore. Install with: pip install minio")
        self._bucket = bucket
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        from io import BytesIO

        self._client.put_object(
            self._bucket,
            key,
            data=BytesIO(content),
            length=len(content),
            content_type=content_type,
        )
        return self.uri_for(key)

    def put_file(self, key: str, file_path: Union[str, Path], content_type: str = "application/octet-stream") -> str:
        self._client.fput_object(self._bucket, key, str(file_path), content_type=content_type)
        return self.uri_for(key)

    def read_bytes(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def download_to_path(self, key: str, destination: Union[str, Path]) -> str:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(self._bucket, key, str(dest))
        return str(dest)

    def delete_object(self, key: str) -> bool:
        try:
            self._client.remove_object(self._bucket, key)
            return True
        except Exception:
            return False

    def uri_for(self, key: str) -> str:
        return f"minio://{self._bucket}/{key}"

    def key_from_uri(self, uri: str) -> Optional[str]:
        prefix = f"minio://{self._bucket}/"
        if not uri.startswith(prefix):
            return None
        return uri[len(prefix):]


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        return default if value is None else value
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return default


def create_object_store(settings: Any) -> ObjectStore:
    """Create object store from typed settings or raw mapping."""
    cfg = getattr(settings, "object_store", settings)
    provider = _cfg_value(cfg, "provider", "local")
    if provider == "minio":
        return MinioObjectStore(
            endpoint=str(_cfg_value(cfg, "endpoint", "")),
            bucket=str(_cfg_value(cfg, "bucket", "modular-rag")),
            access_key=str(_cfg_value(cfg, "access_key", "")),
            secret_key=str(_cfg_value(cfg, "secret_key", "")),
            secure=bool(_cfg_value(cfg, "secure", False)),
        )
    return LocalObjectStore(root=str(_cfg_value(cfg, "local_root", "data/object_store")))


class ObjectImageStorage:
    """Object-store-backed image storage with PostgreSQL metadata registry."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS image_assets (
            image_id TEXT PRIMARY KEY,
            object_key TEXT NOT NULL,
            collection TEXT,
            doc_hash TEXT,
            page_num INTEGER,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_image_collection ON image_assets(collection)",
        "CREATE INDEX IF NOT EXISTS idx_pg_image_doc_hash ON image_assets(doc_hash)",
    ]

    def __init__(
        self,
        *,
        dsn: str,
        object_store: ObjectStore,
        images_prefix: str = "images",
        cache_root: str = "data/images_cache",
    ) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)
        self._store = object_store
        self._images_prefix = images_prefix.strip("/") or "images"
        self._cache_root = Path(cache_root)
        self._cache_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        return None

    def _build_key(self, image_id: str, collection: Optional[str], extension: str) -> str:
        coll = collection or "default"
        return f"{self._images_prefix}/{coll}/{image_id}.{extension}"

    def save_image(
        self,
        image_id: str,
        image_data: Union[bytes, Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        extension: str = "png",
    ) -> str:
        key = self._build_key(image_id, collection, extension)
        if isinstance(image_data, bytes):
            uri = self._store.put_bytes(key, image_data, content_type=f"image/{extension}")
        else:
            uri = self._store.put_file(key, image_data, content_type=f"image/{extension}")
        self._upsert_metadata(image_id, key, collection, doc_hash, page_num)
        return uri

    def register_image(
        self,
        image_id: str,
        file_path: Union[Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        extension = path.suffix.lstrip(".") or "png"
        return self.save_image(
            image_id=image_id,
            image_data=path,
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
            extension=extension,
        )

    def _upsert_metadata(
        self,
        image_id: str,
        object_key: str,
        collection: Optional[str],
        doc_hash: Optional[str],
        page_num: Optional[int],
    ) -> None:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO image_assets (image_id, object_key, collection, doc_hash, page_num, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (image_id) DO UPDATE
                SET object_key = EXCLUDED.object_key,
                    collection = EXCLUDED.collection,
                    doc_hash = EXCLUDED.doc_hash,
                    page_num = EXCLUDED.page_num
                """,
                (image_id, object_key, collection, doc_hash, page_num, datetime.now(timezone.utc)),
            )

    def get_image_path(self, image_id: str) -> Optional[str]:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT object_key FROM image_assets WHERE image_id = %s", (image_id,))
            row = cur.fetchone()
        if not row:
            return None
        object_key = row[0]
        digest = hashlib.sha256(object_key.encode("utf-8")).hexdigest()[:12]
        ext = Path(object_key).suffix or ".bin"
        cached = self._cache_root / f"{digest}{ext}"
        if not cached.exists():
            self._store.download_to_path(object_key, cached)
        return str(cached)

    def image_exists(self, image_id: str) -> bool:
        return self.get_image_path(image_id) is not None

    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        query = """
            SELECT image_id, object_key, collection, doc_hash, page_num, created_at
            FROM image_assets
            WHERE 1 = 1
        """
        params: list[Any] = []
        if collection:
            query += " AND collection = %s"
            params.append(collection)
        if doc_hash:
            query += " AND doc_hash = %s"
            params.append(doc_hash)
        query += " ORDER BY created_at DESC"
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            {
                "image_id": row[0],
                "file_path": self._store.uri_for(row[1]),
                "object_key": row[1],
                "collection": row[2],
                "doc_hash": row[3],
                "page_num": row[4],
                "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            }
            for row in rows
        ]

    def delete_image(self, image_id: str, remove_file: bool = True) -> bool:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT object_key FROM image_assets WHERE image_id = %s", (image_id,))
            row = cur.fetchone()
            if not row:
                return False
            object_key = row[0]
            cur.execute("DELETE FROM image_assets WHERE image_id = %s", (image_id,))
        if remove_file:
            self._store.delete_object(object_key)
        return True
