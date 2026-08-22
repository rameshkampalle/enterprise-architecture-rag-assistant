from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Pinecone
    pinecone_api_key: SecretStr
    pinecone_index_name: str = "corp-enterprise-arch-python"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "corpus-v1-python"
    pinecone_index_dimension: int = 3072
    pinecone_metadata_filter: Dict[str, Any] = Field(default_factory=dict)
    pinecone_text_field: str = "text"

    # Models
    gemini_api_key: SecretStr
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = 0.0

    # Corpus and chunking
    corpus_path: Path = Field(default=Path("data/corpus"))
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_ingest_batch_size: int = 10

    # Retrieval / generation
    top_k_retrieval: int = 5
    eval_batch_size: int = 5
    eval_sleep_seconds: float = 0.0

    # Logging
    log_level: str = "INFO"

    @field_validator("corpus_path", mode="before")
    def _expand_corpus_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @field_validator("pinecone_metadata_filter", mode="before")
    def _parse_filter(cls, value: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}
