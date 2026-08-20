import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/signalrank")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ce_model: str = os.getenv("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    embedding_dim: int = 384
    top_k_retrieval: int = 100
    top_k_final: int = 10
    rrf_k: int = 60
    hf_token: str | None = os.getenv("HF_TOKEN")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
