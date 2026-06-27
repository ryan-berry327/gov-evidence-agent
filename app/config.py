"""
Central configuration, read from environment variables (and a .env file in dev).
pydantic-settings validates types at startup so a bad value fails on boot rather
than deep inside a request.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider: "ollama" (local, free, default) or "anthropic" (prod)
    llm_provider: str = "ollama"
    ollama_model: str = "llama3.1"
    anthropic_api_key: str = ""
    chat_model: str = "claude-sonnet-4-6"
    embed_model: str = "voyage-3"  # placeholder; we use a local embedder by default

    # Retrieval knobs
    chunk_size: int = 800        # characters per chunk
    chunk_overlap: int = 150     # characters shared between adjacent chunks
    top_k: int = 5               # how many chunks to retrieve per query

    # Agent
    max_agent_steps: int = 6     # safety bound so the loop can't run forever


settings = Settings()
