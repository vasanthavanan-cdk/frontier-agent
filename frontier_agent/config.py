"""Runtime configuration loaded from environment variables and .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. All fields map to environment variables of the same name (uppercase)."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    escalation_threshold: float = 0.70
    max_local_retries: int = 2


settings = Settings()
