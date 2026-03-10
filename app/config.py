from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Docker Compose variables (e.g. SYSTEM_CA_BUNDLE) are not app settings
    )

    # Database
    database_url: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Auth
    jwt_secret_key: str = "change-this-in-production-use-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Rate limiting (fixed-window, Redis-backed, keyed by client IP)
    auth_register_rate_limit_requests: int = 5
    auth_register_rate_limit_window: int = 300
    auth_token_rate_limit_requests: int = 10
    auth_token_rate_limit_window: int = 60

    # CORS — restrict to the actual frontend origin(s) in production
    # Accepts a JSON array in the env var: CORS_ALLOW_ORIGINS='["https://app.example.com"]'
    cors_allow_origins: list[str] = ["http://localhost:7860"]

    # Upload limits
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB

    # App
    app_env: str = "development"
    log_level: str = "INFO"


settings = Settings()
