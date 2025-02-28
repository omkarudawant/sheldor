from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Model settings
    DEFAULT_LLM_MODEL: str = "llama2"
    DEFAULT_EMBEDDING_MODEL: str = "mxbai-embed-large"

    # Vector store settings
    DEFAULT_SEARCH_K: int = 3

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "sheldor.log"

    # Personality settings
    SHELDON_SARCASM_DETECTION: bool = False  # Sheldon doesn't understand sarcasm
    SHELDON_VERBOSITY: int = 3  # 1-3, controls detail level in explanations
    SHELDON_SCIENTIFIC_REFERENCES: bool = True  # Include scientific paper references

    class Config:
        env_prefix = "SHELDOR_"
