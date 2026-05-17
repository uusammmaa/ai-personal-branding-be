# backend/config.py
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "OPEN_AI_API_KEY"),
    )
    pinecone_api_key: str
    pinecone_index_name: str = "rag-documents"
    supabase_url: str = ""
    supabase_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SUPABASE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ),
    )
    vector_store: str = "pinecone"  # "pinecone" or "supabase"


settings = Settings()
