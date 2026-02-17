import yaml
import os
from pydantic import BaseModel, Field
from typing import Optional, List

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

class AIConfig(BaseModel):
    api_key: str
    model: str = "gemini-pro"

class ObsidianConfig(BaseModel):
    vault_path: str
    persist_directory: Optional[str] = None
    api_key: Optional[str] = None
    url: str = "https://127.0.0.1:27124"

class FilesystemConfig(BaseModel):
    allowed_paths: List[str] = Field(default_factory=list)

class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    ai: AIConfig
    obsidian: ObsidianConfig
    filesystem: Optional[FilesystemConfig] = None

def load_config(config_path: str = "../../workspace/config.yaml") -> AppConfig:
    """
    Loads configuration from a YAML file and validates it using Pydantic.
    Allows overriding with environment variables if necessary.
    """
    # Try to find config in the current directory if relative fails
    if not os.path.isabs(config_path):
        # Resolve relative to the app root (one level up from src)
        possible_paths = [
            config_path, # Standard relative path from src
            os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "config.yaml"), # Absolute reference
            "/app/workspace/config.yaml" # Docker path
        ]
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                break

    if not os.path.exists(config_path):
        # Fallback to current directory for development convenience
        if os.path.exists("workspace/config.yaml"):
            config_path = "workspace/config.yaml"
        else:
            raise FileNotFoundError(f"Configuration file not found in any of: {possible_paths}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Example of env override for sensitive data:
    env_api_key = os.getenv("GOOGLE_API_KEY")
    if env_api_key:
        if "ai" not in config_dict:
            config_dict["ai"] = {}
        config_dict["ai"]["api_key"] = env_api_key

    config = AppConfig(**config_dict)
    
    # Automate persist_directory resolution
    if config.obsidian:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        if not config.obsidian.persist_directory:
            # Default to 'chromadb' in the same directory as config.yaml
            config.obsidian.persist_directory = os.path.join(config_dir, "chromadb")
        elif not os.path.isabs(config.obsidian.persist_directory):
            # Resolve relative path against config directory
            config.obsidian.persist_directory = os.path.abspath(
                os.path.join(config_dir, config.obsidian.persist_directory)
            )
            
    return config
