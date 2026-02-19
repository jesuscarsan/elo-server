import json
import os
from pydantic import BaseModel, Field
from typing import Optional, List

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
    reload: bool = True

class AIConfig(BaseModel):
    api_key: str
    model: str = "gemini-2.0-flash"

class ObsidianConfig(BaseModel):
    vault_path: Optional[str] = None
    persist_directory: Optional[str] = None
    api_key: Optional[str] = None
    url: str = "http://host.docker.internal:27123"
    last_index_datetime: Optional[str] = None

class FilesystemConfig(BaseModel):
    allowed_paths: List[str] = Field(default_factory=list)

class MCPConfig(BaseModel):
    name: str
    active: bool = True

class ToolConfig(BaseModel):
    name: str
    active: bool = True

class PathsConfig(BaseModel):
    root: str
    workspace: str
    assets: str
    mcps: str
    local_tools: List[str] = Field(default_factory=list)

class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    ai: AIConfig
    obsidian: ObsidianConfig
    filesystem: Optional[FilesystemConfig] = None
    paths: PathsConfig
    activated_mcps: List[MCPConfig] = Field(default_factory=list)
    activated_tools: List[ToolConfig] = Field(default_factory=list)

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Loads configuration by merging:
    1. Base structure from elo.config.json
    2. Sensitive data from environment variables (.env)
    3. Docker-specific path automation
    """
    # 1. Find and load elo-config.json
    possible_json_paths = [
        "elo-config.json",
        "/app/elo-config.json",
        "/app/workspace/elo-config.json",
        os.path.join(os.path.dirname(__file__), "..", "..", "elo-config.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "elo-config.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "elo-config.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "workspace", "elo-config.json")
    ]
    
    config_dict = {}
    for p in possible_json_paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                config_dict = json.load(f)
            break
            
    # 2. Map JSON keys to AppConfig structure (handling flattening/renaming)
    
    # AI Config
    ai_data = config_dict.get("ai", {})
    ai_config = {
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "model": ai_data.get("model", os.getenv("AI_MODEL", "gemini-2.0-flash"))
    }
    
    # Obsidian Config
    obs_data = config_dict.get("obsidian", {})
    obs_config = {
        "vault_path": os.getenv("VAULT_PATH"),
        "api_key": os.getenv("OBSIDIAN_API_KEY"),
        "url": obs_data.get("url", obs_data.get("url", os.getenv("OBSIDIAN_URL", "http://host.docker.internal:27123"))),
        "persist_directory": os.getenv("PERSIST_DIRECTORY"),
        "last_index_datetime": obs_data.get("lastIndexDatetime")
    }
    
    # Server Config
    server_config = {
        "host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "port": int(os.getenv("SERVER_PORT", 8001)),
        "reload": os.getenv("SERVER_RELOAD", "true").lower() == "true"
    }

    # Path Resolution Logic
    # Determine root of the project (apps/elo-server)
    # We assume this file is in src/infrastructure/config.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Handle Docker paths vs Local paths
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true"
    
    if is_docker:
        project_root = "/app"
        workspace_path = "/app/workspace"
        assets_path = "/app/assets"
    else:
        # Local development: we need to find where 'workspace' and 'assets' effectively are.
        # usually ../../../workspace relative to elo-server if in mono-repo structure
        # or just inside elo-server if standalone.
        # Based on user info: /Users/joshua/my-docs/code/elocuency-v3/apps/elo-server
        # Workspace is at /Users/joshua/my-docs/code/elocuency-v3/workspace
        
        # Check standard locations
        potential_workspace = os.path.join(project_root, "workspace")
        if not os.path.exists(potential_workspace):
             # Try one level up (monorepo root)
             monorepo_root = os.path.dirname(project_root)
             potential_workspace = os.path.join(monorepo_root, "workspace")
        
        workspace_path = potential_workspace
        
        potential_assets = os.path.join(project_root, "assets")
        if not os.path.exists(potential_assets):
             monorepo_root = os.path.dirname(project_root)
             potential_assets = os.path.join(monorepo_root, "assets")
             
        assets_path = potential_assets

    mcps_path = os.path.join(workspace_path, "mcps")
    
    # Tool directories
    tool_dirs = []
    
    # 1. Assets tools (git tracked)
    assets_tools = os.path.join(assets_path, "langchain", "tools")
    tool_dirs.append(assets_tools)
    
    # 2. Workspace tools (user defined)
    workspace_tools = os.path.join(workspace_path, "langchain", "tools")
    tool_dirs.append(workspace_tools)

    paths_config = {
        "root": project_root,
        "workspace": workspace_path,
        "assets": assets_path,
        "mcps": mcps_path,
        "local_tools": tool_dirs
    }

    full_config_dict = {
        "server": server_config,
        "ai": ai_config,
        "obsidian": obs_config,
        "paths": paths_config,
        "activated_mcps": config_dict.get("mcps", []),
        "activated_tools": config_dict.get("langchainTools", [])
    }
    
    config = AppConfig(**full_config_dict)
    
    # 3. Automate vault_path for Docker
    if not config.obsidian.vault_path:
        if is_docker and os.path.exists("/data/vault"):
            config.obsidian.vault_path = "/data/vault"

    # 4. Automate filesystem allowed_paths for Docker
    if is_docker:
        config.filesystem = FilesystemConfig(allowed_paths=["/app", "/data/vault"])
    elif not config.filesystem:
         # Allow workspace and assets locally
         config.filesystem = FilesystemConfig(allowed_paths=[workspace_path, assets_path])


    # 5. Automate persist_directory resolution
    if config.obsidian:
        # Default to 'workspace/chromadb' if not set
        if not config.obsidian.persist_directory:
             config.obsidian.persist_directory = os.path.join(workspace_path, "chromadb")
            
    return config

def update_obsidian_last_index(config_path: str, last_index: str):
    """
    Updates only the lastIndexDatetime field in elo.config.json.
    """
    if not os.path.exists(config_path):
        return
        
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    if "obsidian" not in config_dict:
        config_dict["obsidian"] = {}
        
    config_dict["obsidian"]["lastIndexDatetime"] = last_index
    
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
