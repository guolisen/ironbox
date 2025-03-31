"""
Configuration module for IronBox.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "llm": {
        "provider": "ollama",
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://192.168.0.103:11434"),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "database": {
        "url": f"sqlite:///{DB_DIR}/ironbox.db",
        "echo": False,
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
    },
    "ui": {
        "port": 8501,
    },
    "mcp": {
        "enabled": True,
        "servers": {
            "weather": {
                "enabled": True,
            }
        }
    },
    "kubernetes": {
        "default_timeout": 30,
    },
    "toolkit": {
        "tools": [],  # Will be populated from config file or auto-discovery
        "agents": [],  # Will be populated from config file or auto-discovery
        "discovery": {
            "tools": {
                "enabled": True,
                "paths": ["ironbox.tools"]
            },
            "agents": {
                "enabled": True,
                "paths": ["ironbox.agents"]
            }
        }
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f)
            if file_config:
                # Recursively update config
                _update_dict(config, file_config)
    
    return config


def _update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _update_dict(d[k], v)
        else:
            d[k] = v
    return d


# Load configuration
config = load_config()
