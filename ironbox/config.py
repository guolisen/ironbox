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
    "agent_frameworks": [
        {
            "name": "route_framework",
            "type": "route",
            "enabled": True,
            "config": {
                "system_prompt": """
                You are a router agent for the IronBox system. Your job is to analyze the user's request and determine which specialized agent should handle it.
                
                Available agents:
                {agent_descriptions}
                
                Respond with the name of the agent that should handle the request.
                """
            }
        },
        {
            "name": "react_framework",
            "type": "react",
            "enabled": True,
            "config": {
                "system_prompt": """
                You are a React agent for the IronBox system. You solve problems by thinking step-by-step and taking actions.
                """,
                "max_iterations": 10
            }
        },
        {
            "name": "plan_framework",
            "type": "plan",
            "enabled": True,
            "config": {
                "planning_prompt": """
                You are a planning agent for the IronBox system. Your job is to create a detailed plan to solve the user's problem.
                
                Available tools:
                {tool_descriptions}
                
                Create a step-by-step plan to solve the following problem:
                {problem}
                
                Your plan should be detailed and include all the necessary steps to solve the problem.
                Format your response as a numbered list of steps.
                """
            }
        }
    ],
    "graph": {
        "entry_point": "framework_selector",
        "edges": [
            {
                "from": "framework_selector",
                "to": "route_framework",
                "condition": "state.agent_outputs.get('framework_selector', {}).get('framework_type') == 'route'"
            },
            {
                "from": "framework_selector",
                "to": "react_framework",
                "condition": "state.agent_outputs.get('framework_selector', {}).get('framework_type') == 'react'"
            },
            {
                "from": "framework_selector",
                "to": "plan_framework",
                "condition": "state.agent_outputs.get('framework_selector', {}).get('framework_type') == 'plan'"
            }
        ]
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
