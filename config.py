"""
config.py
---------
YAML-based configuration system with CLI override support.


Usage
-----
    from config import load_config
    cfg = load_config("configs/swinunet.yaml", overrides={"training.lr": 1e-4})
"""


import yaml
import copy
from pathlib import Path
from typing import Any



def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result



def _set_nested(d: dict, dotted_key: str, value: Any):
    """Set a value in a nested dict using dot notation: 'training.lr' -> d['training']['lr']."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Try to cast to appropriate type
    d[keys[-1]] = _auto_cast(value)



def _auto_cast(value: str) -> Any:
    """Auto-cast string values from CLI to appropriate Python types."""
    if isinstance(value, str):
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        if value.lower() == "null" or value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value



class Config(dict):
    """Dict subclass with dot-notation access for convenience."""


    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, Config):
                val = Config(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


    def __setattr__(self, key, value):
        self[key] = value


    def __repr__(self):
        return f"Config({super().__repr__()})"



def load_config(
    config_path: str = None,
    overrides: dict = None,
    cli_overrides: list = None,
) -> Config:
    """
    Load configuration from YAML with optional overrides.


    Parameters
    ----------
    config_path   : Path to YAML config (if None, loads default)
    overrides     : Dict of overrides (supports nested dicts)
    cli_overrides : List of "key=value" strings from CLI (dot notation)


    Returns
    -------
    Config object with dot-notation access
    """
    # Load default config
    default_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)


    # Merge experiment-specific config
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            experiment_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, experiment_cfg)


    # Apply programmatic overrides
    if overrides:
        cfg = _deep_merge(cfg, overrides)


    # Apply CLI dot-notation overrides: ["training.lr=1e-4", "model.name=unet"]
    if cli_overrides:
        for item in cli_overrides:
            if "=" in item:
                key, value = item.split("=", 1)
                _set_nested(cfg, key.strip(), value.strip())


    return Config(cfg)