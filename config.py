"""
config.py
---------
YAML configuration with layered overrides and schema validation.

Precedence, lowest to highest::

    configs/default.yaml  ->  experiment YAML  ->  programmatic overrides  ->  CLI

Usage
-----
    from config import load_config

    cfg = load_config("configs/swinunet.yaml")
    cfg = load_config("configs/swinunet.yaml", cli_overrides=["training.lr=1e-4"])
    cfg = load_config(overrides={"training": {"epochs": 100}})

Validation
----------
:func:`load_config` validates the merged result before returning it. Every
misconfiguration caught here is one that would otherwise surface either as an
obscure stack trace hours into a run, or — much worse — as a run that trains
perfectly well on the wrong settings. Unknown top-level sections and unknown
keys within known sections are reported as errors rather than ignored, because
a typo'd ``lr`` that silently keeps the default is indistinguishable from a
successful override until you compare two runs and find them identical.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

__all__ = ["Config", "ConfigError", "load_config", "validate_config"]


class ConfigError(ValueError):
    """Raised when a configuration is structurally invalid."""


# ---------------------------------------------------------------------------
# Merging and casting
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``; ``override`` wins."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _auto_cast(value: str) -> Any:
    """
    Cast a CLI string to a Python value.

    Handles bools, null, ints, floats, and comma-separated or bracketed lists,
    so ``data.acceleration=[4,8]`` and ``training.betas=0.9,0.98`` both work.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None

    if (text.startswith("[") and text.endswith("]")) or "," in text:
        inner = text[1:-1] if text.startswith("[") else text
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if parts:
            return [_auto_cast(p) for p in parts]

    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    return text


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set ``d['a']['b'] = value`` from the dotted key ``'a.b'``."""
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ConfigError(f"Empty config key in override: {dotted_key!r}")

    cursor = d
    for k in keys[:-1]:
        nxt = cursor.get(k)
        if not isinstance(nxt, dict):
            if nxt is not None:
                raise ConfigError(
                    f"Cannot set {dotted_key!r}: {k!r} is a value, not a section."
                )
            nxt = {}
            cursor[k] = nxt
        cursor = nxt
    cursor[keys[-1]] = _auto_cast(value)


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


class Config(dict):
    """
    ``dict`` with attribute access, so ``cfg.training.lr`` works alongside
    ``cfg["training"]["lr"]``.

    Nested dicts are wrapped lazily on first access and the wrapper is written
    back, so repeated access does not re-allocate.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no key {key!r}. Available: {sorted(self.keys())}"
            ) from None
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        value = super().get(key, default)
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def to_dict(self) -> dict:
        """Plain nested ``dict``, safe for YAML/JSON serialisation."""

        def plain(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [plain(v) for v in obj]
            return obj

        return plain(dict(self))

    def __repr__(self) -> str:
        return f"Config({dict.__repr__(self)})"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# section -> set of accepted keys. Nested sections are validated separately.
_SCHEMA: dict[str, set[str]] = {
    "model": {"name", "complex", "params", "data_consistency"},
    "data": {
        "train_dir",
        "val_dir",
        "center_fraction",
        "acceleration",
        "crop_size",
        "mask_type",
        "slice_mode",
        "normalization",
        "undersample_domain",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "cache_dataset",
        "max_files",
    },
    "training": {
        "epochs",
        "batch_size",
        "seed",
        "deterministic",
        "val_split",
        "optimizer",
        "lr",
        "weight_decay",
        "betas",
        "scheduler",
        "warmup_epochs",
        "warmup_steps",
        "eta_min_factor",
        "loss",
        "gradient_clip",
        "augmentation",
        "patience",
        "amp",
        "amp_dtype",
        "ema",
        "gradient_accumulation",
        "compile",
        "channels_last",
        "monitor",
        "monitor_mode",
        "max_steps_per_epoch",
    },
    "logging": {
        "output_dir",
        "experiment_name",
        "tensorboard",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "log_interval",
        "save_top_k",
        "log_images",
        "unique_run_dir",
    },
    "inference": {"checkpoint", "export_onnx", "onnx_path", "tta", "mc_dropout_samples"},
}

_LOSS_KEYS = {
    "l1_weight",
    "ssim_weight",
    "freq_weight",
    "edge_weight",
    "perceptual_weight",
    "charbonnier",
}
_EMA_KEYS = {"enabled", "decay", "warmup_steps", "include_buffers"}
_DC_KEYS = {"enabled", "mode", "n_cascades", "share_weights", "hard"}


def _check_keys(section: str, payload: dict, allowed: set[str], errors: list[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        errors.append(
            f"{section}: unknown key(s) {sorted(unknown)}. Accepted: {sorted(allowed)}"
        )


def validate_config(cfg: dict) -> list[str]:
    """
    Check a merged configuration and return a list of problems.

    Returns an empty list when the config is valid. :func:`load_config` raises
    on a non-empty result; call this directly to inspect without raising.
    """
    errors: list[str] = []

    unknown_sections = set(cfg) - set(_SCHEMA)
    if unknown_sections:
        errors.append(
            f"Unknown top-level section(s) {sorted(unknown_sections)}. "
            f"Accepted: {sorted(_SCHEMA)}"
        )

    for section, allowed in _SCHEMA.items():
        payload = cfg.get(section)
        if payload is None:
            continue
        if not isinstance(payload, dict):
            errors.append(f"{section}: expected a mapping, got {type(payload).__name__}")
            continue
        _check_keys(section, payload, allowed, errors)

    model = cfg.get("model", {})
    if isinstance(model, dict):
        from models.registry import available_models

        name = model.get("name")
        if name is not None and str(name).lower() not in available_models():
            errors.append(
                f"model.name: unknown model {name!r}. "
                f"Choose from: {', '.join(available_models())}"
            )
        dc = model.get("data_consistency") or {}
        if isinstance(dc, dict):
            _check_keys("model.data_consistency", dc, _DC_KEYS, errors)
            mode = dc.get("mode")
            if mode is not None and str(mode).lower() not in ("cascade", "single"):
                errors.append(
                    f"model.data_consistency.mode: expected 'cascade' or 'single', got {mode!r}"
                )
            n_cascades = dc.get("n_cascades")
            if n_cascades is not None and int(n_cascades) < 1:
                errors.append(f"model.data_consistency.n_cascades must be >= 1, got {n_cascades}")

    training = cfg.get("training", {})
    if isinstance(training, dict):
        _check_keys("training.loss", training.get("loss") or {}, _LOSS_KEYS, errors)
        _check_keys("training.ema", training.get("ema") or {}, _EMA_KEYS, errors)

        for key, minimum in (
            ("epochs", 1),
            ("batch_size", 1),
            ("gradient_accumulation", 1),
            ("patience", 1),
        ):
            value = training.get(key)
            if value is not None and int(value) < minimum:
                errors.append(f"training.{key} must be >= {minimum}, got {value}")

        lr = training.get("lr")
        if lr is not None and not (0 < float(lr) < 1):
            errors.append(f"training.lr should be in (0, 1), got {lr}")

        val_split = training.get("val_split")
        if val_split is not None and not (0 < float(val_split) < 1):
            errors.append(f"training.val_split must be in (0, 1), got {val_split}")

        optimizer = training.get("optimizer")
        if optimizer is not None and str(optimizer).lower() not in ("adam", "adamw", "sgd"):
            errors.append(f"training.optimizer: expected adam, adamw or sgd, got {optimizer!r}")

        monitor_mode = training.get("monitor_mode")
        if monitor_mode is not None and monitor_mode not in ("min", "max"):
            errors.append(f"training.monitor_mode must be 'min' or 'max', got {monitor_mode!r}")

        loss_cfg = training.get("loss") or {}
        weights = [v for k, v in loss_cfg.items() if k.endswith("_weight")]
        if weights and all(float(w) <= 0 for w in weights):
            errors.append("training.loss: at least one component weight must be positive")

    data = cfg.get("data", {})
    if isinstance(data, dict):
        cf = data.get("center_fraction")
        if cf is not None and not (0 < float(cf) < 1):
            errors.append(f"data.center_fraction must be in (0, 1), got {cf}")

        acceleration = data.get("acceleration")
        if acceleration is not None:
            accs = acceleration if isinstance(acceleration, list) else [acceleration]
            for a in accs:
                if int(a) < 1:
                    errors.append(f"data.acceleration must be >= 1, got {a}")

        crop = data.get("crop_size")
        if crop is not None and (not isinstance(crop, (list, tuple)) or len(crop) != 2):
            errors.append(f"data.crop_size must be a two-element list, got {crop!r}")

        for key, options in (
            ("mask_type", ("random", "equispaced", "magic")),
            ("normalization", ("zf_max", "attr_max", "none")),
            ("undersample_domain", ("cropped", "full")),
        ):
            value = data.get(key)
            if value is not None and str(value) not in options:
                errors.append(f"data.{key}: expected one of {options}, got {value!r}")

    # Cross-section consistency: this combination silently produced a no-op in
    # the original code, so it is now an explicit error.
    dc_enabled = bool((model or {}).get("data_consistency", {}).get("enabled", False))
    if dc_enabled and str(data.get("undersample_domain", "cropped")) == "full":
        errors.append(
            "model.data_consistency.enabled=true requires "
            "data.undersample_domain='cropped': in 'full' mode the returned "
            "k-space is not the measured data, so data consistency is invalid."
        )

    return errors


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
    cli_overrides: list[str] | None = None,
    validate: bool = True,
) -> Config:
    """
    Load and merge configuration.

    Parameters
    ----------
    config_path
        Experiment YAML layered on top of ``configs/default.yaml``. A missing
        path is an error, not a silent fallback to defaults — the original code
        ignored unreadable paths, so a typo'd filename trained the default model
        while appearing to honour the request.
    overrides
        Nested dict merged after the file.
    cli_overrides
        ``"dotted.key=value"`` strings, applied last.
    validate
        Run :func:`validate_config` and raise on failure.

    Returns
    -------
    A validated :class:`Config`.
    """
    default_path = Path(__file__).parent / "configs" / "default.yaml"
    if not default_path.exists():
        raise ConfigError(f"Base config not found at {default_path}")

    with open(default_path) as f:
        cfg = yaml.safe_load(f) or {}

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(
                f"Config file not found: {path}. "
                f"Available: {sorted(p.name for p in default_path.parent.glob('*.yaml'))}"
            )
        with open(path) as f:
            experiment = yaml.safe_load(f) or {}
        if not isinstance(experiment, dict):
            raise ConfigError(f"{path} must contain a YAML mapping at the top level")
        cfg = _deep_merge(cfg, experiment)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    for item in cli_overrides or []:
        if "=" not in item:
            raise ConfigError(
                f"Malformed override {item!r}; expected 'section.key=value' "
                f"(for example 'training.lr=1e-4')."
            )
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), value.strip())

    if validate:
        errors = validate_config(cfg)
        if errors:
            bullet = "\n  - ".join(errors)
            raise ConfigError(f"Invalid configuration:\n  - {bullet}")

    return Config(cfg)
