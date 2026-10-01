"""User config loading: omlx settings key, models.yaml (+ presets/profiles), pricing.yaml."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
ALLOWED_SERVER_KEYS = {
    "hot_cache_max_size": str, "paged_ssd_cache_max_size": str, "hot_cache_write_through": bool,
    "initial_cache_blocks": int, "no_cache": bool, "hot_cache_only": bool,
}


def omlx_settings_key():
    try:
        return json.loads((Path.home() / ".omlx" / "settings.json").read_text())["auth"]["api_key"] or None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Config:
    model_root: str
    models: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    presets: dict = field(default_factory=dict)


def _err(path, msg):
    raise ValueError(f"{path}: {msg}")


def load_models(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model_root = raw.get("model_root", "~/models/omlx-bench")
    models = {}
    for name, spec in (raw.get("models") or {}).items():
        if not NAME_RE.match(name):
            _err(f"models.{name}", "invalid model name")
        spec = spec or {}
        if "repo" not in spec or not isinstance(spec["repo"], str):
            _err(f"models.{name}.repo", "required (str)")
        color = spec.get("color")
        if color is not None and not isinstance(color, str):
            _err(f"models.{name}.color", "must be a hex string")
        settings = spec.get("settings", {}) or {}
        if not isinstance(settings, dict):
            _err(f"models.{name}.settings", "must be a dict")
        models[name] = {"repo": spec["repo"], "color": color, "settings": settings}

    profiles = {}
    for name, spec in (raw.get("profiles") or {}).items():
        if not NAME_RE.match(name):
            _err(f"profiles.{name}", "invalid profile name")
        spec = spec or {}
        server = spec.get("server", {}) or {}
        if not isinstance(server, dict):
            _err(f"profiles.{name}.server", "must be a dict")
        for k, v in server.items():
            if k not in ALLOWED_SERVER_KEYS:
                _err(f"profiles.{name}.server.{k}", "unknown key")
            t = ALLOWED_SERVER_KEYS[k]
            if t is bool and not isinstance(v, bool):
                _err(f"profiles.{name}.server.{k}", "must be a bool")
            if t is int and not isinstance(v, int):
                _err(f"profiles.{name}.server.{k}", "must be an int")
            if t is str and not isinstance(v, str):
                _err(f"profiles.{name}.server.{k}", "must be a str")
        if server.get("hot_cache_only") \
                and (server.get("hot_cache_max_size") in (None, "0") or server.get("no_cache")):
            _err(f"profiles.{name}.server.hot_cache_only",
                 "requires hot_cache_max_size != '0' and no_cache not true")
        model_settings = spec.get("model_settings", {}) or {}
        if not isinstance(model_settings, dict):
            _err(f"profiles.{name}.model_settings", "must be a dict")
        profiles[name] = {"server": server, "model_settings": model_settings}

    presets = raw.get("presets") or {}
    return Config(model_root=model_root, models=models, profiles=profiles, presets=presets)


def load_pricing(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def effective_model_settings(cfg: Config, model_name, profile_name):
    m = cfg.models.get(model_name, {})
    p = cfg.profiles.get(profile_name, {})
    return {**(m.get("settings") or {}), **(p.get("model_settings") or {})}
