"""Configuration helpers for the VisHarness trajectory runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Small dict wrapper that preserves the legacy ``config.foo`` style."""

    def __init__(self, value: dict[str, Any] | None = None):
        super().__init__()
        for key, item in (value or {}).items():
            self[key] = self._wrap(item)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_plain_dict(self) -> dict[str, Any]:
        return _plain(self)


def _plain(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        if config_path.suffix.lower() == ".json":
            payload = json.load(file)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(file)
        else:
            raise ValueError(f"Config file should be json/yaml, got {config_path}")
    return ConfigNode(payload or {})


def as_plain_dict(value: Any) -> dict[str, Any]:
    return _plain(value) or {}
