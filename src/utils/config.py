"""Configuration loading and snapshot utilities."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Configuration files make experiments reproducible and auditable. This
    matters for a research repository because paper claims should be traceable
    to explicit assumptions about data, planning costs, and stability weights.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError("Config file does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return config


def save_config_snapshot(config: Dict[str, Any], output_dir: Union[str, Path], file_name: str = "config_snapshot.yaml") -> Path:
    """Save a YAML snapshot of the active experiment configuration.

    Config snapshots make later tables and figures reproducible. They are
    especially important for this project because planning utility depends on
    cost weights, execution capacity, and stability thresholds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_path / file_name
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return snapshot_path
