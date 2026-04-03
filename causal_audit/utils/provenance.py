"""Provenance tracking and hashing utilities for reproducibility."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
import yaml


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate SHA256 hash of DataFrame contents.

    Args:
        df: Input DataFrame

    Returns:
        Hexadecimal hash string
    """
    # Use pandas built-in hash for consistency
    hash_obj = hashlib.sha256()

    # Hash column names
    hash_obj.update(str(df.columns.tolist()).encode("utf-8"))

    # Hash data types
    hash_obj.update(str(df.dtypes.tolist()).encode("utf-8"))

    # Hash values (handle NaNs consistently)
    for col in df.columns:
        col_bytes = df[col].fillna("__NA__").astype(str).to_numpy().tobytes()
        hash_obj.update(col_bytes)

    return hash_obj.hexdigest()


def hash_config(config_path: str) -> str:
    """
    Generate SHA256 hash of configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Hexadecimal hash string
    """
    with open(config_path, "r") as f:
        content = f.read()

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_dict(data: Dict[str, Any]) -> str:
    """
    Generate SHA256 hash of dictionary (sorted keys for consistency).

    Args:
        data: Dictionary to hash

    Returns:
        Hexadecimal hash string
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def generate_provenance(
    input_data_hash: str,
    config_hashes: Dict[str, str],
    random_seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate complete provenance record for reproducibility.

    Args:
        input_data_hash: SHA256 hash of input DataFrame
        config_hashes: Dictionary mapping config file names to their hashes
        random_seed: Random seed used (if any)
        metadata: Optional additional metadata

    Returns:
        Provenance dictionary with all tracking information
    """
    from .. import __version__

    provenance = {
        "schema_version": "1.0.0",
        "causal_audit_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_data_hash": input_data_hash,
        "config_hashes": config_hashes,
    }

    if random_seed is not None:
        provenance["random_seed"] = random_seed

    if metadata:
        provenance["metadata"] = metadata

    # Generate provenance hash (hash of all provenance fields)
    provenance["provenance_hash"] = hash_dict(provenance)

    return provenance


def verify_provenance(
    provenance: Dict[str, Any], df: pd.DataFrame, config_paths: Dict[str, str]
) -> bool:
    """
    Verify that provenance matches current data and config.

    Args:
        provenance: Provenance record to verify
        df: Current DataFrame
        config_paths: Dictionary mapping config names to file paths

    Returns:
        True if provenance is valid, False otherwise
    """
    # Check data hash
    current_data_hash = hash_dataframe(df)
    if current_data_hash != provenance.get("input_data_hash"):
        return False

    # Check config hashes
    for config_name, config_path in config_paths.items():
        current_hash = hash_config(config_path)
        if current_hash != provenance.get("config_hashes", {}).get(config_name):
            return False

    return True
