"""Input/output utilities for saving and loading artifacts."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import warnings


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(
    data: Dict[str, Any],
    filepath: str,
    indent: int = 2,
    validate_writable: bool = True
) -> None:
    """
    Save dictionary as JSON file with pretty printing.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
        validate_writable: If True, check directory is writable before saving
        
    Raises:
        IOError: If directory is not writable
    """
    filepath = Path(filepath)
    
    # Ensure directory exists
    ensure_dir(filepath.parent)
    
    # Check if writable
    if validate_writable and not os.access(filepath.parent, os.W_OK):
        raise IOError(f"Directory {filepath.parent} is not writable")
    
    # Save with proper formatting
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str, validate_exists: bool = True) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath: Input file path
        validate_exists: If True, raise error if file doesn't exist
        
    Returns:
        Dictionary from JSON
        
    Raises:
        FileNotFoundError: If file doesn't exist and validate_exists=True
    """
    filepath = Path(filepath)
    
    if validate_exists and not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_markdown(
    content: str,
    filepath: str,
    append: bool = False
) -> None:
    """
    Save markdown content to file.
    
    Args:
        content: Markdown string
        filepath: Output file path
        append: If True, append to existing file; otherwise overwrite
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    mode = 'a' if append else 'w'
    with open(filepath, mode) as f:
        f.write(content)


def save_csv(
    df,  # pandas DataFrame
    filepath: str,
    **kwargs
) -> None:
    """
    Save DataFrame as CSV.
    
    Args:
        df: pandas DataFrame
        filepath: Output file path
        **kwargs: Additional arguments for df.to_csv()
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    df.to_csv(filepath, **kwargs)


def check_output_directory_writable(output_dir: str) -> bool:
    """
    Check if output directory exists and is writable.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        True if writable, False otherwise
    """
    output_dir = Path(output_dir)
    
    # Create if doesn't exist
    if not output_dir.exists():
        try:
            ensure_dir(output_dir)
        except Exception as e:
            warnings.warn(f"Cannot create output directory {output_dir}: {e}")
            return False
    
    # Check if writable
    return os.access(output_dir, os.W_OK)


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath: File path
        
    Returns:
        File size in MB
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0.0
    
    return filepath.stat().st_size / (1024 * 1024)


def validate_output_structure(output_dir: str, expected_files: list) -> Dict[str, bool]:
    """
    Validate that expected output files were created.
    
    Args:
        output_dir: Output directory
        expected_files: List of expected relative file paths
        
    Returns:
        Dictionary mapping file paths to existence status
    """
    output_dir = Path(output_dir)
    status = {}
    
    for expected_file in expected_files:
        filepath = output_dir / expected_file
        status[expected_file] = filepath.exists()
    
    return status

