"""JSON schema validation for output artifacts."""

from typing import Dict, Any, List, Optional
import warnings


# JSON Schema definitions for output artifacts
AUDIT_EVIDENCE_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "timestamp", "diagnostics", "t_eff", "safe_tau_max", "provenance"],
    "properties": {
        "schema_version": {"type": "string"},
        "timestamp": {"type": "string"},
        "diagnostics": {"type": "object"},
        "t_eff": {"type": "object"},
        "safe_tau_max": {"type": "object"},
        "provenance": {"type": "object"}
    }
}

RISK_PROFILE_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "timestamp", "risks", "risk_ledger", "provenance"],
    "properties": {
        "schema_version": {"type": "string"},
        "timestamp": {"type": "string"},
        "risks": {"type": "object"},
        "risk_ledger": {"type": "object"},
        "covariance_matrix": {"type": "object"},
        "provenance": {"type": "object"}
    }
}

POLICY_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "timestamp", "decision", "provenance"],
    "properties": {
        "schema_version": {"type": "string"},
        "timestamp": {"type": "string"},
        "decision": {"type": "string", "enum": ["recommend", "abstain"]},
        "recommended_method": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "policy_reason_codes": {"type": "array"},
        "method_config": {"type": "object"},
        "provenance": {"type": "object"}
    }
}


def _validate_schema(data: Dict[str, Any], schema: Dict[str, Any], artifact_name: str) -> bool:
    """
    Simple schema validation (checks required fields and types).
    
    Args:
        data: Data to validate
        schema: JSON schema definition
        artifact_name: Name of artifact for error messages
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check required fields
    required_fields = schema.get("required", [])
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(
            f"{artifact_name} missing required fields: {missing_fields}"
        )
    
    # Check field types (basic validation)
    properties = schema.get("properties", {})
    for field, field_schema in properties.items():
        if field in data:
            expected_type = field_schema.get("type")
            value = data[field]
            
            # Handle nullable types (if type is a list with "null")
            if isinstance(expected_type, list):
                # Type can be one of multiple types (e.g., ["string", "null"])
                if value is None and "null" in expected_type:
                    continue  # None is allowed
                # Check against non-null types
                valid_types = [t for t in expected_type if t != "null"]
                if not valid_types:
                    continue
                expected_type = valid_types[0]  # Use first non-null type for checking
            
            # Skip validation if None and field allows null
            if value is None:
                continue  # Assume None is acceptable for optional fields
            
            # Handle type validation
            if expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"{artifact_name}.{field} must be string, got {type(value)}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"{artifact_name}.{field} must be number, got {type(value)}")
            elif expected_type == "object" and not isinstance(value, dict):
                raise ValueError(f"{artifact_name}.{field} must be object, got {type(value)}")
            elif expected_type == "array" and not isinstance(value, list):
                raise ValueError(f"{artifact_name}.{field} must be array, got {type(value)}")
            
            # Handle enums
            if "enum" in field_schema and value not in field_schema["enum"]:
                raise ValueError(
                    f"{artifact_name}.{field} must be one of {field_schema['enum']}, got {value}"
                )
    
    return True


def validate_audit_evidence(data: Dict[str, Any]) -> bool:
    """
    Validate AuditEvidence JSON output.
    
    Args:
        data: AuditEvidence dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    return _validate_schema(data, AUDIT_EVIDENCE_SCHEMA, "AuditEvidence")


def validate_risk_profile(data: Dict[str, Any]) -> bool:
    """
    Validate RiskProfile JSON output.
    
    Args:
        data: RiskProfile dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    return _validate_schema(data, RISK_PROFILE_SCHEMA, "RiskProfile")


def validate_policy(data: Dict[str, Any]) -> bool:
    """
    Validate Policy JSON output.
    
    Args:
        data: Policy dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    return _validate_schema(data, POLICY_SCHEMA, "Policy")


def check_schema_version_compatibility(
    artifact_version: str,
    package_version: str,
    warn_only: bool = True
) -> bool:
    """
    Check if artifact schema version is compatible with package version.
    
    Args:
        artifact_version: Schema version from artifact
        package_version: Current package schema version
        warn_only: If True, only warn on mismatch; if False, raise error
        
    Returns:
        True if compatible
    """
    # Simple semantic versioning check (major.minor.patch)
    artifact_parts = artifact_version.split('.')
    package_parts = package_version.split('.')
    
    # Major version must match
    if artifact_parts[0] != package_parts[0]:
        msg = f"Incompatible schema versions: artifact {artifact_version}, package {package_version}"
        if warn_only:
            warnings.warn(msg)
            return False
        else:
            raise ValueError(msg)
    
    # Minor version mismatch is a warning
    if artifact_parts[1] != package_parts[1]:
        warnings.warn(
            f"Schema version mismatch (minor): artifact {artifact_version}, package {package_version}. "
            "Forward compatibility not guaranteed."
        )
    
    return True

