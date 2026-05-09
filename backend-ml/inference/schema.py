"""
Request schemas and frontend metadata mappings.
"""

def get_features_schema(metadata: dict) -> dict:
    """
    Returns the expected input fields and their metadata for frontend form generation.
    Pulls from the model metadata if available, otherwise returns empty fallbacks.
    """
    return {
        "status": "success",
        "fields": metadata.get("form_field_mapping", {}),
        "raw_feature_schema": metadata.get("raw_feature_schema", [])
    }
