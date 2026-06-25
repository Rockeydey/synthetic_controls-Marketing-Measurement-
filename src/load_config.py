def load_json_config(file_path):
    """
    Load a JSON configuration file and return parameter defaults.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        dict: A dictionary of parameter keys and their default values.
              Example: {"kpi": "sales"}
    """
    import json

    with open(file_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    parameters = config.get("parameters", {})
    return {
        key: value.get("default")
        for key, value in parameters.items()
        if isinstance(value, dict) and "default" in value
    }