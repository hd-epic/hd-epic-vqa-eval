import json


def load_check_config(config_fn):
    try:
        with open(config_fn, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_fn} not found")
        exit(1)

    # Check for required fields
    assert config["mode"] in [
        "visual_text",
        "text",
        "narration_text",
        "narration_verb_noun_text",
    ]

    # TODO: Add more checks here

    return config
