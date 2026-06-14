import jsonschema


REASONING_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "screen_summary",
        "user_stories",
        "acceptance_criteria",
        "gap_analysis",
        "truncated",
        "raw_response",
        "image_path",
    ],
    "properties": {
        "screen_summary": {"type": "string"},
        "user_stories": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "story", "scenario"],
                "properties": {
                    "title": {"type": "string"},
                    "story": {"type": "string"},
                    "assembled_story": {"type": "string"},
                    "scenario": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "acceptance_criteria": {"type": "string"},
        "gap_analysis": {"type": "string"},
        "truncated": {"type": "boolean"},
        "raw_response": {"type": "string"},
        "image_path": {"type": "string"},
    },
    "additionalProperties": True,
}


def validate_reasoning_output(artefacts: dict) -> dict:
    """
    Validate LLaVA reasoning output.

    Performs:
    1. Structural validation using JSON schema.
    2. Completeness validation for non-empty BA artefact fields.

    Args:
        artefacts (dict): Output returned by analyse_frame().

    Returns:
        dict:
            {
                "valid": bool,
                "errors": list[str]
            }
    """

    errors = []

    # 1. Schema validation
    try:
        jsonschema.validate(
            instance=artefacts,
            schema=REASONING_OUTPUT_SCHEMA,
        )
    except jsonschema.ValidationError as exc:
        errors.append(f"Schema validation error: {exc.message}")

    # 2. Non-empty text checks
    required_text_fields = [
        "screen_summary",
        "acceptance_criteria",
        "gap_analysis",
    ]

    for field in required_text_fields:
        value = artefacts.get(field, "")

        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} is empty or missing")

    # 3. User story checks
    user_stories = artefacts.get("user_stories", [])

    if not isinstance(user_stories, list) or len(user_stories) == 0:
        errors.append("user_stories is empty or missing")
    else:
        for idx, user_story in enumerate(user_stories, start=1):
            title = user_story.get("title", "")
            story = user_story.get("story", "")
            scenario = user_story.get("scenario", "")
            assembled_story = user_story.get("assembled_story", "")

            if not isinstance(title, str) or not title.strip():
                errors.append(f"user_stories[{idx}].title is empty or missing")

            if not isinstance(story, str) or not story.strip():
                errors.append(f"user_stories[{idx}].story is empty or missing")

            if not isinstance(scenario, str) or not scenario.strip():
                errors.append(f"user_stories[{idx}].scenario is empty or missing")

            if not isinstance(assembled_story, str) or not assembled_story.strip():
                errors.append(f"user_stories[{idx}].assembled_story is empty or missing")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }