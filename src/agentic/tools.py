ATTACH_TOOL = {
    "name": "attach_image",
    "description": (
        "Show yourself an image you already saved into $OUTPUT_DIR from the "
        "container. This does NOT read the container filesystem — the file must "
        "already have been captured by a completed bash command. Pass the bare "
        "filename, e.g. 'rgb_code_1.png'. Call this in a turn of its own, after "
        "the bash command that wrote the file has returned."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Bare filename, no path."}
        },
        "required": ["filename"],
    },
}

CODE_EXEC_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}