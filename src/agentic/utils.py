from pathlib import Path
import base64

from agentic.constants import BETAS

def encode_raw(path):
    """Send original JPEG bytes — no re-encode, no second lossy pass."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def upload(client, path, label):
    p = Path(path)
    with open(p, "rb") as fh:
        return client.beta.files.upload(
            file=(f"{label}{p.suffix}", fh, "image/jpeg"), betas=BETAS
        )


def harvest(client, response, registry):
    """Add every file the container captured to the filename -> file_id map."""
    for block in response.content:
        if block.type != "bash_code_execution_tool_result":
            continue
        c = block.content
        if getattr(c, "type", None) != "bash_code_execution_result":
            continue
        for out in c.content:
            meta = client.beta.files.retrieve_metadata(out.file_id)
            registry[meta.filename] = out.file_id
            print(f"   [registry] {meta.filename} -> {out.file_id}")


def attach_image(filename, registry):
    fid = registry.get(filename)
    if fid is None:
        known = ", ".join(sorted(registry)) or "(nothing captured yet)"
        return {
            "content": f"No captured file named '{filename}'. Available: {known}. "
                       f"Did you cp it into $OUTPUT_DIR?",
            "is_error": True,
        }
    return {
        "content": [
            {"type": "text", "text": f"{filename}:"},
            {"type": "image", "source": {"type": "file", "file_id": fid}},
        ],
        "is_error": False,
    }


def trace(response):
    for b in response.content:
        if b.type == "text":
            print("[text]", b.text)
        elif b.type == "server_tool_use":
            print("[bash]", b.input.get("command", b.input))
        elif b.type == "bash_code_execution_tool_result":
            c = b.content
            print("[out ] rc =", getattr(c, "return_code", "?"))
            print(getattr(c, "stdout", "")[:1200])
            if getattr(c, "stderr", ""):
                print("[err ]", c.stderr[:400])
        elif b.type == "tool_use":
            print(f"[call] {b.name}({b.input})")