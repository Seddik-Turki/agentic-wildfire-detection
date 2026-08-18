from pathlib import Path
from tqdm.auto import tqdm

from src.agentic.constants import BETAS, MAX_TOKENS, MAX_TURNS, MODEL, THINKING_EFFORT
from src.agentic.tools import CODE_EXEC_TOOL,ATTACH_TOOL
from src.agentic.prompt import PROMPT, BBOX_PROMPT
from src.schema import DetectionResult
from src.data_utils import encode_raw, R


def upload(client, path, label):
    p = Path(path)
    with open(p, "rb") as fh:
        return client.beta.files.upload(
            file=(f"{label}{p.suffix}", fh, "image/jpeg"), betas=BETAS
        )


def harvest(client, response, registry, verbose = True):
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
            if verbose:
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

        if b.type in ("thinking", "redacted_thinking"):
            txt = getattr(b, "thinking", "") or "[redacted]"
            print(f"\n[🧠] {txt}\n")

        elif b.type == "text":
            print(f"\n[👾] {b.text}\n")

        elif b.type == "server_tool_use":
            print(f"\n[🐧] {b.input.get("command", b.input)}")

        elif b.type == "bash_code_execution_tool_result":
            c = b.content
            print("\n[>_] rc =", getattr(c, "return_code", "?"))
            print(getattr(c, "stdout", "")[:1200])
            if getattr(c, "stderr", ""):
                print("\n[❌]", c.stderr[:400])

        elif b.type == "tool_use":
            print(f"\n[🔧] {b.name}({b.input})\n")



def build_messages(client, stem, split):

    rgb_file = upload(client, R / f'rgb/{split}/{stem}.jpg', "rgb")
    thermal_file  = upload(client, R / f'thermal/{split}/{stem}.jpg', "thermal")

    return [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Image 1 — RGB:'},
            {'type': 'image', 'source': {
                'type': 'base64', 'media_type': 'image/jpeg',
                'data': encode_raw(R / f'rgb/{split}/{stem}.jpg')}},

            {'type': 'text', 'text': 'Image 2 — Thermal (LWIR):'},
            {'type': 'image', 'source': {
                'type': 'base64', 'media_type': 'image/jpeg',
                'data': encode_raw(R / f'thermal/{split}/{stem}.jpg')}},

            # Container Uploaded Files
            {"type": "container_upload", "file_id": rgb_file.id},
            {"type": "container_upload", "file_id": thermal_file.id},

            # cached: images + prompt are re-sent on every turn
            {'type': 'text', 'text': PROMPT,
             'cache_control': {'type': 'ephemeral'}},
        ],
    }]




def run(client, messages):
    registry = {}
    container = None


    for turn in tqdm(range(MAX_TURNS)):
        print(f"\n{'=' * 60}\nTURN {turn}\n{'=' * 60}")
        kwargs = {"container": container} if container else {}
        
        r = client.beta.messages.create(
            model=MODEL, 
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": THINKING_EFFORT},
            betas=BETAS, 
            
            tools=[CODE_EXEC_TOOL,ATTACH_TOOL],
            messages=messages, 
            **kwargs,
        )
        trace(r)
        print(f"stop={r.stop_reason}  tokens={r.usage.input_tokens}in/"
              f"{r.usage.output_tokens}out")

        if r.container:
            container = r.container.id
        harvest(client, r, registry)
        messages.append({"role": "assistant", "content": r.content})

        if r.stop_reason == "pause_turn":
            continue

        calls = [b for b in r.content if b.type == "tool_use"]

        if not calls:
            messages.append({
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': BBOX_PROMPT},
                    ],
                })
            
            r = client.beta.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": THINKING_EFFORT},
                output_format=DetectionResult,
                messages=messages,
                betas=BETAS
            )

            trace(r)


            dets = r.parsed_output.detections
            return dets

        results = []
        for b in calls:
            if b.name == "attach_image":
                out = attach_image(b.input["filename"], registry)
            else:
                out = {"content": f"Unknown tool: {b.name}", "is_error": True}

            print(f"   [result] {b.input.get('filename')} "
                  f"{'ERROR: ' + out['content'] if out['is_error'] else 'ok'}")
            
            results.append({"type": "tool_result", "tool_use_id": b.id, **out})
        
        messages.append({"role": "user", "content": results})

    print("\nregistry:", list(registry))
