from pathlib import Path

from src.schema import DetectionResult
from src.data_utils import encode_raw, R
from src.basic_prompting.prompt import PROMPT
from src.basic_prompting.constants import MODEL, MAX_TOKENS



def trace(response):
    for b in response.content:

        if b.type in ("thinking", "redacted_thinking"):
            txt = getattr(b, "thinking", "") or "[redacted]"
            print(f"\n[🧠] {txt}\n")

        elif b.type == "text":
            print(f"\n[👾] {b.text}\n")

        elif b.type == "server_tool_use":
            print(f"\n[🐧] {b.input.get('command', b.input)}")

        elif b.type == "bash_code_execution_tool_result":
            c = b.content
            print("\n[>_] rc =", getattr(c, "return_code", "?"))
            print(getattr(c, "stdout", "")[:1200])
            if getattr(c, "stderr", ""):
                print("\n[❌]", c.stderr[:400])

        elif b.type == "tool_use":
            print(f"\n[🔧] {b.name}({b.input})\n")


def detect(client, stem, split):
    """One frame -> (detections, raw response)."""
    r = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "high"},
        output_format=DetectionResult,
        messages=[{
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
                {'type': 'text', 'text': PROMPT},
            ],
        }],
    )
    return r.parsed_output.detections, r
