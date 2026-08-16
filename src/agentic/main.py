from pathlib import Path
from tqdm.auto import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt

from schema import DetectionResult
from data_utils import load_frames, load_gt, draw_boxes

from agentic.constants import MAX_TOKENS, MODEL, BETAS
from agentic.tools import ATTACH_TOOL, CODE_EXEC_TOOL
from agentic.prompt import PROMPT, BBOX_PROMPT
from agentic.utils import trace, harvest, attach_image, encode_raw, upload


R = Path(__file__).parents[2] / "rgbt-3m"

from dotenv import load_dotenv
load_dotenv()

import anthropic
client = anthropic.Anthropic()


def run(client, messages):
    registry = {}
    container = None


    for turn in tqdm(range(12)):
        print(f"\n{'=' * 60}\nTURN {turn}\n{'=' * 60}")
        kwargs = {"container": container} if container else {}
        
        r = client.beta.messages.create(
            model=MODEL, 
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
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
            print("\n--- DONE ---")
            break

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


def main():
    frames = load_frames()
    vid_id = random.choice([2,3,4,5])
    _, stem, split = random.choice(frames[f'video{vid_id}'])
    img = Image.open(R / f'rgb/{split}/{stem}.jpg').convert('RGB')
    thermal = Image.open(R / f'thermal/{split}/{stem}.jpg')
    gt = load_gt(stem, split)


    rgb = upload(R / f'rgb/{split}/{stem}.jpg', "rgb")
    th  = upload(R / f'thermal/{split}/{stem}.jpg', "thermal")

    messages=[{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Image 1 — RGB:'},
            {'type': 'image', 
            'source': {
                'type': 'base64', 
                'media_type': 'image/jpeg', 
                'data': encode_raw(R / f'rgb/{split}/{stem}.jpg')}
            },
            {'type': 'text', 'text': 'Image 2 — Thermal (LWIR)'},
            {'type': 'image', 'source': {
                'type': 'base64', 
                'media_type': 'image/jpeg', 
                'data': encode_raw(R / f'thermal/{split}/{stem}.jpg')}
            },
            {"type": "container_upload", "file_id": rgb.id},
            {"type": "container_upload", "file_id": th.id},
            {'type': 'text', 'text': PROMPT},
        ],
    }]


    run(client, messages)

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
        output_config={"effort": "high"},
        output_format=DetectionResult,
        messages=messages,
        betas=BETAS
    )

    dets = r.parsed_output.detections
    pred = [(d.label, *d.box) for d in dets]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(draw_boxes(img, gt));   ax[0].set_title(f'GT ({len(gt)})');   ax[0].axis('off')
    ax[1].imshow(draw_boxes(img, pred)); ax[1].set_title(f'Claude ({len(pred)})'); ax[1].axis('off')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()