from pathlib import Path
import random

from PIL import Image
import matplotlib.pyplot as plt

from data_utils import load_frames, load_gt, draw_boxes
from schema import DetectionResult

from basic_prompting.utils import encode_raw
from basic_prompting.prompt import PROMPT
from basic_prompting.constants import MODEL, MAX_TOKENS

R = Path(__file__).parents[2] / "rgbt-3m"

from dotenv import load_dotenv

load_dotenv()

import anthropic

client = anthropic.Anthropic()


def detect(split, stem):
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
                {'type': 'text', 'text': PROMPT},
            ],
        }],
    )
    return r.parsed_output.detections, r


def main():
    frames = load_frames()
    vid_id = random.choice([2,3,4,5])
    _, stem, split = random.choice(frames[f'video{vid_id}'])
    img = Image.open(R / f'rgb/{split}/{stem}.jpg').convert('RGB')
    # thermal = Image.open(R / f'thermal/{split}/{stem}.jpg')
    gt = load_gt(stem, split)


    dets, resp = detect(split, stem)
    print(dets)
    print('tokens:', resp.usage.input_tokens, '->', resp.usage.output_tokens)

    pred = [(d.label, *d.box) for d in dets]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(draw_boxes(img, gt));   ax[0].set_title(f'GT ({len(gt)})');   ax[0].axis('off')
    ax[1].imshow(draw_boxes(img, pred)); ax[1].set_title(f'Claude ({len(pred)})'); ax[1].axis('off')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()