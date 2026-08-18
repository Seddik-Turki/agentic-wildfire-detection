"""Single-shot baseline: one request, both modalities, structured output.

    python -m src.basic_prompting.main
"""
import random
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
import anthropic

from src.data_utils import load_frames, load_gt, draw_boxes, R
from src.basic_prompting.utils import detect, trace

load_dotenv()
client = anthropic.Anthropic()

def main():
    frames = load_frames()
    _, stem, split = random.choice(frames[f'video{random.choice([2, 3, 4, 5])}'])
    print(f"frame: {stem} ({split})")

    dets, resp = detect(client, stem, split)
    trace(resp)
    print('tokens:', resp.usage.input_tokens, '->', resp.usage.output_tokens)
    for d in dets:
        print(f"  {d.label:7} {d.confidence:.2f}  {d.box}")

    img = Image.open(R / f'rgb/{split}/{stem}.jpg').convert('RGB')
    gt = load_gt(stem, split)
    pred = [(d.label, *d.box) for d in dets]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(draw_boxes(img, gt));   ax[0].set_title(f'GT ({len(gt)})');       ax[0].axis('off')
    ax[1].imshow(draw_boxes(img, pred)); ax[1].set_title(f'Claude ({len(pred)})'); ax[1].axis('off')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()