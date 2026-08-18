import random
from PIL import Image
import matplotlib.pyplot as plt

from src.data_utils import load_frames, load_gt, draw_boxes
from src.agentic.utils import run
from src.data_utils import R


from dotenv import load_dotenv
load_dotenv()

import anthropic
client = anthropic.Anthropic()


def main():
    frames = load_frames()
    vid_id = random.choice([2,3,4,5])
    _, stem, split = random.choice(frames[f'video{vid_id}'])
    img = Image.open(R / f'rgb/{split}/{stem}.jpg').convert('RGB')
    thermal = Image.open(R / f'thermal/{split}/{stem}.jpg')
    gt = load_gt(stem, split)


    
    dets, _ = run(client, stem, split)

    
    pred = [(d.label, *d.box) for d in dets]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(draw_boxes(img, gt));   ax[0].set_title(f'GT ({len(gt)})');   ax[0].axis('off')
    ax[1].imshow(draw_boxes(img, pred)); ax[1].set_title(f'Claude ({len(pred)})'); ax[1].axis('off')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()