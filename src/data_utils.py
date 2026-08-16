from pathlib import Path
import re

from PIL import ImageDraw
from tqdm.auto import tqdm

from constants import NAMES, W, H

R = Path(__file__).parent / "rgbt-3m"
COLORS = {'smoke': (128, 200, 255), 'fire': (255, 80, 0), 'person': (0, 255, 120)}

def load_frames():
    frames = dict()

    for vid_id in [2,3,4,5]:
        VID = f'video{vid_id}'
        frames[VID] = []

    for vid_id in tqdm([2,3,4,5]):
        VID = f'video{vid_id}'
        
        for split in tqdm(['train', 'test'], leave=False):
            for p in tqdm((R/f'labels/{split}').glob(f'{VID}_frame_*.txt'), leave=False):
                idx = int(re.search(r'frame_(\d+)', p.stem).group(1))
                frames[VID].append((idx, p.stem, split))

        print(f"[{VID}] {len(frames[VID])}")


def load_gt(stem, split):
    """YOLO normalized cxcywh -> absolute xyxy."""
    out = []
    for line in (R / f'labels/{split}/{stem}.txt').read_text().split('\n'):
        t = line.split()
        if len(t) == 5:
            c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
            out.append((NAMES[c], (cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H))
    return out

def draw_boxes(img, boxes, width=2):
    """boxes: list of (label, x1, y1, x2, y2). Returns a copy."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for label, x1, y1, x2, y2 in boxes:
        col = COLORS.get(label, (255, 255, 0))
        d.rectangle([x1, y1, x2, y2], outline=col, width=width)
        d.text((x1 + 3, max(y1 - 12, 0)), label, fill=col)
    return out