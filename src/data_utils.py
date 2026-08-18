from pathlib import Path
import re
import base64

from PIL import ImageDraw

from src.constants import NAMES, W, H


R = Path(__file__).parents[1] / "rgbt-3m"

COLORS = {'smoke': (128, 200, 255), 'fire': (255, 80, 0), 'person': (0, 255, 120)}
VIDEOS = [2, 3, 4, 5]



def encode_raw(path):
    """Send original JPEG bytes — no re-encode, no second lossy pass."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def load_frames(videos=VIDEOS, verbose=True):
    """-> {'video2': [(frame_idx, stem, split), ...], ...}, sorted by frame."""
    frames = {}
    for vid_id in videos:
        vid = f'video{vid_id}'
        found = []
        for split in ('train', 'test'):
            for p in (R / f'labels/{split}').glob(f'{vid}_frame_*.txt'):
                idx = int(re.search(r'frame_(\d+)', p.stem).group(1))
                found.append((idx, p.stem, split))
        frames[vid] = sorted(found)
        if verbose:
            print(f"[{vid}] {len(frames[vid])}")
    return frames


def load_gt(stem, split):
    """YOLO normalised cxcywh -> [(label, x1, y1, x2, y2), ...] absolute px."""
    out = []
    for line in (R / f'labels/{split}/{stem}.txt').read_text().split('\n'):
        t = line.split()
        if len(t) == 5:
            c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
            out.append((NAMES[c], (cx - w/2)*W, (cy - h/2)*H,
                        (cx + w/2)*W, (cy + h/2)*H))
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