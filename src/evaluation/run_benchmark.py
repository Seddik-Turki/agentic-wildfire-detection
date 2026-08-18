"""Batch runner for both Claude methods, with resume and inline scoring.

    python -m src.run_benchmark agentic --videos video2 --limit 5 --workers 4
    python -m src.run_benchmark agentic --videos video2 --workers 32
    python -m src.run_benchmark basic   --videos video2 --workers 32
    python -m src.run_benchmark agentic --score-only

One JSON per frame under results/<method>/frames/, so a crash at frame 600
doesn't cost the first 600 and re-scoring never means re-paying for inference.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.data_utils import load_frames, load_gt
from src.evaluation import cost
from src.evaluation.metrics import report, score

load_dotenv()


def get_detect(method):
    """-> fn(client, stem, split) -> (detections, turns)"""
    if method == 'agentic':
        from src.agentic.utils import run

        def f(client, stem, split):
            return run(client, stem, split, verbose=False)
        return f

    from src.basic_prompting.utils import detect as basic_detect

    def f(client, stem, split):
        dets, r = basic_detect(client, stem, split)
        turns = [dict(
            i=0, stop=r.stop_reason,
            inp=r.usage.input_tokens, out=r.usage.output_tokens,
            cache_read=getattr(r.usage, 'cache_read_input_tokens', 0) or 0,
            cache_write=getattr(r.usage, 'cache_creation_input_tokens', 0) or 0,
            bash=0, attach=0,
        )]
        return dets, turns
    return f


def ramp(sem, n, per_sec):
    """Release workers gradually. Rate limits are enforced sub-minute, so a
    simultaneous launch trips 429s even when the per-minute average is fine."""
    for _ in range(n):
        sem.release()
        time.sleep(1.0 / per_sec)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', choices=['agentic', 'basic'])
    ap.add_argument('--videos', nargs='+', default=['video2'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--ramp', type=float, default=4.0,
                    help='workers released per second at startup')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--score-only', action='store_true',
                    help='skip inference, score what is already on disk')
    return ap.parse_args()


def run_inference(args, out):
    vids = [int(v.replace('video', '')) for v in args.videos]
    frames = load_frames(vids, verbose=False)
    sample = [f for v in vids for f in frames[f'video{v}']][::args.stride]
    if args.limit:
        sample = sample[:args.limit]

    todo = [(stem, split) for _, stem, split in sample
            if not (out / 'frames' / f'{stem}.json').exists()]
    print(f'{len(sample)} frames, {len(todo)} to do, {args.workers} workers')
    if not todo:
        return

    detect = get_detect(args.method)
    client = anthropic.Anthropic(max_retries=8)

    sem = threading.Semaphore(0)
    threading.Thread(target=ramp, args=(sem, args.workers, args.ramp),
                     daemon=True).start()

    def work(stem, split):
        sem.acquire()
        try:
            dets, turns = detect(client, stem, split)
            rec = dict(stem=stem, split=split, turns=turns,
                       dets=[d.model_dump() for d in dets])
            (out / 'frames' / f'{stem}.json').write_text(json.dumps(rec))
        finally:
            sem.release()

    ok, fail, t0 = 0, {}, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s, sp): s for s, sp in todo}
        pbar = tqdm(as_completed(futs), total=len(futs))
        for fut in pbar:
            stem = futs[fut]
            try:
                fut.result()
                ok += 1
            except Exception as e:
                fail[stem] = f'{type(e).__name__}: {e}'
                pbar.write(f'[{stem}] {type(e).__name__}: {e}')
            pbar.set_postfix(ok=ok, fail=len(fail))

    print(f'\n{ok} ok, {len(fail)} failed in {(time.time() - t0) / 60:.1f} min')
    if fail:
        (out / 'failures.json').write_text(json.dumps(fail, indent=2))


def load_run(out):
    results, meta, turnlogs = {}, {}, {}
    for f in sorted((out / 'frames').glob('*.json')):
        rec = json.loads(f.read_text())
        results[rec['stem']] = rec['dets']
        meta[rec['stem']] = (rec['stem'], rec['split'])
        turnlogs[rec['stem']] = rec['turns']
    return results, meta, turnlogs


def main():
    args = parse_args()
    out = args.out or Path(f'results/{args.method}')
    (out / 'frames').mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        run_inference(args, out)

    results, meta, turnlogs = load_run(out)
    if not results:
        print('nothing to score')
        return

    print()
    m = score(results, meta, load_gt)
    report(m, args.method)
    print()
    t = cost.tally(turnlogs)
    cost.report(t, args.method)

    (out / 'summary.json').write_text(
        json.dumps(dict(metrics=m, cost=t), indent=2))
    print(f'\nwrote {out}/summary.json')


if __name__ == '__main__':
    main()