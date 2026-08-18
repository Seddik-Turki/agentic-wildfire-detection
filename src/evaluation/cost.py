"""Token and cost accounting. Opus 5 pricing, USD per million tokens."""

IN, CACHE_WRITE, CACHE_READ, OUT = 5.0, 6.25, 0.50, 25.0


def tally(turnlogs):
    """turnlogs: {key: [turn dicts]} -> totals and per-frame cost."""
    t = dict(frames=len(turnlogs), turns=0, inp=0, out=0,
             cache_read=0, cache_write=0, bash=0, attach=0)
    for turns in turnlogs.values():
        t['turns'] += len([x for x in turns if x.get('i') != 'commit'])
        for x in turns:
            for k in ('inp', 'out', 'cache_read', 'cache_write', 'bash', 'attach'):
                t[k] += x.get(k, 0)

    # input_tokens ALREADY excludes cache reads — do not subtract them
    t['cost'] = (t['inp'] / 1e6 * IN + t['cache_read'] / 1e6 * CACHE_READ
                 + t['cache_write'] / 1e6 * CACHE_WRITE + t['out'] / 1e6 * OUT)
    n = max(t['frames'], 1)
    t['cost_per_frame'] = t['cost'] / n
    t['turns_per_frame'] = t['turns'] / n
    return t


def report(t, label=''):
    print(f"{label}  {t['frames']} frames")
    print(f"  in={t['inp']:,}  out={t['out']:,}  "
          f"cache_read={t['cache_read']:,}  cache_write={t['cache_write']:,}")
    print(f"  {t['turns_per_frame']:.1f} turns/frame, "
          f"{t['bash']/max(t['frames'],1):.1f} bash, "
          f"{t['attach']/max(t['frames'],1):.1f} attach")
    print(f"  ${t['cost']:.2f} total, ${t['cost_per_frame']:.2f}/frame")