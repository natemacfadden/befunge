import argparse, os, re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))

import pyarrow.parquet as pq

SANITIZE_RE = re.compile(r'\\x[0-9a-fA-F]{2}')

def raw_len(sanitized):
    """Length of the unescaped output in bytes (each \\xNN counts as 1)."""
    return len(SANITIZE_RE.sub('.', sanitized))

def bucket(n, bin_width):
    return (n // bin_width) * bin_width

def render(counts, bin_width, width=60):
    if not counts:
        print('(no records)')
        return
    items = sorted(counts.items())
    max_c = max(counts.values())
    for lo, n in items:
        bar = '#' * round(width * n / max_c) if max_c else ''
        print(f'  {lo:>5}-{lo+bin_width-1:<5}  {n:>7}  {bar}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default=os.path.join(_HERE, 'dataset.parquet'))
    p.add_argument('--bin', type=int, default=50, help='histogram bin width (bytes)')
    p.add_argument('--field', default='output', choices=['output', 'program'])
    p.add_argument('--by-status', action='store_true', help='separate hist per status')
    p.add_argument('--png', default=None, help='also save matplotlib plot to this path')
    p.add_argument('--log', action='store_true', help='log-scale y-axis on png')
    args = p.parse_args()

    lengths = []
    statuses = []
    pf = pq.ParquetFile(args.in_path)
    cols = [args.field, 'status'] if 'status' in pf.schema_arrow.names else [args.field]
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=cols)
        vals = tbl.column(args.field).to_pylist()
        sts = tbl.column('status').to_pylist() if 'status' in cols else ['?']*len(vals)
        for v, st in zip(vals, sts):
            n = raw_len(v) if args.field == 'output' else len(v)
            lengths.append(n)
            statuses.append(st)

    print(f'{len(lengths)} records from {args.in_path} ({args.field} length, bin={args.bin})')
    if not lengths:
        raise SystemExit
    print(f'  min={min(lengths)}  max={max(lengths)}  mean={sum(lengths)/len(lengths):.1f}')
    s = sorted(lengths)
    for q, label in [(0.5, 'p50'), (0.9, 'p90'), (0.99, 'p99'), (1.0, 'p100')]:
        print(f'  {label}={s[min(len(s)-1, int(q*len(s)))]}')
    print(f'  zero-length: {sum(1 for x in lengths if x == 0)} '
          f'({100*sum(1 for x in lengths if x == 0)/len(lengths):.1f}%)')

    print()
    if args.by_status:
        groups = {}
        for L, st in zip(lengths, statuses):
            groups.setdefault(st, []).append(L)
        for st, ls in sorted(groups.items()):
            print(f'--- {st} ({len(ls)} records) ---')
            c = Counter(bucket(L, args.bin) for L in ls)
            render(c, args.bin)
            print()
    else:
        c = Counter(bucket(L, args.bin) for L in lengths)
        render(c, args.bin)

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        bins = list(range(0, max(lengths) + args.bin, args.bin))
        fig, ax = plt.subplots(figsize=(10, 5))
        if args.by_status:
            groups = {}
            for L, st in zip(lengths, statuses):
                groups.setdefault(st, []).append(L)
            ax.hist([groups[k] for k in sorted(groups)], bins=bins,
                    stacked=True, label=sorted(groups))
            ax.legend()
        else:
            ax.hist(lengths, bins=bins)
        if args.log:
            ax.set_yscale('log')
        ax.set_xlabel(f'{args.field} length (bytes)')
        ax.set_ylabel('records')
        ax.set_title(f'{args.in_path}  ({len(lengths)} records)')
        fig.tight_layout()
        fig.savefig(args.png)
        print(f'\nsaved plot to {args.png}')
