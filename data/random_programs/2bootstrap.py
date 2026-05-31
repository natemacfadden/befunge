# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Add one more "round" of training data. Each round is 1M fresh
#               programs (--no-halt --local), run through the interpreter and
#               filtered to a uniques threshold. Round indices are deterministic
#               in the program seed range so different rounds never overlap
# -----------------------------------------------------------------------------

import argparse, glob, os, re, subprocess, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----- defaults (override via CLI) -------------------------------------------
DEFAULT_ROUND_SIZE  = 1_000_000  # programs per round
DEFAULT_MIN_UNIQUES = 10         # aggressive filter
DEFAULT_SEED        = 0          # shared across all rounds — what varies is `start`
DEFAULT_KEEP_FULL   = True       # keep the unfiltered run for later threshold revisits
DEFAULT_KEEP_PROGS  = True       # keep the raw programs file (large but lets you re-run
                                 # the interpreter with different settings without regenerating)

ROUND_RE = re.compile(r'dataset_agg_round(\d+)\.parquet$')

def next_round_idx(dirpath):
    """Smallest non-negative int not already present as dataset_agg_round{N}.parquet."""
    seen = set()
    for f in glob.glob(os.path.join(dirpath, 'dataset_agg_round*.parquet')):
        m = ROUND_RE.search(os.path.basename(f))
        if m:
            seen.add(int(m.group(1)))
    i = 0
    while i in seen:
        i += 1
    return i

def filter_aggressive(in_path, out_path, min_uniques):
    pf = pq.ParquetFile(in_path)
    writer = None
    kept = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        sub = tbl.filter(pc.greater_equal(tbl.column('output_uniques'), min_uniques))
        if sub.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(out_path, sub.schema, compression='zstd')
        writer.write_table(sub)
        kept += sub.num_rows
    if writer:
        writer.close()
    return kept

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round-size',  type=int, default=DEFAULT_ROUND_SIZE)
    p.add_argument('--min-uniques', type=int, default=DEFAULT_MIN_UNIQUES,
                   help='filter threshold — keep rows with output_uniques >= this')
    p.add_argument('--seed',        type=int, default=DEFAULT_SEED)
    p.add_argument('--round',       type=int, default=None,
                   help='explicit round index (default: next unused)')
    p.add_argument('--no-keep-full', dest='keep_full', action='store_false',
                   help='delete the unfiltered run after filtering '
                        '(default: keep for later threshold revisits)')
    p.add_argument('--no-keep-programs', dest='keep_programs', action='store_false',
                   help='delete the raw programs file after the interpreter run '
                        '(default: keep — file is large but lets you re-run with '
                        'different interpreter settings without regenerating)')
    p.set_defaults(keep_full=DEFAULT_KEEP_FULL, keep_programs=DEFAULT_KEEP_PROGS)
    args = p.parse_args()

    idx   = args.round if args.round is not None else next_round_idx(_HERE)
    start = idx * args.round_size
    progs = os.path.join(_HERE, f'programs_round{idx}.parquet')
    full  = os.path.join(_HERE, f'dataset_round{idx}.parquet')
    agg   = os.path.join(_HERE, f'dataset_agg_round{idx}.parquet')

    print(f'=== round {idx}  (seed={args.seed}, start={start}, count={args.round_size}) ===')

    if not os.path.exists(full):
        # generate
        subprocess.run([sys.executable, os.path.join(_HERE, '0generate.py'),
                        str(args.round_size),
                        '--seed',  str(args.seed),
                        '--start', str(start),
                        '--no-halt', '--local',
                        '--out',   progs], check=True)
        # run
        subprocess.run([sys.executable, os.path.join(_HERE, '1run.py'),
                        '--in',  progs,
                        '--out', full], check=True)
        if not args.keep_programs:
            # the raw-programs file is large (~1.2 GB) but reconstructable from (seed, start, count)
            os.remove(progs)
            print(f'deleted {progs}')
    else:
        print(f'reusing existing {full}')

    # filter
    kept = filter_aggressive(full, agg, args.min_uniques)
    print(f'wrote {kept} rows (uniques >= {args.min_uniques}) to {agg}')

    if not args.keep_full:
        os.remove(full)
        print(f'deleted {full}')
