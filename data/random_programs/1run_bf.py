# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Run programs through the interpreter and save outputs to
#               dataset.parquet
# -----------------------------------------------------------------------------

import argparse, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root

import pyarrow as pa
import pyarrow.parquet as pq

from befunge import run_traced, prune_program

# ----- defaults (override via CLI) -------------------------------------------
DEFAULT_IN             = os.path.join(_HERE, 'programs.parquet')
DEFAULT_OUT            = os.path.join(_HERE, 'dataset.parquet')
DEFAULT_MAX_STEPS      = 20000
DEFAULT_MAX_OUTPUT     = 4096
DEFAULT_WORKERS        = os.cpu_count()
DEFAULT_PROGRESS_EVERY = 10000
DEFAULT_BATCH_SIZE     = 50000
DEFAULT_JIT            = True

def sanitize(s):
    out = []
    for c in s:
        o = ord(c)
        if c in '\n\t' or 32 <= o < 127:
            out.append(c)
        else:
            out.append(f'\\x{o:02x}')
    return ''.join(out)

def process_record(args_tuple):
    rec, max_steps, max_output, jit = args_tuple
    program = rec['program']
    try:
        status, raw_str, visited, final_stack = run_traced(program, max_steps=max_steps, jit=jit)
    except Exception:
        status, raw_str, visited, final_stack = 'error', '', None, []
    raw_str = raw_str[:max_output]
    rec_out = dict(rec)
    # Only prune when the program halted naturally — then `visited` is the
    # complete set of cells the program ever needed and the pruned version
    # is exactly equivalent. For `step_limit` or `error`, we don't know
    # what cells would have mattered past the truncation point, so we
    # leave the program untouched.
    if status == 'ok' and visited is not None:
        rec_out['program'] = prune_program(program, visited)
    else:
        rec_out['program'] = program
    rec_out['output'] = sanitize(raw_str)
    rec_out['status'] = status
    rec_out['final_stack'] = final_stack
    return rec_out

def iter_programs(path):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        for rec in tbl.to_pylist():
            yield rec

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default=DEFAULT_IN)
    p.add_argument('--out',                 default=DEFAULT_OUT)
    p.add_argument('--max-steps',      type=int,   default=DEFAULT_MAX_STEPS)
    p.add_argument('--max-output',     type=int,   default=DEFAULT_MAX_OUTPUT)
    p.add_argument('--workers',        type=int,   default=DEFAULT_WORKERS)
    p.add_argument('--progress-every', type=int,   default=DEFAULT_PROGRESS_EVERY)
    p.add_argument('--batch-size',     type=int,   default=DEFAULT_BATCH_SIZE,
                   help='records per parquet row-group write')
    p.add_argument('--no-jit', dest='jit', action='store_false',
                   help='disable the numba-JIT interpreter (default: on)')
    p.set_defaults(jit=DEFAULT_JIT)
    args = p.parse_args()

    pf = pq.ParquetFile(args.in_path)
    total_records = pf.metadata.num_rows
    print(f'{total_records} programs from {args.in_path}, {args.workers} workers')

    counts = {'ok': 0, 'error': 0, 'step_limit': 0}
    seen = set()  # exact program strings we've already written
    dropped = 0
    t0 = time.time()

    work_iter = ((rec, args.max_steps, args.max_output, args.jit)
                 for rec in iter_programs(args.in_path))

    writer = None
    batch = []

    def flush(batch):
        global writer
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression='zstd')
        writer.write_table(table)

    with ProcessPoolExecutor(args.workers) as pool:
        for i, rec_out in enumerate(
                pool.map(process_record, work_iter, chunksize=64), 1):
            # Dedup by the (post-pruning) program text. After pruning, many
            # source programs collapse to the same minimal form — keep one.
            if rec_out['program'] in seen:
                dropped += 1
            else:
                seen.add(rec_out['program'])
                batch.append(rec_out)
                counts[rec_out['status']] += 1
            if i % args.progress_every == 0:
                rate = i / (time.time() - t0)
                eta = (total_records - i) / rate
                print(f'  [{i}/{total_records}]  {rate:.0f} rec/s  eta {eta:.0f}s, '
                      f'{dropped} dup')
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []

    flush(batch)
    if writer:
        writer.close()

    dt = time.time() - t0
    kept = total_records - dropped
    print(f'\nwrote {args.out}: {counts}  '
          f'({kept} unique kept, {dropped} dups dropped, '
          f'{dt:.1f}s, {total_records/max(dt,1e-9):.0f} rec/s)')
