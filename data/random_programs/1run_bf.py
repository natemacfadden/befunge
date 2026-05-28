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
        status, raw_str, visited = run_traced(program, max_steps=max_steps, jit=jit)
    except Exception:
        status, raw_str, visited = 'error', '', None
    raw_str = raw_str[:max_output]
    rec_out = dict(rec)
    # Replace `program` with the pruned version — cells that the IP never
    # visited (and `g` never read) become spaces. We drop the un-pruned
    # original since it carries no information the interpreter ever used.
    rec_out['program'] = prune_program(program, visited) if visited is not None else program
    rec_out['output'] = sanitize(raw_str)
    rec_out['status'] = status
    return rec_out

def iter_programs(path):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        for rec in tbl.to_pylist():
            yield rec

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default=os.path.join(_HERE, 'programs.parquet'))
    p.add_argument('--out', default=os.path.join(_HERE, 'dataset.parquet'))
    p.add_argument('--max-steps', type=int, default=20000)
    p.add_argument('--max-output', type=int, default=4096)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--progress-every', type=int, default=10000)
    p.add_argument('--jit', action='store_true', help='use numba-JIT interpreter')
    p.add_argument('--batch-size', type=int, default=50000,
                   help='records per parquet row-group write')
    args = p.parse_args()

    pf = pq.ParquetFile(args.in_path)
    total_records = pf.metadata.num_rows
    print(f'{total_records} programs from {args.in_path}, {args.workers} workers')

    counts = {'ok': 0, 'error': 0, 'step_limit': 0}
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
            batch.append(rec_out)
            counts[rec_out['status']] += 1
            if i % args.progress_every == 0:
                rate = i / (time.time() - t0)
                eta = (total_records - i) / rate
                print(f'  [{i}/{total_records}]  {rate:.0f} rec/s  eta {eta:.0f}s')
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []

    flush(batch)
    if writer:
        writer.close()

    dt = time.time() - t0
    print(f'\nwrote {args.out}: {counts}  ({dt:.1f}s, {total_records/max(dt,1e-9):.0f} rec/s)')
