# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Generate random Befunge programs and save to programs.parquet
# -----------------------------------------------------------------------------

import argparse, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from befunge import INSTRUCTIONS

# Drop interactive-input opcodes (block on stdin) and `?` (nondeterministic).
CHARS       = ''.join(c for c in INSTRUCTIONS if c not in '&~?') + ' '
CHARS_BYTES = np.frombuffer(CHARS.encode('ascii'), dtype=np.uint8)
N_CHARS     = len(CHARS)
_SPACE = np.uint8(ord(' '))
_AT    = np.uint8(ord('@'))
_NL    = np.uint8(ord('\n'))

def generate_batch(seed, start, count, w=80, h=25, density=0.7):
    """Generate `count` programs in one vectorized numpy call. Programs are
    indexed start..start+count-1, but they share entropy from a single batch
    RNG — i.e. no longer reproducible by (seed, idx) alone, only by the
    full (seed, batch_start, batch_count) tuple."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, start]))
    mask = rng.random((count, h, w)) < density
    idx  = rng.integers(0, N_CHARS, size=(count, h, w))
    # Work as a uint8 grid throughout — much faster string assembly via
    # `tobytes().decode()` than per-program Python `''.join` loops.
    grid = np.where(mask, CHARS_BYTES[idx], _SPACE).astype(np.uint8)
    rows = rng.integers(0, h, size=count)
    cols = rng.integers(0, w, size=count)
    grid[np.arange(count), rows, cols] = _AT
    # Append a newline column so each program-row ends with '\n', then flatten.
    nl_col = np.full((count, h, 1), _NL, dtype=np.uint8)
    flat = np.concatenate([grid, nl_col], axis=2).reshape(count, h * (w + 1))
    return [flat[i].tobytes().decode('ascii').rstrip('\n') for i in range(count)]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('count', type=int, nargs='?', default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=os.path.join(_HERE, 'programs.parquet'))
    p.add_argument('--batch-size', type=int, default=50000)
    args = p.parse_args()

    writer = None
    for batch_start in range(0, args.count, args.batch_size):
        batch_count = min(args.batch_size, args.count - batch_start)
        programs = generate_batch(args.seed, batch_start, batch_count)
        batch = [{'index': batch_start + j, 'seed': args.seed, 'program': prog}
                 for j, prog in enumerate(programs)]
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression='zstd')
        writer.write_table(table)
        print(f'  [{batch_start + batch_count}/{args.count}]')
    if writer:
        writer.close()
    print(f'wrote {args.count} programs to {args.out}')
