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

# ----- defaults (override via CLI) -------------------------------------------
DEFAULT_COUNT      = 10
DEFAULT_SEED       = 0
DEFAULT_OUT        = os.path.join(_HERE, 'programs.parquet')
DEFAULT_BATCH_SIZE = 50000
DEFAULT_DENSITY    = 1.0
DEFAULT_NO_HALT    = False
DEFAULT_LOCAL_ONLY = False  # if True, also drops `g`/`p` (nonlocal mem ops)
DEFAULT_START      = 0      # first program index — bump to append to a prior run

# drop interactive-input opcodes (block on stdin) and `?` (nondeterministic)
CHARS = ''.join(c for c in INSTRUCTIONS if c not in '&~?') + ' '
_SPACE = np.uint8(ord(' '))
_AT    = np.uint8(ord('@'))
_NL    = np.uint8(ord('\n'))

def _char_pool(allow_halt, allow_mem):
    """Build the uint8 char pool with the requested instructions removed."""
    pool = CHARS
    if not allow_halt:
        pool = pool.replace('@', '')
    if not allow_mem:
        pool = pool.replace('g', '').replace('p', '')
    return np.frombuffer(pool.encode('ascii'), dtype=np.uint8)

def generate_batch(seed, start, count, w=80, h=25, density=1.0,
                   allow_halt=True, allow_mem=True):
    """Generate `count` programs in one vectorized numpy call. Programs are
    indexed start..start+count-1, but they share entropy from a single batch
    RNG — i.e. no longer reproducible by (seed, idx) alone, only by the
    full (seed, batch_start, batch_count) tuple.

    `allow_halt=False` removes `@` from the character pool and skips planting
    a guaranteed `@` cell — every generated program will run until it hits
    the step limit.

    `allow_mem=False` removes `g` and `p` (the nonlocal read/write opcodes)
    so all execution is purely local along the IP path."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, start]))
    chars = _char_pool(allow_halt, allow_mem)
    mask = rng.random((count, h, w)) < density
    idx  = rng.integers(0, len(chars), size=(count, h, w))
    # work as a uint8 grid throughout — much faster string assembly via
    # `tobytes().decode()` than per-program Python `''.join` loops
    grid = np.where(mask, chars[idx], _SPACE).astype(np.uint8)
    if allow_halt:
        # guarantee at least one `@` per program so some fraction actually halts
        rows = rng.integers(0, h, size=count)
        cols = rng.integers(0, w, size=count)
        grid[np.arange(count), rows, cols] = _AT
    # append a newline column so each program-row ends with '\n', then flatten
    nl_col = np.full((count, h, 1), _NL, dtype=np.uint8)
    flat = np.concatenate([grid, nl_col], axis=2).reshape(count, h * (w + 1))
    return [flat[i].tobytes().decode('ascii').rstrip('\n') for i in range(count)]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('count',        type=int, nargs='?', default=DEFAULT_COUNT)
    p.add_argument('--seed',       type=int,            default=DEFAULT_SEED)
    p.add_argument('--out',                             default=DEFAULT_OUT)
    p.add_argument('--batch-size', type=int,            default=DEFAULT_BATCH_SIZE)
    p.add_argument('--density',    type=float,          default=DEFAULT_DENSITY,
                   help='per-cell probability of picking a random char from '
                        'CHARS; cells that don\'t pick stay as space. Default '
                        '1.0 means every cell is randomized (CHARS still '
                        'includes a literal space, so ~3%% will be space).')
    p.add_argument('--no-halt', action='store_true', default=DEFAULT_NO_HALT,
                   help='exclude `@` from the char pool — every program will '
                        'run until the step limit')
    p.add_argument('--local', action='store_true', default=DEFAULT_LOCAL_ONLY,
                   help='also exclude `g` and `p` (nonlocal mem ops) so '
                        'execution stays on the IP path')
    p.add_argument('--start', type=int, default=DEFAULT_START,
                   help='first program index. Programs are deterministic in '
                        '(seed, batch_start), so e.g. `--start 1000000` '
                        'produces fresh programs that pick up where a prior '
                        '`--count 1000000 --start 0` left off')
    args = p.parse_args()

    writer = None
    end = args.start + args.count
    for batch_start in range(args.start, end, args.batch_size):
        batch_count = min(args.batch_size, end - batch_start)
        programs = generate_batch(args.seed, batch_start, batch_count,
                                  density=args.density,
                                  allow_halt=not args.no_halt,
                                  allow_mem=not args.local)
        batch = [{'index': batch_start + j, 'seed': args.seed, 'program': prog}
                 for j, prog in enumerate(programs)]
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression='zstd')
        writer.write_table(table)
        print(f'  [{batch_start + batch_count - args.start}/{args.count}]')
    if writer:
        writer.close()
    print(f'wrote {args.count} programs to {args.out} '
          f'(indices {args.start}..{end - 1})')
