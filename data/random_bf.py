import argparse, os, random, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pyarrow as pa
import pyarrow.parquet as pq

from interpreter.befunge import INSTRUCTIONS

# Drop interactive-input opcodes (block on stdin) and `?` (nondeterministic).
CHARS = ''.join(c for c in INSTRUCTIONS if c not in '&~?') + ' '

def generate(rng, w=80, h=25, density=0.7):
    grid = [[' '] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if rng.random() < density:
                grid[y][x] = rng.choice(CHARS)
    grid[rng.randrange(h)][rng.randrange(w)] = '@'
    return '\n'.join(''.join(row) for row in grid)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('count', type=int, nargs='?', default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=os.path.join(_HERE, 'programs.parquet'))
    p.add_argument('--batch-size', type=int, default=50000)
    args = p.parse_args()

    writer = None
    batch = []
    for i in range(args.count):
        rng = random.Random(f'{args.seed}-{i}')
        batch.append({'index': i, 'seed': args.seed, 'program': generate(rng)})
        if len(batch) >= args.batch_size:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression='zstd')
            writer.write_table(table)
            print(f'  [{i+1}/{args.count}]')
            batch = []
    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression='zstd')
        writer.write_table(table)
    if writer:
        writer.close()
    print(f'wrote {args.count} programs to {args.out}')
