# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Filter the dataset to records whose outputs are entirely in
#               the model vocab
# -----------------------------------------------------------------------------

import argparse, os, re, sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root

import pyarrow as pa
import pyarrow.parquet as pq

from befunge import ALPHABET

# ----- defaults (override via CLI) -------------------------------------------
DEFAULT_IN            = os.path.join(_HERE, 'dataset.parquet')
DEFAULT_OUT           = os.path.join(_HERE, 'dataset_clean.parquet')
DEFAULT_MAX_UNK_FRAC  = 0.0
DEFAULT_BATCH_SIZE    = 50000

ALPHABET_SET = set(ALPHABET)
SANITIZE_RE = re.compile(r'\\x([0-9a-fA-F]{2})')

def unsanitize(s):
    out = []
    i = 0
    while i < len(s):
        m = SANITIZE_RE.match(s, i)
        if m:
            out.append(chr(int(m.group(1), 16)))
            i = m.end()
        else:
            out.append(s[i]); i += 1
    return ''.join(out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default=DEFAULT_IN)
    p.add_argument('--out',                default=DEFAULT_OUT)
    p.add_argument('--max-unk-frac', type=float, default=DEFAULT_MAX_UNK_FRAC)
    p.add_argument('--batch-size',   type=int,   default=DEFAULT_BATCH_SIZE)
    args = p.parse_args()

    kept = dropped = 0
    by_status = Counter()
    drop_status = Counter()
    pf = pq.ParquetFile(args.in_path)
    writer = None
    batch = []

    def flush():
        global writer
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression='zstd')
        writer.write_table(table)
        batch.clear()

    for rg in range(pf.num_row_groups):
        for rec in pf.read_row_group(rg).to_pylist():
            raw = unsanitize(rec['output'])
            unk = sum(1 for c in raw if c not in ALPHABET_SET)
            frac = unk / len(raw) if raw else 0.0
            if frac > args.max_unk_frac:
                dropped += 1
                drop_status[rec.get('status', '?')] += 1
            else:
                kept += 1
                by_status[rec.get('status', '?')] += 1
                batch.append(rec)
                if len(batch) >= args.batch_size:
                    flush()
    flush()
    if writer:
        writer.close()

    total = kept + dropped
    print(f'kept {kept}/{total} ({100*kept/total:.1f}%)  ->  {args.out}')
    print(f'dropped {dropped}/{total} ({100*dropped/total:.1f}%)')
    print(f'kept by status:    {dict(by_status)}')
    print(f'dropped by status: {dict(drop_status)}')
