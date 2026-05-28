"""Load a dataset .parquet into a pandas DataFrame with derived columns.

Usage:
    from load import load
    df = load('dataset_clean.parquet')
    df.sort_values('output_len', ascending=False).head(20)

Or run directly to drop into Python with `df` preloaded:
    python -i load.py dataset_clean.parquet
"""
import os, re, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pandas as pd

from interpreter.befunge import PLAYFIELD

PLAYFIELD_SET = set(PLAYFIELD)
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

def load(path):
    df = pd.read_parquet(path)
    df['output_raw'] = df['output'].map(unsanitize)
    df['output_len'] = df['output_raw'].str.len()
    df['unk_count'] = df['output_raw'].map(
        lambda s: sum(1 for c in s if c not in PLAYFIELD_SET))
    df['unk_frac'] = (df['unk_count'] / df['output_len'].clip(lower=1)).where(
        df['output_len'] > 0, 0.0)
    return df

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'dataset.parquet')
    df = load(path)
    print(f'loaded {len(df)} records from {path}')
    print(df.describe(include='all'))
    print('\nTop 10 longest ok:')
    print(df[df.status == 'ok']
          .sort_values('output_len', ascending=False)
          .head(10)[['index', 'output_len', 'unk_count', 'output']])
