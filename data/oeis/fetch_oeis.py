# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Fetch OEIS sequences and write them to oeis.parquet
# -----------------------------------------------------------------------------

"""Fetch OEIS stripped.gz + names.gz and write an oeis.parquet of (number, name, sequence).

Run:
    python data/oeis/fetch_oeis.py

Output: data/oeis/oeis.parquet with columns
    number   int64        e.g. 1 for A000001
    name     str          short description from names.gz
    sequence list[str]    sequence terms (strings, since OEIS terms can exceed int64)
"""
import gzip
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STRIPPED_URL = 'https://oeis.org/stripped.gz'
NAMES_URL = 'https://oeis.org/names.gz'
STRIPPED_PATH = os.path.join(HERE, 'stripped.gz')
NAMES_PATH = os.path.join(HERE, 'names.gz')
OUT_PATH = os.path.join(HERE, 'oeis.parquet')


def fetch(url, path):
    if os.path.exists(path):
        print(f'  using cached {path}')
        return
    print(f'  downloading {url} -> {path}')
    req = urllib.request.Request(url, headers={'User-Agent': 'befunge-oeis-fetch/1.0'})
    with urllib.request.urlopen(req) as r, open(path, 'wb') as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)


def parse_stripped(path):
    """Yield (number, [terms...]) from stripped.gz. Lines look like:
        A000001 ,0,1,1,1,2,1,2,1,5,...,
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            aid, _, rest = line.partition(' ')
            terms = [t for t in rest.strip().strip(',').split(',') if t]
            yield int(aid[1:]), terms


def parse_names(path):
    """Yield (number, name) from names.gz. Lines look like:
        A000001 Number of groups of order n.
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            aid, _, name = line.partition(' ')
            yield int(aid[1:]), name


def main():
    print('fetching:')
    fetch(STRIPPED_URL, STRIPPED_PATH)
    fetch(NAMES_URL, NAMES_PATH)

    print('parsing names...')
    names = dict(parse_names(NAMES_PATH))
    print(f'  {len(names)} names')

    print('parsing sequences...')
    rows = []
    for number, terms in parse_stripped(STRIPPED_PATH):
        rows.append((number, names.get(number, ''), terms))
    print(f'  {len(rows)} sequences')

    df = pd.DataFrame(rows, columns=['number', 'name', 'sequence'])
    df.to_parquet(OUT_PATH, index=False)
    print(f'wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
