# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Load a parquet dataset into a pandas DataFrame. Supports
#               random_programs datasets (adds derived columns for output
#               length and UNK stats) and OEIS sequences
# -----------------------------------------------------------------------------

import os, re, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pandas as pd

from befunge import ALPHABET

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

def _resolve(subdir, default_name, path):
    """If `path` is None use the default; if it's a bare filename, look under data/<subdir>/."""
    if path is None:
        path = default_name
    if os.sep not in path and '/' not in path:
        path = os.path.join(_HERE, subdir, path)
    return path

def load_programs(path=None):
    path = _resolve('random_programs', 'dataset.parquet', path)
    df = pd.read_parquet(path)
    df['output_raw'] = df['output'].map(unsanitize)
    df['output_len'] = df['output_raw'].str.len()
    df['unk_count'] = df['output_raw'].map(
        lambda s: sum(1 for c in s if c not in ALPHABET_SET))
    df['unk_frac'] = (df['unk_count'] / df['output_len'].clip(lower=1)).where(
        df['output_len'] > 0, 0.0)
    return df

def to_clipboard(text):
    """Copy `text` to the system clipboard. Run the GUI separately and Cmd+V
    to paste — handy for inspecting a program from a DataFrame."""
    import platform, subprocess
    sysname = platform.system()
    if sysname == 'Darwin':
        cmd = ['pbcopy']
    elif sysname == 'Linux':
        cmd = ['xclip', '-selection', 'clipboard']
    else:
        cmd = ['clip']
    subprocess.run(cmd, input=text, text=True, check=True)
    print(text)


def load_oeis(path=None, parse_ints=True):
    df = pd.read_parquet(_resolve('oeis', 'oeis.parquet', path))
    if parse_ints:
        # OEIS terms can exceed int64 (factorials, etc.), so use Python ints.
        df['sequence'] = df['sequence'].map(lambda s: tuple(int(x) for x in s))
    return df
