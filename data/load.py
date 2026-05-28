# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Load a parquet dataset into a pandas DataFrame. Supports
#               random_programs datasets (adds raw + length derived columns)
#               and OEIS sequences
# -----------------------------------------------------------------------------

import os, re, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

def load_programs(path=None, n=None):
    """Load the random-programs dataset.

    `n`: optional row limit. With `n` set, reads only the first n rows from
    the parquet via streaming row-group iteration (way faster than loading
    the whole file and slicing)."""
    path = _resolve('random_programs', 'dataset.parquet', path)
    if n is None:
        df = pd.read_parquet(path)
    else:
        pf = pq.ParquetFile(path)
        chunks, rows = [], 0
        for batch in pf.iter_batches(batch_size=min(n, 50000)):
            chunks.append(batch)
            rows += batch.num_rows
            if rows >= n:
                break
        df = pa.Table.from_batches(chunks).to_pandas().iloc[:n].copy()
    # Older datasets carry both `program` (raw) and `pruned_program`. The
    # raw form has cells the interpreter never touched, so we prefer the
    # pruned one and drop the raw column.
    if 'pruned_program' in df.columns:
        df['program'] = df['pruned_program']
        df = df.drop(columns=['pruned_program'])
    # Count "active" (non-space, non-newline) chars via two replaces + len.
    # Faster on ~1M rows than separate str.count calls.
    sizes = (df['program'].str.replace(' ', '', regex=False)
                          .str.replace('\n', '', regex=False)
                          .str.len())
    df.insert(df.columns.get_loc('program') + 1, 'program_size', sizes)
    # Output length in actual bytes: each `\xNN` escape (4 chars) was 1 byte,
    # so subtract 3 chars per escape from the sanitized length.
    out_sizes = (df['output'].str.len()
                 - 3 * df['output'].str.count(r'\\x[0-9a-fA-F]{2}'))
    df.insert(df.columns.get_loc('output') + 1, 'output_size', out_sizes)
    return df

def to_clipboard(text):
    """Copy `text` to the system clipboard and print it. Run the GUI separately
    and Cmd+V to paste — handy for inspecting a program from a DataFrame.

    If `text` looks like a Python repr of a string (since `load_programs`
    repr-wraps the program column for display), it's auto-unwrapped first."""
    import ast, platform, subprocess
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        try:
            text = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
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
