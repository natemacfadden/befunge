"""Numba-JIT version of `befunge.run`. Reuses the dispatch loop from
`befunge._run_core` — only the compilation step lives here."""
import sys

import numpy as np
from numba import njit

from befunge import _run_core, src_to_grid, AT, run as _run

_run_core_jit = njit(cache=True)(_run_core)

# Reusable buffers shared across calls in the same process.
_STACK = np.zeros(65536, dtype=np.int64)
_OUTBUF = np.zeros(8192, dtype=np.int32)


def run(src, max_steps=None, out=None):
    if out is None:
        out = sys.stdout
    if max_steps is None:
        max_steps = 1 << 62
    grid = src_to_grid(src)
    status, n = _run_core_jit(grid, max_steps, _STACK, _OUTBUF)
    if n > 0:
        out.write(''.join(chr(int(b)) for b in _OUTBUF[:n]))
    return 'ok' if status == 0 else 'step_limit'


# Warm the JIT so first real call doesn't pay compile cost.
_warm = np.full((25, 80), 32, dtype=np.int32)
_warm[0, 0] = AT
_run_core_jit(_warm, 10, _STACK, _OUTBUF)
