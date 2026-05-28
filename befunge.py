import sys

import numpy as np
from numba import njit

# Befunge-93 instruction set, grouped by category. Space is a no-op and pads
# the playfield; newline separates rows in source.
DIGITS       = '0123456789'
ARITHMETIC   = '+*-/%'
LOGIC        = '!`'
MOVEMENT     = '><^v?_|'
STRING_MODE  = '"'
STACK_OPS    = ':\\$'
OUTPUT       = '.,'
SKIP         = '#'
MEMORY       = 'gp'
INPUT        = '&~'
HALT         = '@'
INSTRUCTIONS = DIGITS + ARITHMETIC + LOGIC + MOVEMENT + STRING_MODE + \
               STACK_OPS + OUTPUT + SKIP + MEMORY + INPUT + HALT
PLAYFIELD    = INSTRUCTIONS + ' \n'  # chars that can appear in a .bf source file

W, H = 80, 25

# ord values used inside _run_core; named to read like the chars they represent.
SPACE = 32; BANG = 33; DQ = 34; HASH = 35; DOLLAR = 36; PCT = 37; AMP = 38
STAR = 42; PLUS = 43; COMMA = 44; MINUS = 45; DOT = 46; SLASH = 47
ZERO = 48; NINE = 57; COLON = 58
LT = 60; GT = 62; QMARK = 63; AT = 64
BSLASH = 92; CARET = 94; UNDER = 95; BACKTICK = 96
G_GET = 103; P_PUT = 112; V_DOWN = 118
PIPE = 124; TILDE = 126


def src_to_grid(src):
    grid = np.full((H, W), SPACE, dtype=np.int32)
    for y, line in enumerate(src.splitlines()[:H]):
        for x, ch in enumerate(line[:W]):
            grid[y, x] = ord(ch)
    return grid


def _run_core(grid, max_steps, stack, out_buf):
    """Shared dispatch loop. Numba-friendly: no Python objects, just int ops on
    pre-allocated arrays. `run(jit=True)` calls a @njit-wrapped copy of this
    function; `run(jit=False)` calls it unjitted. Returns (status, out_len);
    status: 0=ok, 1=step_limit.

    `&` and `~` (interactive input) are treated as `push(0)` since the core
    can't block on stdin — random programs don't generate these anyway."""
    sp = 0
    out_len = 0
    x = 0; y = 0
    dx = 1; dy = 0
    string_mode = False
    steps = 0
    stack_cap = stack.shape[0]
    out_cap = out_buf.shape[0]

    while steps < max_steps:
        steps += 1
        c = grid[y, x]

        if string_mode:
            if c == DQ:
                string_mode = False
            elif sp < stack_cap:
                stack[sp] = c; sp += 1
        elif c == SPACE:
            pass
        elif ZERO <= c <= NINE:
            if sp < stack_cap:
                stack[sp] = c - ZERO; sp += 1
        elif c == PLUS:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = a + b; sp += 1
        elif c == STAR:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = a * b; sp += 1
        elif c == MINUS:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = b - a; sp += 1
        elif c == SLASH:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = (b // a) if a != 0 else 0; sp += 1
        elif c == PCT:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = (b % a) if a != 0 else 0; sp += 1
        elif c == GT:
            dx = 1; dy = 0
        elif c == LT:
            dx = -1; dy = 0
        elif c == CARET:
            dx = 0; dy = -1
        elif c == V_DOWN:
            dx = 0; dy = 1
        elif c == QMARK:
            r = np.random.randint(0, 4)
            if r == 0:   dx = 1;  dy = 0
            elif r == 1: dx = -1; dy = 0
            elif r == 2: dx = 0;  dy = 1
            else:        dx = 0;  dy = -1
        elif c == UNDER:
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            if v == 0: dx = 1;  dy = 0
            else:      dx = -1; dy = 0
        elif c == PIPE:
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            if v == 0: dx = 0; dy = 1
            else:      dx = 0; dy = -1
        elif c == BANG:
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = 0 if v != 0 else 1; sp += 1
        elif c == BACKTICK:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            stack[sp] = 1 if b > a else 0; sp += 1
        elif c == DQ:
            string_mode = True
        elif c == COLON:
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            if sp < stack_cap:
                stack[sp] = v; sp += 1
            if sp < stack_cap:
                stack[sp] = v; sp += 1
        elif c == BSLASH:
            sp -= 1
            a = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            b = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            if sp < stack_cap:
                stack[sp] = a; sp += 1
            if sp < stack_cap:
                stack[sp] = b; sp += 1
        elif c == DOLLAR:
            if sp > 0: sp -= 1
        elif c == DOT:
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            s = str(v)
            for ch in s:
                if out_len < out_cap:
                    out_buf[out_len] = ord(ch); out_len += 1
            if out_len < out_cap:
                out_buf[out_len] = SPACE; out_len += 1
        elif c == COMMA:
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            if out_len < out_cap:
                out_buf[out_len] = v % 256; out_len += 1
        elif c == HASH:
            x = (x + dx) % W
            y = (y + dy) % H
        elif c == G_GET:
            sp -= 1
            gy = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            gx = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            if sp < stack_cap:
                stack[sp] = grid[gy % H, gx % W]; sp += 1
        elif c == P_PUT:
            sp -= 1
            py = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            px = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            sp -= 1
            v = stack[sp] if sp >= 0 else 0
            if sp < 0: sp = 0
            grid[py % H, px % W] = v % 256
        elif c == AMP or c == TILDE:
            if sp < stack_cap:
                stack[sp] = 0; sp += 1
        elif c == AT:
            return 0, out_len

        x = (x + dx) % W
        y = (y + dy) % H

    return 1, out_len


# Lazily compiled JIT version of _run_core. First `run(..., jit=True)` call
# pays the compile cost (~1s, cached after); subsequent calls are fast.
_run_core_jit = njit(cache=True)(_run_core)

# Reusable buffers — only one set per process. Not threadsafe; this is fine
# under multiprocessing (one process per worker) but would need rethinking
# if called from multiple threads.
_STACK = np.zeros(65536, dtype=np.int64)
_OUTBUF = np.zeros(8192, dtype=np.int32)


def run(src, max_steps=None, out=None, jit=False):
    """Run a Befunge program. Set `jit=True` for the numba-compiled hot path."""
    if out is None:
        out = sys.stdout
    if max_steps is None:
        max_steps = 1 << 62
    grid = src_to_grid(src)
    core = _run_core_jit if jit else _run_core
    status, n = core(grid, max_steps, _STACK, _OUTBUF)
    if n > 0:
        out.write(''.join(chr(int(b)) for b in _OUTBUF[:n]))
    return 'ok' if status == 0 else 'step_limit'


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Run a Befunge program, or launch the GUI when no file is given.')
    p.add_argument('file', nargs='?', help='.bf source file (omit to open the GUI)')
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--jit', action='store_true')
    args = p.parse_args()
    if args.file is None:
        from gui import App
        App().run()
    else:
        with open(args.file) as f:
            status = run(f.read(), max_steps=args.max_steps, jit=args.jit)
        if status == 'step_limit':
            sys.stderr.write(f'\n[step limit {args.max_steps} reached]\n')
