# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Befunge-93 interpreter; runs a .bf file or launches the GUI
# -----------------------------------------------------------------------------

import sys

import numpy as np
from numba import njit


# =============================================================================
# Language
# =============================================================================
# Befunge-93 in brief (see https://esolangs.org/wiki/Befunge for the full
# reference). A program runs on a "playfield" — for us, a fixed 80x25 grid of
# ASCII characters. An "instruction pointer" (IP) travels through that grid,
# executing whatever character it lands on. Most instructions read and write a
# stack of int64 values ("in the manner of Forth"). The IP has inertia: it
# keeps moving in its current cardinal direction until an instruction changes
# it. While "string mode" is active (toggled by `"`), characters are pushed
# to the stack as ASCII values instead of being executed.
#
# A couple of implementation details worth pinning down:
#   - The playfield doubles as memory: programs can read/write any cell with
#     `g`/`p`.
#   - SP is the stack pointer — index where the next push will land; the top
#     of the stack is stack[SP-1].
#
W, H = 80, 25  # playfield dimensions (columns, rows)

# INSTRUCTIONS below maps each instruction character to its ord value, with
# a brief description per op. ALPHABET adds the two chars that can appear
# on the grid but aren't instructions (space, newline).
INSTRUCTIONS = {
    # digits — push value 0..9
    '0': ord('0'), '1': ord('1'), '2': ord('2'), '3': ord('3'), '4': ord('4'),
    '5': ord('5'), '6': ord('6'), '7': ord('7'), '8': ord('8'), '9': ord('9'),

    '+':  ord('+'),  # pop a, pop b, push b+a
    '*':  ord('*'),  # pop a, pop b, push b*a
    '-':  ord('-'),  # pop a, pop b, push b-a
    '/':  ord('/'),  # pop a, pop b, push b//a   (0 if a==0)
    '%':  ord('%'),  # pop a, pop b, push b%a    (0 if a==0)

    '!':  ord('!'),  # logical not: pop v, push 1 if v==0 else 0
    '`':  ord('`'),  # greater-than: pop a, pop b, push 1 if b>a else 0

    '>':  ord('>'),  # IP right
    '<':  ord('<'),  # IP left
    '^':  ord('^'),  # IP up
    'v':  ord('v'),  # IP down
    '?':  ord('?'),  # random of the 4 cardinal directions
    '_':  ord('_'),  # horizontal if: pop v, go right if v==0 else left
    '|':  ord('|'),  # vertical if:   pop v, go down  if v==0 else up

    '"':  ord('"'),  # toggle stringmode

    ':':  ord(':'),  # duplicate top of stack
    '\\': ord('\\'), # swap top two
    '$':  ord('$'),  # pop and discard

    '.':  ord('.'),  # pop v, output str(v) + ' '
    ',':  ord(','),  # pop v, output chr(v % 256)

    '#':  ord('#'),  # bridge: skip next cell along IP direction

    'g':  ord('g'),  # get: pop y, pop x, push grid[y%H, x%W]
    'p':  ord('p'),  # put: pop y, pop x, pop v, grid[y%H, x%W] = v % 256

    '&':  ord('&'),  # read integer from stdin (we push 0)
    '~':  ord('~'),  # read char from stdin    (we push 0)

    '@':  ord('@'),  # halt
}

ALPHABET = {
    **INSTRUCTIONS,
    ' ':  ord(' '),   # no-op padding
    '\n': ord('\n'),  # row separator in .bf source files
}

# Aliases for _run_core's dispatch. Reading from the enclosing scope at compile
# time lets numba fold each comparison to a literal int compare. Names mirror
# the char they encode.
SPACE    = ALPHABET[' ']
BANG     = ALPHABET['!']
DQ       = ALPHABET['"']
HASH     = ALPHABET['#']
DOLLAR   = ALPHABET['$']
PCT      = ALPHABET['%']
AMP      = ALPHABET['&']
STAR     = ALPHABET['*']
PLUS     = ALPHABET['+']
COMMA    = ALPHABET[',']
MINUS    = ALPHABET['-']
DOT      = ALPHABET['.']
SLASH    = ALPHABET['/']
ZERO     = ALPHABET['0']
NINE     = ALPHABET['9']
COLON    = ALPHABET[':']
LT       = ALPHABET['<']
GT       = ALPHABET['>']
QMARK    = ALPHABET['?']
AT       = ALPHABET['@']
BSLASH   = ALPHABET['\\']
CARET    = ALPHABET['^']
UNDER    = ALPHABET['_']
BACKTICK = ALPHABET['`']
G_GET    = ALPHABET['g']
P_PUT    = ALPHABET['p']
V_DOWN   = ALPHABET['v']
PIPE     = ALPHABET['|']
TILDE    = ALPHABET['~']


# =============================================================================
# Source parsing
# =============================================================================

def str_to_grid(src):
    """Lay out a .bf source string onto an (H, W) int32 grid, padded with spaces."""
    grid = np.full((H, W), SPACE, dtype=np.int32)
    for y, line in enumerate(src.splitlines()[:H]):
        for x, ch in enumerate(line[:W]):
            grid[y, x] = ord(ch)
    return grid


# =============================================================================
# Runtime state
# =============================================================================
# The interpreter's pausable runtime state lives in a small int64 array,
# mutated in place by _run_core so the GUI can pause between steps and so
# numba can compile the dispatch loop. Indexes into that array:
S_SP          = 0  # stack pointer
S_OUT_LEN     = 1  # bytes written to the output buffer
S_X           = 2  # IP column
S_Y           = 3  # IP row
S_DX          = 4  # IP horizontal direction (-1, 0, +1)
S_DY          = 5  # IP vertical direction   (-1, 0, +1)
S_STRING_MODE = 6  # 0 or 1
STATE_SIZE    = 7


def new_state():
    """Initial interpreter state: IP at (0,0) heading right."""
    s = np.zeros(STATE_SIZE, dtype=np.int64)
    s[S_DX] = 1
    return s


# =============================================================================
# Interpreter
# =============================================================================

def _run_core(grid, max_steps, stack, out_buf, state):
    """Shared dispatch loop. Numba-friendly: no Python objects, just int ops on
    pre-allocated arrays. `run(jit=True)` calls a @njit-wrapped copy of this
    function; `run(jit=False)` calls it unjitted. Resumable: state is mutated
    in place so callers can drive the interpreter step-by-step (the GUI does
    exactly that). Returns status (0=halted, 1=step budget exhausted).

    `&` and `~` (interactive input) are treated as `push(0)` since the core
    can't block on stdin — random programs don't generate these anyway."""
    sp          = int(state[S_SP])
    out_len     = int(state[S_OUT_LEN])
    x           = int(state[S_X])
    y           = int(state[S_Y])
    dx          = int(state[S_DX])
    dy          = int(state[S_DY])
    string_mode = state[S_STRING_MODE] != 0
    stack_cap   = stack.shape[0]
    out_cap     = out_buf.shape[0]
    steps       = 0
    halted      = False

    while steps < max_steps and not halted:
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
            halted = True

        if not halted:
            x = (x + dx) % W
            y = (y + dy) % H

    state[S_SP]          = sp
    state[S_OUT_LEN]     = out_len
    state[S_X]           = x
    state[S_Y]           = y
    state[S_DX]          = dx
    state[S_DY]          = dy
    state[S_STRING_MODE] = 1 if string_mode else 0
    return 0 if halted else 1


# Lazily compiled JIT version of _run_core. First `run(..., jit=True)` call
# pays the compile cost (~1s, cached after); subsequent calls are fast.
_run_core_jit = njit(cache=True)(_run_core)

# Reusable buffers — only one set per process. Not threadsafe; this is fine
# under multiprocessing (one process per worker) but would need rethinking
# if called from multiple threads.
_STACK = np.zeros(65536, dtype=np.int64)
_OUTBUF = np.zeros(8192, dtype=np.int32)


# =============================================================================
# Entry points
# =============================================================================

def run(src, max_steps=None, out=None, jit=False):
    """Run a Befunge program. Set `jit=True` for the numba-compiled hot path."""
    if out is None:
        out = sys.stdout
    if max_steps is None:
        max_steps = 1 << 62
    grid = str_to_grid(src)
    state = new_state()
    # _STACK/_OUTBUF are module-level reusable buffers; we only read up to
    # state[S_OUT_LEN], so stale data past it is harmless.
    core = _run_core_jit if jit else _run_core
    status = core(grid, max_steps, _STACK, _OUTBUF, state)
    n = int(state[S_OUT_LEN])
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
        from viz.gui import App
        App().run()
    else:
        with open(args.file) as f:
            status = run(f.read(), max_steps=args.max_steps, jit=args.jit)
        if status == 'step_limit':
            sys.stderr.write(f'\n[step limit {args.max_steps} reached]\n')
