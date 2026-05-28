import sys, random

def run(src, max_steps=None):
    grid = [[' '] * 80 for _ in range(25)]
    for y, line in enumerate(src.splitlines()[:25]):
        for x, ch in enumerate(line[:80]):
            grid[y][x] = ch

    stack = []
    x, y, dx, dy = 0, 0, 1, 0
    string_mode = False
    steps = 0

    def pop():
        return stack.pop() if stack else 0

    while True:
        if max_steps is not None and steps >= max_steps:
            sys.stderr.write(f'\n[step limit {max_steps} reached]\n')
            return
        steps += 1
        c = grid[y][x]

        if string_mode:
            if c == '"': string_mode = False
            else: stack.append(ord(c))
        elif c.isdigit(): stack.append(int(c))
        elif c == '+': stack.append(pop() + pop())
        elif c == '*': stack.append(pop() * pop())
        elif c == '-': a, b = pop(), pop(); stack.append(b - a)
        elif c == '/': a, b = pop(), pop(); stack.append(b // a if a else 0)
        elif c == '%': a, b = pop(), pop(); stack.append(b % a if a else 0)
        elif c == '!': stack.append(0 if pop() else 1)
        elif c == '`': a, b = pop(), pop(); stack.append(1 if b > a else 0)
        elif c == '>': dx, dy = 1, 0
        elif c == '<': dx, dy = -1, 0
        elif c == '^': dx, dy = 0, -1
        elif c == 'v': dx, dy = 0, 1
        elif c == '?': dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        elif c == '_': dx, dy = (1, 0) if pop() == 0 else (-1, 0)
        elif c == '|': dx, dy = (0, 1) if pop() == 0 else (0, -1)
        elif c == '"': string_mode = True
        elif c == ':': v = pop(); stack += [v, v]
        elif c == '\\': a, b = pop(), pop(); stack += [a, b]
        elif c == '$': pop()
        elif c == '.': sys.stdout.write(str(pop()) + ' '); sys.stdout.flush()
        elif c == ',': sys.stdout.write(chr(pop() % 256)); sys.stdout.flush()
        elif c == '#': x, y = (x + dx) % 80, (y + dy) % 25
        elif c == 'g': gy, gx = pop(), pop(); stack.append(ord(grid[gy % 25][gx % 80]))
        elif c == 'p': py, px, v = pop(), pop(), pop(); grid[py % 25][px % 80] = chr(v % 256)
        elif c == '&': stack.append(int(input()))
        elif c == '~': stack.append(ord(sys.stdin.read(1)))
        elif c == '@': return

        x, y = (x + dx) % 80, (y + dy) % 25

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('--max-steps', type=int, default=None)
    args = p.parse_args()
    with open(args.file) as f:
        run(f.read(), max_steps=args.max_steps)
