import argparse, os, random

CHARS = '0123456789+*-/%!`><^v?_|:\\$.,#gp@" '

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
    p.add_argument('--dir', default='random')
    args = p.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        rng = random.Random(f'{args.seed}-{i}')
        path = os.path.join(args.dir, f'random_{i}.bf')
        with open(path, 'w') as f:
            f.write(generate(rng) + '\n')
        print(f'wrote {path}')
