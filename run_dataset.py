import argparse, glob, json, os, subprocess

def sanitize(s):
    out = []
    for c in s:
        o = ord(c)
        if c in '\n\t' or 32 <= o < 127:
            out.append(c)
        else:
            out.append(f'\\x{o:02x}')
    return ''.join(out)

def run_one(path, max_steps):
    r = subprocess.run(
        ['python3', 'befunge.py', path, '--max-steps', str(max_steps)],
        capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        return r.stdout, 'error'
    if b'[step limit' in r.stderr:
        return r.stdout, 'step_limit'
    return r.stdout, 'ok'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='random')
    p.add_argument('--out', default='dataset.jsonl')
    p.add_argument('--max-steps', type=int, default=100000)
    p.add_argument('--max-output', type=int, default=4096)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, '*.bf')))
    counts = {'ok': 0, 'error': 0, 'step_limit': 0}
    with open(args.out, 'w') as out:
        for path in files:
            raw, status = run_one(path, args.max_steps)
            raw = raw[:args.max_output]
            output = sanitize(raw.decode('utf-8', errors='replace'))
            with open(path) as src:
                program = src.read()
            out.write(json.dumps({
                'file': path,
                'program': program,
                'output': output,
                'status': status,
            }) + '\n')
            counts[status] += 1
            print(f'{path}: {status} ({len(raw)} bytes)  {output[:60]!r}')
    print(f'\nwrote {args.out}: {counts}')
