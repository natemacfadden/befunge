# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Encoder-decoder transformer that maps an output-byte sequence
#               to a sparse-triple representation of a Befunge program. The
#               decoder emits (y, x, char) triples in random order until a
#               <done> token
# -----------------------------------------------------------------------------

import argparse, math, os, random, re, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from befunge import W, H

# ----- defaults (override via CLI) -------------------------------------------
DEFAULT_DATA       = [os.path.join(os.path.dirname(_HERE),
                                   'data', 'random_programs',
                                   'dataset_agg_round0.parquet')]
DEFAULT_OUT        = os.path.join(_HERE, 'triples_model.pt')
DEFAULT_EPOCHS     = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 3e-4
DEFAULT_D_MODEL    = 128
DEFAULT_NHEAD      = 4
DEFAULT_N_ENC      = 4
DEFAULT_N_DEC      = 6
DEFAULT_MAX_IN     = 512   # max input-bytes length
DEFAULT_MAX_OUT    = 256   # max triple-token length (allows ~85 cells + specials)
DEFAULT_SEED       = 0

# ----- vocab -----------------------------------------------------------------
# single unified token table so encoder and decoder share embeddings
#   0..255       byte values (both encoder input chars and decoder char tokens)
#   256..280     y position (decoder only)             — H = 25 slots
#   281..360     x position (decoder only)             — W = 80 slots
#   361          <bos>
#   362          <done>
#   363          <pad>
Y_BASE     = 256
X_BASE     = 256 + H
BOS        = X_BASE + W       # 256 + 25 + 80 = 361
DONE       = BOS + 1
PAD        = BOS + 2
VOCAB_SIZE = PAD + 1

# ----- helpers ---------------------------------------------------------------
SANITIZE_RE = re.compile(r'\\x([0-9a-fA-F]{2})')

def unsanitize(s):
    """Reverse the '\\xNN' escape used in the output column → raw bytes."""
    out = bytearray()
    i = 0
    while i < len(s):
        m = SANITIZE_RE.match(s, i)
        if m:
            out.append(int(m.group(1), 16))
            i = m.end()
        else:
            out.append(ord(s[i]) & 0xff)
            i += 1
    return bytes(out)

def program_to_triples(program):
    """List of (y, x, char_ord) for each non-space cell of `program`."""
    triples = []
    for y, row in enumerate(program.splitlines()):
        for x, ch in enumerate(row):
            if ch != ' ':
                triples.append((y, x, ord(ch)))
    return triples


# ----- dataset ---------------------------------------------------------------
class TripleDataset(Dataset):
    """One (encoder_input, decoder_target) example per program. The decoder
    target shuffles the triple order on each __getitem__ call so the model
    sees the same cells in many orderings across epochs."""

    def __init__(self, parquet_paths, max_in=DEFAULT_MAX_IN, max_out=DEFAULT_MAX_OUT):
        self.max_in  = max_in
        self.max_out = max_out
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]
        self.programs, self.outputs, self.per_file_lengths = [], [], []
        for path in parquet_paths:
            tbl = pq.read_table(path, columns=['program', 'output'])
            n0 = len(self.programs)
            self.programs.extend(tbl.column('program').to_pylist())
            self.outputs.extend(tbl.column('output').to_pylist())
            n = len(self.programs) - n0
            self.per_file_lengths.append(n)
            print(f'  loaded {n:>7,} rows from {os.path.basename(path)}')

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, i):
        # encoder side: output bytes, truncated to max_in-2 (for BOS/EOS-style padding)
        out_bytes = unsanitize(self.outputs[i])[:self.max_in]
        src = list(out_bytes)
        if not src:
            src = [0]  # avoid empty sequence

        # decoder side: triples in random order
        triples = program_to_triples(self.programs[i])
        random.shuffle(triples)
        # cap to (max_out - 2) / 3 cells so we always fit BOS + cells + DONE
        max_cells = (self.max_out - 2) // 3
        triples = triples[:max_cells]
        tgt = [BOS]
        for (y, x, c) in triples:
            tgt.extend([Y_BASE + y, X_BASE + x, c])
        tgt.append(DONE)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate(batch):
    """Pad both src and tgt to the max length in the batch."""
    srcs, tgts = zip(*batch)
    src_len = max(len(s) for s in srcs)
    tgt_len = max(len(t) for t in tgts)
    B = len(batch)
    src = torch.full((B, src_len), PAD, dtype=torch.long)
    tgt = torch.full((B, tgt_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)] = s
        tgt[i, :len(t)] = t
    return src, tgt


# ----- model -----------------------------------------------------------------
class TripleTransformer(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, d_model=DEFAULT_D_MODEL,
                 nhead=DEFAULT_NHEAD, n_enc=DEFAULT_N_ENC, n_dec=DEFAULT_N_DEC,
                 max_in=DEFAULT_MAX_IN, max_out=DEFAULT_MAX_OUT, dropout=0.1):
        super().__init__()
        self.emb     = nn.Embedding(vocab, d_model)
        self.enc_pos = nn.Embedding(max_in,  d_model)
        self.dec_pos = nn.Embedding(max_out, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.ln_out  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied

    def _emb_with_pos(self, x, pos_emb):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return self.emb(x) + pos_emb(pos)

    def forward(self, src, tgt):
        src_emb = self._emb_with_pos(src, self.enc_pos)
        tgt_emb = self._emb_with_pos(tgt, self.dec_pos)
        src_kp  = (src == PAD)
        tgt_kp  = (tgt == PAD)
        T = tgt.size(1)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=tgt.device),
                            diagonal=1)
        memory = self.encoder(src_emb, src_key_padding_mask=src_kp)
        out = self.decoder(tgt_emb, memory,
                           tgt_mask=causal,
                           tgt_key_padding_mask=tgt_kp,
                           memory_key_padding_mask=src_kp)
        return self.head(self.ln_out(out))


# ----- train -----------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f'device: {device}')

    full = TripleDataset(args.data, max_in=args.max_in, max_out=args.max_out)
    # split per-file so that each file's val set stays fixed across resumes —
    # otherwise newly-added rounds would contaminate the round0 val set
    val_idxs, train_idxs = [], []
    offset = 0
    for fi, n in enumerate(full.per_file_lengths):
        file_idxs = list(range(offset, offset + n))
        random.Random(f'{args.seed}-{fi}').shuffle(file_idxs)
        nv = max(1, n // 20)
        val_idxs.extend(file_idxs[:nv])
        train_idxs.extend(file_idxs[nv:])
        offset += n
    random.Random(args.seed).shuffle(train_idxs)
    train_set  = torch.utils.data.Subset(full, train_idxs)
    val_set    = torch.utils.data.Subset(full, sorted(val_idxs))
    print(f'examples: {len(full)} ({len(train_set)} train, {len(val_set)} val)')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=2)

    model = TripleTransformer(
        d_model=args.d_model, nhead=args.nhead,
        n_enc=args.n_enc, n_dec=args.n_dec,
        max_in=args.max_in, max_out=args.max_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # resume: load model + optimizer state, pick up the epoch counter
    history = []
    completed = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
        completed = ckpt.get('completed_epochs', 0)
        history   = ckpt.get('history', [])
        print(f'resumed {args.resume} @ epoch {completed}')
    if args.start_epoch is not None:
        completed = args.start_epoch
        print(f'overriding start epoch -> {completed}')

    for epoch in range(completed + 1, completed + args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_tok = 0.0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])              # predict tgt[1:]
            target = tgt[:, 1:]
            mask = (target != PAD)
            loss = F.cross_entropy(logits[mask], target[mask], reduction='mean')
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * mask.sum().item()
            n_tok += mask.sum().item()
        train_loss = total_loss / max(1, n_tok)

        model.eval()
        total_loss, n_tok = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt[:, :-1])
                target = tgt[:, 1:]
                mask = (target != PAD)
                if not mask.any(): continue
                total_loss += F.cross_entropy(logits[mask], target[mask],
                                              reduction='sum').item()
                n_tok += mask.sum().item()
        val_loss = total_loss / max(1, n_tok)
        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss,
                        'data': [os.path.basename(p) for p in args.data]})
        print(f'epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  '
              f'({time.time()-t0:.1f}s)')
        # save after every epoch so a crash doesn't lose progress
        torch.save({'state_dict':        model.state_dict(),
                    'optimizer':         opt.state_dict(),
                    'completed_epochs':  epoch,
                    'history':           history,
                    'config':            vars(args)}, args.out)
    print(f'saved {args.out}')


# ----- sampling --------------------------------------------------------------
@torch.no_grad()
def sample(model, src_bytes, device, max_new=DEFAULT_MAX_OUT, temperature=0.0):
    """Generate triples for a given input-byte sequence. Returns a grid string."""
    model.eval()
    src = torch.tensor([list(src_bytes)], dtype=torch.long, device=device)
    ids = torch.tensor([[BOS]], dtype=torch.long, device=device)
    triples_out = []
    pending = []  # accumulate (y, x, c) one token at a time
    for _ in range(max_new):
        logits = model(src, ids)[:, -1, :]
        if temperature <= 0:
            tok = logits.argmax(-1).item()
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
        if tok == DONE:
            break
        pending.append(tok)
        if len(pending) == 3:
            y_tok, x_tok, c_tok = pending
            if (Y_BASE <= y_tok < Y_BASE + H and
                X_BASE <= x_tok < X_BASE + W and
                0      <= c_tok < 256):
                triples_out.append((y_tok - Y_BASE, x_tok - X_BASE, c_tok))
            pending = []

    # render to grid
    grid = [[' '] * W for _ in range(H)]
    for y, x, c in triples_out:
        ch = chr(c)
        grid[y][x] = ch
    return '\n'.join(''.join(row) for row in grid)


# ----- CLI -------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',         nargs='+', default=DEFAULT_DATA,
                   help='one or more parquet files; rows are concatenated')
    p.add_argument('--out',          default=DEFAULT_OUT)
    p.add_argument('--resume',       default=None,
                   help='checkpoint to load (state_dict + optimizer + epoch counter)')
    p.add_argument('--start-epoch',  type=int, default=None,
                   help='override the epoch counter (useful when resuming from '
                        'an old checkpoint that predates the counter)')
    p.add_argument('--epochs',     type=int,   default=DEFAULT_EPOCHS,
                   help='additional epochs to run (in addition to any already '
                        'completed in --resume)')
    p.add_argument('--batch-size', type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument('--lr',         type=float, default=DEFAULT_LR)
    p.add_argument('--d-model',    type=int,   default=DEFAULT_D_MODEL)
    p.add_argument('--nhead',      type=int,   default=DEFAULT_NHEAD)
    p.add_argument('--n-enc',      type=int,   default=DEFAULT_N_ENC)
    p.add_argument('--n-dec',      type=int,   default=DEFAULT_N_DEC)
    p.add_argument('--max-in',     type=int,   default=DEFAULT_MAX_IN)
    p.add_argument('--max-out',    type=int,   default=DEFAULT_MAX_OUT)
    p.add_argument('--seed',       type=int,   default=DEFAULT_SEED)
    train(p.parse_args())
