import os
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

# ---------- Settings ----------
batch_size = 32
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
eval_iters = 50

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using", device)

# ---------- Chunk A: letters -> numbers ----------
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "data.txt"), "r", encoding="utf-8") as f:
    text = f.read()
text = text[:5_000_000]

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------- Chunk B: practice pieces ----------
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------- Chunk C: the mini GPT ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, h):
        B, T, C = h.shape
        k = self.key(h)
        q = self.query(h)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(h)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)

    def forward(self, h):
        out = torch.cat([hd(h) for hd in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, h):
        return self.net(h)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHead(n_head, n_embd // n_head)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, h):
        h = h + self.attn(self.ln1(h))
        h = h + self.ff(self.ln2(h))
        return h

class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=device))
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if y is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits, _ = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return x

# ---------- Steady loss measurement ----------
@torch.no_grad()
def estimate_loss():
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, l = model(xb, yb)
            losses[k] = l.item()
        out[split] = losses.mean().item()
    return out

# ---------- Chunk D: train, save, then write ----------
model = MiniGPT(vocab_size).to(device)
print(sum(p.numel() for p in model.parameters()), "parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(5001):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        e = estimate_loss()
        print(f"step {step}: train {e['train']:.3f}, val {e['val']:.3f}")

torch.save(model.state_dict(), os.path.join(here, "minigpt_stories.pt"))

start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 500)[0].tolist()))