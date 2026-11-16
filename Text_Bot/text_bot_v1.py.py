import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)

# ---------- Chunk A: letters -> numbers ----------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

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
batch_size = 32
block_size = 8

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# ---------- Chunk C: the model ----------
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.table(x)
        loss = None
        if y is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return x

# ---------- Chunk D: train, then write ----------
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(5001):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step}: loss {loss.item():.3f}")

start = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(start, 300)[0].tolist()))