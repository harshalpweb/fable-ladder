"""O-1 chemistry specialist: a small model that predicts the product of a reaction, trained from scratch on human-reported
patent reactions (USPTO, Jin et al. 2017 split; data up to 2016). No other lab's model is used.
Outputs: results.json, model.pt (half precision), vocab.json."""
import json, math, os, random, re, subprocess, sys, time, urllib.request, zipfile
T0 = time.time(); SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."); WORK = os.environ.get("WORK_DIR", "/kaggle/temp" if os.path.isdir("/kaggle") else "./work")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
BUDGET = float(os.environ.get("BUDGET_HOURS", "0.05" if SMOKE else "7.0")) * 3600
RES = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), log=[])
def save(): json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1)
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{(time.time()-T0)/60:6.1f} min] {s}", flush=True); RES["log"].append(s); save()
try: from rdkit import Chem, RDLogger
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"]); from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"; log("device", DEV)
z = f"{WORK}/uspto.zip"
if not os.path.exists(z):
    src = os.environ.get("USPTO_ZIP")
    if src: z = src
    else: urllib.request.urlretrieve("https://raw.githubusercontent.com/wengong-jin/nips17-rexgen/master/USPTO/data.zip", z)
def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    for a in m.GetAtoms(): a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)
def load(name, limit=None):
    out = []
    with zipfile.ZipFile(z) as zf:
        for k, line in enumerate(zf.read(f"data/{name}.txt").decode().splitlines()):
            if limit and k >= limit: break
            rx = line.split(" ")[0]
            if ">>" not in rx: continue
            a, b = rx.split(">>"); a, b = canon(a), canon(b)
            if a and b: out.append((a, b))
    return out
TOK = re.compile(r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
def toks(s): return TOK.findall(s)
train = load("train", 6000 if SMOKE else None); test = load("test", 300 if SMOKE else None); log("reactions", len(train), len(test))
vocab = ["<pad>", "<sep>", "<eos>", "<unk>"] + sorted({t for a, b in train for t in toks(a) + toks(b)}); stoi = {t: i for i, t in enumerate(vocab)}
json.dump(vocab, open(f"{OUT}/vocab.json", "w")); V = len(vocab); CTX = 320
def enc(s): return [stoi.get(t, 3) for t in toks(s)]
def pack(a, b): x = enc(a) + [1] + enc(b) + [2]; return x, len(enc(a)) + 1
data = [pack(a, b) for a, b in train]; keep = [(x, s) for x, s in data if len(x) <= CTX]; log("train kept", len(keep), "of", len(data), "vocab", V)
RES["train_kept_share"] = len(keep) / len(data)
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.f1 = nn.Linear(d, 4 * d, bias=False); s.f2 = nn.Linear(4 * d, d, bias=False)
    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + s.f2(F.gelu(s.f1(s.n2(x))))
class GPT(nn.Module):
    def __init__(s, d, L, h, ctx, V):
        super().__init__(); s.ctx = ctx; s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(ctx, d); s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); s.nf = nn.LayerNorm(d)
        s.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)
        for b in s.blocks: nn.init.zeros_(b.o.weight); nn.init.zeros_(b.f2.weight)
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(x.shape[1], device=x.device))
        for b in s.blocks: h = b(h)
        return s.nf(h) @ s.emb.weight.t()
CFG = dict(d=128, L=2, h=4, ctx=CTX, V=V) if SMOKE else dict(d=384, L=8, h=8, ctx=CTX, V=V); RES["config"] = CFG
m = GPT(**CFG).to(DEV); RES["params"] = sum(p.numel() for p in m.parameters()); log("parameters", RES["params"])
keep.sort(key=lambda t: len(t[0])); BS = 32 if SMOKE else 128
batches = [keep[i:i + BS] for i in range(0, len(keep), BS)]
def tens(batch):
    L = max(len(x) for x, _ in batch); X = torch.zeros(len(batch), L, dtype=torch.long); M = torch.zeros(len(batch), L - 1)
    for i, (x, s) in enumerate(batch): X[i, :len(x)] = torch.tensor(x); M[i, s - 1:len(x) - 1] = 1   # loss only on product tokens and <eos>
    return X.to(DEV), M.to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.01); scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
EVAL_RESERVE = 60 if SMOKE else 1500; train_budget = BUDGET - EVAL_RESERVE - (time.time() - T0); t1 = time.time(); step = 0; ep = 0; last = 0.0; R = random.Random(0)
while time.time() - t1 < train_budget:
    R.shuffle(batches); ep += 1
    for b in batches:
        frac = (time.time() - t1) / train_budget
        if frac >= 1: break
        lr = 1e-3 * min(1.0, (step + 1) / 2000) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups: g["lr"] = lr
        X, M = tens(b)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            lg = m(X[:, :-1]).float(); loss = (F.cross_entropy(lg.transpose(1, 2), X[:, 1:], reduction="none") * M).sum() / M.sum()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
        step += 1; last = loss.item() if step == 1 else 0.99 * last + 0.01 * loss.item()
        if step % (50 if SMOKE else 2000) == 0: log(f"epoch {ep} step {step} loss {last:.4f} lr {lr:.2e}")
RES.update(epochs=ep, steps=step, train_loss=last, train_minutes=(time.time() - t1) / 60); log("training done", ep, "epochs", step, "steps")
torch.save({k: v.half() for k, v in m.state_dict().items()}, f"{OUT}/model.pt")
@torch.no_grad()
def predict(a, max_new=160):
    ids = enc(a) + [1]
    if len(ids) >= CTX - 2: return None
    n0 = len(ids)
    for _ in range(min(max_new, CTX - n0)):
        x = torch.tensor([ids], device=DEV)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"): nxt = int(m(x)[0, -1].argmax())
        if nxt == 2: break
        ids.append(nxt)
    return "".join(vocab[i] for i in ids[n0:])
m.eval(); R2 = random.Random(1); sample = R2.sample(test, min(len(test), 100 if SMOKE else 2000)); ok = valid = 0
for a, b in sample:
    p = predict(a); c = canon(p) if p else None; valid += c is not None; ok += (c == b)
RES.update(test_n=len(sample), test_top1=ok / len(sample), test_valid=valid / len(sample), finished=time.strftime("%Y-%m-%d %H:%M:%S"), total_minutes=(time.time() - T0) / 60)
log(f"USPTO test top-1 {ok}/{len(sample)} = {ok/len(sample):.3f}; valid SMILES {valid/len(sample):.3f}"); save(); print("done")
