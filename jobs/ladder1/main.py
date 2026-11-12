"""Compute ladder scored on the real exam (plan: exam + ratchet; 12 Nov 2026).
Trains one model per rung on human-written and program-generated text only, then scores it on exam v1.
No other lab's model is used anywhere. Runs on a Kaggle GPU (or CPU with SMOKE=1 for a tiny local test).
Outputs: results.json (after every rung) and a printed table."""
import base64, gzip, io, json, math, os, random, re, subprocess, sys, tempfile, time, urllib.request
T_START = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
WORK = os.environ.get("WORK_DIR", "/kaggle/temp" if os.path.isdir("/kaggle") else "./work"); os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
BUDGET_S = float(os.environ.get("BUDGET_HOURS", "0.2" if SMOKE else "10.5")) * 3600
RES = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), smoke=SMOKE, log=[], rungs=[])
def save(): json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1)
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{(time.time()-T_START)/60:6.1f} min] {s}", flush=True); RES["log"].append(s); save()
try:
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from tokenizers import ByteLevelBPETokenizer
except Exception as e:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "numpy"]); import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from tokenizers import ByteLevelBPETokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else f"cpu threads {torch.get_num_threads()}")

# ---------------------------------------------------------------- exam (embedded by the build step, or read from a file)
import ast, csv, hashlib, zipfile
def _get(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f: f.write(r.read())
    return path
def build_exam(work):
    csv.field_size_limit(10**9); R = random.Random(20261112); items = []
    def add(bench, kind, rows, n):
        rows = list(rows); R.shuffle(rows)
        for k, r in enumerate(rows[:n]): r.update(id=f"{bench}-{k}", bench=bench, kind=kind, split="dev" if k % 2 == 0 else "hidden"); items.append(r)
    z = zipfile.ZipFile(_get("https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip", f"{work}/gpqa.zip"))
    raw = z.read("dataset/gpqa_diamond.csv", pwd=b"deserted-untie-orchid").decode("utf-8"); g = []
    for r in csv.DictReader(io.StringIO(raw, newline="")):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]; o = opts[:]; R.shuffle(o)
        g.append(dict(question=r["Question"], options=o, answer="ABCD"[o.index(opts[0])], sub=r["Subdomain"]))
    add("gpqa", "mc", g, 198)
    z = zipfile.ZipFile(_get("https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/main/eval_results/model_outputs_Llama-2-7b-hf_5shots.zip", f"{work}/mp.zip"))
    mp = json.loads(z.read([n for n in z.namelist() if n.endswith(".json")][0]))
    add("mmlupro", "mc", [dict(question=x["question"], options=ast.literal_eval(x["options"]) if isinstance(x["options"], str) else x["options"],
                               answer=x["answer"], sub=x["category"]) for x in mp], 300)
    gs = []
    for l in open(_get("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl", f"{work}/gsm_test.jsonl")):
        d = json.loads(l); gs.append(dict(question=d["question"], answer=d["answer"].rsplit("####", 1)[1].strip().replace(",", "")))
    add("gsm8k", "num", gs, 200)
    nq = _get("https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl", f"{work}/nq_dev.jsonl")
    add("nqopen", "text", [dict(question=d["question"], answer=d["answer"]) for d in map(json.loads, open(nq))], 300)
    he = [json.loads(l) for l in gzip.open(_get("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz", f"{work}/he.jsonl.gz"), "rt")]
    add("humaneval", "code", [dict(question=d["prompt"], test=d["test"], entry=d["entry_point"], answer=d["canonical_solution"]) for d in he], 164)
    sha = hashlib.sha256("\n".join(json.dumps(i, sort_keys=True) for i in items).encode()).hexdigest()
    return items, sha
def load_exam():
    f = os.environ.get("EXAM_FILE")
    if f and os.path.exists(f): return [json.loads(l) for l in open(f)]
    items, sha = build_exam(WORK); RES["exam_sha"] = sha
    if not sha.startswith("483e9569cf4f235c"): log("WARNING: rebuilt exam differs from exam v1 (hash", sha[:16], "); scores are for a different sample")
    return items
EXAM = load_exam()
if SMOKE:
    seen = {}; EXAM = [it for it in EXAM if seen.setdefault(it["bench"], 0) < 6 and not seen.update({it["bench"]: seen[it["bench"]] + 1})]
log("exam items", len(EXAM))

# ---------------------------------------------------------------- data: human-written text and program-generated maths
def fetch(url, path, timeout=120):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b)
    return path
def detok(s):
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "), (" : ", ": "),
                 (" 's", "'s"), (" n't", "n't"), ("( ", "("), (" )", ")"), (" ?", "?"), (" !", "!")):
        s = s.replace(a, b)
    return s
def wiki_texts(max_chars):
    """Human-written encyclopedia text. First choice: English Wikipedia snapshot of March 2022 (before chat models were
    public). Fallback: WikiText-103 (2016)."""
    got = 0
    if not SMOKE:
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
            for k in range(41):
                p = hf_hub_download("legacy-datasets/wikipedia", f"data/20220301.en/train-{k:05d}-of-00041.parquet", repo_type="dataset")
                for bt in pq.ParquetFile(p).iter_batches(batch_size=2000, columns=["title", "text"]):
                    for ti, tx in zip(bt.column(0).to_pylist(), bt.column(1).to_pylist()):
                        t = ti + "\n" + tx + "\n\n"; got += len(t); yield t
                    if got >= max_chars: break
                try: os.remove(os.path.realpath(p))
                except Exception: pass
                if got >= max_chars or time.time() - T_START > 3600: break
            if got > 5e7: log("wikipedia 20220301.en chars", got, "files", k + 1); return
        except Exception as e: log("wikipedia download failed", repr(e)[:200])
    local = os.environ.get("WIKI_FILE")
    if not local:
        local = fetch("https://media.githubusercontent.com/media/tp6fu6m3/wikitext103/main/wiki.train.tokens", f"{WORK}/wt103.txt", 600)
    cur = []
    for line in open(local, encoding="utf-8"):
        if re.match(r"^ = [^=].* = $", line) and cur:
            t = "".join(cur) + "\n"; got += len(t); yield t; cur = []
            if got >= max_chars: break
        if line.strip() and not re.match(r"^ (= )+.* (= )+$", line[1:] if False else line) or re.match(r"^ = [^=].* = $", line):
            cur.append(detok(line.strip().strip("=").strip() if line.startswith(" = ") else line.strip()) + "\n")
    log("wikitext-103 fallback chars", got)

from fractions import Fraction
NAMES = ["Asha", "Ravi", "Meera", "Arjun", "Lena", "Omar", "Sara", "Ken", "Priya", "Tom", "Nadia", "Vikram", "Ana", "Ben", "Kavya", "Leo"]
ITEMS = ["apples", "books", "pencils", "marbles", "stickers", "coins", "cards", "shells", "tickets", "cookies"]
def fr(x): x = Fraction(x); return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
def gen_problem(R):
    k = R.randrange(6); n, n2 = R.sample(NAMES, 2); it = R.choice(ITEMS)
    if k == 0:
        a, b, c, d = R.randint(5, 60), R.randint(2, 30), R.randint(2, 9), R.randint(1, 20)
        return (f"{n} has {a} {it}. {n} buys {b} more, then the number is multiplied by {c}, and {n} gives away {d}. How many {it} does {n} have now?",
                f"{a} + {b} = {a+b}. {a+b} * {c} = {(a+b)*c}. {(a+b)*c} - {d} = {(a+b)*c-d}.", str((a+b)*c-d))
    if k == 1:
        p, pct = R.randint(2, 50) * 10, R.choice([5, 10, 20, 25, 50]); off = Fraction(p * pct, 100)
        return (f"A jacket costs {p} dollars and is on sale for {pct}% off. What is the sale price?", f"{pct}% of {p} = {fr(off)}. {p} - {fr(off)} = {fr(p-off)}.", fr(p - off))
    if k == 2:
        a, b, m = R.randint(1, 7), R.randint(1, 7), R.randint(2, 12)
        return (f"{n} and {n2} share {(a+b)*m} sweets in the ratio {a}:{b}. How many does {n2} get?", f"Parts = {a} + {b} = {a+b}. One part = {(a+b)*m} / {a+b} = {m}. {b} * {m} = {b*m}.", str(b * m))
    if k == 3:
        v, t = R.randint(20, 90), R.randint(2, 9)
        return (f"A train travels at {v} km per hour for {t} hours. How far does it travel?", f"{v} * {t} = {v*t}.", str(v * t))
    if k == 4:
        a, m, y = R.randint(3, 15), R.randint(2, 4), R.randint(2, 10)
        return (f"{n} is {m} times as old as {n2}, who is {a}. How old will {n} be in {y} years?", f"{m} * {a} = {m*a}. {m*a} + {y} = {m*a+y}.", str(m * a + y))
    xs = [R.randint(10, 99) for _ in range(R.randint(3, 5))]
    return (f"The scores are {', '.join(map(str, xs))}. What is the total?", f"{' + '.join(map(str, xs))} = {sum(xs)}.", str(sum(xs)))
def fmt_math(p, s, a): return f"Problem: {p}\nSolution: {s}\nAnswer: {a}\n\n"
def math_texts(n_generated):
    out = []
    try:
        p = fetch("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl", f"{WORK}/gsm_train.jsonl")
        for l in open(p):
            d = json.loads(l); sol, ans = d["answer"].rsplit("####", 1)
            out.append(fmt_math(d["question"].strip(), re.sub(r"<<[^>]*>>", "", sol).strip(), ans.strip().replace(",", "")))
        log("gsm8k train problems", len(out))
    except Exception as e: log("gsm8k train failed", repr(e)[:200])
    try:
        p = fetch("https://raw.githubusercontent.com/deepmind/AQuA/master/train.json", f"{WORK}/aqua_train.json", 300); k = 0
        for l in open(p):
            try: d = json.loads(l)
            except Exception: continue
            out.append(f"Problem: {d['question'].strip()}\nOptions: {' '.join(d['options'])}\nSolution: {d.get('rationale','').strip()}\nAnswer: {d['correct'].strip()}\n\n"); k += 1
            if SMOKE and k > 2000: break
        log("aqua train problems", k)
    except Exception as e: log("aqua failed", repr(e)[:200])
    R = random.Random(11); out += [fmt_math(*gen_problem(R)) for _ in range(n_generated)]
    return out
def qa_texts():
    out = []
    try:
        p = fetch("https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl", f"{WORK}/nq_train.jsonl", 300)
        for l in open(p):
            d = json.loads(l); out.append(f"Question: {d['question']}?\nAnswer: {d['answer'][0]}\n\n")
        log("nq-open train questions", len(out))
    except Exception as e: log("nq train failed", repr(e)[:200])
    return out

WIKI_CHARS = 4e6 if SMOKE else 3.2e9
log("reading text")
chunks, buf, n = [], [], 0
wiki_sample = []
TOKF = f"{WORK}/bpe"
math_t = math_texts(2000 if SMOKE else 400000); qa_t = qa_texts()
gen = wiki_texts(WIKI_CHARS); first = []; c = 0
for t in gen:
    first.append(t); c += len(t)
    if c >= (2e6 if SMOKE else 2.5e8): break
VOCAB = 2048 if SMOKE else 16384
samp = f"{WORK}/tok_sample.txt"
with open(samp, "w", encoding="utf-8") as f:
    f.write("".join(first)); f.write("".join(math_t[:60000])); f.write("".join(qa_t[:40000]))
    ex_code = [it["question"] for it in EXAM if it["kind"] == "code"]  # prompts only (no solutions), so code symbols get tokens
    f.write("\n".join(ex_code))
tok = ByteLevelBPETokenizer(); tok.train([samp], vocab_size=VOCAB, min_frequency=2, special_tokens=[]); os.remove(samp)
log("tokenizer trained, vocab", tok.get_vocab_size())
def enc_all(texts, B=2000):
    parts = []
    for i in range(0, len(texts), B):
        parts.append(np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in tok.encode_batch(texts[i:i + B])] or [np.zeros(0, np.uint16)]))
    return np.concatenate(parts) if parts else np.zeros(0, np.uint16)
wparts = [enc_all(first)]; del first; batch_t, bc = [], 0
for t in gen:
    batch_t.append(t); bc += len(t)
    if bc >= 5e7: wparts.append(enc_all(batch_t)); batch_t, bc = [], 0
if batch_t: wparts.append(enc_all(batch_t))
W = np.concatenate(wparts); del wparts
M = enc_all(math_t); Q = enc_all(qa_t) if qa_t else np.zeros(0, np.uint16)
log(f"tokens: wiki {len(W)/1e6:.1f}M, maths {len(M)/1e6:.1f}M, qa {len(Q)/1e6:.1f}M")
SRC = [(W, 0.83), (M, 0.15), (Q, 0.02)] if len(Q) > 1000 else [(W, 0.85), (M, 0.15)]
RES["data"] = dict(wiki_tokens=int(len(W)), math_tokens=int(len(M)), qa_tokens=int(len(Q)), vocab=VOCAB, mix=[p for _, p in SRC])

# ---------------------------------------------------------------- model
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
    def flops_per_token(s):  # 6 x (block parameters + output matrix); embeddings are tied
        return 6 * (sum(p.numel() for b in s.blocks for p in b.parameters()) + s.emb.weight.numel())

def get_batch(R, B, T):
    xs = []
    ps = np.array([p for _, p in SRC]); ps = ps / ps.sum()
    for k in R.choice(len(SRC), size=B, p=ps):
        a = SRC[k][0]; i = R.integers(0, len(a) - T - 1); xs.append(a[i:i + T + 1].astype(np.int64))
    x = torch.from_numpy(np.stack(xs)).to(DEV); return x[:, :-1], x[:, 1:]
def train(m, C, lr, B, T, deadline):
    tokens = C / m.flops_per_token(); steps = max(10, int(tokens // (B * T))); R = np.random.default_rng(0)
    dec = [p for p in m.parameters() if p.dim() >= 2]; nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda"); warm = max(1, steps // 50); t0 = time.time(); done = 0; last = 0.0
    for s in range(steps):
        f = (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        for g in opt.param_groups: g["lr"] = lr * f
        x, y = get_batch(R, B, T)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(m(x).flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
        done = s + 1; last = 0.98 * last + 0.02 * loss.item() if s else loss.item()
        if s == 50 or (s % max(1, steps // 10) == 0 and s):
            rate = done * B * T / (time.time() - t0); log(f"  step {done}/{steps} loss {last:.3f} {rate/1e3:.1f}k tok/s")
        if time.time() > deadline: log("  rung cut short by the time budget"); break
        if not math.isfinite(loss.item()): log("  loss not finite, stopping rung"); break
    return dict(steps_planned=steps, steps_done=done, tokens=done * B * T, C_actual=done * B * T * m.flops_per_token(), train_loss=last, seconds=time.time() - t0,
                tok_per_s=done * B * T / max(1e-9, time.time() - t0))

# ---------------------------------------------------------------- scoring on the exam
def norm(s): return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).split())
def num(s):
    mm = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", "")); return float(mm[-1]) if mm else None
def correct(it, out):
    k = it["kind"]
    if k == "mc": return str(out).strip()[:1].upper() == it["answer"]
    if k == "num": a, b = num(out), num(it["answer"]); return a is not None and abs(a - b) < 1e-6
    if k == "text": return norm(out) != "" and norm(out) in {norm(a) for a in it["answer"]}
    if k == "code":
        src = it["question"] + str(out) + "\n\n" + it["test"] + f"\n\ncheck({it['entry']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src)
        try: return subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=10).returncode == 0
        except Exception: return False
        finally: os.unlink(f.name)
@torch.no_grad()
def logp_tail(m, prefix, tail):
    """Sum log-probability of `tail` tokens given `prefix` (cropped to the context)."""
    a, b = tok.encode(prefix).ids, tok.encode(tail).ids
    if not b: return -1e9
    ids = (a + b)[-(m.ctx):]; nb = min(len(b), len(ids) - 1)
    if nb <= 0: return -1e9
    x = torch.tensor([ids], device=DEV)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        lp = F.log_softmax(m(x[:, :-1]).float(), -1)
    return lp[0, -nb:].gather(-1, x[0, -nb:, None]).sum().item()
@torch.no_grad()
def greedy(m, prompt, max_new, stop):
    ids = tok.encode(prompt).ids; n0 = len(ids)
    for _ in range(max_new):
        x = torch.tensor([ids[-m.ctx:]], device=DEV)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            nxt = int(m(x)[0, -1].argmax())
        ids.append(nxt); g = tok.decode(ids[n0:])
        if stop(g): break
    return tok.decode(ids[n0:])
def answer(m, it):
    k = it["kind"]
    if k == "mc":
        sc = [logp_tail(m, f"Question: {it['question']}\nAnswer:", " " + str(o)) - logp_tail(m, "Answer:", " " + str(o)) for o in it["options"]]
        return "ABCDEFGHIJ"[max(range(len(sc)), key=lambda i: sc[i])]
    if k == "num":
        g = greedy(m, f"Problem: {it['question']}\nSolution:", 160, lambda g: "Answer:" in g and "\n" in g.split("Answer:", 1)[1])
        return g.split("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in g else ""
    if k == "text":
        q = it["question"]; g = greedy(m, f"Question: {q}{'' if q.endswith('?') else '?'}\nAnswer:", 24, lambda g: "\n" in g.strip() or g.endswith("\n"))
        return g.strip().split("\n")[0]
    g = greedy(m, it["question"], 120, lambda g: "\ndef " in g or "\n\n\n" in g or "\nclass " in g)
    return g.split("\ndef ")[0].split("\nclass ")[0]
def score(m):
    m.eval(); r = {}
    for it in EXAM:
        try: ok = bool(correct(it, answer(m, it)))
        except Exception: ok = False
        r.setdefault(f"{it['bench']}_{it['split']}", []).append(ok)
    m.train(); return {k: [int(sum(v)), len(v)] for k, v in r.items()}

# ---------------------------------------------------------------- the ladder
RUNGS = [(1e12, 64, 2, 2, 128, 16, 2e-3)] if SMOKE else [
    # (compute, width, layers, heads, context, batch, learning rate)
    (1e15, 192, 6, 6, 512, 64, 1.5e-3), (1e16, 320, 8, 8, 512, 64, 1.2e-3), (1e17, 512, 10, 8, 512, 48, 8e-4), (3e17, 640, 12, 10, 512, 40, 6e-4)]
for C, d, L, h, ctx, B, lr in RUNGS:
    left = BUDGET_S - (time.time() - T_START)
    if left < 900 and not SMOKE: log(f"rung {C:.0e} skipped: {left/60:.0f} minutes left"); continue
    try:
        m = GPT(d, L, h, ctx, VOCAB).to(DEV); log(f"rung {C:.0e}: width {d}, layers {L}, FLOPs/token {m.flops_per_token():.2e}, planned tokens {C/m.flops_per_token()/1e6:.1f}M")
        eval_reserve = 600 if SMOKE else 1500
        try: tr = train(m, C, lr, B, ctx, T_START + BUDGET_S - eval_reserve)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            log("  out of memory; halving the batch"); del m; torch.cuda.empty_cache(); m = GPT(d, L, h, ctx, VOCAB).to(DEV); tr = train(m, C, lr, B // 2, ctx, T_START + BUDGET_S - eval_reserve)
        t0 = time.time(); sc = score(m); tr.update(C=C, width=d, layers=L, scores=sc, eval_seconds=time.time() - t0); RES["rungs"].append(tr)
        log(f"rung {C:.0e} scores", json.dumps(sc)); del m
        if DEV == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        log(f"rung {C:.0e} failed:", repr(e)[:300]); RES["rungs"].append(dict(C=C, error=repr(e)[:300]))
RES["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["total_minutes"] = (time.time() - T_START) / 60; save()
print("\n=== RESULT TABLE (correct / questions; dev and hidden halves) ===")
for r in RES["rungs"]:
    if "scores" in r: print(f"C={r['C_actual']:.2e} width {r['width']}: " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(r["scores"].items())))
print("done")
