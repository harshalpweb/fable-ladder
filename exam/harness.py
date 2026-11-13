"""Exam harness. A system is a module with answer(item) -> str (item has no 'answer'/'test' fields).
Usage: python3 harness.py <system_module> [dev|hidden]   -> prints score per benchmark, saves runs/<system>_<split>.json"""
import importlib, json, os, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/systems")
def norm(s): return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).split())
def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", "")); return float(m[-1]) if m else None
def correct(it, out):
    k = it["kind"]
    if k == "mc": return str(out).strip()[:1].upper() == it["answer"]
    if k == "num": a, b = num(out), num(it["answer"]); return a is not None and abs(a - b) < 1e-6
    if k == "text": return norm(out) in {norm(a) for a in it["answer"]} and norm(out) != ""
    if k == "code":
        src = it["question"] + str(out) + "\n\n" + it["test"] + f"\n\ncheck({it['entry']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src)
        try: return subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired: return False
        finally: os.unlink(f.name)
def main(name, split="dev"):
    sysm = importlib.import_module(name); items = [json.loads(l) for l in open(f"{HERE}/exam_v1.jsonl")]
    res, t0, words = {}, time.time(), 0
    for it in items:
        if it["split"] != split: continue
        pub = {k: v for k, v in it.items() if k not in ("answer", "test")}
        try: out = sysm.answer(pub)
        except Exception as e: out = ""
        words += len(str(out).split()); ok = bool(correct(it, out)); res.setdefault(it["bench"], []).append((it["id"], ok))
    summ = {b: round(sum(o for _, o in v) / len(v), 4) for b, v in res.items()}; summ["_n"] = {b: len(v) for b, v in res.items()}
    summ["_seconds"] = round(time.time() - t0, 1); os.makedirs(f"{HERE}/runs", exist_ok=True)
    json.dump(dict(summary=summ, detail=res), open(f"{HERE}/runs/{name}_{split}.json", "w")); print(name, split, summ)
if __name__ == "__main__": main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev")
