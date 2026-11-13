"""Rebuilds exam v1 (1,162 real questions, half dev, half hidden) from public sources, bit for bit.
Usage: python3 exam_builder.py   -> writes exam_v1.jsonl next to this file and checks it against the known hash.
The question file is never committed here: the GPQA authors ask that their questions not be posted openly."""
import ast, csv, gzip, hashlib, io, json, os, random, zipfile, urllib.request
EXAM_SHA = "483e9569cf4f235c0dcb753ae5fefcd490474044162b4272c6360e827a9bfab0"
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
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__)); work = os.path.join(here, "_sources"); os.makedirs(work, exist_ok=True)
    items, sha = build_exam(work)
    if sha != EXAM_SHA: raise SystemExit(f"Exam differs from exam v1: got {sha[:16]}, expected {EXAM_SHA[:16]}. A public source has changed.")
    with open(os.path.join(here, "exam_v1.jsonl"), "w") as f: f.write("\n".join(json.dumps(i) for i in items) + "\n")
    n = {}
    for i in items: n[i["bench"]] = n.get(i["bench"], 0) + 1
    print("exam v1 rebuilt:", len(items), "questions", n, "hash", sha[:16], "OK")
