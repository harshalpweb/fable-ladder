# Direct file listing and download of the March 2022 Wikipedia snapshot (no streaming API). Hard time limits.
import json, signal, time
r = {}
def lim(sec):
    def h(*a): raise TimeoutError(f"over {sec}s")
    signal.signal(signal.SIGALRM, h); signal.alarm(sec)
try:
    lim(120); from huggingface_hub import list_repo_files, hf_hub_download
    for repo in ("legacy-datasets/wikipedia", "wikimedia/wikipedia"):
        try:
            fs = list_repo_files(repo, repo_type="dataset"); en = [f for f in fs if "20220301.en" in f]
            r[repo] = dict(total_files=len(fs), en2022=en[:3], n_en2022=len(en), sample=[f for f in fs if f.endswith(".parquet")][:3])
        except Exception as e: r[repo] = repr(e)[:200]
    signal.alarm(0)
    lim(120)
    try:
        fs = list_repo_files("legacy-datasets/wikipedia", repo_type="dataset", revision="refs/convert/parquet"); en = [f for f in fs if f.startswith("20220301.en")]
        r["convert_branch"] = dict(n=len(en), first=en[:2])
    except Exception as e: r["convert_branch"] = repr(e)[:200]
    signal.alarm(0)
    cand = None
    if isinstance(r.get("legacy-datasets/wikipedia"), dict) and r["legacy-datasets/wikipedia"]["en2022"]:
        cand = ("legacy-datasets/wikipedia", [f for f in r["legacy-datasets/wikipedia"]["en2022"] if f.endswith(".parquet")][0], None)
    elif isinstance(r.get("convert_branch"), dict) and r["convert_branch"]["first"]:
        cand = ("legacy-datasets/wikipedia", r["convert_branch"]["first"][0], "refs/convert/parquet")
    if cand:
        lim(600); t = time.time(); p = hf_hub_download(cand[0], cand[1], repo_type="dataset", revision=cand[2]); signal.alarm(0)
        import pyarrow.parquet as pq; tb = pq.read_table(p); d = tb.slice(0, 2).to_pylist()
        r["download"] = dict(file=cand[1], seconds=round(time.time() - t, 1), rows=tb.num_rows, columns=tb.column_names, first_title=d[0].get("title"), chars=sum(len(x) for x in tb.column("text").to_pylist()[:2000]))
except Exception as e: r["fatal"] = repr(e)[:300]
json.dump(r, open("results.json", "w"), indent=1); print(json.dumps(r, indent=1))
