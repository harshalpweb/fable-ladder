# Which way of reading the March 2022 Wikipedia snapshot works on Kaggle? No training.
import json, time
r = {}
def try_(name, fn):
    t = time.time()
    try:
        ds = fn(); n = c = 0; first = None
        for ex in ds:
            first = first or ex.get("title"); n += 1; c += len(ex.get("text", ""))
            if n >= 3000: break
        r[name] = dict(ok=True, first=first, articles=n, chars=c, seconds=round(time.time() - t, 1))
    except Exception as e: r[name] = dict(ok=False, error=repr(e)[:300])
from datasets import load_dataset
try_("legacy_named", lambda: load_dataset("legacy-datasets/wikipedia", "20220301.en", split="train", streaming=True))
try_("legacy_parquet_glob", lambda: load_dataset("parquet", data_files="hf://datasets/legacy-datasets/wikipedia/data/20220301.en/*.parquet", split="train", streaming=True))
try_("wikipedia_named", lambda: load_dataset("wikipedia", "20220301.en", split="train", streaming=True))
try:
    from huggingface_hub import list_repo_files
    fs = [f for f in list_repo_files("legacy-datasets/wikipedia", repo_type="dataset") if "20220301.en" in f]; r["files"] = fs[:5] + [len(fs)]
except Exception as e: r["files"] = repr(e)[:200]
json.dump(r, open("results.json", "w"), indent=1); print(json.dumps(r, indent=1))
