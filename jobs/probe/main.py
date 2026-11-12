# GPU access probe: reports what the Kaggle machine offers. No data, no training.
import json, subprocess, time, urllib.request
r = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
try: r["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e: r["nvidia_smi"] = "error: " + repr(e)
try:
    import torch
    r["torch"] = torch.__version__; r["cuda"] = torch.cuda.is_available(); r["gpus"] = torch.cuda.device_count()
    if r["cuda"]:
        x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16); torch.cuda.synchronize(); t = time.time()
        for _ in range(20): y = x @ x
        torch.cuda.synchronize(); r["matmul_tflops_fp16"] = round(20 * 2 * 8192 ** 3 / (time.time() - t) / 1e12, 1)
except Exception as e: r["torch_error"] = repr(e)
for name, url in (("huggingface", "https://huggingface.co/api/datasets/legacy-datasets/wikipedia"), ("github", "https://raw.githubusercontent.com/openai/human-eval/master/README.md")):
    try: r["net_" + name] = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"}), timeout=20).status
    except Exception as e: r["net_" + name] = "error: " + repr(e)[:120]
for mod in ("tokenizers", "datasets", "numpy"):
    try: r["has_" + mod] = __import__(mod).__version__
    except Exception as e: r["has_" + mod] = False
json.dump(r, open("results.json", "w"), indent=1); print(json.dumps(r, indent=1))
