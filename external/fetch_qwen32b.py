"""Download Qwen3-VL-32B-Instruct weights into /l/users/<user>/hf_cache3 (run as a CPU Slurm job;
two login-node attempts stalled). Plain HTTP (xet disabled), resumable, prints progress per file."""
import os, sys, time
os.environ["HF_HOME"] = "/l/users/<user>/hf_cache3"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import HfApi, hf_hub_download
repo = "Qwen/Qwen3-VL-32B-Instruct"
files = [f.rfilename for f in HfApi().model_info(repo, files_metadata=True).siblings]
print(len(files), "files", flush=True)
t0 = time.time()
for i, f in enumerate(sorted(files)):
    for attempt in range(5):
        try:
            hf_hub_download(repo, f); break
        except Exception as e:
            print(f"  retry {f}: {e}", flush=True); time.sleep(10)
    print(f"[{i + 1}/{len(files)}] {f}  {time.time() - t0:.0f}s", flush=True)
print("DOWNLOAD DONE", flush=True)
