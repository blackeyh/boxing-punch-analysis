"""Download the public OWLv2 weights (safetensors + config/tokenizer files only) into the Hugging Face cache.

Needed because the installed Brotli (1.0.9) breaks httpx2's brotli decoding: we ask the server for gzip only.
    python bagroute/fetch_model.py
"""
import httpx2
import huggingface_hub as hf

MODEL = "google/owlv2-base-patch16-ensemble"
hf.set_client_factory(lambda: httpx2.Client(headers={"Accept-Encoding": "gzip"}, follow_redirects=True, timeout=60))
p = hf.snapshot_download(MODEL, allow_patterns=["*.json", "*.txt", "*.safetensors"])
print(p)
