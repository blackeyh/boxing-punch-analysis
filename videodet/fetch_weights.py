"""Download the pinned pretrained weights into the HF / torch caches (run once on the login node,
which has internet; jobs then run with HF_HUB_OFFLINE=1). Prints SHA-256 for videodet/SOURCES.md."""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import MODEL_IDS  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.txt", "vocab*", "merges*", "tokenizer*"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for name in sys.argv[1:] or MODEL_IDS:
    rid, rev = MODEL_IDS[name]
    d = Path(snapshot_download(rid, revision=rev, allow_patterns=PATTERNS))
    for f in sorted(d.glob("*.safetensors")):
        print(f"{name} {rid}@{rev} {f.name} {f.stat().st_size} sha256={sha(f)}", flush=True)

import torchvision  # noqa: E402
w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w)
p = Path.home() / ".cache/torch/hub/checkpoints" / Path(w.url).name
print(f"detector {w.url} {p.stat().st_size} sha256={sha(p)} torchvision={torchvision.__version__}")
