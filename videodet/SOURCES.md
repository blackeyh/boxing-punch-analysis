# Third-party models used by videodet

All weights were downloaded over verified HTTPS on the HPC login node (Hugging Face Hub via
`huggingface_hub.snapshot_download` with a pinned commit, and download.pytorch.org via
torchvision) by `videodet/fetch_weights.py`, into `~/.cache/huggingface` / `~/.cache/torch` on
the HPC home (NFS). Jobs run with `HF_HUB_OFFLINE=1`. Only `.safetensors` weights are loaded from
Hugging Face (`use_safetensors=True`); no remote code (`trust_remote_code` is never set). No weights are
committed to git.

| Use | Model | Source / revision | Licence | File | SHA-256 |
|---|---|---|---|---|---|
| Person boxes (crop only) | Faster R-CNN ResNet-50 FPN v2, COCO | torchvision 0.24.0 `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`, https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth | BSD-3-Clause (torchvision code); weights trained on COCO | 175,221,657 B | `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf` (torchvision also checks the `dd69338a` prefix) |
| Main features + zero-shot | VideoMAE-Base, Kinetics-400 fine-tuned (16 frames, 224) | `MCG-NJU/videomae-base-finetuned-kinetics` @ `488eb9a0565f257b32866000305c8178965eb9f6` | CC-BY-NC-4.0 (research / non-commercial) | model.safetensors 346,161,616 B | `065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73` |
| Main features + zero-shot | VideoMAE-Large, Kinetics-400 fine-tuned (16 frames, 224) | `MCG-NJU/videomae-large-finetuned-kinetics` @ `0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4` | CC-BY-NC-4.0 (research / non-commercial) | model.safetensors 1,217,128,032 B | `92d33bedfe4f171705af418b5ddf059a2beb8ef4d1532621c2ae0a6693dbb8bc` |
| Zero-shot video-text | X-CLIP B/32, 8 frames, Kinetics-400 | `microsoft/xclip-base-patch32` @ `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963` | MIT | model.safetensors 786,414,772 B | `abf286e8cdd0612761c3e42d3a55eca998382dfa67a04a0f3fdcdfa4f150cdbb` |
| VLM second opinion (subset) | Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 | model-00001-of-00004.safetensors 4,902,275,944 B; -00002 4,915,962,496 B; -00003 4,999,831,048 B; -00004 2,716,270,024 B | `d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa`; `8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5`; `83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192`; `0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5` (match the Hub's content-addressed blob names) |

Software: HPC conda env `punchvid` (Python 3.12.7, torch 2.9.0+cu128, torchvision 0.24.0,
transformers 4.57.1, opencv-python-headless 4.11, numpy 1.26.4, scikit-learn 1.7.2), built by
`~/punchvid_setup.sh` (see `videodet/REPORT.md`). Licence note: the VideoMAE checkpoints are
**non-commercial**; fine for this research project, but a commercial app would need to swap them
(e.g. for an Apache/MIT video backbone) and re-run `extract.py` + `score.py`.
