# 3D lifting: sources and provenance

## Model
- **VideoPose3D** (Pavllo et al., *3D human pose estimation in video with temporal convolutions and semi-supervised training*, CVPR 2019). Licence: **CC BY-NC 4.0** (non-commercial).
- Official repo cloned for reference only into `third_party/VideoPose3D` (git-ignored), commit `1afb1ca0f1237776518469876342fc8669d3f6a9` (2020-08-05).
- **We do not run their code.** `lift3d/vp3d_model.py` is our own re-implementation of `TemporalModel` (architecture copied by reading `common/model.py`), with identical parameter names so the official weights load with `strict=True`. Running cloned third-party code was blocked by the Claude Code safety classifier, and re-implementing a ~60-line model is the safer route anyway.
- Inference recipe follows the official `INFERENCE.md` / `run.py`: screen normalisation, edge padding by (receptive field − 1)/2 = 121 frames, flip test-time augmentation, root-relative output.

## Checkpoint
- `pretrained_h36m_detectron_coco.bin`: the official "in the wild" model. **Input: COCO-17 2D keypoints** (our format). **Output: Human3.6M-17 3D joints.** Architecture `-arc 3,3,3,3,3` (receptive field 243 frames).
- Official URL: `https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin`
- Expected location: `third_party/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin`
- **Status (2026-11-12): NOT downloaded.** The server's TLS certificate expired on 2026-11-11 23:59:59 GMT (issuer DigiCert Global G2 TLS RSA SHA256 2020 CA1; checked from the Mac and from the HPC, same result). An insecure download (`curl -k`) was blocked by the safety classifier. No independent mirror found (search 2026-11-12). **Awaiting the user's decision.**
- Whatever the source, `lift.py` loads it with `torch.load(..., weights_only=True)`, which reads tensors only and cannot execute code from the file, and records the file's SHA-256 in the output.

## Known limitations
- Our clips are 3–25 frames, but the model's window is 243 frames, so most of each window is edge-replicated frames: little temporal context. V6 is a continuous stream and could be lifted with full context later.
- Aspect ratio assumed 16:9 (`--aspect`). The video-3D agent will report each video's true resolution; rerun with the right value if needed.
- The lifter was trained on Human3.6M (everyday motion, no boxing) and is least reliable side-on, where we need it most.
