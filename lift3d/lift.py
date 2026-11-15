"""Lift every 2D clip to 3D with VideoPose3D (COCO-input checkpoint), then save.

Recipe follows the official VideoPose3D inference (run.py / INFERENCE.md):
  * input: COCO-17 2D keypoints in screen coords, normalised with
    normalize_screen_coordinates: x -> 2x-1, y -> 2y*(h/w) - h/w   (our data is 0-1 of w and h)
  * model: TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3]) -> receptive field 243 frames,
    each clip edge-padded by 121 frames on both sides (our clips are only 3-25 frames long,
    so most of the window is replicated edge frames: a known limitation)
  * test-time flip augmentation: also lift the mirrored input, un-mirror, average
  * output: Human3.6M-17 joints, root-relative (pelvis = joint 0 set to 0), camera axes,
    roughly metres. H36M order: 0 pelvis, 1 R hip, 2 R knee, 3 R foot, 4 L hip, 5 L knee,
    6 L foot, 7 spine, 8 thorax, 9 neck/nose, 10 head, 11 L shoulder, 12 L elbow, 13 L wrist,
    14 R shoulder, 15 R elbow, 16 R wrist.

Input is the ORTHODOX data (V4 mirrored in 2D before lifting), so rows match the main pipeline.

    /opt/anaconda3/bin/python lift3d/lift.py                    # needs the checkpoint
    /opt/anaconda3/bin/python lift3d/lift.py --random-weights   # pipeline test only, output is meaningless
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mem_guard  # noqa: F401,E402  before torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines import to_orthodox  # noqa: E402
from progress import progress  # noqa: E402
from punch_data import load_dataset  # noqa: E402

from lift3d.vp3d_model import TemporalModel  # noqa: E402  our own re-implementation

CKPT = ROOT / "third_party" / "VideoPose3D" / "checkpoint" / "pretrained_h36m_detectron_coco.bin"
KPS_LEFT, KPS_RIGHT = [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]   # COCO input
JOINTS_LEFT, JOINTS_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]          # H36M output
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def screen_norm(xy01, aspect):
    """Our 0-1 coords (x/w, y/h) -> VideoPose3D screen coords; only h/w matters."""
    out = np.empty_like(xy01)
    out[..., 0] = 2 * xy01[..., 0] - 1
    out[..., 1] = (2 * xy01[..., 1] - 1) * aspect
    return out


@torch.no_grad()
def lift_clip(model, seq, pad):
    """seq: (L, 17, 2) normalised. Returns (L, 17, 3) root-relative, flip-averaged."""
    x = np.pad(seq, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    flipped = x.copy()
    flipped[..., 0] *= -1
    flipped[:, KPS_LEFT + KPS_RIGHT] = flipped[:, KPS_RIGHT + KPS_LEFT]
    batch = torch.tensor(np.stack([x, flipped]), dtype=torch.float32, device=DEVICE)
    out = model(batch)                                      # (2, L, 17, 3)
    out[1, :, :, 0] *= -1
    out[1, :, JOINTS_LEFT + JOINTS_RIGHT] = out[1, :, JOINTS_RIGHT + JOINTS_LEFT]
    out = out.mean(0)
    out = out - out[:, :1]                                  # root-relative (pelvis = 0)
    return out.cpu().numpy()


@torch.no_grad()
def lift_frames(model, frames, rf, batch=512):
    """PER-FRAME mode. frames: (F, 17, 2) normalised. Each frame is repeated over the whole
    receptive field ("a person holding this pose"), so its 3D is estimated on its own.
    Needed because our clips (3-25 frames) are far shorter than the 243-frame window:
    edge-padding a clip makes ~95% of every window a frozen pose and the temporal model
    smooths the punch away (measured: jab wrist moved ~2 cm instead of tens of cm).
    Flip test-time augmentation as in the official recipe. Returns (F, 17, 3) root-relative."""
    outs = []
    for s in range(0, len(frames), batch):
        f = frames[s:s + batch]
        fl = f.copy(); fl[..., 0] *= -1; fl[:, KPS_LEFT + KPS_RIGHT] = fl[:, KPS_RIGHT + KPS_LEFT]
        both = torch.tensor(np.concatenate([f, fl]), dtype=torch.float32, device=DEVICE)  # (2B, 17, 2)
        seq = both[:, None].expand(-1, rf, -1, -1).contiguous()                          # (2B, rf, 17, 2)
        out = model(seq)[:, 0]                                                            # (2B, 17, 3)
        a, b = out[:len(f)], out[len(f):].clone()
        b[..., 0] *= -1; b[:, JOINTS_LEFT + JOINTS_RIGHT] = b[:, JOINTS_RIGHT + JOINTS_LEFT]
        o = (a + b) / 2
        outs.append((o - o[:, :1]).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspect", type=float, default=9 / 16, help="video height / width (16:9 assumed)")
    ap.add_argument("--random-weights", action="store_true", help="pipeline test without the checkpoint")
    ap.add_argument("--mode", choices=["frame", "clip"], default="frame",
                    help="frame: lift each frame on its own (default); clip: whole clip edge-padded (smooths motion away)")
    ap.add_argument("--out", default=str(ROOT / "lift3d" / "lifted_h36m.npz"))
    args = ap.parse_args()

    model = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024)
    if args.random_weights:
        torch.manual_seed(0)
        ck_info = "RANDOM WEIGHTS (pipeline test only)"
    else:
        if not CKPT.exists():
            sys.exit(f"checkpoint missing: {CKPT}\nsee lift3d/SOURCES.md for how to obtain it")
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)   # safe mode: tensors only, no code
        model.load_state_dict(ck["model_pos"])
        ck_info = "sha256 " + hashlib.sha256(CKPT.read_bytes()).hexdigest()
    model = model.to(DEVICE).eval()
    pad = (model.receptive_field() - 1) // 2
    print(f"receptive field {model.receptive_field()} frames (pad {pad}), device {DEVICE}, {ck_info}")

    X, y, video_id, lengths = load_dataset()
    Xo = to_orthodox(X, video_id)
    X3 = np.zeros((len(X), X.shape[1], 17, 3), dtype=np.float32)
    if args.mode == "clip":
        progress.start("lift3d clip-mode", total=len(X))
        for i in range(len(X)):
            L = lengths[i]
            X3[i, :L] = lift_clip(model, screen_norm(Xo[i, :L], args.aspect), pad)
            if i % 100 == 0 or i == len(X) - 1:
                progress.tick(min(100, len(X) - progress.done), note=f"clip {i + 1}/{len(X)}")
    else:
        real = np.arange(X.shape[1])[None, :] < lengths[:, None]           # (N, 25)
        frames = screen_norm(Xo[real], args.aspect)                         # (F, 17, 2) all real frames
        progress.start("lift3d frame-mode", total=len(frames))
        out, step = [], 4096
        for s in range(0, len(frames), step):
            out.append(lift_frames(model, frames[s:s + step], model.receptive_field()))
            progress.tick(len(frames[s:s + step]), note=f"frame {min(s + step, len(frames))}/{len(frames)}")
        X3[real] = np.concatenate(out)
    progress.finish()

    np.savez_compressed(args.out, X3=X3.astype(np.float16), lengths=lengths, y=y.astype(str),
                        video_id=video_id, aspect=args.aspect, checkpoint=ck_info, mode=args.mode)
    print(f"saved {args.out}  X3 {X3.shape}")


if __name__ == "__main__":
    main()
