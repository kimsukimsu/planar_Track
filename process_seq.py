"""
python process_seq.py
    --root /workspace/WOFT/pot        
    --config pytracking/configs/WOFT.py        
    --gpu 0 
"""

import os, sys, logging, argparse
from pathlib import Path
from typing  import List
import numpy as np
import cv2
import time
from tqdm import tqdm

from pytracking.utils.config import load_config
from pytracking.utils        import io as io_utils

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s:%(message)s")
log = logging.getLogger("dump")

# ───────────────────────── corner util ─────────────────────────
def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (4,2)  → (4,2)  with homography"""
    hom = np.hstack([pts, np.ones((4,1))]).T     # 3×4
    proj = H @ hom
    proj /= proj[2,:]
    return proj[:2,:].T       # (4,2)

# ───────────────────────── single sequence ─────────────────────
def process_sequence(seq_dir: Path,
                     tracker_cls,
                     cfg,
                     out_root: Path) -> None:
    name    = seq_dir.name           # V01_1 …
    img_dir = seq_dir / "img"
    imgs    = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        log.warning("[skip] no images in %s", seq_dir); return

    # --- 템플릿 코너(픽셀) 불러오기 ---------------------------
    gt_arr = np.loadtxt(seq_dir / f"{name}_gt_points.txt")

    # ① 첫 줄이 8개 숫자인 경우 (8,) or (1,8)
    if gt_arr.ndim == 1 and gt_arr.size == 8:
        tpl_xy = gt_arr.reshape(4, 2)

    elif gt_arr.ndim == 2 and gt_arr.shape[1] == 8:  # 여러 줄(프레임) 저장
        # 첫 줄이 전부 0 → 다음 유효 줄 선택
        for row in gt_arr:
            if not np.allclose(row, 0.0):
                tpl_xy = row.reshape(4, 2)
                break
        else:
            raise ValueError("all GT rows are zero")

    # ② 이미 (4,2) 모양
    elif gt_arr.shape == (4, 2):
        tpl_xy = gt_arr

    else:
        raise ValueError("unexpected gt_points shape")

    first = cv2.imread(str(imgs[0]))
    h, w  = first.shape[:2]

    # --- init mask (convex quad) ------------------------------
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [tpl_xy.astype(np.int32).reshape(-1,1,2)], 255)

    # --- tracker ----------------------------------------------
    tracker = tracker_cls(cfg)
    tracker.init(first, mask)

    out_path = out_root / f"{name}_WOFT_re.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w")
    np.set_printoptions(floatmode="fixed", precision=4, suppress=True)
    f.write(" ".join(f"{v:.4f}" for v in tpl_xy.reshape(-1)) + "\n")

    cap = io_utils.GeneralVideoCapture(img_dir)
    cap.read()                
    frame_idx = 1                
    last_H    = np.eye(3)
    
    t0 = time.perf_counter()  #시작 시간
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            H_cur2init, _ = tracker.track(frame)
            last_H = H_cur2init.copy()
        except Exception:
            H_cur2init = last_H.copy()

        cur_xy = apply_H(np.linalg.inv(H_cur2init), tpl_xy)
        f.write(" ".join(f"{v:.4f}" for v in cur_xy.reshape(-1)) + "\n")
        frame_idx += 1

    elapsed = time.perf_counter() - t0                     
    fps = frame_idx / elapsed if elapsed > 0 else 0.0    
    csv_fp.write(f"{name},{frame_idx},{fps:.2f}\n")        
    csv_fp.flush()  
    f.close()
    log.info("%s saved (%d frames, %.2f FPS)",name, frame_idx, fps)

# ──────────────────────────── main ────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        "Dump WOFT-predicted corners for POT-280",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root",   required=True, type=Path,
                    help="POT-280 root (e.g. /workspace/WOFT/pot)")
    ap.add_argument("--config", type=Path,
                    default=Path("pytracking/configs/WOFT.py"))
    ap.add_argument("--gpu",    type=str,
                    help="CUDA_VISIBLE_DEVICES value (optional)")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/workspace/WOFT/WOFT_re"),
                    help="output folder for *_WOFT_re.txt")
    ap.add_argument("--csv", type=Path,                    
                    default=Path("/workspace/WOFTseq_fps.csv"),
                    help="per-sequence FPS csv")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = load_config(args.config)
    tracker_cls = cfg.tracker_class

    seq_dirs: List[Path] = sorted({p.parent
                                   for p in args.root.rglob("*_gt_points.txt")})
    log.info("found %d sequences", len(seq_dirs))

    with args.csv.open("w") as csv_fp:                    
        csv_fp.write("sequence,n_frames,fps\n")
    for seq in tqdm(seq_dirs, desc="Sequences"):
        process_sequence(seq, tracker_cls, cfg,args.out_dir, csv_fp)       


if __name__ == "__main__":
    main()