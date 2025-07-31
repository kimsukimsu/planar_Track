import numpy as np, csv, argparse, logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s:%(message)s")

# ─────────────────── util ────────────────────
def read_pred_xy(txt: Path) -> np.ndarray:
    """
    txt 각 줄마다 x1 y1 x2 y2 x3 y3 x4 y4 ->(N,4,2) float32
    """
    rows = []
    with open(txt, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            nums = [float(v) for v in ln.replace('\t', ' ').split()]
            if len(nums) == 9:          # frameNo + 8
                nums = nums[1:]
            if len(nums) != 8:
                raise ValueError(f"{txt.name}: line에 좌표 8개가 아님 ({len(nums)})")
            rows.append(nums)
    if not rows:
        raise ValueError(f"{txt.name}: empty")
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4, 2)

def eal_pts(gt_xy: np.ndarray, pred_xy: np.ndarray) -> np.ndarray:
    """
    gt_xy, pred_xy : (N,4,2) → eAL (N,)  float32
    """
    diff = gt_xy - pred_xy             # (N,4,2)
    return np.sqrt((diff**2).sum(-1).mean(-1))  # RMSE 4 pts

# ───────────────── eval one sequence ─────────────────
def eval_seq(seq_dir: Path, woft_txt: Path):
    try:
        name     = seq_dir.name
        gt_raw   = np.loadtxt(seq_dir / f"{name}_gt_points.txt")  # (N,8)
        gt_H     = np.loadtxt(seq_dir / f"{name}_gt_homography.txt").reshape(-1, 3, 3)
        pred_xy  = read_pred_xy(woft_txt)                         # (N,4,2)

        # ── 프레임 필터: GT 포인트 8개 모두 0 → 미주(frames not annotated) ->URP용 필터
        valid = ~(gt_raw == 0).all(1)
        if valid.sum() == 0:
            raise ValueError("all GT rows are zero")

        gt_raw  = gt_raw [valid]                 # (M,8)
        gt_H    = gt_H  [valid]                  # (M,3,3)  (안 써도 길이 맞추기)
        pred_xy = pred_xy[valid]                 # (M,4,2)

        # GT 이미 픽셀 좌표로 주어짐 → 바로 4점 reshaping
        gt_xy = gt_raw.reshape(-1, 4, 2).astype(np.float32)  # (M,4,2)

        if len(gt_xy) != len(pred_xy):
            raise ValueError("GT / Pred length mismatch")

        eals = eal_pts(gt_xy, pred_xy)                # (M,)
        p5   = float((eals <=  5).mean()*100)
        p15  = float((eals <= 15).mean()*100)
        return p5, p15, float(eals.mean()), len(eals)

    except Exception as e:
        logging.warning("[skip] %s (%s)", seq_dir.name, e)
        return None

# ───────────────── main ─────────────────
def main(root: Path, woft_dir: Path, output_csv: Path):
    seq_dirs = sorted({p.parent for p in root.rglob("*_gt_homography.txt")})
    rows, p5_sum, p15_sum, n_total, skipped = [], 0.0, 0.0, 0, 0

    for seq in tqdm(seq_dirs, desc="Sequences"):
        res = eval_seq(seq, woft_dir / f"{seq.name}_WOFT_re.txt")
        if res is None:
            skipped += 1
            continue
        p5, p15, mean_e, n = res
        logging.info(f"{seq.name:15s} P@5={p5:.2f}%  P@15={p15:.2f}%  mean_e={mean_e:.3f}  (N={n})")
        rows.append([seq.name, n, f"{p5:.2f}", f"{p15:.2f}", f"{mean_e:.3f}"])
        p5_sum  += p5  * n
        p15_sum += p15 * n
        n_total += n

    if n_total == 0:
        logging.error("all sequences skipped (%d)", skipped); return

    p5_avg, p15_avg = p5_sum/n_total, p15_sum/n_total
    logging.info("======== SUMMARY ========")
    logging.info(f"Overall P@5  = {p5_avg:.2f}%")
    logging.info(f"Overall P@15 = {p15_avg:.2f}%")
    logging.info(f"Skipped sequences: {skipped}")

    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f);  w.writerow(["sequence","n_ann","P@5(%)","P@15(%)","mean_eAL"])
        w.writerows(rows);   w.writerow(["OVERALL", n_total, f"{p5_avg:.2f}", f"{p15_avg:.2f}", "-"])

# ───────────────── CLI ─────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",      required=True, type=Path,
                    help="GT 데이터셋 루트 (pot)")
    ap.add_argument("--woft_dir",  required=True, type=Path,
                    help="WOFT txt 폴더")
    ap.add_argument("--output_csv", default="woft_paper_eval.csv", type=Path)
    args = ap.parse_args()
    main(args.root, args.woft_dir, args.output_csv)
