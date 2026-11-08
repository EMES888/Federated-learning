#!/usr/bin/env python
"""Summarise and plot all finished runs under a directory.

  python compare.py --runs runs --out runs/_summary [--pattern "camvid_*"]
"""
import argparse
import os

from fedseg import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.runs, "_summary")
    os.makedirs(out, exist_ok=True)
    runs = viz.load_runs(a.runs, a.pattern)
    if not runs:
        print("no finished runs found")
        return
    df = viz.summary_table(runs)
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    agg = viz.aggregate_seeds(df)
    agg.to_csv(os.path.join(out, "summary_over_seeds.csv"), index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:  # tabulate not installed
        md = df.to_string(index=False)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    for ds in sorted(df["dataset"].unique()):
        sel = [r for r in runs if r["config"]["dataset"] == ds]
        viz.plot_comparison(sel, os.path.join(out, f"{ds}_val_mIoU.png"), "val_mIoU", f"{ds}: validation mIoU")
        viz.plot_comparison(sel, os.path.join(out, f"{ds}_train_loss.png"), "train_loss", f"{ds}: train loss")
        viz.plot_comparison(sel, os.path.join(out, f"{ds}_mIoU_vs_comm.png"), "val_mIoU",
                            f"{ds}: mIoU vs communication", x="comm_MB")
    print(f"saved summary to {out}")


if __name__ == "__main__":
    main()
