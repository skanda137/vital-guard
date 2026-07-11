"""
run_analysis.py
Wrapper around vitalguard_analysis.py that patches the spo2_hr function
to skip SpO2 when those columns are NaN (as is the case with the PPG-BP dataset).
Run this instead of vitalguard_analysis.py:
    python run_analysis.py --data ./measured --out ./figures_real
"""

import argparse
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Import everything from the original toolkit ──────────────────────
from vitalguard_analysis import (
    bp_validation,
    motion_gating,
    sos_latency,
    risk_model,
    synth_gating,
    synth_sos,
    _watermark,
    _save,
    ACCENT,
    ACCENT2,
    GREY,
)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})


def spo2_hr_patched(df, outdir, synthetic, log):
    """
    Plots only the columns that are not entirely NaN.
    SpO2 is skipped when unavailable (PPG-BP dataset has no SpO2 reference).
    """
    plots_made = []

    for ref_c, dev_c, label, unit, color in [
        ("ref_spo2", "dev_spo2", "SpO\u2082", "%",   ACCENT),
        ("ref_hr",   "dev_hr",   "Heart rate", "bpm", ACCENT2),
    ]:
        if ref_c not in df.columns or df[ref_c].isna().all():
            log.append(f"[{label}]  Skipped — column '{ref_c}' is all NaN (not in dataset)")
            continue

        ref  = df[ref_c].dropna().to_numpy(float)
        dev  = df[dev_c].dropna().to_numpy(float)
        n    = min(len(ref), len(dev))
        ref  = ref[:n]
        dev  = dev[:n]

        diff = dev - ref
        rms  = np.sqrt(np.mean(diff ** 2))
        mae  = np.mean(np.abs(diff))
        r, _ = stats.pearsonr(ref, dev)

        fig, ax = plt.subplots(figsize=(6, 5))
        lo = min(ref.min(), dev.min()) - 2
        hi = max(ref.max(), dev.max()) + 2
        ax.scatter(ref, dev, s=24, color=color, alpha=0.7, edgecolor="white", lw=0.5)
        ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1.2, label="identity")
        m, b = np.polyfit(ref, dev, 1)
        ax.plot([lo, hi], [m*lo+b, m*hi+b], color="black", lw=1.4,
                label=f"fit (r={r:.3f})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Reference {label} ({unit})")
        ax.set_ylabel(f"Estimated {label} ({unit})")
        ax.set_title(f"{label}: RMSE={rms:.2f} {unit}, MAE={mae:.2f} {unit} (n={n})")
        ax.legend(frameon=False, fontsize=9)

        _watermark(fig, synthetic)
        fname = f"03_{label.lower().replace(' ','_').replace('\u2082','2')}_accuracy.png"
        p = _save(fig, outdir, fname)
        plots_made.append(os.path.basename(p))

        log.append(textwrap.dedent(f"""\
            [{label}]  n={n}
              RMSE={rms:.2f} {unit}  MAE={mae:.2f} {unit}  Pearson r={r:.3f}
              -> {os.path.basename(p)}"""))

    if not plots_made:
        log.append("[SpO2/HR]  No valid columns found — check spo2_hr.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out",  default="figures_real")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    synthetic = args.data is None
    log = []
    log.append("VitalGuard technical-effect summary\n" + "=" * 38)
    if synthetic:
        log.append("MODE: ILLUSTRATIVE (synthetic data)\n")
    else:
        log.append(f"MODE: mixed — real data from {args.data} "
                   f"(motion gating and SOS latency are simulation-based)\n")

    def load(name, gen):
        if synthetic:
            return gen()
        path = os.path.join(args.data, name)
        if os.path.exists(path):
            return pd.read_csv(path)
        print(f"  [warn] {name} not found, using synthetic fallback")
        return gen()

    from vitalguard_analysis import synth_bp, synth_risk
    bp_validation(load("bp_validation.csv", synth_bp),   args.out, synthetic, log)
    motion_gating(load("motion_gating_waveform.csv", synth_gating), args.out, True,      log)
    spo2_hr_patched(load("spo2_hr.csv",             lambda: pd.DataFrame()), args.out, synthetic, log)
    sos_latency(  load("sos_latency.csv",            synth_sos),    args.out, True,      log)
    risk_model(   load("risk_model_predictions.csv", synth_risk),  args.out, synthetic, log)

    summary = "\n\n".join(log)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()