import argparse
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
 
# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
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
ACCENT = "#1f6feb"
ACCENT2 = "#d1495b"
GREEN = "#2e9e5b"
GREY = "#6b7280"
 
 
def _watermark(fig, synthetic):
    if not synthetic:
        return
    fig.text(0.5, 0.5,
             "ILLUSTRATIVE — SYNTHETIC DATA\nREPLACE WITH MEASURED RESULTS",
             fontsize=26, color="grey", alpha=0.18,
             ha="center", va="center", rotation=28, fontweight="bold")
 
 
def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
# ======================================================================
# 1. BLOOD-PRESSURE ACCURACY VALIDATION  (ISO 81060-2 / AAMI style)
#    -> Bland-Altman + correlation. This is the primary accuracy proof.
# ======================================================================
def bp_validation(df, outdir, synthetic, log):
    """
    df columns: subject_id, ref_sys, ref_dia, dev_sys, dev_dia
      ref_* = reference cuff (validated upper-arm monitor)
      dev_* = VitalGuard wrist-cuff device
    """
    for kind, ref_c, dev_c, color in [("Systolic", "ref_sys", "dev_sys", ACCENT),
                                       ("Diastolic", "ref_dia", "dev_dia", ACCENT2)]:
        ref = df[ref_c].to_numpy(float)
        dev = df[dev_c].to_numpy(float)
        diff = dev - ref
        mean = (dev + ref) / 2
        bias = diff.mean()
        sd = diff.std(ddof=1)
        loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd
        r, _ = stats.pearsonr(ref, dev)
 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
 
        # Correlation / scatter
        lo = min(ref.min(), dev.min()) - 5
        hi = max(ref.max(), dev.max()) + 5
        ax1.scatter(ref, dev, s=28, color=color, alpha=0.7, edgecolor="white", lw=0.5)
        ax1.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1.2, label="line of identity")
        m, b = np.polyfit(ref, dev, 1)
        ax1.plot([lo, hi], [m * lo + b, m * hi + b], color="black", lw=1.4,
                 label=f"fit (r={r:.3f})")
        ax1.set_xlabel("Reference monitor (mmHg)")
        ax1.set_ylabel("VitalGuard device (mmHg)")
        ax1.set_title(f"{kind} BP — Correlation (n={len(ref)})")
        ax1.legend(frameon=False, fontsize=9)
        ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
 
        # Bland-Altman
        ax2.scatter(mean, diff, s=28, color=color, alpha=0.7, edgecolor="white", lw=0.5)
        ax2.axhline(bias, color="black", lw=1.4, label=f"bias {bias:+.1f}")
        ax2.axhline(loa_hi, color=ACCENT2, ls="--", lw=1.2,
                    label=f"+1.96 SD ({loa_hi:+.1f})")
        ax2.axhline(loa_lo, color=ACCENT2, ls="--", lw=1.2,
                    label=f"-1.96 SD ({loa_lo:+.1f})")
        ax2.axhspan(-5, 5, color=GREEN, alpha=0.07)
        ax2.set_xlabel("Mean of device & reference (mmHg)")
        ax2.set_ylabel("Device − Reference (mmHg)")
        ax2.set_title(f"{kind} BP — Bland–Altman agreement")
        ax2.legend(frameon=False, fontsize=8, loc="upper right")
 
        _watermark(fig, synthetic)
        p = _save(fig, outdir, f"01_bp_{kind.lower()}_accuracy.png")
 
        passed = abs(bias) <= 5 and sd <= 8  # ISO 81060-2 thresholds
        log.append(textwrap.dedent(f"""\
            [BP {kind}]  n={len(ref)}
              Mean error (bias) : {bias:+.2f} mmHg   (ISO 81060-2 limit |mean| <= 5)
              SD of error       : {sd:.2f} mmHg      (ISO 81060-2 limit SD <= 8)
              95% limits of agr.: {loa_lo:+.1f} to {loa_hi:+.1f} mmHg
              Pearson r         : {r:.3f}
              ISO 81060-2 (basic) : {'PASS' if passed else 'CHECK'}
              -> {os.path.basename(p)}"""))
    return
 
 
# ======================================================================
# 2. PPG-GATED MOTION-ARTIFACT REJECTION  (THE CORE INVENTIVE STEP)
#    -> oscillometric envelope WITHOUT vs WITH PPG gating.
# ======================================================================
def motion_gating(df, outdir, synthetic, log):
    """
    df columns: t, cuff_pressure, oscillation, ppg_quality
      t            = time (s) during a single deflation
      cuff_pressure= cuff pressure (mmHg) ramping down
      oscillation  = band-passed cuff oscillation amplitude (a.u.)
      ppg_quality  = 0..1 PPG signal-quality / beat-gate flag
    """
    t = df["t"].to_numpy(float)
    cuff = df["cuff_pressure"].to_numpy(float)
    osc = df["oscillation"].to_numpy(float)
    q = df["ppg_quality"].to_numpy(float)
 
    # Oscillometric envelope = peak-to-peak oscillation amplitude binned by cuff
    # pressure. This mirrors real processing and avoids transform edge artifacts.
    def binned_envelope(mask=None):
        edges = np.arange(40, 182, 4.0)           # 4 mmHg bins, 40..180
        centers = (edges[:-1] + edges[1:]) / 2
        amp = np.full(centers.size, np.nan)
        for i in range(centers.size):
            sel = (cuff >= edges[i]) & (cuff < edges[i + 1])
            if mask is not None:
                sel &= mask
            if sel.sum() > 3:
                seg = osc[sel]
                amp[i] = seg.max() - seg.min()      # peak-to-peak in the bin
        good = ~np.isnan(amp)
        # interpolate small gaps (e.g. motion windows removed by the gate)
        if good.sum() > 2:
            amp = np.interp(centers, centers[good], amp[good])
        return centers, amp
 
    p_raw, env_raw = binned_envelope(mask=None)            # all samples
    p_g, env_gated = binned_envelope(mask=(q > 0.6))       # PPG-gated samples only
 
    def oscillometric_bp(pressure, envelope):
        # pressure ascends (40..180). MAP at envelope peak; SBP on the higher-
        # pressure side, DBP on the lower-pressure side (standard ratios).
        i_map = int(np.nanargmax(envelope))
        peak = envelope[i_map]
        map_p = pressure[i_map]
        sys_ratio, dia_ratio = 0.55, 0.85
        low = envelope[:i_map + 1]                 # lower pressures -> diastolic
        high = envelope[i_map:]                    # higher pressures -> systolic
        dbp = pressure[np.argmin(np.abs(low - dia_ratio * peak))] if low.size else np.nan
        sbp = pressure[i_map + np.argmin(np.abs(high - sys_ratio * peak))]
        return sbp, map_p, dbp
 
    sbp_r, map_r, dbp_r = oscillometric_bp(p_raw, env_raw)
    sbp_g, map_g, dbp_g = oscillometric_bp(p_g, env_gated)
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
 
    axes[0, 0].plot(t, cuff, color="black", lw=1.4)
    axes[0, 0].set_title("Cuff deflation profile")
    axes[0, 0].set_xlabel("Time (s)"); axes[0, 0].set_ylabel("Cuff pressure (mmHg)")
 
    axes[0, 1].plot(t, q, color=GREEN, lw=1.2)
    axes[0, 1].axhline(0.6, ls="--", color=GREY, lw=1)
    axes[0, 1].fill_between(t, 0, 1, where=q <= 0.6, color=ACCENT2, alpha=0.15,
                            label="rejected (motion)")
    axes[0, 1].set_title("PPG signal-quality gate")
    axes[0, 1].set_xlabel("Time (s)"); axes[0, 1].set_ylabel("PPG quality (0–1)")
    axes[0, 1].legend(frameon=False, fontsize=9)
 
    axes[1, 0].plot(p_raw, env_raw, color=ACCENT2, lw=1.8, marker="o", ms=3)
    axes[1, 0].axvline(sbp_r, ls=":", color=GREY); axes[1, 0].axvline(dbp_r, ls=":", color=GREY)
    axes[1, 0].set_title(f"WITHOUT PPG gating\nSBP≈{sbp_r:.0f} / DBP≈{dbp_r:.0f} (motion-corrupted)")
    axes[1, 0].set_xlabel("Cuff pressure (mmHg)"); axes[1, 0].set_ylabel("Oscillation amplitude (a.u.)")
    axes[1, 0].invert_xaxis()
 
    axes[1, 1].plot(p_g, env_gated, color=ACCENT, lw=2.0, marker="o", ms=3)
    axes[1, 1].axvline(sbp_g, ls=":", color=GREY); axes[1, 1].axvline(dbp_g, ls=":", color=GREY)
    axes[1, 1].annotate("MAP", (map_g, np.nanmax(env_gated)), textcoords="offset points",
                        xytext=(6, -4), fontsize=9, color=GREY)
    axes[1, 1].set_title(f"WITH PPG gating (invention)\nSBP≈{sbp_g:.0f} / DBP≈{dbp_g:.0f} (clean)")
    axes[1, 1].set_xlabel("Cuff pressure (mmHg)"); axes[1, 1].set_ylabel("Oscillation amplitude (a.u.)")
    axes[1, 1].invert_xaxis()
 
    fig.suptitle("Technical effect: PPG-gated rejection of wrist-motion artifact in oscillometric BP",
                 fontsize=14, fontweight="bold")
    _watermark(fig, synthetic)
    p = _save(fig, outdir, "02_ppg_motion_gating_core_inventive_step.png")
 
    # Quantify improvement: roughness (deviation from a smooth single-lobe fit).
    def roughness(env):
        env = np.nan_to_num(env)
        smooth = np.convolve(env, np.ones(5) / 5, mode="same")
        return np.std(env - smooth) / (np.nanmax(env) + 1e-9)
 
    rough_raw = roughness(env_raw)
    rough_gated = roughness(env_gated)
    log.append(textwrap.dedent(f"""\
        [PPG MOTION GATING — core inventive step]
          Envelope roughness without gating : {rough_raw:.3f}
          Envelope roughness with gating    : {rough_gated:.3f}  (lower = cleaner)
          SBP/DBP (ungated)                 : {sbp_r:.0f} / {dbp_r:.0f} mmHg
          SBP/DBP (PPG-gated)               : {sbp_g:.0f} / {dbp_g:.0f} mmHg
          MAP (gated)                       : {map_g:.0f} mmHg
          -> {os.path.basename(p)}
          NOTE: report the gated SBP/DBP against a reference cuff to show
                the accuracy recovered by gating during motion."""))
    return
 
 
# ======================================================================
# 3. SpO2 / HEART-RATE ACCURACY vs reference pulse oximeter
# ======================================================================
def spo2_hr(df, outdir, synthetic, log):
    """ df columns: ref_spo2, dev_spo2, ref_hr, dev_hr """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ref_c, dev_c, label, unit, color in [
        (axes[0], "ref_spo2", "dev_spo2", "SpO₂", "%", ACCENT),
        (axes[1], "ref_hr", "dev_hr", "Heart rate", "bpm", ACCENT2)]:
        ref = df[ref_c].to_numpy(float); dev = df[dev_c].to_numpy(float)
        diff = dev - ref
        rms = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        lo = min(ref.min(), dev.min()) - 2; hi = max(ref.max(), dev.max()) + 2
        ax.scatter(ref, dev, s=24, color=color, alpha=0.7, edgecolor="white", lw=0.5)
        ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1.2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Reference {label} ({unit})")
        ax.set_ylabel(f"Device {label} ({unit})")
        ax.set_title(f"{label}: RMSE={rms:.2f} {unit}, MAE={mae:.2f} {unit}")
        log.append(f"[{label}]  RMSE={rms:.2f}{unit}  MAE={mae:.2f}{unit}  n={len(ref)}")
    _watermark(fig, synthetic)
    p = _save(fig, outdir, "03_spo2_hr_accuracy.png")
    log.append(f"  -> {os.path.basename(p)}")
    return
 
 
# ======================================================================
# 4. EMERGENCY (SOS) DETECTION-TO-DISPATCH LATENCY
# ======================================================================
def sos_latency(df, outdir, synthetic, log):
    """ df columns: event_type, detection_to_dispatch_s """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lat = df["detection_to_dispatch_s"].to_numpy(float)
    ax1.hist(lat, bins=15, color=ACCENT, alpha=0.85, edgecolor="white")
    ax1.axvline(np.median(lat), color=ACCENT2, lw=1.6, label=f"median {np.median(lat):.1f}s")
    ax1.set_xlabel("Detection → dispatch latency (s)"); ax1.set_ylabel("Count")
    ax1.set_title("End-to-end SOS latency"); ax1.legend(frameon=False, fontsize=9)
 
    types = df["event_type"].unique()
    data = [df.loc[df.event_type == t, "detection_to_dispatch_s"] for t in types]
    bp = ax2.boxplot(data, labels=types, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor=ACCENT, alpha=0.5)
    ax2.set_ylabel("Latency (s)"); ax2.set_title("By event type")
    ax2.tick_params(axis="x", rotation=20)
    _watermark(fig, synthetic)
    p = _save(fig, outdir, "04_sos_latency.png")
    log.append(textwrap.dedent(f"""\
        [SOS LATENCY]  n={len(lat)}
          median={np.median(lat):.2f}s  mean={lat.mean():.2f}s  p95={np.percentile(lat,95):.2f}s
          -> {os.path.basename(p)}"""))
    return
 
 
# ======================================================================
# 5. PREDICTIVE RISK-MODEL PERFORMANCE  (ROC + confusion matrix)
# ======================================================================
def risk_model(df, outdir, synthetic, log):
    """ df columns: y_true (0/1), y_score (0..1 risk prob) """
    y = df["y_true"].to_numpy(int)
    s = df["y_score"].to_numpy(float)
    yhat = (s >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, yhat)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("Health-risk classifier ROC"); ax1.legend(frameon=False, fontsize=10)
 
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax2.set_yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax2.set_title("Confusion matrix"); ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    _watermark(fig, synthetic)
    p = _save(fig, outdir, "05_risk_model_performance.png")
    log.append(textwrap.dedent(f"""\
        [RISK MODEL]  n={len(y)}
          AUC={roc_auc:.3f}  acc={accuracy_score(y,yhat):.3f}
          precision={precision_score(y,yhat,zero_division=0):.3f}
          recall(sens)={recall_score(y,yhat,zero_division=0):.3f}
          F1={f1_score(y,yhat,zero_division=0):.3f}
          -> {os.path.basename(p)}"""))
    return
 
 
# ======================================================================
# Synthetic-data generators (ILLUSTRATIVE ONLY)
# ======================================================================
def synth_bp(n=42, seed=1):
    rng = np.random.default_rng(seed)
    ref_sys = rng.normal(132, 18, n).clip(95, 185)
    ref_dia = rng.normal(84, 11, n).clip(60, 115)
    dev_sys = ref_sys + rng.normal(-1.8, 6.5, n)
    dev_dia = ref_dia + rng.normal(1.1, 5.5, n)
    return pd.DataFrame({"subject_id": np.arange(1, n + 1),
                         "ref_sys": ref_sys.round(0), "ref_dia": ref_dia.round(0),
                         "dev_sys": dev_sys.round(0), "dev_dia": dev_dia.round(0)})
 
 
def synth_gating(seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 30, 3000)
    cuff = 180 - (180 - 40) * (t / t.max())          # linear deflation 180->40
    hr = 72 / 60
    pulses = np.sin(2 * np.pi * hr * t)
    true_env = np.exp(-((cuff - 100) ** 2) / (2 * 20 ** 2))  # peak at MAP~100
    osc_clean = 1.0 * true_env * pulses
    q = np.ones_like(t)
    motion = np.zeros_like(t)
    for c in [6.5, 17, 24]:
        burst = np.exp(-((t - c) ** 2) / (2 * 0.45 ** 2))     # compact artifact
        guard = np.exp(-((t - c) ** 2) / (2 * 1.10 ** 2))     # wider quality dip
        motion += burst * rng.normal(0, 4, t.size)
        q -= guard
    q = q.clip(0, 1)
    osc = osc_clean + motion
    return pd.DataFrame({"t": t, "cuff_pressure": cuff,
                         "oscillation": osc, "ppg_quality": q})
 
 
def synth_spo2_hr(n=50, seed=5):
    rng = np.random.default_rng(seed)
    ref_spo2 = rng.normal(96, 2.2, n).clip(88, 100)
    ref_hr = rng.normal(78, 14, n).clip(50, 130)
    return pd.DataFrame({"ref_spo2": ref_spo2.round(0),
                         "dev_spo2": (ref_spo2 + rng.normal(0.3, 1.6, n)).round(0),
                         "ref_hr": ref_hr.round(0),
                         "dev_hr": (ref_hr + rng.normal(-0.4, 2.6, n)).round(0)})
 
 
def synth_sos(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for et, base in [("Fall", 3.2), ("Cardiac anomaly", 4.0), ("Hypertensive crisis", 4.6)]:
        for _ in range(18):
            rows.append((et, max(1.5, rng.normal(base, 0.8))))
    return pd.DataFrame(rows, columns=["event_type", "detection_to_dispatch_s"])
 
 
def synth_risk(n=180, seed=9):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = np.clip(rng.normal(0.35 + 0.32 * y, 0.17), 0, 1)
    return pd.DataFrame({"y_true": y, "y_score": score.round(3)})
 
 
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="folder with measured CSVs; omit to generate illustrative figures")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    synthetic = args.data is None
    log = []
    log.append("VitalGuard technical-effect summary\n" + "=" * 38)
    if synthetic:
        log.append("MODE: ILLUSTRATIVE (synthetic data) — DO NOT SUBMIT THESE NUMBERS.\n")
    else:
        log.append(f"MODE: measured data from {args.data}\n")
 
    def load(name, gen):
        if synthetic:
            return gen()
        return pd.read_csv(os.path.join(args.data, name))
 
    bp_validation(load("bp_validation.csv", synth_bp), args.out, synthetic, log)
    motion_gating(load("motion_gating_waveform.csv", synth_gating), args.out, synthetic, log)
    spo2_hr(load("spo2_hr.csv", synth_spo2_hr), args.out, synthetic, log)
    sos_latency(load("sos_latency.csv", synth_sos), args.out, synthetic, log)
    risk_model(load("risk_model_predictions.csv", synth_risk), args.out, synthetic, log)
 
    summary = "\n\n".join(log)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
 
 
if __name__ == "__main__":
    main()