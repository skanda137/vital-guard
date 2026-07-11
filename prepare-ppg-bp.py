import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR     = r"Data File\0_subject"
XLSX         = r"Data File\PPG-BP dataset.xlsx"
OUT_DIR      = "measured"
FS           = 1000  # Hz
feature_cols = [
    "hr_est", "mean_amp", "std_amp", "mean_width",
    "auc_ppg", "sig_range", "kurtosis", "skewness",
    "max_slope", "min_slope", "rr_mean"
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load metadata ───────────────────────────────────────────────────────
meta = pd.read_excel(XLSX, header=1)
meta.columns = meta.columns.str.strip()
meta = meta.rename(columns={
    "subject_ID":                     "subject_id",
    "Systolic Blood Pressure(mmHg)":  "sbp",
    "Diastolic Blood Pressure(mmHg)": "dbp",
    "Heart Rate(b/m)":                "hr_ref",
    "Age(year)":                      "age",
    "Sex(M/F)":                       "sex",
    "BMI(kg/m^2)":                    "bmi",
    "Hypertension":                   "hypertension",
})
meta = meta.dropna(subset=["subject_id", "sbp", "dbp"])
meta["subject_id"] = meta["subject_id"].astype(int)
print(f"Metadata loaded: {len(meta)} subjects")

# ── PPG signal processing helpers ──────────────────────────────────────
def bandpass(sig, fs=FS, low=0.5, high=8.0):
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def extract_features(sig, fs=FS):
    filtered = bandpass(sig, fs)
    peaks, _ = find_peaks(filtered, distance=int(fs * 0.4), prominence=10)

    if len(peaks) < 2:
        return None

    rr     = np.diff(peaks) / fs
    hr_est = 60.0 / np.mean(rr)

    amplitudes = filtered[peaks]
    mean_amp   = np.mean(amplitudes)
    std_amp    = np.std(amplitudes)

    # Pulse width at half amplitude (arterial stiffness proxy)
    half_amp = mean_amp / 2
    widths = []
    for pk in peaks:
        left_seg  = filtered[:pk][::-1]
        right_seg = filtered[pk:]
        l_idx = np.searchsorted(left_seg < half_amp, True)
        r_idx = np.searchsorted(right_seg < half_amp, True)
        widths.append(l_idx + r_idx)
    mean_width = np.mean(widths) / fs * 1000  # ms

    # Area under curve (cardiac output proxy)
    auc_ppg = np.trapezoid(np.abs(filtered)) / len(filtered)

    # Signal range and shape
    sig_range = filtered.max() - filtered.min()
    kurtosis  = float(pd.Series(filtered).kurtosis())
    skewness  = float(pd.Series(filtered).skew())

    # Derivative features (augmentation index proxy)
    d1        = np.diff(filtered)
    max_slope = float(np.max(d1))
    min_slope = float(np.min(d1))

    # RR mean in ms
    rr_mean = float(np.mean(rr) * 1000)

    return {
        "hr_est":     hr_est,
        "mean_amp":   mean_amp,
        "std_amp":    std_amp,
        "mean_width": mean_width,
        "auc_ppg":    auc_ppg,
        "sig_range":  sig_range,
        "kurtosis":   kurtosis,
        "skewness":   skewness,
        "max_slope":  max_slope,
        "min_slope":  min_slope,
        "rr_mean":    rr_mean,
    }

# ── Load all PPG segments ───────────────────────────────────────────────
records = []

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".txt"):
        continue

    parts = fname.replace(".txt", "").split("_")
    if len(parts) != 2:
        continue

    subj_id = int(parts[0])
    seg_num = int(parts[1])

    row = meta[meta["subject_id"] == subj_id]
    if row.empty:
        continue

    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "r") as f:
        raw = f.read().strip().split("\t")

    try:
        sig = np.array([float(x) for x in raw if x.strip()])
    except ValueError:
        continue

    if len(sig) < 500:
        continue

    feats = extract_features(sig)
    if feats is None:
        continue

    feats["subject_id"] = subj_id
    feats["segment"]    = seg_num
    feats["sbp"]        = float(row["sbp"].values[0])
    feats["dbp"]        = float(row["dbp"].values[0])
    feats["hr_ref"]     = float(row["hr_ref"].values[0])
    records.append(feats)

# ── Clean ───────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df = df.dropna(subset=feature_cols).reset_index(drop=True)

print(f"Segments after cleaning: {len(df)}")
print(f"Unique subjects: {df['subject_id'].nunique()}")
print(f"SBP range: {df['sbp'].min():.0f} - {df['sbp'].max():.0f} mmHg")

# ── Per-subject train/test split ────────────────────────────────────────
subjects   = df["subject_id"].unique()
np.random.seed(42)
np.random.shuffle(subjects)
split      = int(len(subjects) * 0.8)
train_subj = set(subjects[:split])

train_mask = df["subject_id"].isin(train_subj).values
test_mask  = ~train_mask

print(f"Train segments: {train_mask.sum()}  |  Test segments: {test_mask.sum()}")

X     = df[feature_cols].values
y_sys = df["sbp"].values
y_dia = df["dbp"].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X[train_mask])
X_test  = scaler.transform(X[test_mask])

# ── Option 2: GradientBoosting instead of Ridge ────────────────────────
reg_sys = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
).fit(X_train, y_sys[train_mask])

reg_dia = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
).fit(X_train, y_dia[train_mask])

dev_sys = reg_sys.predict(X_test)
dev_dia = reg_dia.predict(X_test)

# ── Option 3: Average across segments per subject ──────────────────────
bp_raw = pd.DataFrame({
    "subject_id": df["subject_id"].values[test_mask],
    "ref_sys":    y_sys[test_mask],
    "ref_dia":    y_dia[test_mask],
    "dev_sys":    dev_sys,
    "dev_dia":    dev_dia,
})

bp_val = bp_raw.groupby("subject_id").agg(
    ref_sys=("ref_sys", "first"),
    ref_dia=("ref_dia", "first"),
    dev_sys=("dev_sys", "mean"),
    dev_dia=("dev_dia", "mean"),
).reset_index()

bp_val.to_csv(os.path.join(OUT_DIR, "bp_validation.csv"), index=False)
print(f"\nbp_validation.csv: {len(bp_val)} subjects (averaged across segments)")

sys_bias = (bp_val["dev_sys"] - bp_val["ref_sys"]).mean()
sys_sd   = (bp_val["dev_sys"] - bp_val["ref_sys"]).std()
dia_bias = (bp_val["dev_dia"] - bp_val["ref_dia"]).mean()
dia_sd   = (bp_val["dev_dia"] - bp_val["ref_dia"]).std()
print(f"  Systolic  bias={sys_bias:+.2f} mmHg  SD={sys_sd:.2f} mmHg")
print(f"  Diastolic bias={dia_bias:+.2f} mmHg  SD={dia_sd:.2f} mmHg")

# ── spo2_hr.csv — HR only, SpO2 unavailable ────────────────────────────
hr_raw = pd.DataFrame({
    "subject_id": df["subject_id"].values[test_mask],
    "dev_hr":     df["hr_est"].values[test_mask],
    "ref_hr":     df["hr_ref"].values[test_mask],
})

hr_val = hr_raw.groupby("subject_id").agg(
    ref_hr=("ref_hr", "first"),
    dev_hr=("dev_hr", "mean"),
).reset_index()
hr_val["ref_spo2"] = np.nan
hr_val["dev_spo2"] = np.nan

hr_val.to_csv(os.path.join(OUT_DIR, "spo2_hr.csv"), index=False)
print(f"spo2_hr.csv: {len(hr_val)} subjects (SpO2 unavailable in this dataset)")

# ── risk_model_predictions.csv ─────────────────────────────────────────
df["y_true"] = ((df["sbp"] >= 140) | (df["dbp"] >= 90)).astype(int)
print(f"\nClass distribution: {df['y_true'].value_counts().to_dict()}")

scaler2 = StandardScaler()
X_all   = scaler2.fit_transform(df[feature_cols].values)
y_all   = df["y_true"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

clf = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
)
clf.fit(X_tr, y_tr)
y_score = clf.predict_proba(X_te)[:, 1]

from sklearn.metrics import roc_auc_score, f1_score
print(f"  Risk AUC: {roc_auc_score(y_te, y_score):.3f}")
print(f"  Risk F1:  {f1_score(y_te, (y_score>=0.5).astype(int), zero_division=0):.3f}")

risk_df = pd.DataFrame({"y_true": y_te, "y_score": y_score.round(4)})
risk_df.to_csv(os.path.join(OUT_DIR, "risk_model_predictions.csv"), index=False)
print(f"risk_model_predictions.csv: {len(risk_df)} rows")

print("\nDone. Now run:")
print("  python run_analysis.py --data ./measured --out ./figures_real")