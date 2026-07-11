import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR = r"Data File\0_subject"
XLSX     = r"Data File\PPG-BP dataset.xlsx"
OUT_DIR  = "measured"
FS       = 1000  # Hz

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load metadata ───────────────────────────────────────────────────────
meta = pd.read_excel(XLSX, header=1)
meta.columns = meta.columns.str.strip()
meta = meta.rename(columns={
    "subject_ID":                    "subject_id",
    "Systolic Blood Pressure(mmHg)": "sbp",
    "Diastolic Blood Pressure(mmHg)":"dbp",
    "Heart Rate(b/m)":               "hr_ref",
    "Age(year)":                     "age",
    "Sex(M/F)":                      "sex",
    "BMI(kg/m^2)":                   "bmi",
    "Hypertension":                  "hypertension",
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

    sdnn  = np.std(rr) * 1000
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000 if len(rr) > 1 else np.nan
    skewness = float(pd.Series(filtered).skew())

    return {
        "hr_est":   hr_est,
        "mean_amp": mean_amp,
        "std_amp":  std_amp,
        "sdnn":     sdnn,
        "rmssd":    rmssd,
        "skewness": skewness,
        "n_peaks":  len(peaks),
    }

# ── Load all PPG segments and extract features ─────────────────────────
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

# ── Clean: drop NaN features BEFORE extracting any arrays ─────────────
feature_cols = ["hr_est", "mean_amp", "std_amp", "sdnn", "rmssd", "skewness"]

df = pd.DataFrame(records)
df = df.dropna(subset=feature_cols).reset_index(drop=True)

print(f"Segments after cleaning: {len(df)}")
print(f"Unique subjects: {df['subject_id'].nunique()}")
print(f"SBP range: {df['sbp'].min():.0f} – {df['sbp'].max():.0f} mmHg")

# ── NOW extract arrays (after dropna, so sizes are consistent) ─────────
X     = df[feature_cols].values
y_sys = df["sbp"].values
y_dia = df["dbp"].values

# ── Per-subject train/test split (no data leakage) ────────────────────
subjects = df["subject_id"].unique()
np.random.seed(42)
np.random.shuffle(subjects)
split      = int(len(subjects) * 0.8)
train_subj = set(subjects[:split])

train_mask = df["subject_id"].isin(train_subj).values
test_mask  = ~train_mask

print(f"Train segments: {train_mask.sum()}  |  Test segments: {test_mask.sum()}")

# ── bp_validation.csv ──────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_mask])
X_test  = scaler.transform(X[test_mask])

reg_sys = Ridge(alpha=1.0).fit(X_train, y_sys[train_mask])
reg_dia = Ridge(alpha=1.0).fit(X_train, y_dia[train_mask])

dev_sys = reg_sys.predict(X_test)
dev_dia = reg_dia.predict(X_test)

bp_val = pd.DataFrame({
    "subject_id": df["subject_id"].values[test_mask],
    "ref_sys":    y_sys[test_mask],
    "ref_dia":    y_dia[test_mask],
    "dev_sys":    dev_sys,
    "dev_dia":    dev_dia,
})
bp_val.to_csv(os.path.join(OUT_DIR, "bp_validation.csv"), index=False)
print(f"\nbp_validation.csv: {len(bp_val)} rows")

# ── spo2_hr.csv ────────────────────────────────────────────────────────
# SpO2 not in this dataset — HR only, honestly labelled
hr_df = df[test_mask][["hr_est", "hr_ref"]].copy().reset_index(drop=True)
hr_df.columns   = ["dev_hr", "ref_hr"]
hr_df["ref_spo2"] = np.nan
hr_df["dev_spo2"] = np.nan
hr_df.to_csv(os.path.join(OUT_DIR, "spo2_hr.csv"), index=False)
print(f"spo2_hr.csv: {len(hr_df)} rows (SpO2 unavailable in this dataset)")

# ── risk_model_predictions.csv ─────────────────────────────────────────
# Hypertension label: SBP >= 140 OR DBP >= 90 (standard clinical threshold)
df["y_true"] = ((df["sbp"] >= 140) | (df["dbp"] >= 90)).astype(int)
print(f"\nClass distribution: {df['y_true'].value_counts().to_dict()}")

scaler2 = StandardScaler()
X_all   = scaler2.fit_transform(df[feature_cols].values)
y_all   = df["y_true"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
clf.fit(X_tr, y_tr)
y_score = clf.predict_proba(X_te)[:, 1]

risk_df = pd.DataFrame({"y_true": y_te, "y_score": y_score.round(4)})
risk_df.to_csv(os.path.join(OUT_DIR, "risk_model_predictions.csv"), index=False)
print(f"risk_model_predictions.csv: {len(risk_df)} rows")

print("\nDone. Now run:")
print("  python vitalguard_analysis.py --data ./measured --out ./figures_real")
