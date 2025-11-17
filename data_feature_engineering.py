# ----------------------------------------------------------- 
# STEP 5: ADVANCED FEATURE ENGINEERING for Risk Score Model
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load the dataset
# -------------------------------
df = pd.read_csv("cleaned_accident_data.csv")
print(f"✅ Loaded dataset: {df.shape}")
print("Columns:", df.columns.tolist())

# -------------------------------
# 2. Clean and map severity
# -------------------------------
if np.issubdtype(df["Accident Severity"].dtype, np.number):
    print("🔹 Numeric severity detected — mapping 0,1,2 → risk scores.")
    df["Risk_Score"] = df["Accident Severity"].map({0: 30, 1: 65, 2: 90})
else:
    print("🔹 Text severity detected — mapping Minor/Serious/Fatal → risk scores.")
    df["Accident Severity"] = df["Accident Severity"].astype(str).str.strip().str.title()
    df["Risk_Score"] = df["Accident Severity"].map({"Minor": 30, "Serious": 65, "Fatal": 90})

df = df.dropna(subset=["Risk_Score"]).reset_index(drop=True)
print(f"✅ Cleaned rows: {df.shape}")

# -------------------------------
# 3. Feature Engineering
# -------------------------------

def extract_hour(time_str):
    """Extracts hour from HH:MM formatted strings."""
    try:
        parts = str(time_str).split(":")
        hour = int(parts[0])
        if 0 <= hour <= 23:
            return hour
    except:
        return np.nan
    return np.nan

# Detect the time column automatically
time_col = next((col for col in df.columns if "time" in col.lower()), None)

if time_col:
    print(f"🕒 Using time column: {time_col}")
    df["Hour"] = df[time_col].astype(str).apply(extract_hour)
else:
    print("⚠️ No time column found — defaulting Hour = 12")
    df["Hour"] = 12

# Fill missing hours with median
median_hour = df["Hour"].median() if df["Hour"].notna().any() else 12
df["Hour"] = df["Hour"].fillna(median_hour)

# Cyclic encoding for Hour
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

# Cyclic encoding for Month (if Month column exists)
if "Month" in df.columns:
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df["Month_Num"] = df["Month"].map(month_map)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
else:
    print("⚠️ Month column not found — skipping month cyclic encoding.")

# Fill missing object columns with "Unknown"
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown").astype(str)

# -------------------------------
# 4. Encode categorical features (One-Hot)
# -------------------------------
categorical_cols = [
    "City Name", "Day of Week", "Vehicle Type Involved", "Weather Conditions",
    "Road Type", "Road Condition", "Lighting Conditions", "Traffic Control Presence",
    "Driver Gender", "Driver License Status", "Alcohol Involvement", "Accident Location Details"
]

# Keep only columns that exist in the dataframe
categorical_cols = [col for col in categorical_cols if col in df.columns]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------------
# 5. Prepare final dataset
# -------------------------------
numeric_cols = [
    'Number of Vehicles Involved', 'Number of Casualties', 'Number of Fatalities',
    'Speed Limit (km/h)', 'Driver Age'
]

# Add cyclic features to numeric columns for scaling
numeric_cols += ["Hour_sin", "Hour_cos"]
if "Month_sin" in df.columns:
    numeric_cols += ["Month_sin", "Month_cos"]

X = df.drop(columns=["Accident Severity", "Risk_Score"])
y = df["Risk_Score"]

# Fill any remaining numeric missing values
for col in X.select_dtypes(include=["int64", "float64"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("💾 Scaler saved as 'scaler.pkl'")

# -------------------------------
# 6. Save processed dataset
# -------------------------------
processed_path = "processed_data.csv"
df_final = X.copy()
df_final["Risk_Score"] = y
df_final.to_csv(processed_path, index=False)
print(f"💾 Processed dataset saved as '{processed_path}'")
print(f"🧩 Final feature matrix shape: {X.shape}, Target shape: {y.shape}")
print("✅ Feature engineering completed successfully!")
