# ---------------------------------------------
# STEP 1: ADVANCED DATA REFINEMENT & CLEANING
# ---------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Load Dataset
# -----------------------
path = r"C:\Users\Arya\Downloads\aiml_dataset\accident_prediction_india.csv"
df = pd.read_csv(path)
print(f"✅ Original dataset loaded: {df.shape}")

# -----------------------
# 1️⃣ Basic Cleaning
# -----------------------
# Replace unknown or missing entries
df.replace(["Unknown", "N/A", "NA", "-", "?"], np.nan, inplace=True)
df["Traffic Control Presence"].fillna("None", inplace=True)
df["Driver License Status"].fillna("Unknown", inplace=True)
df["City Name"].fillna("Other", inplace=True)

# Drop irrelevant columns (if any)
if "State Name" in df.columns:
    df.drop("State Name", axis=1, inplace=True)

# -----------------------
# 2️⃣ Handle Outliers
# -----------------------
df = df[(df["Speed Limit (km/h)"] >= 20) & (df["Speed Limit (km/h)"] <= 160)]
df = df[(df["Driver Age"] >= 18) & (df["Driver Age"] <= 80)]
print(f"✅ After outlier removal: {df.shape}")

# -----------------------
# 3️⃣ Standardize categorical text
# -----------------------
def clean_text(x):
    if isinstance(x, str):
        x = x.strip().lower()
        x = x.replace("-", " ").replace("_", " ")
        return " ".join([w.capitalize() for w in x.split()])
    return x

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(clean_text)

# -----------------------
# 4️⃣ Encode categorical columns
# -----------------------
categorical_cols = df.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"✅ Encoded {len(categorical_cols)} categorical columns.")

# -----------------------
# 5️⃣ Handle class imbalance
# -----------------------
X = df.drop("Accident Severity", axis=1)
y = df["Accident Severity"]

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

df_balanced = pd.concat([X_bal, y_bal], axis=1)
print(f"✅ After balancing: {df_balanced.shape}")

# -----------------------
# 6️⃣ Feature Scaling
# -----------------------
num_cols = X_bal.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df_balanced[num_cols] = scaler.fit_transform(df_balanced[num_cols])

# -----------------------
# 7️⃣ Correlation Analysis
# -----------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df_balanced.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------
# 8️⃣ Save refined dataset
# -----------------------
refined_path = "refined_accident_data.csv"
df_balanced.to_csv(refined_path, index=False)
print(f"💾 Refined dataset saved as: {refined_path}")
print(f"✅ Final shape: {df_balanced.shape}")

# -----------------------
# 9️⃣ Sanity Check
# -----------------------
print("\n🔹 Final columns:")
print(df_balanced.columns.tolist())
print("\n🔹 Sample rows:")
print(df_balanced.head())
