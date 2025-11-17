# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# -------------------------------------------
import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\Arya\Downloads\aiml_dataset\accident_prediction_india.csv"
df = pd.read_csv(file_path)

print("✅ Dataset loaded:", df.shape)

# -------------------------------
# 1️⃣  HANDLE MISSING VALUES
# -------------------------------
df["Traffic Control Presence"].fillna("Unknown", inplace=True)
df["Driver License Status"].fillna("Unknown", inplace=True)
df["City Name"].replace("Unknown", "Other", inplace=True)

# -------------------------------
# 2️⃣  CONVERT TIME OF DAY INTO CATEGORIES
# -------------------------------
def time_to_period(t):
    try:
        hour = int(str(t).split(":")[0])
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    except:
        return "Unknown"

df["Time Period"] = df["Time of Day"].apply(time_to_period)

# -------------------------------
# 3️⃣  ADD DERIVED FEATURES
# -------------------------------

# Weekend flag
df["Is_Weekend"] = df["Day of Week"].apply(lambda x: 1 if x in ["Saturday", "Sunday"] else 0)

# Peak hour flag (morning/evening rush)
df["Is_Peak_Hour"] = df["Time Period"].apply(lambda x: 1 if x in ["Morning", "Evening"] else 0)

# Bad weather flag
df["Bad_Weather_Flag"] = df["Weather Conditions"].apply(
    lambda x: 1 if x.lower() in ["rainy", "foggy", "snowy", "storm", "hail"] else 0
)

# Road risk score: lighting + condition
def road_risk(row):
    cond = row["Road Condition"].lower()
    light = row["Lighting Conditions"].lower()
    score = 0
    if "wet" in cond or "damaged" in cond:
        score += 1
    if "night" in light and "unlit" in light:
        score += 1
    return score

df["Road_Risk_Score"] = df.apply(road_risk, axis=1)

# Urban or rural flag
df["Urban_or_Rural"] = df["Road Type"].apply(lambda x: 1 if "city" in x.lower() else 0)

# -------------------------------
# 4️⃣  TARGET ENCODING
# -------------------------------
severity_map = {"Minor": 0, "Serious": 1, "Fatal": 2}
df["Accident Severity"] = df["Accident Severity"].map(severity_map)

# -------------------------------
# 5️⃣  DROP UNNEEDED / REDUNDANT COLUMNS
# -------------------------------
drop_cols = ["State Name", "Year", "Time of Day"]  # can add more later
df.drop(columns=drop_cols, inplace=True)

print("\n✅ Cleaned dataset shape:", df.shape)

# -------------------------------
# 6️⃣  SAVE CLEANED DATA
# -------------------------------
df.to_csv("cleaned_accident_data.csv", index=False)
print("\n💾 Cleaned data saved as cleaned_accident_data.csv")

# Quick check of new features
print("\n🧩 Columns after preprocessing:\n", df.columns.tolist())
print("\n🔹 Sample rows:\n", df.head())
