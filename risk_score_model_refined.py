# ---------------------------------------------------
# STEP 7: ADVANCED RISK CLASSIFICATION MODEL (High Accuracy)
# ---------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Try to import xgboost; if unavailable, set a flag and fall back to sklearn's GradientBoostingClassifier later
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("⚠️ xgboost not installed or could not be imported; falling back to sklearn's GradientBoostingClassifier.")

# ---------------------------------------------------
# 1. Load processed dataset
# ---------------------------------------------------
df = pd.read_csv("processed_data.csv")
print(f"✅ Loaded dataset: {df.shape}")

# ---------------------------------------------------
# 2. Additional feature engineering
# ---------------------------------------------------
print("🧠 Adding engineered ratio and group-based features...")

df["Vehicle_to_Casualty_Ratio"] = df["Number of Vehicles Involved"] / (df["Number of Casualties"] + 1)
df["Fatality_Rate"] = df["Number of Fatalities"] / (df["Number of Casualties"] + 1)
df["Driver_Age_Group"] = pd.cut(
    df["Driver Age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=["Teen", "Young", "Adult", "MiddleAge", "Senior"]
)

# Encode age group numerically
df["Driver_Age_Group"] = df["Driver_Age_Group"].astype(str).map({
    "Teen": 0, "Young": 1, "Adult": 2, "MiddleAge": 3, "Senior": 4
})

# ---------------------------------------------------
# 3. Convert continuous Risk_Score → categorical Risk_Level
# ---------------------------------------------------
# Explicit bins based on score ranges
bins = [0, 50, 80, 100]
labels = ["Low", "Medium", "High"]
df["Risk_Level"] = pd.cut(df["Risk_Score"], bins=bins, labels=labels, include_lowest=True)


# ---------------------------------------------------
# 4. Prepare features and target
# ---------------------------------------------------
X = df.drop(columns=["Risk_Score", "Risk_Level"])
y = df["Risk_Level"]

# Load encoders and scaler if available
try:
    encoders = joblib.load("feature_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    print("🔄 Loaded encoders and scaler successfully.")
except:
    print("⚠️ Could not load encoders/scaler — proceeding without them.")

# ---------------------------------------------------
# 5. Split data
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"🧩 Train/Test Split: {X_train.shape}, {X_test.shape}")
# 🧹 Handle missing values before SMOTE
print("🧹 Filling missing values before SMOTE...")
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

# ---------------------------------------------------
# 6. Handle imbalance (SMOTE)
# ---------------------------------------------------
# 🧹 Handle missing values before SMOTE
print("🧹 Filling missing values before SMOTE...")
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

# ⚖️ Apply SMOTE
# ⚖️ Apply SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)

print("⚖️ Applying SMOTE oversampling...")
X_train, y_train = sm.fit_resample(X_train, y_train)



# ---------------------------------------------------
# 7. Random Forest with hyperparameter tuning
# ---------------------------------------------------
print("🔍 Running GridSearchCV for Random Forest tuning...")

param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [6, 10, 15],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced"]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)
grid.fit(X_train, y_train)
rf_best = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)

# ---------------------------------------------------
# 8. Evaluate Random Forest
# ---------------------------------------------------
rf_pred = rf_best.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="macro")
print("\n🚀 Training XGBoost classifier..." if XGBOOST_AVAILABLE else "\n🚀 Training Gradient Boosting classifier (xgboost not installed)...")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

if XGBOOST_AVAILABLE:
    # Encode categorical labels numerically for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Train XGBoost
    xgb_model = XGBClassifier(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb_model.fit(X_train, y_train_enc)

    # Predict numeric classes
    xgb_pred = xgb_model.predict(X_test)

    # Decode both y_test_enc and predictions back to original labels
    y_test_decoded = le.inverse_transform(y_test_enc)
    xgb_pred_decoded = le.inverse_transform(xgb_pred)
else:
    # Fall back to sklearn's GradientBoostingClassifier when xgboost is not available
    from sklearn.ensemble import GradientBoostingClassifier

    # Train using original categorical labels
    xgb_model = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_decoded, xgb_pred_decoded, labels=["Low", "Medium", "High"]),
            annot=True, fmt="d", cmap="Purples",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix - XGBoost Accident Risk Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(f"Accuracy: {xgb_acc:.3f}")
print(f"F1 Macro: {xgb_f1:.3f}")
print(classification_report(y_test_decoded, xgb_pred_decoded))
xgb_acc = accuracy_score(y_test_decoded, xgb_pred_decoded)
xgb_f1 = f1_score(y_test_decoded, xgb_pred_decoded, average="macro")

print("\n📊 XGBoost Model Performance:")
print(f"Accuracy: {xgb_acc:.3f}")
print(f"F1 Macro: {xgb_f1:.3f}")
print(classification_report(y_test_decoded, xgb_pred_decoded))

# ---------------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, xgb_pred, labels=["Low", "Medium", "High"]),
            annot=True, fmt="d", cmap="Purples",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix - XGBoost Accident Risk Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------------
# 11. Feature Importance (XGBoost)
# ---------------------------------------------------
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_imp.head(15), x="Importance", y="Feature", palette="magma")
plt.title("Top 15 Important Features (XGBoost)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 12. Save best model
# ---------------------------------------------------
best_model = xgb_model if xgb_acc > rf_acc else rf_best
joblib.dump(best_model, "risk_model_refined.pkl")
print("\n💾 Saved best model as 'risk_model_refined.pkl'")

# ---------------------------------------------------
# 13. Summary
# ---------------------------------------------------
print("\n🔍 Final Summary:")
print(f"Random Forest → Acc: {rf_acc:.3f}, F1: {rf_f1:.3f}")
print(f"XGBoost → Acc: {xgb_acc:.3f}, F1: {xgb_f1:.3f}")

if max(rf_acc, xgb_acc) > 0.75:
    print("✅ Excellent model — strong predictive capability.")
elif max(rf_acc, xgb_acc) > 0.55:
    print("🟡 Moderate model — acceptable performance, can improve with more contextual data.")
else:
    print("⚠️ Low model accuracy — more diverse or higher-quality data required.")
