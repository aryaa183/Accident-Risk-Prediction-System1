# --------------------------------------------------------
# STEP 2: MODEL TRAINING ON REFINED DATA
# --------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Load refined dataset
# -----------------------------
df = pd.read_csv("refined_accident_data.csv")
print(f"✅ Refined dataset loaded: {df.shape}")

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop("Accident Severity", axis=1)
y = df["Accident Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"🧩 Train/Test split: {X_train.shape}, {X_test.shape}")

# -----------------------------
# Train multiple models
# -----------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlOrRd")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if acc > best_acc:
        best_acc = acc
        best_model = model

# -----------------------------
# Save best model
# -----------------------------
joblib.dump(best_model, "accident_model_refined.pkl")
print(f"\n✅ Best model saved as 'accident_model_refined.pkl' with accuracy: {best_acc:.2f}")

# -----------------------------
# Feature Importance
# -----------------------------
if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance.head(15))
   
    print("\n🧠 Top Important Features Influencing Risk:")
    print(importance.head(10))
