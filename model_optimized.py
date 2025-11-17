# STEP 4: MODEL OPTIMIZATION & BALANCING
# --------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_accident_data.csv")
print("✅ Cleaned dataset loaded:", df.shape)

# Define target and features
X = df.drop(columns=["Accident Severity"])
y = df["Accident Severity"]

# Encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "encoders.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("📊 Train/Test Split:", X_train.shape, X_test.shape)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("🔁 After SMOTE balancing:", X_train.shape)

# Try two models: RandomForest and XGBoost
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42
    )
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    if acc > best_acc:
        best_acc = acc
        best_model = model

# Confusion Matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title(f"Best Model Confusion Matrix ({type(best_model).__name__})")
plt.show()

# Save final model
joblib.dump(best_model, "accident_model_optimized.pkl")
print(f"\n✅ Best model saved! Accuracy = {best_acc*100:.2f}%")
