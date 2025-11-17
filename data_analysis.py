# STEP 1: LOAD AND INSPECT DATA
# -------------------------------------
import pandas as pd

# 1️⃣ Load the CSV file
file_path = r"C:\Users\Arya\Downloads\aiml_dataset\accident_prediction_india.csv"
df = pd.read_csv(file_path)

# 2️⃣ Display basic info
print("🔹 Shape of dataset:", df.shape)
print("\n🔹 Column Names:\n", df.columns.tolist())

# 3️⃣ Show first 5 rows
print("\n🔹 Sample Data:\n", df.head())

# 4️⃣ Data types and non-null counts
print("\n🔹 Data Info:")
print(df.info())

# 5️⃣ Missing value summary
print("\n🔹 Missing Values per Column:\n", df.isnull().sum())

# 6️⃣ Quick unique-value overview for first few columns
print("\n🔹 Unique values check (first 10 unique entries per column):")
for col in df.columns[:10]:   # Adjust range if needed
    print(f"\nColumn: {col}")
    print(df[col].unique()[:10])
