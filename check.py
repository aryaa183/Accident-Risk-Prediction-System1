import pandas as pd

df = pd.read_csv("cleaned_accident_data.csv")
print(df["Accident Severity"].unique())
