import pandas as pd
import json
import time

# Read the CSV file with multi-level headers
data = pd.read_csv("infy_2min_60days.csv", header=[0, 1], index_col=0)
print("Data shape:", data.shape)
print("First few rows:")
print(data.head())

# Flatten the multi-level column index
data.columns = ['_'.join(col).strip() for col in data.columns.values]

print("\nProcessing rows as JSON:")
print("-" * 50)

for index, row in data.iterrows():
    row_dict = row.to_dict()
    row_json = json.dumps(row_dict)
    print(f"Index: {index}")
    print(row_json)
    print("-" * 30)
    time.sleep(0.5)