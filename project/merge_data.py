import os
import pandas as pd

labels = ["Are", "Good", "Hello","How","Is","My","Name","What","You"]  # 👈 Add your labels here

all_data = []
for label in labels:
    filename = f"{label}_data.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, header=None)
        df['label'] = label
        all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("asl_dataset.csv", index=False)
print("[✓] All data merged into 'asl_dataset.csv'")
