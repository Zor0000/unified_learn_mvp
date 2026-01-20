import pandas as pd
from datasets import Dataset

# Load your exported eval CSV (from main env)
df = pd.read_csv("copilot_studio_eval_v1.csv")

# IMPORTANT: contexts must be a list of strings
# If contexts is missing, we create a placeholder
if "contexts" not in df.columns:
    df["contexts"] = df["ground_truth"].apply(lambda x: [x])

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

print("✅ Dataset ready for RAGAS")
print(dataset)
