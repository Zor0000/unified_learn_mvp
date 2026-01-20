import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

print("🚀 Running RAGAS evaluation...\n")

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv("copilot_studio_eval_v1.csv")

# -------------------------------
# Ensure required columns
# -------------------------------

# If answer column is missing, use ground_truth as a proxy
if "answer" not in df.columns:
    df["answer"] = df["ground_truth"]

# If contexts column is missing, create minimal context
if "contexts" not in df.columns:
    df["contexts"] = df["ground_truth"].apply(lambda x: [x])

# -------------------------------
# Convert to Dataset
# -------------------------------
dataset = Dataset.from_pandas(df)

print("📄 Dataset preview:")
print(dataset)

# -------------------------------
# Run RAGAS
# -------------------------------
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print("\n🎉 RAGAS evaluation complete!\n")
print(results)
