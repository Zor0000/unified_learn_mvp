import math
import json
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langsmith import Client

# ==============================
# CONFIG
# ==============================

INPUT_CSV = "copilot_studio_eval_ragas.csv"

# Outputs
AGGREGATE_JSON = "ragas_results_summary.json"
PER_SAMPLE_CSV = "ragas_results_per_sample.csv"

LANGSMITH_PROJECT = "milvus-rag-evaluation"

# ==============================
# MAIN
# ==============================

def main():
    print("🚀 Running RAGAS evaluation (with persistence + LangSmith)...\n")

    # ------------------------------
    # Load dataset
    # ------------------------------
    df = pd.read_csv(INPUT_CSV)

    # Ensure contexts are proper lists
    df["contexts"] = df["contexts"].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    dataset = Dataset.from_pandas(df)

    print("📄 Dataset preview:")
    print(dataset)

    # ------------------------------
    # Run RAGAS
    # ------------------------------
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # ------------------------------
    # 1️⃣ Save AGGREGATE results
    # ------------------------------
    aggregate_scores = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
        "context_recall": float(results["context_recall"]),
    }

    with open(AGGREGATE_JSON, "w", encoding="utf-8") as f:
        json.dump(aggregate_scores, f, indent=2)

    print("\n📊 Aggregate RAGAS metrics saved:")
    print(aggregate_scores)

    # ------------------------------
    # 2️⃣ Save PER-SAMPLE results
    # ------------------------------
    per_sample_df = pd.DataFrame({
        "question": df["question"],
        "faithfulness": results.scores["faithfulness"],
        "answer_relevancy": results.scores["answer_relevancy"],
        "context_precision": results.scores["context_precision"],
        "context_recall": results.scores["context_recall"],
    })

    per_sample_df.to_csv(PER_SAMPLE_CSV, index=False)

    print(f"\n🧾 Per-sample RAGAS results saved to: {PER_SAMPLE_CSV}")

    # ------------------------------
    # 3️⃣ Log RAGAS metrics to LangSmith
    # ------------------------------
    client = Client()

    for i, row in df.iterrows():
        run_id = row.get("run_id")

        # ✅ Skip missing / invalid run_ids (NaN becomes float)
        if run_id is None:
            continue
        if isinstance(run_id, float) and math.isnan(run_id):
            continue

        run_id = str(run_id)  # ensure UUID string

        # Weighted aggregate score (customizable)
        aggregate_score = (
            0.35 * float(results.scores["faithfulness"][i])
            + 0.30 * float(results.scores["answer_relevancy"][i])
            + 0.20 * float(results.scores["context_precision"][i])
            + 0.15 * float(results.scores["context_recall"][i])
        )

        client.create_feedback(
            run_id=run_id,
            key="ragas_metrics",
            score=aggregate_score,
            value={
                "faithfulness": float(results.scores["faithfulness"][i]),
                "answer_relevancy": float(results.scores["answer_relevancy"][i]),
                "context_precision": float(results.scores["context_precision"][i]),
                "context_recall": float(results.scores["context_recall"][i]),
            },
            comment="RAGAS evaluation metrics",
        )

    print("\n📊 RAGAS metrics logged to LangSmith successfully!")
    print("✅ RAGAS evaluation fully persisted.")


if __name__ == "__main__":
    main()
