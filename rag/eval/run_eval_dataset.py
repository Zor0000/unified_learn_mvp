import pandas as pd
from rag.eval.run_rag import run_rag_question

# ======================================================
# CONFIG
# ======================================================

INPUT_CSV = "copilot_studio_eval_v1.csv"
OUTPUT_CSV = "copilot_studio_eval_ragas.csv"


# ======================================================
# MAIN
# ======================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    results = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
        "run_id": [],              # 🔹 NEW
    }

    print("🚀 Running RAG pipeline to build RAGAS dataset (Milvus + LangSmith)...\n")

    for idx, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]

        print(f"🔹 [{idx + 1}/{len(df)}] {question}")

        # -------------------------------
        # Run RAG (Milvus-backed)
        # -------------------------------
        rag_output = run_rag_question(question)

        contexts = rag_output["contexts"]     # List[str]
        answer = rag_output["answer"]         # str
        run_id = rag_output.get("run_id")     # str | None

        # -------------------------------
        # Store results
        # -------------------------------
        results["question"].append(question)
        results["contexts"].append(contexts)
        results["answer"].append(answer)
        results["ground_truth"].append(ground_truth)
        results["run_id"].append(run_id)      # 🔹 NEW

        print("   ✅ Done\n")

    # -------------------------------
    # Save RAGAS-ready CSV
    # -------------------------------
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("🎉 RAGAS dataset created!")
    print(f"💾 Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
