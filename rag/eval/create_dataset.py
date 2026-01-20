from datasets import Dataset
import pandas as pd

data = {
    "question": [
        "What is Microsoft Copilot Studio?",
        "How do I publish a copilot in Copilot Studio?",
        "Why is authentication required in Copilot Studio?",
        "How is Copilot Studio different from Power Virtual Agents?",
        "Does Copilot Studio support on-prem Kubernetes deployment?"
    ],
    "ground_truth": [
        "Microsoft Copilot Studio is a tool used to build, customize, and manage copilots using conversational AI.",
        "To publish a copilot, you configure channels, validate the copilot, and then publish it so users can access it.",
        "Authentication ensures secure access to resources and protects user data and APIs used by the copilot.",
        "Copilot Studio extends Power Virtual Agents with generative AI, deeper orchestration, and better integration.",
        "Copilot Studio does not provide native support for on-prem Kubernetes deployment."
    ]
}

# Create HuggingFace dataset
dataset = Dataset.from_dict(data)

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Save as CSV
output_path = "copilot_studio_eval_v1.csv"
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved to {output_path}")
