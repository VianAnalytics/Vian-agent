import pandas as pd
from agents.reasoning_agent import run_reasoning

df = pd.read_csv("data/input.csv")

results = []
for _, row in df.iterrows():
    summary = run_reasoning(row['gene'], row['drug'], row['mechanism'], row['abstract'])
    results.append({
        "gene": row['gene'],
        "drug": row['drug'],
        "mechanism": row['mechanism'],
        "abstract": row['abstract'],
        "biogpt_summary": summary
    })

out_df = pd.DataFrame(results)
out_df.to_csv("data/biogpt_output.csv", index=False)