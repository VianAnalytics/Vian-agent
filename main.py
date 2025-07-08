import pandas as pd
from agents.planner_agent import generate_plan, save_plan
from agents.reasoning_agent import process_pubmed
import os

def main():
    os.makedirs("outputs", exist_ok=True)

    print("📥 Loading gene input...")
    df = pd.read_csv("data/input_genes.csv")
    gene_list = df["gene"].tolist()

    print("🧠 Generating plan...")
    plan = generate_plan(gene_list)
    save_plan(plan)

    print("🤖 Running BioGPT reasoning on PubMed data...")
    process_pubmed()

if __name__ == "__main__":
    main()