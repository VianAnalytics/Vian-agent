import pandas as pd
import json

def generate_plan(gene_list):
    """
    Creates a planning JSON object for search or reasoning tasks.
    """
    plan = []
    for gene in gene_list:
        plan.append({
            "gene": gene,
            "task_type": "evidence_lookup",
            "search_queries": [
                f"What drugs are associated with {gene}?",
                f"What is the mechanism of action for drugs targeting {gene}?"
            ]
        })
    return plan

def save_plan(plan, path="outputs/plan_tasks.json"):
    with open(path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"✅ Plan saved at {path}")