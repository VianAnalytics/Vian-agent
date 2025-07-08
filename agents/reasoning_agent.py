import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_biogpt():
    print("⏳ Loading BioGPT...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

def run_reasoning(pipe, gene, drug, mechanism, abstract):
    prompt = (
        f"Gene: {gene}\n"
        f"Drug: {drug}\n"
        f"Mechanism: {mechanism}\n"
        f"Abstract: {abstract}\n"
        f"Is this drug effective for {gene}-related disease? Justify."
    )
    result = pipe(prompt, max_new_tokens=80, do_sample=True)[0]["generated_text"]
    return result.strip()

def process_pubmed(input_csv="data/pubmed_parsed.csv", output_csv="outputs/biogpt_output.csv"):
    df = pd.read_csv(input_csv)
    pipe = load_biogpt()

    df["biogpt_summary"] = df.apply(
        lambda row: run_reasoning(pipe, row["gene"], row["drug"], row["mechanism"], row["abstract"]),
        axis=1
    )

    df.to_csv(output_csv, index=False)
    print(f"✅ Output saved at {output_csv}")