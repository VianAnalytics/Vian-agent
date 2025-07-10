from transformers import pipeline
from utils.prompt_utils import make_prompt

def run_reasoning(gene, drug, mechanism, abstract):
    prompt = make_prompt(gene, drug, mechanism, abstract)

    pipe = pipeline("text-generation", model="microsoft/biogpt", tokenizer="microsoft/biogpt")
    result = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)

    return result[0]['generated_text'].strip()