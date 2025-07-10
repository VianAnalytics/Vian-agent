def make_prompt(gene, drug, mechanism, abstract):
    return f"""Gene: {gene}
Drug: {drug}
Mechanism: {mechanism}
Abstract: {abstract}

Is this drug effective for {gene}-related disease? Justify."""