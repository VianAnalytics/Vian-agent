import boto3
import io
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import json
import re

# === CONFIG ===
bucket_name = 'vian-mvp-outputs-raisha-ohio'
embedding_folder = 'embeddings/'
drug_gene_key = 'drug_gene_mapping.csv'
model = SentenceTransformer('malteos/scincl')
s3 = boto3.client('s3')

# === LOAD DRUG-GENE CSV ===
obj = s3.get_object(Bucket=bucket_name, Key=drug_gene_key)
dg_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
drug_set = set(dg_df['Drug'].dropna().str.upper().str.strip())

gene_set = set()
for gene_str in dg_df['Gene']:
    if pd.isna(gene_str) or gene_str == "No gene found":
        continue
    try:
        genes = json.loads(gene_str.replace("'", '"')) 
        gene_set.update(g.strip().upper() for g in genes)
    except Exception as e:
        continue

print(f"Loaded {len(drug_set)} drugs and {len(gene_set)} genes.")

# print("\n=== Extracted Gene Names ===")
# for gene in sorted(gene_set):
#     print(gene)
# print(f"\nTotal unique genes extracted: {len(gene_set)}")
# print("\n=== Extracted Drug Names ===")
# for drug in sorted(drug_set):
#     print(drug)
# print(f"\nTotal unique drugs extracted: {len(drug_set)}")

# === LIST FILES IN BUCKET ===
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix='pm_xmls/')

csv_files = []
for page in pages:
    contents = page.get('Contents', [])
    for obj in contents:
        key = obj['Key']
        if key.endswith('.csv'):
            name = key.split('/')[-1]
            csv_files.append(key)

# def process_csv(csv_key):
#     try:
#         print(f"\nProcessing: {csv_key}")
#         obj = s3.get_object(Bucket=bucket_name, Key=csv_key)
#         df = pd.read_csv(io.BytesIO(obj['Body'].read()))

#         df = df.dropna(subset=['abstract', 'authors', 'pubdate'])

#         texts, keys, titles = [], [], []
#         for row in df.itertuples(index=False):
#             title = str(row.title).strip()
#             abstract = str(row.abstract).strip()
#             authors = str(row.authors).strip()
#             pubdate = str(row.pubdate).strip()

#             combined = (title + " " + abstract).upper()
#             if not any(p.search(combined) for p in drug_patterns) and not any(p.search(combined) for p in gene_patterns):
#                 continue

#             first_author = authors.split(';')[0].split('|')[0].strip()
#             year = pubdate[:4] if len(pubdate) >= 4 else "0000"
#             key = f"{first_author}{year}"

#             texts.append(abstract)
#             keys.append(key)
#             titles.append(title)

#         if not texts:
#             print(f"No matches in {csv_key}")
#             return

#         embeddings = model.encode(texts, batch_size=650, show_progress_bar=False, convert_to_tensor=True)

#         pt_buffer = io.BytesIO()
#         torch.save({'keys': keys, 'texts': texts, 'titles': titles, 'embeddings': embeddings}, pt_buffer)
#         pt_buffer.seek(0)
#         pt_key = f"{embedding_folder}{csv_key.replace('.csv', '_onco.pt')}"
#         s3.upload_fileobj(pt_buffer, Bucket=bucket_name, Key=pt_key)
#         print(f"Uploaded {pt_key}")
#     except Exception as e:
#         print(f"Error processing {csv_key}: {e}")

for csv_key in csv_files:
    print(f"\nProcessing: {csv_key}")
    obj = s3.get_object(Bucket=bucket_name, Key=csv_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    drug_patterns = [re.compile(rf"\b{re.escape(drug)}\b") for drug in drug_set]
    gene_patterns = [re.compile(rf"\b{re.escape(gene)}\b") for gene in gene_set]
    
    def contains_drug_or_gene(text):
        return any(p.search(text) for p in drug_patterns) or any(p.search(text) for p in gene_patterns)
    
    df = df.dropna(subset=['abstract', 'authors', 'pubdate'])
    
    texts, keys, titles = [], [], []
    for row in df.itertuples(index=False):
        title = str(row.title).strip()
        abstract = str(row.abstract).strip()
        authors = str(row.authors).strip()
        pubdate = str(row.pubdate).strip()
    
        combined = (title + " " + abstract).upper()
        if not contains_drug_or_gene(combined):
            continue
    
        first_author = authors.split(';')[0].split('|')[0].strip()
        year = pubdate[:4] if len(pubdate) >= 4 else "0000"
        key = f"{first_author}{year}"
    
        texts.append(abstract)
        keys.append(key)
        titles.append(title)


    if not texts:
        print("No drug/gene-related abstracts found, skipping.")
        continue

    print(f"Encoding {len(texts)} abstracts...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)

    # === UPLOAD ===
    pt_buffer = io.BytesIO()
    torch.save({
        'keys': keys,
        'texts': texts,
        'titles': titles,
        'embeddings': embeddings
    }, pt_buffer)
    pt_buffer.seek(0)
    pt_key = f"{embedding_folder}{csv_key.replace('.csv', '_onco.pt')}"
    s3.upload_fileobj(pt_buffer, Bucket=bucket_name, Key=pt_key)
    print(f"Uploaded {pt_key} to S3.")

# from concurrent.futures import ThreadPoolExecutor

# drug_patterns = [re.compile(rf"\b{re.escape(drug)}\b") for drug in drug_set]
# gene_patterns = [re.compile(rf"\b{re.escape(gene)}\b") for gene in gene_set]

# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(process_csv, csv_files)


print("\nDone filtering and embedding PubMed abstracts.")

# # DrugCentral Embeddings
# # === COLLECT _dc.csv FILES FROM BUCKET ===
# response = s3.list_objects_v2(Bucket=bucket_name, Prefix='dc/')
# dc_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('_dc.csv')]

# all_keys = []
# all_texts = []

# print(f"Found {len(dc_files)} DrugCentral CSVs in bucket.")

# for dc_key in dc_files:
#     print(f"\nProcessing: {dc_key}")
#     try:
#         obj = s3.get_object(Bucket=bucket_name, Key=dc_key)
#         df = pd.read_csv(io.BytesIO(obj['Body'].read()), dtype=str).fillna('')
#     except Exception as e:
#         print(f"Error reading {dc_key}: {e}")
#         continue

#     if df.shape[1] < 2:
#         print(f"Skipping {dc_key} — needs at least 2 columns")
#         continue

#     keys = df.iloc[:, 0].tolist()
#     texts = df.iloc[:, 1:].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1).tolist()

#     all_keys.extend(keys)
#     all_texts.extend(texts)

# # === EMBED & UPLOAD TO S3 ===
# if not all_keys:
#     print("No valid data found in any dc files.")
# else:
#     print(f"\nTotal valid entries: {len(all_keys)}")
#     embeddings = model.encode(
#         all_texts, batch_size=650, show_progress_bar=True, convert_to_tensor=True
#     )

#     pt_buffer = io.BytesIO()
#     torch.save({
#         'keys': all_keys,
#         'texts': all_texts,
#         'embeddings': embeddings
#     }, pt_buffer)
#     pt_buffer.seek(0)

#     pt_key = f"{embedding_folder}drugcentral_embeddings.pt"
#     s3.upload_fileobj(pt_buffer, Bucket=bucket_name, Key=pt_key)
#     print(f"\nUploaded embeddings to: {pt_key}")

# print("\nDone processing all DrugCentral records.")