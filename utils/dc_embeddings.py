import boto3
import io
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# === CONFIG ===
bucket_name = 'vian-mvp-outputs-raisha-ohio'
dc_folder = 'embeddings/dc/'
model = SentenceTransformer("malteos/scincl")
s3 = boto3.client('s3')

# === COLLECT _dc.csv FILES FROM BUCKET ===
response = s3.list_objects_v2(Bucket=bucket_name, Prefix='dc/')
dc_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('_dc.csv')]

all_keys = []
all_texts = []

print(f"Found {len(dc_files)} DrugCentral CSVs in bucket.")

for dc_key in dc_files:
    print(f"\nProcessing: {dc_key}")
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=dc_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), dtype=str).fillna('')
    except Exception as e:
        print(f"Error reading {dc_key}: {e}")
        continue

    if df.shape[1] < 2:
        print(f"Skipping {dc_key} — needs at least 2 columns")
        continue

    keys = df.iloc[:, 0].tolist()
    texts = df.iloc[:, 1:].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1).tolist()

    all_keys.extend(keys)
    all_texts.extend(texts)

# === EMBED & UPLOAD TO S3 ===
if not all_keys:
    print("No valid data found in any _dc.csv files.")
else:
    print(f"\nTotal valid entries: {len(all_keys)}")
    embeddings = model.encode(
        all_texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True, device='cuda'
    )

    # Save into in-memory buffer
    pt_buffer = io.BytesIO()
    torch.save({
        'keys': all_keys,
        'texts': all_texts,
        'embeddings': embeddings
    }, pt_buffer)
    pt_buffer.seek(0)

    pt_key = f"{dc_folder}dc_embeddings.pt"
    s3.upload_fileobj(pt_buffer, Bucket=bucket_name, Key=pt_key)
    print(f"\nUploaded embeddings to: {pt_key}")
