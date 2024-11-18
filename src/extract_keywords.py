import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF

model = SentenceTransformer(
    "intfloat/multilingual-e5-large", prompts=dict(passage="passage: ", query="query: ")
)
model.default_prompt_name = "query"

embeddings = np.load("dat/hpv_embeddings.npy")
records = []
with Path("dat/hpv_query_data.jsonl").open() as in_file:
    for line in in_file:
        records.append(json.loads(line))

data = pd.DataFrame.from_records(records)
data["date"] = pd.to_datetime(data["date"])

corpus = list(data.content)
embeddings = embeddings[data.index]

keynmf = KeyNMF(10, encoder=model, random_state=42, top_n=100)

print("Extracting keywords")
keywords_path = Path("dat/hpv_keywords.jsonl")
keywords = keynmf.extract_keywords(corpus, embeddings)
print("Saving")

with keywords_path.open("w") as out_file:
    for keys in keywords:
        out_file.write(json.dumps(keys) + "\n")
print("DONE")
