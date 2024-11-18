import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF


def load_jsonl(path: Path | str) -> list[dict]:
    entries = []
    with Path(path).open() as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


trf = SentenceTransformer(
    "intfloat/multilingual-e5-large", prompts=dict(passage="passage: ", query="query: ")
)
trf.default_prompt_name = "query"

embeddings = np.load("dat/hpv_embeddings.npy")
records = load_jsonl("dat/hpv_query_data.jsonl")
keywords = load_jsonl("dat/hpv_keywords.jsonl")

data = pd.DataFrame.from_records(records)
data["date"] = pd.to_datetime(data["date"])

corpus = list(data.content)
embeddings = embeddings[data.index]

model = KeyNMF(20, encoder=trf, random_state=42, top_n=15)
doc_topic_matrix = model.fit_transform_dynamic(
    corpus, keywords=keywords, timestamps=data["date"], bins=30
)

model.print_topics()

events = pd.DataFrame(
    {
        "date": [
            "2009-01-01",
            "2009-06-01",
            "2012-09-01",
            "2012-12-01",
            "2013-02-01",
            "2015-03-01",
            "2013-09-01",
            "2015-07-01",
            "2015-12-01",
        ],
        "event_name": [
            "Start of routine vaccincation",
            "Promotion campaign",
            "Catch-up programme",
            "Actor dies",
            "Critical articles",
            "De vaccinerede piger",
            "Safety signal EMA",
            "PRAC request",
            "PRAC report",
        ],
    }
)
fig = model.plot_topics_over_time()
events["date"] = pd.to_datetime(events["date"])
for idx, row in events.iterrows():
    fig = fig.add_vline(x=row["date"])
    fig = fig.add_annotation(
        x=row["date"],
        y=np.random.normal(0.25, 0.05),
        text=row["event_name"],
        showarrow=False,
        textangle=-90,
        xshift=0,
        xanchor="right",
    )
# fig = fig.update_layout(xaxis_range=[datetime(2008, 6, 15), datetime(2016, 5, 1)])
fig.show()

model.export_topics_over_time(format="csv")
