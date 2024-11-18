import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from turftopic import SemanticSignalSeparation

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
events["date"] = pd.to_datetime(events["date"])

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
data = data.sort_values("date")

corpus = list(data.content)
embeddings = embeddings[data.index]

topic_model = SemanticSignalSeparation(
    10, encoder=model, vectorizer=CountVectorizer(max_features=8000), random_state=42
)
document_topic_matrix = topic_model.fit_transform(corpus, embeddings=embeddings)

topic_model.print_topics(top_k=10)

topic_model.print_representative_documents(8, corpus, document_topic_matrix)

relevant_topic = -document_topic_matrix[:, 8]
days = data.assign(signal=relevant_topic).groupby("date")[["signal"]].mean()
days = days.rolling(window=30).mean()
days = days.dropna().reset_index()

fig = px.line(days, x="date", y="signal", template="plotly_white")
for idx, row in events.iterrows():
    fig = fig.add_vline(x=row["date"])
    fig = fig.add_annotation(
        x=row["date"],
        y=np.random.normal(1, 0.5),
        text=row["event_name"],
        showarrow=False,
        textangle=-90,
        xshift=0,
        xanchor="right",
    )
fig = fig.update_layout(template="plotly_white")
fig.show()
