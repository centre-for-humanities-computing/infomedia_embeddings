import json
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cosine
import plotly.graph_objects as go

import numpy as np
import pandas as pd


def load_jsonl(path: Path | str) -> list[dict]:
    entries = []
    with Path(path).open() as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def novelty(a: np.ndarray, window_size: int, metric=cosine) -> np.ndarray:
    res = []
    for j in range(len(a)):
        if j < window_size:
            res.append(np.nan)
        else:
            res.append(np.mean([metric(a[j], a[j-d]) for d in range(window_size)]))
    return np.array(res)

def transience(a: np.ndarray, window_size: int, metric=cosine) -> np.ndarray:
    res = []
    for j in range(len(a)):
        if j > (len(a) - window_size - 1):
            res.append(np.nan)
        else:
            res.append(np.mean([metric(a[j], a[j+d]) for d in range(window_size)]))
    return np.array(res)

records = load_jsonl("dat/hpv_query_data.jsonl")

data = pd.DataFrame.from_records(records)
data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].map(lambda dt: datetime(year=dt.year, month=dt.month, day=1))

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
models = ["keynmf_20", "s3_20"]
for model in models:
    topics = pd.read_csv(f"topics/{model}_doc_topic_matrix.csv", index_col=0)
    topics = topics.join(data[["month"]])
    topics = topics.groupby("month").mean()
    topics = topics.sort_index()
    topic_novelty = novelty(topics.to_numpy(), window_size=10)
    topic_transience = transience(topics.to_numpy(), window_size=10)
    topic_resonance = topic_novelty - topic_transience
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="novelty", y=topic_novelty, x=topics.index))
    fig.add_trace(go.Scatter(name="resonance", y=topic_resonance, x=topics.index))
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
    fig.update_layout(template="plotly_white")
    fig.show()

