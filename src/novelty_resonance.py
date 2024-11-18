import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine


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
            res.append(np.mean([metric(a[j], a[j - d]) for d in range(window_size)]))
    return np.array(res)


def transience(a: np.ndarray, window_size: int, metric=cosine) -> np.ndarray:
    res = []
    for j in range(len(a)):
        if j > (len(a) - window_size - 1):
            res.append(np.nan)
        else:
            res.append(np.mean([metric(a[j], a[j + d]) for d in range(window_size)]))
    return np.array(res)


records = load_jsonl("dat/hpv_query_data.jsonl")

data = pd.DataFrame.from_records(records)
data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].map(
    lambda dt: datetime(year=dt.year, month=dt.month, day=1)
)

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
models = ["keynmf", "s3"]
fig = make_subplots(rows=2, cols=1, subplot_titles=["Novelty", "Resonance"])
for model in models:
    topics = pd.read_csv(f"topics/{model}_20_doc_topic_matrix.csv", index_col=0)
    topics = topics.join(data[["month"]])
    topics = topics.groupby("month").mean()
    topics = topics.sort_index()
    topic_novelty = novelty(topics.to_numpy(), window_size=20)
    topic_transience = transience(topics.to_numpy(), window_size=20)
    topic_resonance = topic_novelty - topic_transience
    fig.add_trace(
        go.Scatter(
            name=model,
            y=topic_novelty,
            x=topics.index,
            line=dict(color="red" if model == "keynmf" else "blue"),
            legendgroup=model,
        ),
        col=1,
        row=1,
    )
    fig.add_trace(
        go.Scatter(
            name=model,
            y=topic_resonance,
            x=topics.index,
            line=dict(color="red" if model == "keynmf" else "blue"),
            legendgroup=model,
            showlegend=False,
        ),
        col=1,
        row=2,
    )
for facet in range(2):
    for idx, row in events.iterrows():
        fig = fig.add_vline(x=row["date"], col=1, row=facet + 1)
        fig = fig.add_annotation(
            x=row["date"],
            y=np.random.normal(0.25, 0.05),
            text=row["event_name"],
            showarrow=False,
            textangle=-90,
            xshift=0,
            xanchor="right",
            col=1,
            row=facet + 1,
        )
fig = fig.update_layout(width=1400, height=800, template="plotly_white")
fig.write_html("figures/novelty_resonance.html")
