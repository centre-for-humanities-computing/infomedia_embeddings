import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF

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


def load_jsonl(path: Path | str) -> list[dict]:
    entries = []
    with Path(path).open() as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    trf = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        prompts=dict(passage="passage: ", query="query: "),
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
    fig = model.plot_topics_over_time(top_k=10)
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
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    fig.show()
    fig.write_html(figures_dir.joinpath("keynmf_20_topics_over_time.html"))
    topics_dir = Path("topics")
    topics_dir.mkdir(exist_ok=True)
    with topics_dir.joinpath("keynmf_20_topics_over_time.csv").open("w") as out_file:
        out_file.write(model.export_topics_over_time(top_k=10, format="csv"))
    with topics_dir.joinpath("keynmf_20_topic_descriptions.csv").open("w") as out_file:
        out_file.write(model.export_topics(top_k=10, format="csv"))
    df_topics = pd.DataFrame(doc_topic_matrix, columns=model.topic_names)
    df_topics.to_csv(topics_dir.joinpath("keynmf_20_doc_topic_matrix.csv"))
    model.push_to_hub("kardosdrur/hpv_keynmf_20")


if __name__ == "__main__":
    main()
