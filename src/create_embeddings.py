import argparse
import json
import pathlib
from typing import List, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

EncodingTypes = Literal[
    "query", "passage"
]  # query for question, passage for text to be encoded for retrieval

class E5Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        sentences: list,
        batch_size: int,
        query_type=Literal["query", "passage"],
        normalize: bool = False,
    ) -> np.ndarray:
        """
        encode sentences with query or passage prompt
        """
        if query_type == "query":
            prompt = "query: "
        elif query_type == "passage":
            prompt = "passage: "

        embeddings = self.model.encode(
            sentences=sentences,
            batch_size=batch_size,
            prompt=prompt,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        return embeddings


def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to use.",
        default="intfloat/multilingual-e5-large",
    )
    args = parser.parse_args()
    return args


def main():
    path = pathlib.Path(__file__)
    data_path = (
        path.parents[3]
        / "infomedia-embedding"
        / "dat"
        / "hpv_data"
    )

    with open(data_path / "hpv_query_data.jsonl", "r") as f:
        hpv_data = [json.loads(line) for line in f]

    texts = [row["content"] for row in hpv_data]

    model = E5Model(model_name="intfloat/multilingual-e5-large")
    embeddings = model.encode(sentences=texts, batch_size=32, query_type="passage")
    
    np.save(data_path / "hpv_embeddings.npy", embeddings)


if __name__ == "__main__":
    main()
