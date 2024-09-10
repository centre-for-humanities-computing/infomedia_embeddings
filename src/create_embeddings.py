"""
Run embeddings on data jsonl file and save the embeddings in a numpy file: 

python src/create_embeddings.py --in_file /path/to/data.jsonl --out_file /path/to/embeddings.npy
"""

import argparse
import json
import pathlib

import numpy as np
from sentence_transformers import SentenceTransformer


def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use.",
        default="intfloat/multilingual-e5-large",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for encoding",
        default=32,
    )

    parser.add_argument("--in_file", type=str, help="Path to file with data to encode")

    parser.add_argument("--out_file", type=str, help="Path to save the embeddings")

    args = parser.parse_args()
    return args


def main():
    args = input_parse()
    path = pathlib.Path(__file__)

    default_paths = {
        "in_file": path.parents[3]
        / "infomedia-embedding"
        / "dat"
        / "hpv_data"
        / "hpv_query_data.jsonl",
        "out_file": path.parents[3]
        / "infomedia-embedding"
        / "dat"
        / "hpv_data"
        / "hpv_embeddings.npy",
    }

    # grab data from default path if not provided
    in_file = (
        pathlib.Path(args.in_file)
        if args.in_file is not None
        else default_paths["in_file"]
    )
    with open(in_file, "r") as f:
        data = [json.loads(line) for line in f]

    # extract sents
    sents = [row["content"] for row in data]

    # encode
    model = SentenceTransformer(
        model_name_or_path=args.model_name,  # defaults to "intfloat/multilingual-e5-large"
        prompts={"query": "query: ", "passage": "passage: "},
        default_prompt_name="passage",
    )

    embeddings = model.encode(
        sentences=sents,
        batch_size=args.batch_size,
        prompt_name="passage",
        show_progress_bar=True,
    )

    # save embeddings
    out_file = (
        pathlib.Path(args.out_file)
        if args.out_file is not None
        else default_paths["out_file"]
    )
    np.save(out_file, embeddings)


if __name__ == "__main__":
    main()
