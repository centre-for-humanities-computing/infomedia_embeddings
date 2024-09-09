"""
Make a query for HPV articles in the corpus (based on original paper https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-018-6268-x)

Run the script to query corpus and save the results in a jsonl file (in dat/hpv_data/files_{date}).

Note that if you re-run the script in the same day, the script will overwrite existing files:
    python hpv_query.py 
"""
import json
import pathlib
from datetime import datetime
from typing import Iterable

from tqdm import tqdm

# terms defined from  paper (https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-018-6268-x)
MAIN_TERM = "hpv"
SECOND_TERMS = [
    "livmoderhalskræft",
    "cervix cancer",
    "cancer",
    "kræft",
    "POTS",
    "CRPS",
    "kønsvorter",
    "kondylomer",
    "condyloma",
    "sygdom",
    "kønssygdom",
]


def contains_terms(text: str) -> bool:
    """
    Check if text contains any of the terms defined in the paper.
    """
    text = text.lower()
    contains_main_term = (
        MAIN_TERM in text
    )  # no exact matching for main term (we want to catch "HPV-vaccine" as well)

    if not contains_main_term:
        return False

    # create vocabulary for exact matching of second terms (set for faster lookupª)
    doc_vocab = set(text.split())

    for term in SECOND_TERMS:
        if term in doc_vocab:
            return True

    return False


def stream_entries(file_path: str) -> Iterable[dict]:
    """
    Stream entries from a jsonl file.
    """
    file = pathlib.Path(file_path)

    with open(file) as f:
        for line in f:
            yield json.loads(line.strip())


def process_one_file(file_path: str, text_key: str = "content") -> Iterable[dict]:
    """
    process one file.

    args:
        file_path: path to json file
        text_key: key to the text in the json file

    returns:
        filter object with entries with entries that contain terms defined in the paper
    """
    entries = stream_entries(file_path)
    queried_entries = filter(lambda entry: contains_terms(entry[text_key]), entries)

    return queried_entries


def process_all_files(
    file_paths: list,
    save_file: pathlib.Path,
    log_file: pathlib.Path,
    text_key="content",
):
    """
    Process all files with an option to overwrite existing results and automatically reprocess the last file in the log if it exists.

    Reprocessing the last file ensures that no data is complete if the process was interrupted mid-file.

    args:
        file_paths: list
        save_file: pathlib.Path
        log_file: pathlib.Path
        text_key: str
    """
    for file in [save_file, log_file]:
        if (
            file.exists()
        ):  # since we are appending to the logfile + savefile, we want to remove it if we re-run the script (to avoid duplicates)
            file.unlink()

    for file_path in tqdm(file_paths):
        queried_entries = process_one_file(file_path, text_key)

        article_count = 0
        with open(save_file, "a") as f:
            for entry in queried_entries:
                f.write(json.dumps(entry) + "\n")
                article_count += 1

        with open(log_file, "a") as f:
            log_data = {"file_path": str(file_path), "hpv_article_count": article_count}
            f.write(json.dumps(log_data) + "\n")


def main():
    path = pathlib.Path(__file__)
    corpus_path = path.parents[3] / "infomedia-embedding" / "dat" / "corpus"

    file_paths = [file for file in corpus_path.iterdir() if file.suffix == ".jsonl"]
    file_paths = sorted(file_paths)

    # name folder with date (to avoid overwriting if the script is unintentionally suddenly re-run at a later time)
    now = datetime.now()
    date = now.strftime("%Y_%m_%d")

    save_path = (
        path.parents[3] / "infomedia-embedding" / "dat" / "hpv_data" / f"files_{date}"
    )
    save_path.mkdir(exist_ok=True, parents=True)

    process_all_files(
        file_paths=file_paths,
        save_file=save_path / "hpv_query_data.jsonl",
        log_file=save_path / "hpv_query_log.jsonl",
        text_key="content",
    )


if __name__ == "__main__":
    main()
