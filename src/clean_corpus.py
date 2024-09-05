import argparse
import json
import re
import zipfile
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

HTML_CLEANER = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def remove_html_tags(text: str) -> str:
    return re.sub(HTML_CLEANER, "", text)


def remove_excess_whitespace(text: str) -> str:
    return " ".join(text.split())


def extract_text_content(entry: dict) -> str:
    columns = ["Heading", "SubHeading", "Paragraph", "BodyText"]
    parts = [entry[col] for col in columns]
    parts = map(remove_html_tags, parts)
    parts = map(remove_excess_whitespace, parts)
    return " ".join(parts)


def failsafe_jsonl_stream(file: Path) -> Iterable[dict]:
    with file.open() as in_file:
        for line in in_file:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                yield entry
            except JSONDecodeError as e:
                print(f"WARNING: Couldn't decode current line: {line}. Reason: {e}")


def stream_entries(newspaper_dir: Path) -> Iterable[dict]:
    year_dirs = [path for path in newspaper_dir.iterdir() if path.is_dir()]
    year_dirs = tqdm(year_dirs, desc="Processing all years in newspaper")
    for year_dir in year_dirs:
        files = [
            path
            for path in year_dir.iterdir()
            if path.is_file() and path.name.endswith(".ndjson")
        ]
        year = year_dir.stem
        for file in files:
            _, date = file.stem.rsplit("_", 1)
            for i_entry, entry in enumerate(failsafe_jsonl_stream(file)):
                # Using double underscore, because some newspapers have underscore
                # in their name
                newspaper_name = newspaper_dir.stem
                entry_id = "__".join([newspaper_name, year, date, str(i_entry)])
                content = extract_text_content(entry)
                entry = {
                    "entry_id": entry_id,
                    "newspaper_name": newspaper_name,
                    "year": year,
                    "date": date,
                    "content": content,
                    **entry,
                }
                yield entry


def main(input_file: str = "infomedia.zip"):
    out_dir = Path(input_file).parent.joinpath("corpus")
    out_dir.mkdir(exist_ok=True)
    in_dir = zipfile.Path(input_file, at="infomedia/")
    newspaper_dirs = [path for path in in_dir.iterdir() if path.is_dir()]
    for newspaper_dir in newspaper_dirs:
        newspaper_name = newspaper_dir.stem
        print("Processing newspaper: ", newspaper_name)
        with out_dir.joinpath(f"{newspaper_name}.jsonl").open("w") as out_file:
            entries = stream_entries(newspaper_dir)
            for entry in entries:
                out_file.write(json.dumps(entry) + "\n")
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Infomedia Cleaner",
        description="Cleans entries in the infomdedia dataset and gives each article a unique ID.",
    )
    parser.add_argument(
        "input_file",
        type=str,
        default="infomedia.zip",
        help="ZIP file to read the corpus from.",
    )
    main()
