import pathlib 
import json
import argparse
from tqdm import tqdm

def hpv_query_terms(): 
    '''
    from original paper: 
    https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-018-6268-x
    '''
    # alwaus included in query 
    main_term = "HPV"

    # any of these are included e.g., "HPV" AND "livmoderhalskræft" and then "HPV" AND "cervix cancer"
    second_term = [
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
            "kønssygdom"
            ]

    # get the term pairs "HPV" + term
    pairs = [(main_term, term) for term in second_term]

    return pairs

def query_json_ids(
                    data, 
                    term_pairs, 
                    text_key="content", 
                    id_key="entry_id"
                    ):
    '''
    query json for all term pairs. 
    '''
    all_ids = []

    for pair in term_pairs:
        term1, term2 = pair

        for row in data: 
            text = row[text_key]

            # reason to not split text when searching for term 1 is that we want to catch "HPV-vaccine" as well. 
            # on the othe rhand, we need to split when searching for term 2, because we don't want to catch "kræft" in e.g., "jeg bekræfter"
            if term1 in text and term2 in text.split(" "):
                all_ids.append(row[id_key])

    # an article can match multiple term pairs, but we only want it once
    unique_ids = list(set(all_ids))

    return unique_ids

def process_one_file(
                    file_path: pathlib.Path, 
                    term_pairs: list, 
                    text_key:str="content", 
                    id_key:str="entry_id"):
    '''
    process one file. 

    args:
        file_path: path to json file
        term_pairs: list of tuples with term pairs (e.g., ("HPV", "cancer"))
        text_key: what the text key is called in the json file
        id_key: what the id key is called in the json file
    '''
    # load data 
    with open(file_path) as f:
        data = [json.loads(line) for line in f]

    # get ids 
    ids = query_json_ids(
                        data = data,
                        term_pairs = term_pairs,
                        text_key=text_key,
                        id_key=id_key
                        )
    
    # select data based on ids 
    hpv_data = [row for row in data if row[id_key] in ids]

    return hpv_data

def process_all_files(
                        file_paths:list, 
                        log_file:pathlib.Path, 
                        save_file:pathlib.Path,
                        term_pairs:tuple, 
                        text_key="content", 
                        id_key="entry_id", 
                        overwrite=False
                        ):
    '''
    Process all files with an option to overwrite existing results and automatically reprocess the last file in the log if it exists.

    Reprocessing the last file ensures that no data is complete if the process was interrupted mid-file.

    args:
        file_paths: list
        log_file: pathlib.Path
        save_file: pathlib.Path
        term_pairs: list
        text_key: str
        id_key: 
        overwrite: whether to overwrite existing results
    '''
    # Initialize variables
    processed_filepaths = []
    all_data = []

    # handle overwrite scenarios
    if overwrite:
        print("[INFO:] Overwriting existing data.")
        # clear existing log and save files
        for file in [log_file, save_file]:
            if file.exists():
                file.unlink()
    
    elif log_file.exists():
        # load the log file
        with open(log_file) as f:
            log = [json.loads(line) for line in f]
            processed_filepaths = [entry["file_path"] for entry in log]

        if log:
            # reprocess the last file in the log
            last_entry = log[-1]
            print(f"[INFO:] Reprocessing {pathlib.Path(last_entry['file_path']).stem} to ensure data is not incomplete.")
            processed_filepaths.remove(last_entry["file_path"])

            # remove the last entry's data from the save file
            if save_file.exists():
                with open(save_file) as f:
                    all_data = [json.loads(line) for line in f if json.loads(line)[id_key] != last_entry["file_path"]]

                # rewrite the save file without the last entry's data
                with open(save_file, "w") as f:
                    for data in all_data:
                        f.write(json.dumps(data) + "\n")

            # rewrite the log file without the last entry
            with open(log_file, "w") as f:
                for entry in log:
                    f.write(json.dumps(entry) + "\n")

    # if not overwriting, load the save file
    if save_file.exists() and not overwrite:
        with open(save_file) as f:
            all_data = [json.loads(line) for line in f]

    # process files in file_paths
    for file_path in tqdm(file_paths):
        # check if the file has been processed, and skip it unless overwrite is enabled
        if str(file_path) in processed_filepaths and not overwrite:
            print(f"[INFO:] File {file_path.stem} already processed. Skipping.")
            continue
        
        # process the file
        hpv_data = process_one_file(file_path, term_pairs, text_key, id_key)
        all_data.extend(hpv_data)

        # append processed data to the save file
        with open(save_file, "a") as f:
            for row in hpv_data:
                f.write(json.dumps(row) + "\n")

        # log processed file
        with open(log_file, "a") as f:
            log_data = {"file_path": str(file_path), "hpv_article_count": len(hpv_data)}
            f.write(json.dumps(log_data) + "\n")

    return all_data

def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data.")
    args = parser.parse_args()
    return args

def main(): 
    args = input_parse()
    path = pathlib.Path(__file__)
    corpus_path = path.parents[3] / "infomedia-embedding" / "dat" / "corpus"

    # get filepaths and sort them
    file_paths = [file for file in corpus_path.iterdir() if file.suffix == ".jsonl"]
    file_paths = sorted(file_paths)

    # get term pairs
    term_pairs = hpv_query_terms()

    # process files
    save_file = path.parents[0] / "hpv_query_data.jsonl"
    log_file = path.parents[0] / "hpv_query_log.jsonl"

    all_data = process_all_files(
                                file_paths=file_paths,
                                log_file=log_file,
                                save_file=save_file,
                                term_pairs=term_pairs,
                                overwrite=args.overwrite
                                )
                                




if __name__ == "__main__": 
    main()

