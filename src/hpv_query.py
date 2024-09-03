import pathlib 
import json

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

def query_json_ids(data, term_pairs, text_key="content", id_key="entry_id"):
    '''
    query json for all term pairs. 
    '''
    all_ids = []

    for pair in term_pairs:
        term1, term2 = pair

        for row in data: 
            text = row[text_key]

            if term1 in text and term2 in text.split(" "):
                all_ids.append(row[id_key])

    # an article can match multiple term pairs, but we only want it once
    unique_ids = list(set(all_ids))

    return unique_ids

def main(): 
    path = pathlib.Path(__file__)
    corpus_path = path.parents[3] / "infomedia-embedding" / "dat" / "corpus" / "Berlingske_Nyhedsbureau_BNB.jsonl"

    with open(corpus_path) as f:
        data = [json.loads(line) for line in f]

    # get terms 
    hpv_term_pairs = hpv_query_terms()

    # get ids
    ids = query_json_ids(
                        data = data,
                        term_pairs = hpv_term_pairs,
                        text_key="content",
                        id_key="entry_id"
                        )

    # get all
    hpv_data = [row for row in data if row["entry_id"] in ids]

    print(hpv_data[10]["content"])
    print(hpv_data[20]["content"])


if __name__ == "__main__": 
    main()

