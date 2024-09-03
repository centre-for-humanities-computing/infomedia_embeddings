import pathlib 
import pandas as pd

def query_terms(): 
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

def query_df(df, term_pairs):
    '''
    query df for all term pairs. save a new df with the ids of the articles that match
    '''
    ids = []

    for term_pair in term_pairs: 
        query = df[df["content"].str.contains(term_pair[0]) & df["content"].str.contains(term_pair[1])]
        
        # save the ids of the articles that match
        query_ids = query["entry_id"].tolist()

        ids.extend(query_ids)

    # filter df for the ids that match, avoid duplicates
    ids = set(ids)
    query_df = df[df["entry_id"].isin(ids)].reset_index(drop=True)

    return query_df


def main(): 
    term_pairs = query_terms()

    path = pathlib.Path(__file__)
    corpus_path = path.parents[3] / "infomedia-embedding" / "dat" / "corpus" / "Berlingske_Nyhedsbureau_BNB.jsonl"

    df = pd.read_json(str(corpus_path), lines=True)

    # query df all terms in term_pairs - return ids for those that match
    filtered_df = query_df(df, term_pairs)

    print(filtered_df["content"][0])
    print(filtered_df["content"][1])
    

if __name__ == "__main__": 
    main()

