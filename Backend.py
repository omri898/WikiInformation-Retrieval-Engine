# Regex
import re

# NLTK
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

from collections import Counter
from inverted_index_gcp import *
import math

# Download text index
# connect to google storage text bucket
text_bucket_name = '4413proj1texttf'
index_src = "text_index.pkl"
index_path = f"text_posting_gcp/{index_src}"
# postings_path = ""
bucket = get_bucket(text_bucket_name)
blob = bucket.blob(index_path)
contents = blob.download_as_bytes()
text_inverted = pickle.loads(contents)

# Download title index
# connect to google storage text bucket
title_bucket_name = '4413proj1titletf'
index_src = "title_index.pkl"
index_path = f"title_posting_gcp/{index_src}"
# postings_path = ""
bucket = get_bucket(title_bucket_name)
blob = bucket.blob(index_path)
contents = blob.download_as_bytes()
title_inverted = pickle.loads(contents)

# Download doc_id - title dictionary
file_name = "docid_title_dict.pkl"
bucket = storage.Client().get_bucket(title_bucket_name)
blob = bucket.get_blob(file_name)
if blob:
    with blob.open("rb") as pkl_file:
        docid_title_dict = pickle.load(pkl_file)

# Download doc_id - text length
file_name = "docid_len_dict.pkl"
bucket = storage.Client().get_bucket(text_bucket_name)
blob = bucket.get_blob(file_name)
if blob:
    with blob.open("rb") as pkl_file:
        docid_len_dict = pickle.load(pkl_file)

N = 6348910


def read_pkl_file_form_bucket(file_name, name_bucket):
    """
        func that read pkl file from the bucket
    Args:
        name_bucket: name of the bucket
        file_name: the name of the pkl file + dir : pagerank\page_rank

    Returns:
            dict
    """
    # access to the bucket
    bucket = storage.Client().get_bucket(name_bucket)
    blob = bucket.get_blob(f'{file_name}.pkl')
    if blob:
        with blob.open("rb") as pkl_file:
            return pickle.load(pkl_file)


def tokenize(query_text, title_query):
    """
    Tokenizes the query text, removes stopwords.

    Args:
        query_text: The text of the query to tokenize.
        title: Boolean indicating if the text is a title.

    Returns:
        List of tokenized and cleaned words from the query.
    """
    # Tokenizer
    # General stop words
    english_stopwords = frozenset(stopwords.words('english'))
    # Corpus specific stop words
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    # combine the two
    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    # Remove stop words from tokens
    filtered_tokens = [token.group() for token in RE_WORD.finditer(query_text.lower())]

    if title_query == False:
        filtered_tokens = [token for token in filtered_tokens if token not in all_stopwords]

    elif title_query == True:
        filtered_tokens = [token for token in filtered_tokens if token not in english_stopwords]

    return filtered_tokens


def create_query_tf_idf_scores(query_tokens):
    """
    Calculates the TF-IDF scores for each token in the query. This function computes the term frequency (TF)
    and inverse document frequency (IDF) for each query token and then calculates the TF-IDF score.

    Args:
        query_tokens: A list of tokenized words from the query.

    Returns:
        A dictionary where each key is a query token and each value is the token's TF-IDF score.
    """
    # Count the frequency of each remaining token
    token_counts = Counter(query_tokens)
    query_len = len(query_tokens)
    query_tf_idf_dict = {}
    for query_token in token_counts.keys():
        tf = token_counts[query_token] / query_len # Normalized tf by query len
        idf = math.log10(N / text_inverted.df[query_token])
        tf_idf_query = tf * idf
        query_tf_idf_dict[query_token] = tf_idf_query

    return query_tf_idf_dict


def title_query_similarity(query_tokens):
    """
    Calculates and returns similarity scores between the query tokens and the document titles.

    Args:
        query_tokens: List of tokenized words from the query.

    Returns:
        List of top 100 tuples (document ID, similarity score) for documents sorted by their title similarity.
    """
    # This dictionary will collect for each document id, how many of the query terms appeared in their titles
    doc_counter_dict = {}
    for token in query_tokens:
        if token in title_inverted.df:
            for doc_id, score in title_inverted.read_a_posting_list(base_dir="", w=token,
                                                                    bucket_name=title_bucket_name):
                if doc_id in doc_counter_dict:
                    doc_counter_dict[doc_id] += 1
                else:
                    doc_counter_dict[doc_id] = 1
                    
    for doc_id in doc_counter_dict.keys():
        not_in_title = len(tokenize(docid_title_dict[doc_id], True)) - doc_counter_dict[doc_id]
        doc_counter_dict[doc_id] = doc_counter_dict[doc_id]/(not_in_title + 1)
            

    # Convert dictionary items to a list of tuples and sort by values in descending order
    sorted_tuples = sorted(doc_counter_dict.items(), key=lambda x: x[1], reverse=True)
    # Get the top 100 items
    top_100 = sorted_tuples[:100]
    return top_100


def text_cosine_similarity(query_tokens):
    """
    Calculates the cosine similarity scores between the query tokens and the text of documents.

    Args:
        query_tokens: List of tokenized words from the query.

    Returns:
        List of top 100 documents sorted by their cosine similarity score, based on the text content.
    """
    sim_doc_dictionary = {}  # Will collect the similarity vector per document with query

    # Create tf dictionary for query terms + calc number of terms in query
    query_term_tf_dict = Counter(query_tokens)
    query_length = len(query_tokens)

    # calculate query tf_idf score vector to calculate the norm
    query_tf_idf_dict = create_query_tf_idf_scores(query_tokens)
    squerd_idf_query_sum = 0
    for term in query_tf_idf_dict.keys():
        squerd_idf_query_sum += (query_tf_idf_dict[term]) ** 2
    norm_factor_query = math.sqrt(squerd_idf_query_sum)

    # Calculate the dot product of tf_idf vectors of both query and document for each document
    squerd_idf_doc_dict = {}
    for query_term in query_tokens:
        posting_list = text_inverted.read_a_posting_list(base_dir="", w=query_term, bucket_name=text_bucket_name)
        for doc_id, tf in posting_list:
            try:
                tf = tf / docid_len_dict[doc_id] 
                tf_q = query_term_tf_dict[query_term] / query_length
                idf = math.log10(N / text_inverted.df[query_term])
                tf_idf_doc = tf * idf
                tf_idf_que = tf_q * idf
                sim_doc_dictionary[doc_id] += tf_idf_que * tf_idf_doc # Dot Product calculation (query*document)

                if doc_id in squerd_idf_doc_dict:
                    squerd_idf_doc_dict[doc_id] += tf_idf_doc ** 2
                else:
                    squerd_idf_doc_dict[doc_id] = tf_idf_doc ** 2

            except ZeroDivisionError:
                continue

    documents_norm_dict = {}
    for doc_id in squerd_idf_doc_dict.keys():
        documents_norm_dict[doc_id] = math.sqrt(squerd_idf_doc_dict[doc_id])

    for doc_id in sim_doc_dictionary.keys():
        if (doc_id in documents_norm_dict):
            normalize_num = documents_norm_dict[doc_id]
            try:
                sim_doc_dictionary[doc_id] = sim_doc_dictionary[doc_id] / (norm_factor_query*normalize_num)
            except ZeroDivisionError:
                continue

    text_top_100 = sorted(sim_doc_dictionary.items(), key=lambda x: x[1], reverse=True)[:100]
    return text_top_100

def text_tf_idf_score(query_tokens):
    doc_text_tf_idf_scores = {}
    for query_term in query_tokens:
        posting_list = text_inverted.read_a_posting_list(base_dir="", w=query_term, bucket_name=text_bucket_name)
        for doc_id, tf in posting_list:
            try:
                tf = tf / docid_len_dict[doc_id]
#                 idf = math.log10(N / text_inverted.df[query_term])
#                 tf_idf_doc = tf * idf
                if doc_id in doc_text_tf_idf_scores:
#                     doc_text_tf_idf_scores[doc_id] += tf_idf_doc
                    doc_text_tf_idf_scores[doc_id] += 1
                else:
#                     doc_text_tf_idf_scores[doc_id] = tf_idf_doc
                    doc_text_tf_idf_scores[doc_id] = 1

            except ZeroDivisionError:
                print("A")
                continue

    text_top_100 = sorted(doc_text_tf_idf_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    return text_top_100

def merge_results(title_scores, text_scores, title_weight=0.5, text_weight=0.5):
    """
    Combines the scores from title and text similarities into a single score for each document.

    Args:
        title_scores: List of tuples (document ID, title similarity score).
        text_scores: List of tuples (document ID, text similarity score).
        title_weight: The weight to assign to title scores in the final combination.
        text_weight: The weight to assign to text scores in the final combination.

    Returns:
        List of the top 100 documents sorted by their combined score.
    """
    scores = {}
    for doc_id, score in title_scores:
        scores[doc_id] = title_weight * score
    for doc_id, score in text_scores:
        if doc_id in scores:
            scores[doc_id] += (text_weight * score)
        else:
            scores[doc_id] = text_weight * score
    top_100_docs = sorted([(doc_id, score) for doc_id, score in scores.items()], key=lambda x: x[1], reverse=True)[:100]
    return top_100_docs


def rate_docs_based_on_query(query_text):
    """
    Rates documents based on their relevance to the query text.

    Args:
        query_text: The text of the user's query.

    Returns:
        List of tuples (document ID, title), sorted by their relevance to the query text.
    """
    # Tokenizer
    query_tokens = tokenize(query_text, False)


    if len(query_tokens) == 1:
        title_weight = 1
        text_weight = 0
    else:
        title_weight = 1
        text_weight = 0

    # Title Contribution
    title_scores = title_query_similarity(query_tokens)
    highest_query_term_appearance_in_title_value = title_scores[0][1]
    title_scores = [(pair[0], pair[1] / highest_query_term_appearance_in_title_value) for pair in
                    title_scores]  # Normilize scores based on max value
    # Text
#     text_scores = text_cosine_similarity(query_tokens)

#     merge = merge_results(title_scores, text_scores, title_weight, text_weight)
    merge = merge_results(title_scores, title_scores, title_weight, text_weight)
    return [(str(doc_id), docid_title_dict[doc_id]) for doc_id, score in merge]