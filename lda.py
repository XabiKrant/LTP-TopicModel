"""lda.py
Heavily relying on this tutorial:
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import time
import math
import dataset_util
import io
import fasttext
import fasttext.util
import argparse
from translate import Translator

def multi_lingual_vectorize(corpus, bin_name, vocab):
    ft = fasttext.load_model(bin_name)
    """
    Vectorize according to the downloadable embeddings from fasttext
    """
    if os.path.exists(bin_name):
        ft = fasttext.load_model(bin_name)
        print("Loaded embeddings: " + bin_name)
    else:
        lang = bin_name.split(".")[1]
        print("Downloading embeddings: " + bin_name)
        filename = fasttext.util.download_model(lang, if_exists='ignore')
        ft = fasttext.load_model(filename)
        fasttext.util.reduce_model(ft, 50)
        ft.save_model(bin_name)
        print("Saved embeddings: " + bin_name)

    indptr = [0]
    data = []
    for d in corpus:
        sentence = []
        for term in word_tokenize(d):
            if (ft.get_word_vector(term).all() != None) and (term in vocab):
                sentence.append(ft.get_word_vector(term))
        if (len(sentence)>0):     
            data.append(np.mean(sentence, axis=0))
        else:
            data.append([0]*50)
    return np.array(data)


# Implementation to do word by word translation
# def translate_sentences(corpus):
#     data = []
#     translator= Translator(to_lang="English")
#     for d in corpus:
#         sentence = []
#         for term in word_tokenize(d):
#             sentence.append(translator.translate(term))
#         data.append(sentence)
#         print(len(data))
#     return np.array(data)


def translate_sentences(corpus):
    data = []
    translator= Translator(to_lang="English")
    for d in corpus:
        data.append(translator.translate(d))
    return np.array(data)


def vectorize(corpus, n_features, stop_words):
    """Vectorizes a corpus of documents with the TfidfVectorizer.
    Words should not occur in more than 75% of documents, and 
    occur in at least 10 documents.
    
    args:
    corpus -- A sequence of documents
    n_features -- The numer of features per document
    stop_words -- The stop words we remove first before vectorizing

    returns:
    X -- A sequence of document vectors
    words -- A sequence of the words from the documents
    """

    # If a word occurs in more than 75% of the documents, we do not include it
    vectorizer = TfidfVectorizer(max_df=0.75,
                                 min_df=10,
                                 max_features=n_features,
                                 stop_words=stop_words,
                                 encoding="utf-8")
    X = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    return X, words

def output_top_words(lda_model, words, n=25):
    """Outputs the top n words that define a latent topic.
    n = 25 by default

    args:
    lda_model -- The LDA object which transformed the documents
    words -- The sequence of words
    n -- The number of words that we want to output per topic
    """
    for topic_index, topic in enumerate(lda_model.components_):
        output_string = f"Topic ({topic_index}): "
        for word_index in topic.argsort()[:-n-1:-1]:
            # argsort get the indices
            # We can use these indices to find the corresponding word
            # in the given words object
            word = words[word_index]
            output_string += word + " "
        print(output_string)

    print("")

def generate_clusters(topic_features, n_clusters):
    """Generates K-Means clusters from the topic features.

    args:
    topic_features -- The features of the topics to be clustered.
                      The features can be made using any Machine Learning method.
    n_cluster -- The number of clusters (k)

    returns:
    kmeans -- A KMeans object which stores the cluster centroids 
              and labels of the data points
    """
    cluster_start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1).fit(topic_features)
    cluster_duration = time.time() - cluster_start_time
    print(f"Clustering took {cluster_duration:.3f} seconds")
    return kmeans

def compute_purity(kmeans, n_clusters, categories):
    """Purity calculation as defined in Introduction to Information Retrieval."""
    counts_per_cluster = [dict() for i in range(n_clusters)]
    categories = categories.reset_index(drop=True)

    for data_index, label in enumerate(kmeans.labels_):
        for category in categories[data_index]:
            try:
                counts_per_cluster[label][category] += 1
            except KeyError:
                counts_per_cluster[label][category] = 1

    dict_sum = 0 
    for label_dict in counts_per_cluster:
        dict_sum += max(label_dict.values())

    purity = dict_sum/len(categories)
    return purity

def compute_purity_star(kmeans, n_clusters, categories):
    """An attempt at a multilabel purity computation."""
    counts_per_cluster = [dict() for i in range(n_clusters)]
    categories = categories.reset_index(drop=True)

    for data_index, label in enumerate(kmeans.labels_):
        for category in categories[data_index]:
            try:
                counts_per_cluster[label][category] += 1
            except KeyError:
                counts_per_cluster[label][category] = 1

    sum_of_cats = 0
    for cat_list in categories:
        sum_of_cats += len(cat_list)

    # Average number of categories per data point
    average_n_cats = math.ceil(sum_of_cats / len(categories))

    # Calculate the actual purity*
    total_numerator = 0
    total_denominator = 0
    for counts in counts_per_cluster:
        sorted_values = sorted(counts.values(), reverse=True)
        total_numerator += sum(sorted_values[:average_n_cats])
        total_denominator += sum(sorted_values)

    purity_star = total_numerator / total_denominator
    return purity_star


def compute_max_purity(n_clusters, categories):
    """"""
    count_dicts = {}
    categories = categories.reset_index(drop=True)

    for category_list in categories:
        for category in category_list:
            try:
                count_dicts[category] += 1
            except KeyError:
                count_dicts[category] = 1

    values = list(count_dicts.values())
    values.sort(reverse=True)
    return sum(values[:n_clusters])/len(categories)


def main():
    program_start_time = time.time()
    parser = argparse.ArgumentParser("A neural network which makes use of document embeddings as input.")
    parser.add_argument("-f", "--file", default='Data/wikicomp-2014_ennl.xml', help="The dataset file (xml)")
    parser.add_argument("--n_documents", type=int, default=1000000, help="The maximal number of documents we want to use for training.")
    parser.add_argument("--n_features", type=int, default=1000, help="The number of features TfidfVectorizer will use")
    parser.add_argument("--top_n_cats", type=int, default=300, help="Number of categories used in validation")
    parser.add_argument("--n_topics", type=int, default=50, help="The number of clusters the KMeans algorithm will make.")
    parser.add_argument("--embeddings", action="store_true", help="Whether to download word embeddings from the internet. This WILL take a long time.")
    args = parser.parse_args()

    if args.embeddings:
        print("It is possible to run the program without embeddings if you think the download takes too long."
              " Please rerun the program without the --embeddings option.")
    else:
        print("It is possible to run the program with embeddings, but the download WILL take a long time."
              " If you still want to try this, please rerun the program with the --embeddings option.")

    df_english, df_dutch = dataset_util.process_dataset(args.file, args.n_documents, args.top_n_cats)
    corpus_english = df_english["content"]
    corpus_dutch = df_dutch["content"]
    corpus_dutch_translated = translate_sentences(corpus_dutch)

    assert corpus_dutch_translated.shape == corpus_dutch.shape
    df_all = pd.concat((df_english, df_dutch))
    corpus_all = df_all["content"]

    stopwords = dataset_util.construct_stopwords()

    X_all, words_all = vectorize(corpus_all, args.n_features, stopwords)
    # Vectorize the corpora using a TfidfVectorizer
    X_english, words_english = vectorize(corpus_english, args.n_features, stopwords)
    X_dutch, words_dutch = vectorize(corpus_dutch, args.n_features, stopwords)
    X_dutch_translated, X_words_dutch_translated = vectorize(corpus_dutch_translated, args.n_features, stopwords)

    # Split in train and test
    train_X_english = X_english[:int(0.8*X_english.shape[0])]
    test_X_english = X_english[int(0.8*X_english.shape[0]):]
    categories_test_english = df_english["categories"][int(0.8*X_english.shape[0]):]

    train_X_dutch = X_dutch[:int(0.8*X_dutch.shape[0])]
    test_X_dutch = X_dutch[int(0.8*X_dutch.shape[0]):]
    categories_test_dutch = df_dutch["categories"][int(0.8*X_dutch.shape[0]):]

    train_X_dutch_translated = X_dutch_translated[:int(0.8*X_dutch_translated.shape[0])]
    test_X_dutch_translated = X_dutch_translated[int(0.8*X_dutch_translated.shape[0]):]
    categories_test_dutch_translated = df_dutch["categories"][int(0.8*X_dutch_translated.shape[0]):]

    train_X_all = X_all[:int(0.8*X_all.shape[0])]
    test_X_all = X_all[int(0.8*X_all.shape[0]):]
    categories_test_all = df_all["categories"][int(0.8*X_all.shape[0]):]

    if args.embeddings:
        X_english_emb = multi_lingual_vectorize(corpus_english, 'Data/cc.en.50.bin', words_english)
        X_dutch_emb = multi_lingual_vectorize(corpus_dutch, 'Data/cc.nl.50.bin', words_dutch)

        train_X_english_emb = X_english_emb[:int(0.8*X_english_emb.shape[0])]
        test_X_english_emb = X_english_emb[int(0.8*X_english_emb.shape[0]):]
        categories_test_english_emb = df_english["categories"][int(0.8*X_english_emb.shape[0]):]

        train_X_dutch_emb = X_dutch_emb[:int(0.8*X_dutch_emb.shape[0])]
        test_X_dutch_emb = X_dutch_emb[int(0.8*X_dutch_emb.shape[0]):]
        categories_test_dutch_emb = df_dutch["categories"][int(0.8*X_dutch_emb.shape[0]):]
        embedded_both = np.concatenate((test_X_english_emb, test_X_dutch_emb))
        categories_embedded_both = pd.concat((categories_test_english_emb, categories_test_dutch_emb))


    # learning_method should be set to 'online' for large datasets
    # random_state set to 0 so we can reproduce results
    lda_english = LatentDirichletAllocation(n_components=args.n_topics, 
                                            learning_method = 'online',
                                            random_state=0)

    lda_english.fit(train_X_english)

    lda_dutch = LatentDirichletAllocation(n_components=args.n_topics, 
                                          learning_method = 'online',
                                          random_state=0)
    lda_dutch.fit(train_X_dutch)

    lda_all = LatentDirichletAllocation(n_components=args.n_topics, 
                                            learning_method = 'online',
                                            random_state=0)

    lda_all.fit(train_X_all)

    lda_translated = LatentDirichletAllocation(n_components=args.n_topics, 
                                            learning_method = 'online',
                                            random_state=0)

    lda_translated.fit(train_X_dutch_translated)

    
    output_top_words(lda_english, words_english)
    output_top_words(lda_dutch, words_dutch)

    # features_english.shape looks like (args.n_documents_in_test_set, args.n_topics)
    # So for each document in the test set, we can say what their distribution over the topics is
    features_english = lda_english.transform(test_X_english)
    features_dutch = lda_dutch.transform(test_X_dutch)
    features_dutch_translated = lda_translated.transform(test_X_dutch_translated)
    features_all = lda_all.transform(test_X_all)

    # Concatenate features and categories
    features_both = np.concatenate((features_english, features_dutch))
    categories_both = pd.concat((categories_test_english, categories_test_dutch))

    # Concatenate features and categories
    features_both_translated = np.concatenate((features_english, features_dutch_translated))
    categories_both_translated = pd.concat((categories_test_english, categories_test_dutch_translated))

    kmeans = generate_clusters(features_both, args.n_topics)
    purity = compute_purity(kmeans, args.n_topics, categories_both)
    purity_star = compute_purity_star(kmeans, args.n_topics, categories_both)
    max_purity = compute_max_purity(args.n_topics, categories_both)

    print("Results for split vectors")
    print(f"The purity of the made clusters is {purity:.3f}, the maximum achievable is {max_purity:.3f}")
    print(f"The purity* of the made clusters is {purity_star:.3f}\n")

    kmeans = generate_clusters(features_all, args.n_topics)
    purity = compute_purity(kmeans, args.n_topics, categories_test_all)
    purity_star = compute_purity_star(kmeans, args.n_topics, categories_test_all)
    max_purity = compute_max_purity(args.n_topics, categories_test_all)

    print("Results for combined vectors")
    print(f"The purity of the made clusters is {purity:.3f}, the maximum achievable is {max_purity:.3f}")
    print(f"The purity* of the made clusters is {purity_star:.3f}\n")

    kmeans = generate_clusters(features_both_translated, args.n_topics)
    purity = compute_purity(kmeans, args.n_topics, categories_both_translated)
    purity_star = compute_purity_star(kmeans, args.n_topics, categories_both_translated)
    max_purity = compute_max_purity(args.n_topics, categories_both_translated)

    print("Results for translated words")
    print(f"The purity of the made clusters is {purity:.3f}, the maximum achievable is {max_purity:.3f}")
    print(f"The purity* of the made clusters is {purity_star:.3f}\n")

    if args.embeddings:
        kmeans = generate_clusters(embedded_both, args.n_topics)
        purity = compute_purity(kmeans, args.n_topics, categories_embedded_both)
        purity_star = compute_purity_star(kmeans, args.n_topics, categories_embedded_both)
        max_purity = compute_max_purity(args.n_topics, categories_embedded_both)


        print("Results for embedded vectors")
        print(f"The purity of the made clusters is {purity:.3f}, the maximum achievable is {max_purity:.3f}")
        print(f"The purity* of the made clusters is {purity_star:.3f}\n")

    program_duration = time.time() - program_start_time
    print(f"It took {program_duration:.3f} seconds to run the entire program.")


if __name__ == "__main__":
    # Prevent global scoping issues
    main()
