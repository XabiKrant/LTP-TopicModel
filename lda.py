"""lda.py

Heavily relying on this tutorial:
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
"""

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


def multi_lingual_vectorize(corpus, bin_name, vocab):
    ft = fasttext.load_model(bin_name)
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


def vectorize(corpus, n_features, stop_words):
    # If a word occurs in more than 75% of the documents, we do not include it
    vectorizer = TfidfVectorizer(max_df=0.75,
                                 min_df=20,
                                 max_features=n_features,
                                 stop_words=stop_words,
                                 encoding="utf-8")
    X = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    return X, words


def output_top_words(lda_model, words, n=25):
    # There are n_topics entries in lda_model.components_
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

    cluster_start_time = time.time()
    # TODO: We might need a different distance metric (Jensen-Shannon divergence?)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1).fit(topic_features)
    cluster_duration = time.time() - cluster_start_time
    print(f"Clustering took {cluster_duration} seconds")
    return kmeans

def compute_purity(kmeans, n_clusters, categories):
    """
    I am not sure how to compute purity in a multilabel problem.
    Right now, I implement it like this:
    For each cluster
        For each data point
            Add all the categories of each data point to the dict
        Calculate average number of categories per data point

        Sum the (average number of categories) highest occuring counts
        That is the numerator for this cluster

    Then when we have all numerators, we can divide it by the total number
    of categories to get the purity
    """
    count_dicts = [dict() for i in range(n_clusters)]
    categories = categories.reset_index(drop=True)

    for data_index, label in enumerate(kmeans.labels_):
        for category in categories[data_index]:
            try:
                count_dicts[label][category] += 1
            except KeyError:
                count_dicts[label][category] = 1

    # print(count_dicts)
    dict_sum = 0 
    for label_dict in count_dicts:
        dict_sum += label_dict[max(label_dict)]

    purity = dict_sum/len(categories)
    print(len(categories))


    sum_of_cats = 0
    for cat_list in categories:
        sum_of_cats += len(cat_list)

   
    # Average number of categories per data point
    average_n_cats = math.ceil(sum_of_cats / len(categories))

    # Calculate the actual purity
    total_numerator = 0
    total_denominator = 0
    for counts in count_dicts:
        sorted_values = sorted(counts.values(), reverse=True)
        total_numerator += sum(sorted_values[:average_n_cats])
        total_denominator += sum(sorted_values)

    averaged_purity = total_numerator / total_denominator
    return purity,averaged_purity


def compute_max_purity(n_clusters, categories):
    """
    I am not sure how to compute purity in a multilabel problem.
    Right now, I implement it like this:
    For each cluster
        For each data point
            Add all the categories of each data point to the dict
        Calculate average number of categories per data point

        Sum the (average number of categories) highest occuring counts
        That is the numerator for this cluster

    Then when we have all numerators, we can divide it by the total number
    of categories to get the purity
    """
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

    n_documents = 100000   # Number of documents (Dutch + English)
    n_features = 500  # Number of words in the document?
    n_topics = 50    # Number of topics we want LDA to use

    xmlfile = 'Data/wikicomp-2014_ennl.xml'

    df_english, df_dutch = dataset_util.process_dataset(xmlfile, n_documents)
    df_all = pd.concat((df_english, df_dutch))

    corpus_english = df_english["content"]
    corpus_dutch = df_dutch["content"]
    corpus_all = df_all["content"]

    # Vectorize the corpora using a TfidfVectorizer
    # MIGHTDO: We might need to add stop words ourselves to these lists!
    stopwords_english = dataset_util.preprocess_stopwords('en')
    stopwords_dutch = dataset_util.preprocess_stopwords('nl')

    X_all, words_all = vectorize(corpus_all, n_features, stopwords_english)

    X_english, words_english = vectorize(corpus_english, n_features, stopwords_english)
    X_dutch, words_dutch = vectorize(corpus_dutch, n_features, stopwords_dutch)
    X_english_emb = multi_lingual_vectorize(corpus_english, 'Data/cc.en.50.bin', words_english)
    X_dutch_emb = multi_lingual_vectorize(corpus_dutch, 'Data/cc.nl.50.bin', words_dutch)

    # Split in train and test
    train_X_english = X_english[:int(0.8*X_english.shape[0])]
    test_X_english = X_english[int(0.8*X_english.shape[0]):]
    categories_test_english = df_english["categories"][int(0.8*X_english.shape[0]):]

    train_X_dutch = X_dutch[:int(0.8*X_dutch.shape[0])]
    test_X_dutch = X_dutch[int(0.8*X_dutch.shape[0]):]
    categories_test_dutch = df_dutch["categories"][int(0.8*X_dutch.shape[0]):]

    train_X_all = X_all[:int(0.8*X_all.shape[0])]
    test_X_all = X_all[int(0.8*X_all.shape[0]):]
    categories_test_all = df_all["categories"][int(0.8*X_all.shape[0]):]

    train_X_english_emb = X_english_emb[:int(0.8*X_english_emb.shape[0])]
    test_X_english_emb = X_english_emb[int(0.8*X_english_emb.shape[0]):]
    categories_test_english_emb = df_english["categories"][int(0.8*X_english_emb.shape[0]):]

    train_X_dutch_emb = X_dutch_emb[:int(0.8*X_dutch_emb.shape[0])]
    test_X_dutch_emb = X_dutch_emb[int(0.8*X_dutch_emb.shape[0]):]
    categories_test_dutch_emb = df_dutch["categories"][int(0.8*X_dutch_emb.shape[0]):]
    embedded_both = np.concatenate((test_X_english_emb, test_X_dutch_emb))
    categories_embedded_both = pd.concat((categories_test_english_emb, categories_test_dutch_emb))


    # train_X_english_multi = multi_lingual_vectorize(corpus_english, "Data/cc.en.300.vec")
    # learning_method should be set to 'online' for large datasets
    # random_state set to 0 so we can reproduce results
    lda_english = LatentDirichletAllocation(n_components=n_topics, 
                                            learning_method = 'online',
                                            random_state=0)

    lda_english.fit(train_X_english)

    lda_dutch = LatentDirichletAllocation(n_components=n_topics, 
                                          learning_method = 'online',
                                          random_state=0)
    lda_dutch.fit(train_X_dutch)

    # lda_{language}.components_ will have the results now
    # lda_{language}.components_.shape looks like (n_documents, n_features)
    # output_top_words(lda_english, words_english)
    # output_top_words(lda_dutch, words_dutch)

    lda_all = LatentDirichletAllocation(n_components=n_topics, 
                                            learning_method = 'online',
                                            random_state=0)

    lda_all.fit(train_X_all)

    # features_english.shape looks like (n_documents_in_test_set, n_topics)
    # So for each document in the test set, we can say what their distribution over the topics is
    features_english = lda_english.transform(test_X_english)
    features_dutch = lda_dutch.transform(test_X_dutch)
    features_all = lda_all.transform(test_X_all)

    # Concatenate features and categories
    features_both = np.concatenate((features_english, features_dutch))
    categories_both = pd.concat((categories_test_english, categories_test_dutch))

    kmeans = generate_clusters(features_both, n_topics)
    purity, averaged_purity = compute_purity(kmeans, n_topics, categories_both)
    max_purity = compute_max_purity(n_topics, categories_both)

    print("Results for splitted vectors")
    print(f"The purity of the made clusters is {purity}, the maximum achievable is {max_purity}")
    print(f"The averaged purity of the made clusters is {averaged_purity}\n")

    kmeans = generate_clusters(features_all, n_topics)
    purity, averaged_purity = compute_purity(kmeans, n_topics, categories_test_all)
    max_purity = compute_max_purity(n_topics, categories_both)

    print("Results for combined vectors")
    print(f"The purity of the made clusters is {purity}, the maximum achievable is {max_purity}")
    print(f"The averaged purity of the made clusters is {averaged_purity}\n")

    print(embedded_both.shape)
    print(categories_embedded_both.shape)
    kmeans = generate_clusters(embedded_both, n_topics)
    purity, averaged_purity = compute_purity(kmeans, n_topics, categories_embedded_both)
    max_purity = compute_max_purity(n_topics, categories_embedded_both)

    print("Results for embedded vectors")
    print(f"The purity of the made clusters is {purity}, the maximum achievable is {max_purity}")
    print(f"The averaged purity of the made clusters is {averaged_purity}\n")

    program_duration = time.time() - program_start_time
    print(f"It took {program_duration} seconds to run the entire program.")


if __name__ == "__main__":
    # Prevent global scoping issues
    main()