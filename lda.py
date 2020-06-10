"""lda.py

Heavily relying on this tutorial:
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import time
import math
import dataset_util

def vectorize(corpus, n_features, stop_words):
    # If a word occurs in more than 95% of the documents, we do not include it
    vectorizer = TfidfVectorizer(max_df=0.95,
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

    purity = total_numerator / total_denominator
    return purity

def main():
    program_start_time = time.time()

    n_documents = 10000   # Number of documents (Dutch + English)
    n_features = 500  # Number of words in the document?
    n_topics = 50    # Number of topics we want LDA to use

    xmlfile = 'Data/wikicomp-2014_ennl.xml'

    df_english, df_dutch = dataset_util.process_dataset(xmlfile, n_documents)
    corpus_english = df_english["content"]
    corpus_dutch = df_dutch["content"]

    # Vectorize the corpora using a TfidfVectorizer
    stopwords_english = dataset_util.preprocess_stopwords('en')
    stopwords_dutch = dataset_util.preprocess_stopwords('nl')

    X_english, words_english = vectorize(corpus_english, n_features, stopwords_english)
    X_dutch, words_dutch = vectorize(corpus_dutch, n_features, stopwords_dutch)

    # Split in train and test
    train_X_english = X_english[:int(0.8*X_english.shape[0])]
    test_X_english = X_english[int(0.8*X_english.shape[0]):]
    categories_test_english = df_english["categories"][int(0.8*X_english.shape[0]):]

    train_X_dutch = X_dutch[:int(0.8*X_dutch.shape[0])]
    test_X_dutch = X_dutch[int(0.8*X_dutch.shape[0]):]
    categories_test_dutch = df_dutch["categories"][int(0.8*X_dutch.shape[0]):]

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
    output_top_words(lda_english, words_english)
    output_top_words(lda_dutch, words_dutch)

    # features_english.shape looks like (n_documents_in_test_set, n_topics)
    # So for each document in the test set, we can say what their distribution over the topics is
    features_english = lda_english.transform(test_X_english)
    features_dutch = lda_dutch.transform(test_X_dutch)

    # Concatenate features and categories
    features_both = np.concatenate((features_english, features_dutch))
    categories_both = pd.concat((categories_test_english, categories_test_dutch))

    kmeans = generate_clusters(features_both, n_topics)
    purity = compute_purity(kmeans, n_topics, categories_both)

    print(f"The purity of the made clusters is {purity}")

    program_duration = time.time() - program_start_time
    print(f"It took {program_duration} seconds to run the entire program.")


if __name__ == "__main__":
    # Prevent global scoping issues
    main()