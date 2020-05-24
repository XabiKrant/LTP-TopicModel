"""test.py

Heavily relying on this tutorial:
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
"""

from lxml import etree
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
import time

from sklearn.decomposition import LatentDirichletAllocation

def parse_data(xmlfile, max_id=10000):
    """This function reads the data from the specified xml file
    and outputs a DataFrame with the data.
    
    args:
        xmlfile -- A string containing the path to the xml-file
        max_id  -- The limit for when we stop parsing.
                   If max_id == None, we parse through the entire dataset.

    returns:
        df -- A pandas DataFrame object. The columns are
              language, name, categories and content
    """
    start_time = time.time()

    start_tag = None
    data_list = []
    for event, element in etree.iterparse(xmlfile, events=('start', 'end'), recover=True):
        if event == 'start' and start_tag is None:
            start_tag = element.tag
        if event == 'end' and element.tag == "articlePair" and (element.attrib.get('id') != None):
            if max_id is not None and int(element.attrib.get('id')) > max_id:
                break
            categories = []
            for article in element.findall('article'):
                categories += article.find('categories').get('name').split("|")
            for article in element.findall('article'):
                content = ""
                for content_child in article.find('content'):
                    if(content_child.text != None):
                        content += content_child.text
                a = {"language": article.attrib.get('lang'), "name": article.attrib.get(
                    'name'), "categories": categories, "content": content}
                data_list.append(a)

    df = pd.DataFrame(data_list)

    parse_time = time.time() - start_time
    print(f"Parsing xml file: {xmlfile} took {parse_time} seconds.")
    return df

def vectorize(corpus, n_features, stop_words):
    # If a word occurs in more than 75% of the documents, we do not include it
    vectorizer = TfidfVectorizer(max_df=0.75,
                                 min_df=5,
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

def main():
    n_documents = 10000   # Number of documents (Dutch + English)
    n_features = 100  # Number of words in the document?
    n_topics = 10     # Number of topics we want LDA to use

    xmlfile = 'Data/wikicomp-2014_ennl.xml'
    df = parse_data(xmlfile, max_id=n_documents)
    df_sample = df[:n_documents]
    df_english = df_sample[df_sample["language"] == "en"]
    df_dutch = df_sample[df_sample["language"] == "nl"]

    corpus_english = df_english["content"]
    corpus_dutch = df_dutch["content"]

    # TODO: We might need to add stop words ourselves to these lists!
    stopwords_english = get_stop_words('en')
    stopwords_dutch = get_stop_words('nl')

    X_english, words_english = vectorize(corpus_english, n_features, stopwords_english)
    X_dutch, words_dutch = vectorize(corpus_dutch, n_features, stopwords_dutch)

    # learning_method should be set to 'online' for large datasets
    # random_state set to 0 so we can reproduce results
    lda_english = LatentDirichletAllocation(n_components=n_topics, 
                                            learning_method = 'online',
                                            random_state=0)
    lda_english.fit(X_english)

    lda_dutch = LatentDirichletAllocation(n_components=n_topics, 
                                          learning_method = 'online',
                                          random_state=0)
    lda_dutch.fit(X_dutch)

    # lda_{language}.components_ will have the results now
    # lda_{language}.components_.shape looks like (n_documents, n_features)
    output_top_words(lda_english, words_english)
    output_top_words(lda_dutch, words_dutch)




if __name__ == "__main__":
    # Prevent scoping issues
    main()