"""network.py
"""

import string
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import dataset_util

class Network(nn.Module):
    def __init__(self, vector_length, n_topics):
        super(Network, self).__init__()

        size_steps = n_topics - vector_length // 3
        # vector_length denotes the length of the document vectors (input)
        self.fc1 = nn.Linear(vector_length, vector_length + size_steps)
        self.fc2 = nn.Linear(vector_length + size_steps, vector_length + 2*size_steps)
        self.fc3 = nn.Linear(vector_length + 2*size_steps, n_topics)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_corpus(unprocessed_corpus, stopwords):
    corpus = []
    for doc in unprocessed_corpus:
        doc = doc.translate(doc.maketrans('', '', string.punctuation))
        doc = doc.replace("\n", " ")
        word_list = doc.split(" ")
        while '' in word_list:
            word_list.remove('')
        for word in stopwords:
            while word in word_list:
                word_list.remove(word)
        corpus.append(word_list)
    return corpus

def main():
    program_start_time = time.time()

    parser = argparse.ArgumentParser("A neural network which makes use of document embeddings as input.")
    parser.add_argument("-vl", "--vector_length", type=int, default=100, help="The length of the document embedding vectors.")
    parser.add_argument("-nt", "--n_topics", type=int, default=50, help="The number of topics the network should classify into.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("-ne", "--n_epochs", type=int, default=100, help="The number of epochs the network should train if --train is specified.")
    parser.add_argument("-f", "--file", default='Data/wikicomp-2014_ennl.xml', help="The dataset file (xml)")
    parser.add_argument("-nd", "--n_documents", type=int, default=10000, help="The maximal number of documents we want to use for training.")
    args = parser.parse_args()

    df_english, df_dutch = dataset_util.process_dataset(args.file, args.n_documents)
    unprocessed_corpus_english = list(df_english["content"])
    unprocessed_corpus_dutch = list(df_dutch["content"])

    # MIGHTDO: We might need to add stop words ourselves to these lists!
    stopwords_english = dataset_util.preprocess_stopwords('en')
    stopwords_dutch = dataset_util.preprocess_stopwords('nl')

    corpus_english = preprocess_corpus(unprocessed_corpus_english, stopwords_english)
    corpus_dutch = preprocess_corpus(unprocessed_corpus_dutch, stopwords_dutch)

    documents_english = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_english)]
    documents_dutch = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_dutch)]

    model_english = Doc2Vec(documents_english, vector_size=args.vector_length, window=2, min_count=1, workers=4, epochs=1)
    model_dutch  = Doc2Vec(documents_dutch, vector_size=args.vector_length, window=2, min_count=1, workers=4, epochs=1)

    # For now, we can just use a normal feed forward Dense Network
    if args.train:
        # If the user specified train, we instantiate a new network
        network = Network(args.vector_length, args.n_topics)
        for epoch in range(args.n_epochs):
            for d_index, data in enumerate(documents_english):
                print(data)
                docvec = model_english.infer_vector(data)
                # After having computed the document vector,
                # we can use it as input to a neural network
                topics = network(docvec)
                print(topics)

                # TODO: Backprop (need targets?)
                exit()


    else:
        # Load a doc2vec model and network from disk
        pass

    # The topic variable will hold a distribution over topics.


    vector_english = model_english.infer_vector(["system", "response"])
    vector_dutch = model_dutch.infer_vector(["systeem", "respons"])
    print(vector_english)
    print(vector_dutch)

    program_duration = time.time() - program_start_time
    print(f"It took {program_duration} seconds to run the entire program.")




if __name__ == "__main__":
    main()