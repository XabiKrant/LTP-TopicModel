"""network.py
"""

import os
import math
import string
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import dataset_util
from lda import generate_clusters, compute_purity, compute_purity_star
import pandas as pd

class Network(nn.Module):
    def __init__(self, vector_length, n_features):
        super(Network, self).__init__()

        # size_steps = n_features - vector_length // 3
        size_steps = 100
        # vector_length denotes the length of the document vectors (input)
        self.fc1 = nn.Linear(vector_length, vector_length + size_steps)
        self.fc2 = nn.Linear(vector_length + size_steps, vector_length + 2*size_steps)
        self.fc3 = nn.Linear(vector_length + 2*size_steps, n_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def split_train_test(df, stopwords):
    unprocessed_corpus = list(df["content"])
    corpus = preprocess_corpus(unprocessed_corpus, stopwords)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    category_2idx = {}
    categories = [[get_index(c,category_2idx) for c in y] for y in df["categories"]]

    nhots = convert_to_nhot(categories, len(category_2idx))
    train_documents = documents[:int(0.8*len(documents))]
    test_documents = documents[int(0.8*len(documents)):]

    return train_documents, test_documents, nhots

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

def get_index(category, category2idx, freeze=False):
    """
    map categories to indices
    keep special OOV token (_UNK) at position 0
    """
    if category in category2idx:
        return category2idx[category]
    else:
        if not freeze:
            category2idx[category]=len(category2idx) #new index
            return category2idx[category]

def convert_to_nhot(categories, num_categories):
    out = []
    for instance in categories:
        n_hot = np.zeros(num_categories)
        for c_idx in instance:
            n_hot[c_idx] = 1
        out.append(n_hot)
    return np.array(out)


def nhot_cross_entropy_loss(output, target, device):
    """There is no categorical cross entropy loss built in PyTorch.
    So we define our own loss here.
    It is almost the same as the standard categorical cross entropy.
    The only difference is that the target can be multilabel.
    """
    if np.sum(target) != 0:
        # Normalize the target tensor first if not already done
        # We have to do this because otherwise, documents with
        # a lot of categories will impact the loss more than documents 
        # with a few categories
        target /= np.sum(target)
    else:
        raise Exception("Sum of target was 0")

    target = torch.Tensor(target).to(device)
    output = F.log_softmax(output, dim=0)
    loss = -1 * torch.sum(output * target)
    return loss

def test(
        cats_english, cats_dutch,
        network, model_english, model_dutch,
        test_documents_eng, test_documents_dut, device, args):
    features_english = []
    features_dutch = []
    for d_index, data in enumerate(test_documents_eng):
        docvec = model_english.infer_vector(data[0])
        doctensor = torch.Tensor(docvec).to(device)
        output_eng = network(doctensor).detach().cpu().numpy()         

        docvec = model_dutch.infer_vector(test_documents_dut[d_index][0])
        doctensor = torch.Tensor(docvec).to(device)
        output_dutch = network(doctensor).detach().cpu().numpy()

        features_english.append(output_eng)
        features_dutch.append(output_dutch)

    features_english = np.array(features_english)
    features_dutch = np.array(features_dutch)
    features_both = np.concatenate((features_english, features_dutch))
    categories_both = pd.concat((cats_english, cats_dutch))

    kmeans = generate_clusters(features_both, args.n_topics)
    purity = compute_purity(kmeans, args.n_topics, categories_both)
    purity_star = compute_purity_star(kmeans, args.n_topics, categories_both)
    print(f"The purity of the made clusters is {purity:.3f}\n"
          f"The purity* of the made clusters is {purity_star:.3f}")

def main():
    program_start_time = time.time()

    parser = argparse.ArgumentParser("A neural network which makes use of document embeddings as input.")
    parser.add_argument("--vector_length", type=int, default=300, help="The length of the document embedding vectors.")
    parser.add_argument("--n_topics", type=int, default=50, help="The number of clusters the KMeans should make.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=10, help="The number of epochs the network should train if --train is specified.")
    parser.add_argument("-f", "--file", default='Data/wikicomp-2014_ennl.xml', help="The dataset file (xml)")
    parser.add_argument("--n_documents", type=int, default=1000000, help="The maximal number of documents we want to use for training.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="The learning rate of the Adam optimizer.")
    parser.add_argument("--n_features", type=int, default=300, help="The number of units in the last fully connected layer of the network.")
    parser.add_argument("--network_name", default="network.pt", help="The filename of the network to be saved or loaded")
    parser.add_argument("--cuda", action="store_true", help="If able to use CUDA and this argument is specified, we use CUDA.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")
    if not os.path.isdir("SavedModels"):
        os.mkdir("SavedModels")


    df_english, df_dutch = dataset_util.process_dataset(args.file, args.n_documents, args.n_features)
    stopwords = dataset_util.construct_stopwords()
    train_documents_eng, test_documents_eng, nhots_eng = split_train_test(df_english, stopwords)
    train_documents_dut, test_documents_dut, nhots_dut = split_train_test(df_dutch, stopwords)
    cats_english = df_english["categories"]
    cats_dutch = df_dutch["categories"]

    if args.train:
        print("Computing document vectors, this can take a while...")
        model_english = Doc2Vec(
            train_documents_eng,
            vector_size=args.vector_length,
            window=2,
            min_count=1,
            workers=4,
            epochs=8,
            seed=0)
        model_dutch  = Doc2Vec(
            train_documents_dut,
            vector_size=args.vector_length,
            window=2,
            min_count=1,
            workers=4,
            epochs=8,
            seed=0)
        model_english.save(f"SavedModels/doc2vec_english")
        model_dutch.save(f"SavedModels/doc2vec_dutch")
        print(f"Saved models: SavedModels/doc2vec_english and SavedModels/doc2vec_dutch")

        # For now, we can just use a normal feed forward Dense Network
        # If the user specified train, we instantiate a new network
        network = Network(args.vector_length, args.n_features).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

        for epoch in range(args.n_epochs):
            losses = []
            for d_index, data in enumerate(train_documents_eng):
                optimizer.zero_grad()
                docvec = model_english.infer_vector(data[0])
                doctensor = torch.Tensor(docvec).to(device)
                raw_topics = network(doctensor)

                # The target is an n-hot vector e.g. [1, 1, 0, 1]
                target = nhots_eng[d_index]
                loss = nhot_cross_entropy_loss(raw_topics, target, device)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())

                optimizer.zero_grad()
                docvec = model_dutch.infer_vector(train_documents_dut[d_index][0])
                doctensor = torch.Tensor(docvec).to(device)
                raw_topics = network(doctensor)

                # The target is an n-hot vector e.g. [1, 1, 0, 1]
                target = nhots_dut[d_index]
                loss = nhot_cross_entropy_loss(raw_topics, target, device)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())

            if epoch == args.n_epochs - 1:
                torch.save(network, f"SavedModels/{args.network_name}")
                print(f"Saved network: SavedModels/{args.network_name}")

            losses = np.array(losses)
            print(f"Epoch #{epoch}: The average loss is {np.average(losses)}")

        test(
            cats_english, cats_dutch, network, 
            model_english, model_dutch, test_documents_eng,
            test_documents_dut, device, args)

    else:
        # Load a doc2vec model and network from disk
        network = torch.load(f"SavedModels/{args.network_name}").to(device)
        model_english = Doc2Vec.load("SavedModels/doc2vec_english")
        model_dutch = Doc2Vec.load("SavedModels/doc2vec_dutch")

        test(
            cats_english, cats_dutch, network,
            model_english, model_dutch, test_documents_eng,
            test_documents_dut, device, args)

    program_duration = time.time() - program_start_time
    print(f"It took {program_duration:.3f} seconds to run the entire program.")

if __name__ == "__main__":
    main()