"""network.py
"""

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
from lda import generate_clusters, compute_purity

class Network(nn.Module):
    def __init__(self, vector_length, n_topics):
        super(Network, self).__init__()

        # size_steps = n_topics - vector_length // 3
        size_steps = 100
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
        else:
            return category2idx["_UNK"]

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

def main():
    program_start_time = time.time()

    parser = argparse.ArgumentParser("A neural network which makes use of document embeddings as input.")
    parser.add_argument("-vl", "--vector_length", type=int, default=100, help="The length of the document embedding vectors.")
    parser.add_argument("-nt", "--n_topics", type=int, default=50, help="The number of topics the network should classify into.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("-ne", "--n_epochs", type=int, default=10, help="The number of epochs the network should train if --train is specified.")
    parser.add_argument("-f", "--file", default='Data/wikicomp-2014_ennl.xml', help="The dataset file (xml)")
    parser.add_argument("-nd", "--n_documents", type=int, default=10000, help="The maximal number of documents we want to use for training.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="The learning rate of the Adam optimizer.")
    parser.add_argument("--network_name", default="network.pt", help="The filename of the network to be saved or loaded")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df_english, df_dutch = dataset_util.process_dataset(args.file, args.n_documents)
    df_english = df_english.reset_index()
    df_dutch = df_dutch.reset_index()

    unprocessed_corpus_english = list(df_english["content"])
    unprocessed_corpus_dutch = list(df_dutch["content"])

    stopwords_english = dataset_util.preprocess_stopwords('en')
    stopwords_dutch = dataset_util.preprocess_stopwords('nl')

    corpus_english = preprocess_corpus(unprocessed_corpus_english, stopwords_english)
    corpus_dutch = preprocess_corpus(unprocessed_corpus_dutch, stopwords_dutch)

    documents_english = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_english)]
    documents_dutch = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_dutch)]

    category_english2idx = {"_UNK": 0}
    categories_english = [[get_index(c,category_english2idx) for c in y] for y in df_english["categories"]]
    nhots_english = convert_to_nhot(categories_english, len(category_english2idx))

    category_dutch2idx = {"_UNK": 0}
    categories_dutch = [[get_index(c,category_dutch2idx) for c in y] for y in df_dutch["categories"]]
    nhots_dutch = convert_to_nhot(categories_dutch, len(category_dutch2idx))

    if args.train:
        print("Computing document vectors, this can take a while...")
        model_english = Doc2Vec(
            documents_english,
            vector_size=args.vector_length,
            window=2,
            min_count=1,
            workers=4,
            epochs=8,
            seed=0)
        model_dutch  = Doc2Vec(
            documents_dutch,
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
        network = Network(args.vector_length, len(category_english2idx)).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

        for epoch in range(args.n_epochs):
            losses = []
            for d_index, data in enumerate(documents_english):
                optimizer.zero_grad()
                docvec = model_english.infer_vector(data[0])
                doctensor = torch.Tensor(docvec).to(device)
                raw_topics = network(doctensor)

                # The target is an n-hot vector e.g. [1, 1, 0, 1]
                target = nhots_english[d_index]
                loss = nhot_cross_entropy_loss(raw_topics, target, device)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())

                optimizer.zero_grad()
                docvec = model_dutch.infer_vector(documents_dutch[d_index][0])
                doctensor = torch.Tensor(docvec).to(device)
                raw_topics = network(doctensor)

                # The target is an n-hot vector e.g. [1, 1, 0, 1]
                target = nhots_dutch[d_index]

                loss = nhot_cross_entropy_loss(raw_topics, target, device)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())

            if epoch == args.n_epochs - 1:
                torch.save(network, f"SavedModels/{args.network_name}")
                print(f"Saved network: SavedModels/{args.network_name}")

            losses = np.array(losses)
            print(f"Epoch #{epoch}: The average loss is {np.average(losses)}")

    else:
        # Load a doc2vec model and network from disk
        network = torch.load(f"SavedModels/{args.network_name}")
        model_english = Doc2Vec.load("SavedModels/doc2vec_english")
        model_dutch = Doc2Vec.load("SavedModels/doc2vec_dutch")

        features_english = []
        features_dutch = []
        for d_index, data in enumerate(documents_english):
            docvec = model_english.infer_vector(data[0])
            doctensor = torch.Tensor(docvec).to(device)
            output_eng = network(doctensor)         

            docvec = model_dutch.infer_vector(documents_dutch[d_index][0])
            doctensor = torch.Tensor(docvec).to(device)
            output_dutch = network(doctensor)

            features_english.append(output_eng)
            features_dutch.append(output_dutch)

        features_english = np.array(features_english)
        features_dutch = np.array(features_dutch)
        features_both = np.concatenate((features_english, features_dutch))
        categories_both = pd.concat((categories_test_english, categories_test_dutch))

        kmeans = generate_clusters(features_both, args.n_topics)
        purity = compute_purity(kmeans, args.n_topics, categories_both)
        print(f"The purity of the made clusters is {purity:.3f}")

    program_duration = time.time() - program_start_time
    print(f"It took {program_duration:.3f} seconds to run the entire program.")

if __name__ == "__main__":
    main()