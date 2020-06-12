# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import numpy as np
import torch

from nltk.tokenize import word_tokenize
import math
from sklearn.cluster import KMeans


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.mappings = trainer.mappings

    def generate_clusters(self, topic_features, n_clusters):
        # set random state for reproducibility
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=1).fit(topic_features)

    def compute_purity(self, kmeans, n_clusters, categories):
        """
        We are not sure how to compute purity in a multilabel problem.
        Right now, we implement it like this:
        For each cluster
            For each data point
                Add all the categories of each data point to the dict
            Calculate average number of categories per data point
            Sum the (average number of categories) highest occuring counts
            That is the numerator for this cluster
        Then when we have all numerators, we can divide it by the total number
        of categories to get the purity
        """
        counts_per_cluster = [dict() for i in range(n_clusters)]
        categories = categories.reset_index(drop=True)

        for data_index, label in enumerate(kmeans.labels_):
            for category in categories[data_index]:
                try:
                    counts_per_cluster[label][category] += 1
                except KeyError:
                    counts_per_cluster[label][category] = 1

        # print(count_dicts)
        dict_sum = 0 
        for label_dict in counts_per_cluster:
            dict_sum += max(label_dict.values())

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
        for counts in counts_per_cluster:
            sorted_values = sorted(counts.values(), reverse=True)
            total_numerator += sum(sorted_values[:average_n_cats])
            total_denominator += sum(sorted_values)

        averaged_purity = total_numerator / total_denominator
        return purity,averaged_purity

    def purity(self, to_log, df, doc2vec):
        features = []
        for index, row in df.iterrows():
            if row["language"] == "en":
                features.append(doc2vec['en'].infer_vector(
                    word_tokenize(row['content'])))
            else:
                nl_vec = doc2vec['nl'].infer_vector(
                    word_tokenize(row['content']))
                mapped_emb = self.mappings['nl'](
                    torch.from_numpy(np.array(nl_vec)).float())
                features.append(mapped_emb.detach().numpy())

        kmeans = self.generate_clusters(features, 50)
        purity, averaged_purity = self.compute_purity(kmeans, 50, df['categories'])
        return purity, averaged_purity
