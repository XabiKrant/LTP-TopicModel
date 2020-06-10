# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import numpy as np

from nltk.tokenize import word_tokenize
from .utils import normalize_embeddings


class Discriminator(nn.Module):

    def __init__(self, params, lang):
        super(Discriminator, self).__init__()

        self.lang = lang
        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


def to_vectors(params, df, doc2vec):
    file_embs = {}
    en_vecs = []
    nl_vecs = []

    for en_doc in df[df["language"] == "en"]["content"]:
        en_vecs.append(doc2vec['en'].infer_vector(word_tokenize(en_doc)))

    file_embs['en'] = torch.from_numpy(np.array(en_vecs)).float()
    file_embs['en'] = file_embs['en'].to(params.device)

    for nl_doc in df[df["language"] == "nl"]["content"]:
        nl_vecs.append(doc2vec['nl'].infer_vector(word_tokenize(nl_doc)))

    file_embs['nl'] = torch.from_numpy(np.array(nl_vecs)).float()
    file_embs['nl'] = file_embs['nl'].to(params.device)

    return file_embs


def build_model(params, with_dis, df, doc2vec):
    """
    Build all components of the model.
    """
    # source embeddings
    params.vocabs, _src_embs, embs = {}, {}, {}
    file_vectors = to_vectors(params, df, doc2vec)
    for i, lang in enumerate(params.src_langs):
        emb = file_vectors[lang]
        params.vocabs[lang] = df[df["language"] == lang]
        _src_embs[lang] = emb
    for i, lang in enumerate(params.src_langs):
        src_emb = nn.Embedding(
            len(params.vocabs[lang]), params.emb_dim, sparse=True)
        src_emb.weight.data.copy_(_src_embs[lang])
        embs[lang] = src_emb
    # target embeddings
    if params.tgt_lang:
        _tgt_emb = file_vectors[params.tgt_lang]
        params.vocabs[params.tgt_lang] = df[df["language"] == params.tgt_lang]
        tgt_emb = nn.Embedding(
            len(params.vocabs[params.tgt_lang]), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
        embs[params.tgt_lang] = tgt_emb
    else:
        tgt_emb = None

    # mappings
    mappings = {lang: nn.Linear(params.emb_dim, params.emb_dim,
                                bias=False) for lang in params.src_langs}
    # set tgt mapping to fixed identity matrix
    tgt_map = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    tgt_map.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    for p in tgt_map.parameters():
        p.requires_grad = False
    mappings[params.tgt_lang] = tgt_map
    if getattr(params, 'map_id_init', True):
        for mapping in mappings.values():
            mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminators
    discriminators = {lang: Discriminator(params, lang)
                      for lang in params.all_langs} if with_dis else None

    for lang in params.all_langs:
        embs[lang] = embs[lang].to(params.device)
        mappings[lang] = mappings[lang].to(params.device)
        if with_dis:
            discriminators[lang] = discriminators[lang].to(params.device)

    # normalize embeddings
    params.lang_mean = {}
    for lang, emb in embs.items():
        params.lang_mean[lang] = normalize_embeddings(
            emb.weight.detach(), params.normalize_embeddings)

    # return embs, mappings, discriminators
    return embs, mappings, discriminators
