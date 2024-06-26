#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it
"""

import os
from nltk.corpus import stopwords
import time
import json
from collections import Counter
import numpy as np
import torch
import torch.utils.data
from tools.logger import *
import dgl
from dgl.data.utils import load_graphs
import torch.nn.functional as F

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``', '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)


class Example(object):
    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label, hedge, hedge_node_nums, hedge_edge_nums):

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        self.hedge = hedge
        self.hedge_sen_node_num = int(hedge_node_nums)
        self.hedge_edge_num = int(hedge_edge_nums)
        
        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
        
        self._pad_encoder_input(vocab.word2id('[PAD]'))    

        self.label = label
        
        label_shape = (len(self.original_article_sents), len(label))  
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return:
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, hedge_path):
        """ Initializes the ExampleSet with the path of data
       
        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py)
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        """

        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        logger.info("[INFO] Loading Whole Dataset from %s ...", data_path)
        self.example_list = readJson(data_path)

        logger.info("[INFO] Loading Hyperedges from %s ...", hedge_path)
        self.hedge_list = readJson(hedge_path)
        
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list)
        
        
        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                lowtfidf_num += 1
            
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)

        self.w2s_path = w2s_path
        self.data_path = data_path
        
        self.hedge_path = hedge_path
        
    def get_example(self, index):
        e = self.example_list[index]
        hedge = self.hedge_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], hedge["hedges"], hedge["length"], hedge["section_length"])
        return example, e["id"], hedge["id"]

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m    

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1
        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes) 
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def CreateGraph(self, input_pad, label, w2s_w, feature_bert=None):
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        
        G.set_e_initializer(dgl.init.zero_initializer)
        for i in range(N):
            c = Counter(input_pad[i])
            
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    
                    
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long() 
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]
        
        return G
    
    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item, input_ids, hedge_ids = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        
        edge = item.hedge
        
        
        hypergraph_sen_node = item.hedge_sen_node_num
        
        
        hypergraph_edge_nums = item.hedge_edge_num
        sec_edge = cluster_to_hedge(edge, hypergraph_sen_node, 0)
        e = sec_edge
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]

        G = self.CreateGraph(input_pad, label, w2s_w)
        actual_node_nums = len(item.enc_sent_input_pad) if len(item.enc_sent_input_pad) <= self.doc_max_timesteps else self.doc_max_timesteps
        return G, index, actual_node_nums, e

    def __len__(self):
        return self.size


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        # print(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def graph_collate_fn(samples):

    graphs, index ,actual_node_nums, hedges = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    max_node_num = max(actual_node_nums)
    actual_node_list = []
    
    if max_node_num > 250:
        max_node_num = 250
    
    for i in range(len(actual_node_nums)):
        pad_len = max_node_num - actual_node_nums[i]
        if pad_len > 0:
            actual_node_list.append(max_node_num - pad_len)
        else:
            actual_node_list.append(max_node_num)
    max_edge_num = max([e.shape[0] for e in hedges]) 
    for i in range(len(hedges)):
        pad_node_len = max_node_num - hedges[i].shape[1]
        pad_edge_len = max_edge_num - hedges[i].shape[0]
        if pad_node_len > 0:
            hedges[i] = F.pad(hedges[i], (0, pad_node_len, 0, 0))
        else:
            hedges[i] = hedges[i][:, :max_node_num]

        if pad_edge_len > 0:
            hedges[i] = F.pad(hedges[i], (0, 0, 0, pad_edge_len))
        else:
            hedges[i] = hedges[i][:max_edge_num, :]
    return batched_graph, [index[idx] for idx in sorted_index], torch.stack([hedges[idx] for idx in sorted_index]), torch.tensor([actual_node_list[idx] for idx in sorted_index])


def cluster_to_hedge(cluster, node_num, threshold_low=3, threshold_high=1000000):
    hedges = []   
    for c in cluster: 
        if threshold_low < len(c) <= threshold_high:
            edge = torch.zeros(node_num)
            edge[c] = 1
            hedges.append(edge)
        else:
            continue
    if len(hedges) == 0:
        hedges.append(torch.zeros(node_num))
    return torch.stack(hedges, dim=0)