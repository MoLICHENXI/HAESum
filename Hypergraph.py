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

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
from module.hypergraph import HGNN_ATT
import torch.nn.functional as F

class Hypergraph(nn.Module):
    """ without sent2sent and add residual connection """
    # 
    def __init__(self, hps, embed):
        """

        :param hps:
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim

        self._init_sn_param()

        self._TFembed = nn.Embedding(10, hps.feat_embed_size) 
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,         
                                out_dim=hps.hidden_size,   
                                num_heads=hps.n_head,      
                                attn_drop_out=hps.atten_dropout_prob,     
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,     
                                ffn_drop_out=hps.ffn_dropout_prob,                   
                                feat_embed_size=hps.feat_embed_size,               
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,                      
                                out_dim=embed_size,                           
                                num_heads=6,                                  
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )

        # Hypergraph
        self.outhypergraph = HGNN_ATT(hps.hidden_size, hps.hidden_size, 64)
        
        # node classification
        self.n_feature = hps.hidden_size + 768      
        
        self.wh = torch.nn.Sequential(
            nn.Linear(64, 64),
            torch.nn.LayerNorm(64), 
            nn.Linear(64, 2)
        )

    def forward(self, graph, hyper_edge, actual_node_list, hyper_feature=None):
        
        word_feature = self.set_wnfeature(graph)  

        sent_feature = self.n_feature_proj(self.set_snfeature(graph)) 

        
        word_state = word_feature

        sent_state = self.word2sent(graph, word_feature, sent_feature)   

        

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)  
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state) 

        
        idx = 0
        pad_node_feature = []
        max_node_nums = actual_node_list[0]
        
        for i in range(len(actual_node_list)):
            count = actual_node_list[i].item()
            need_tensor = sent_state[idx:idx + count]
            if count < max_node_nums:
                pad_nums = max_node_nums - count
                need_tensor = F.pad(need_tensor, (0, 0, 0, pad_nums))
                
                pad_node_feature.append(need_tensor)
            else:
                pad_node_feature.append(need_tensor)
            idx += count
        node_feature_input = torch.stack(pad_node_feature, dim = 0)  
        
        output = self.outhypergraph(node_feature_input, hyper_edge)
        actual_node = []
        for i in range(len(actual_node_list)):
            count = actual_node_list[i].item()
            actual_node.append(output[i, : count, :])
        result = torch.cat(actual_node, dim = 0)  
        result = self.wh(result)                  
        return result

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding) 
        return cnn_feature

 
    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0)) 
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0) 
        wid = graph.nodes[wnode_id].data["id"]  
        w_embed = self._embed(wid)  
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        return w_embed

    def set_snfeature(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  
        return node_feature

def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen
