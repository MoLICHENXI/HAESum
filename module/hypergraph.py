import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=True)


    def forward(self, x, H):
        x_res = x
        x = self.gat1(x, H) + x_res
        x = F.dropout(x, self.dropout, training=self.training)
        x_res = x
        x = self.gat2(x, H) + x_res
        return x
    
    

class HyperGraphAttentionLayerSparse(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer
        self.edge_node_fusion_layer = nn.Linear(2 * out_features, out_features)

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj.nonzero().t()

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)
        pair_h = torch.cat((q1, x1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)

        edge = torch.matmul(attention_edge, x)
        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)
        node_attention = torch.matmul(x_4att, edge_4att.permute(0, 2, 1))
        node_attention = node_attention * (1 / self.out_features ** (1/2))
        zero_vec = -9e15 * torch.ones_like(node_attention)
        node_attention = torch.where(adj.permute(0, 2, 1) > 0, node_attention, zero_vec)
        attention_node = F.softmax(node_attention, dim = -1)
        
        node = torch.matmul(attention_node, edge)
        node = torch.cat((node, x), dim=-1)
        node = F.dropout(self.edge_node_fusion_layer(node), self.dropout, training=self.training)
        
        if self.concat:
            node = F.elu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'