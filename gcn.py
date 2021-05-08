#!/usr/bin/env python
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):  # 图卷积层
    def __init__(self, input_dim, output_dim, support, act_func=None, featureless=False, dropout_rate=0., bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):  # 初始化GCN层的权重参数 如果是gcn_cheby会有多个邻接矩阵 同时需要初始化多个W权重矩阵
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))  # W(input_dim, output_dim)

        if bias:  # 偏置参数
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func  # 激活函数
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  # x(train_size+vocab_size+test_size,input_dim)
        x = self.dropout(x)

        for i in range(len(self.support)):  # gcn_cheby会有多个邻接矩阵；gcn只有一个
            if self.featureless:  # featureless=True 表示X为初始图的特征矩阵 节点使用one-hot形式(顺序)
                # 初始特征矩阵是一个单位矩阵(n*n) XW = W;n=train_size+vocab_size+test_size
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))

            if i == 0:
                out = self.support[i].mm(pre_sup)  # A(XW) = AW  A为引入自连接、归一化的邻接矩阵 n*n n=train_size+vocab_size+test_size
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:  # 通过激活函数
            out = self.act_func(out)

        self.embedding = out  # （train_size+vocab_size+test_size,output_dim）
        return out


class GCN(nn.Module):
    def __init__(self, input_dim, support, dropout_rate=0., num_classes=10):
        super(GCN, self).__init__()

        # GraphConvolution
        # input_dim = train_size+vocab_size+test_size
        # support 归一化、引入自连接的邻接矩阵
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, 200, support, act_func=nn.ReLU())
        # self.layer3 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)

    def forward(self, x):  # x (train_size+vocab_size+test_size,train_size+vocab_size+test_size) 单位矩阵
        out = self.layer1(x)  # (train_size+vocab_size+test_size,200)
        out = self.layer2(out)  # (train_size+vocab_size+test_size,num_classes)
        # out = self.layer3(out)
        return out



