import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from copy import deepcopy

"""
    LAYERS: GCNConv and ChebNetConv
"""


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, init):
        super(GCNConv, self).__init__()
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters(init)

    def reset_parameters(self, init):
      stdv = 1. / math.sqrt(self.weight.size(1))
      if init == "Xavier Uniform":
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.linear.weight.data)
      elif init == "Xavier Normal":
        nn.init.xavier_normal_(self.weight.data)
        nn.init.kaiming_uniform_(self.linear.weight.data)
      elif init == "Kaiming Uniform":
        nn.init.kaiming_uniform_(self.weight.data)
        nn.init.kaiming_uniform_(self.linear.weight.data)
      elif init == "Kaiming Normal":
        nn.init.kaiming_normal_(self.weight.data)
        nn.init.kaiming_normal_(self.linear.weight.data)
      else:
          self.weight.data.uniform_(-stdv, stdv)
          nn.init.normal_(self.linear.weight, std=0.05)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency_hat, x)
        return x


class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor):
        x = self.__transform_to_chebyshev(x, laplacian)
        x = self.linear(x)
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x.unsqueeze(2)
        x0 = x

        if self.K > 1:
            x1 = torch.sparse.mm(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            for _ in range(2, self.K):
                x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), 2)
                x0, x1 = x1, x2

        cheb_x = cheb_x.reshape([x.shape[0], -1])
        return cheb_x


"""
    MODELS
"""


class TwoLayerGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(TwoLayerGCN, self).__init__()

        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = self.dropout(x)
        x = self.conv1(x, adjacency_hat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adjacency_hat)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, init, output_size, num_hidden_layers=0, dropout=0.1, residual=False):
        super(GCN, self).__init__()
        self.init = init
        self.dropout = dropout
        self.residual = residual
        # self.opt = opt
        self.num_hidden_layers = num_hidden_layers
        if num_hidden_layers > 1:
          self.input_conv = GCNConv(input_size, hidden_size, init)
          self.hidden_convs = nn.ModuleList([GCNConv(hidden_size, hidden_size, init) for _ in range(num_hidden_layers - 2)])
          self.output_conv = GCNConv(hidden_size, output_size, init)
        else:
          self.output_conv = GCNConv(input_size, output_size, init)



    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        if self.num_hidden_layers > 1:
        
          x = F.dropout(x, p=self.dropout, training=self.training)
          x = F.relu(self.input_conv(x, adjacency_hat))
          # print(len(self.hidden_convs))
          for conv in self.hidden_convs:
              if self.residual:
                  x = F.relu(conv(x, adjacency_hat)) + x
              else:
                  x = F.relu(conv(x, adjacency_hat))
          embed = x
          h = F.dropout(x, p=self.dropout, training=self.training)
          h = self.output_conv(h, adjacency_hat)
        else:
          h = self.output_conv(h, adjacency_hat)
        if labels is None:
            return h

        loss = nn.CrossEntropyLoss()(h, labels)
        return h, loss, embed


class TwoLayerChebNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, k=2):
        super(TwoLayerChebNet, self).__init__()

        self.conv1 = ChebNetConv(input_size, hidden_size, k)
        self.conv2 = ChebNetConv(hidden_size, output_size, k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = self.dropout(x)
        x = self.conv1(x, laplacian)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, laplacian)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss


class ChebNetGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, dropout=0.1, residual=False, k=2):
        super(ChebNetGCN, self).__init__()

        self.dropout = dropout
        self.residual = residual

        self.input_conv = ChebNetConv(input_size, hidden_size, k)
        self.hidden_convs = nn.ModuleList([ChebNetConv(hidden_size, hidden_size, k) for _ in range(num_hidden_layers)])
        self.output_conv = ChebNetConv(hidden_size, output_size, k)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_conv(x, laplacian))
        for conv in self.hidden_convs:
            if self.residual:
                x = F.relu(conv(x, laplacian)) + x
            else:
                x = F.relu(conv(x, laplacian))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_conv(x, laplacian)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss