# Simple GNN model for node classification
import torch.nn as nn
import torch

class Node(nn.Module):
    def __init__(self, num_nodes, dim_features, idx):
        super(Node, self).__init__()
        self.num_nodes = num_nodes
        self.dim_features = dim_features
        self.idx = idx
    

class GCNLayer(nn.Module):
    def __init__(self, nin, nout):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.tanh = nn.Tanh()

    def forward(self, H, adj_matrix):
        # H is n x dim (dim is dimension of each feature vector, and it's equal to nin)
        # adj_matrix is nxn

        # create identity matrix
        I = torch.eye(adj_matrix.size(0), device=adj_matrix.device)

        # add self loop
        adj_matrix = adj_matrix + I     # nxn + nxn = nxn

        # normalize
        deg = torch.sum(adj_matrix, dim=0)      # nxn
        deg_inv_sqrt = torch.pow(deg, -0.5)                 # nxn
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0      # remove inf

        adj_matrix = torch.mm(torch.mm(deg_inv_sqrt.diag(), adj_matrix), deg_inv_sqrt.diag())     # nxn * nxn * nxn = nxn

        H = torch.mm(adj_matrix, H)     # nxn * n x nin = n x nin
        H = self.linear(H)              # n x nin * nin x nout = n x nout
        H = self.tanh(H)                # n x nout

        return H
    
class GCN(nn.Module):
    def __init__(self, nin, nout, nhid, nclass):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(nin, nhid)
        self.layer2 = GCNLayer(nhid, nout)
        self.linear = nn.Linear(nout, nclass)

    def forward(self, H0, adj_matrix):
        H1 = self.layer1(H0, adj_matrix)
        H2 = self.layer2(H1, adj_matrix)
        H3 = self.linear(H2)
        return H3


    


    
    
    

