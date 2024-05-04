import dgl
import torch
from dgl import load_graphs
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx

from dgl.nn.pytorch import GraphConv, GATConv

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from dgl.dataloading import GraphDataLoader

class TrainDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name="folio-graph")
        graphs, lbls = load_graphs("./dataset/folio-train-graphs.bin")

        self.graphs = graphs
        self.labels = lbls['truth']
        self.num_classes = 3

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class TestDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name="folio-graph")
        graphs, lbls = load_graphs("./dataset/folio-validation-graphs.bin")

        self.graphs = graphs
        self.labels = lbls['truth']
        self.num_classes = 3

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata["z"] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.embedding = nn.Embedding(4098, 64)
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.gattn = GAT(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_heads=2)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        g = dgl.add_self_loop(g)
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = self.embedding(g.ndata['x'])
        # Perform graph convolution and activation function.
        h = F.gelu(self.conv1(g, h))
        h = F.gelu(self.gattn(g, h))
        h = F.gelu(self.conv2(g, h))

        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


trainset = TrainDataset()
testset = TestDataset()

data_loader = GraphDataLoader(trainset, batch_size=1, shuffle=True)
test_loader = GraphDataLoader(testset, batch_size=1)

# Create model
model = Classifier(64, 64, trainset.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(20):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)


model.eval()
for iter, (bg, label) in enumerate(test_loader):
    prediction = model(bg)
    print(torch.argmax(prediction), " - ", label)