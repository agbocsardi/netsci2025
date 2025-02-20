import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(nn.Module):
    def __init__(
        self, num_node_features, num_edge_features, hidden_size=32, target_size=1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.target_size = target_size
        self.convs = [
            GATConv(
                self.num_node_features,
                self.hidden_size,
                edge_dim=self.num_edge_features,
            ),
            GATConv(
                self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features
            ),
        ]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)

        return F.relu(x)
