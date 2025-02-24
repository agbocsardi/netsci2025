import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


class GATModel(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,  # Now used in the model
        hidden_size=10,
        target_size=1,
        num_attention_heads=3,
        dropout=0.0,
    ):
        super().__init__()

        # First GAT layer with edge features
        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_size,
            heads=num_attention_heads,
            dropout=dropout,
            edge_dim=num_edge_features,  # Incorporate edge features
        )

        # Second GAT layer, adjust the input size for multi-heads
        self.conv2 = GATv2Conv(
            in_channels=hidden_size * num_attention_heads,
            out_channels=target_size,
            heads=1,
            concat=False,  # No concatenation since we're doing regression
            dropout=dropout,
            edge_dim=num_edge_features,  # Incorporate edge features
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GAT layer with edge attributes
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)

        # Optional: Apply a non-linearity (e.g., ELU) between layers
        x = F.elu(x)

        # Second GAT layer with edge attributes
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x
