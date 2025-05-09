import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.model_type = model_type.lower()
        self.convs = torch.nn.ModuleList()

        if self.model_type == 'gin':
            nn = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn))
            for _ in range(num_layers - 1):
                nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
                self.convs.append(GINConv(nn))
        else:
            conv_layer = {
                'gcn': GCNConv,
                'gat': GATConv,
                'sage': SAGEConv
            }.get(self.model_type)
            if conv_layer is None:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            self.convs.append(conv_layer(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(conv_layer(hidden_dim, hidden_dim))

        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)
