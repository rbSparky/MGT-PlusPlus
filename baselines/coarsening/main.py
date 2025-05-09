
from torch_geometric.datasets import QM9
import copy
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_networkx
import numpy as np
from coarsening import solver_v2
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
import time

target = 0
dim = 64

class MyTransform:
    def __call__(self, data):
        data = copy.copy(data)
        data.y = data.y[:, target]
        return data

class Complete:
    def __call__(self, data):
        data = copy.copy(data)
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

class CoarsenedDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.', transform=None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def coarsen_graph(data, k_frac=0.9, solver_iters=10):
    graph = to_networkx(data, to_undirected=True)
    p = data.x.shape[0]
    W = np.zeros((p, p))
    for (x, y) in graph.edges:
        W[x][y] = np.random.randint(1, 10)
    W_t = W + W.T
    L = np.diag(W_t @ np.ones(W_t.shape[0])) - W_t
    X_np = data.x.numpy()
    k = int(p * k_frac)
    solver = solver_v2(X_np, L, k, 500, 0, 500, X_np.shape[1] / 2)
    C, X_tilde, _ = solver.fit(solver_iters)

    x_coarse = torch.tensor(X_tilde, dtype=torch.float)
    edge_index_coarse = torch.combinations(torch.arange(k), r=2).T  # fully connected
    edge_index_coarse = torch.cat([edge_index_coarse, edge_index_coarse[[1, 0]]], dim=1)

    return Data(x=x_coarse, edge_index=edge_index_coarse, y=data.y)

path = "./"
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform).shuffle()

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

num_nodes = 100
test_dataset = dataset[:10000]
val_dataset = dataset[10000:10000+10]
train_dataset = dataset[20000:20000+num_nodes]
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

from tqdm import tqdm

print("Coarsening train dataset...")
start = time.time()
coarsened_train = [coarsen_graph(d, k_frac=1) for d in tqdm(train_dataset)]
end = time.time()
print(f"Time taken for coarsening: {end-start:.4f}")
train_dataset_coarse = CoarsenedDataset(coarsened_train)
train_loader = DataLoader(train_dataset_coarse, batch_size=32, shuffle=True)

class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

model = GNN(in_dim=11, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

from sklearn.metrics import mean_absolute_error

def train_epoch(loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_epoch(loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y.cpu())
            y_pred.append(out.squeeze().cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    mae = mean_absolute_error(y_true, y_pred)
    return mae

# Example training
start = time.time()
for epoch in range(1, 101):
    loss = train_epoch(train_loader)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    if epoch % 50 == 0:
        end = time.time()
        mae = test_epoch(test_loader)
        print(f"Test MAE at epoch {epoch}: {mae:.4f}, time: {end-start:.4f}")
        exit(0)