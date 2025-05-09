import argparse
import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Complete
from torch_geometric.loader import DataLoader
from models import GNN
from utils import train, evaluate, plot_losses

def main():
    parser = argparse.ArgumentParser(description='Train GNN on QM9 dataset')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN architecture')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load QM9 dataset
    dataset = QM9(root='data/QM9', transform=Complete())
    dataset = dataset.shuffle()
    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = GNN(args.model, dataset.num_node_features, args.hidden_dim, 1, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')

    test_loss = evaluate(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}')

    plot_losses(train_losses, val_losses)

if __name__ == '__main__':
    main()
