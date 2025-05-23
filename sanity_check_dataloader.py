import os
import argparse
import torch
from torch.utils.data import DataLoader
from utils.datasets import StructureDataset
import dgl
from tqdm import tqdm # Import tqdm

def sanity_check_dataloader(args):
    """
    Performs a sanity check on the dataloader to identify issues
    and print the mofid if an exception occurs during batch processing.
    Includes a progress bar with time left.
    """
    print(f"Starting dataloader sanity check with root: {args.root}")

    # Initialize the dataset
    dataset = StructureDataset(args, process=args.process)
    print(f"Total dataset size: {len(dataset)}")

    # Split the dataset
    n_train = int(len(dataset) * args.train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(train_ds,
                              collate_fn=dataset.collate_tt,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=args.prefetch_factor)

    val_loader = DataLoader(val_ds,
                            collate_fn=dataset.collate_tt,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=args.prefetch_factor)

    print("\n--- Testing Train DataLoader ---")
    # Wrap the loader with tqdm for a progress bar
    with tqdm(total=len(train_loader), desc="Checking Train Data", unit="batch") as pbar_train:
        for i, (g, lg, fg, target, mofid_list) in enumerate(train_loader):
            try:
                # Basic checks
                _ = g.batch_size
                _ = target.shape
            except Exception as e:
                print(f"\nError encountered in training batch {i}!")
                print(f"MOFIDs in this batch: {mofid_list}")
                print(f"Exception details: {e}")
                return # Exit after the first error
            pbar_train.update(1) # Update the progress bar

    print("\n--- Testing Validation DataLoader ---")
    # Wrap the loader with tqdm for a progress bar
    with tqdm(total=len(val_loader), desc="Checking Val Data", unit="batch") as pbar_val:
        for i, (g, lg, fg, target, mofid_list) in enumerate(val_loader):
            try:
                # Basic checks
                _ = g.batch_size
                _ = target.shape
            except Exception as e:
                print(f"\nError encountered in validation batch {i}!")
                print(f"MOFIDs in this batch: {mofid_list}")
                print(f"Exception details: {e}")
                return # Exit after the first error
            pbar_val.update(1) # Update the progress bar

    print("\nDataLoader sanity check completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for the dataloader.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of the dataset to use for training.")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of the dataset to use for validation.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for the dataloader.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches to prefetch.")
    parser.add_argument("--process", type=int, default=1, choices=[0, 1],
                        help="Whether to process the data (1) or load pre-processed (0).")

    # Arguments specific to StructureDataset or model that might be needed for initialization
    parser.add_argument("--max_nei_num", type=int, default=12)
    parser.add_argument("--local_radius", type=int, default=8)
    parser.add_argument("--periodic", type=int, default=1, choices=[0, 1])
    parser.add_argument("--periodic_radius", type=int, default=12)
    parser.add_argument("--num_atom_fea", type=int, default=90)
    parser.add_argument("--num_edge_fea", type=int, default=1)
    parser.add_argument("--num_angle_fea", type=int, default=1)
    parser.add_argument("--num_pe_fea", type=int, default=10)
    parser.add_argument("--num_clmb_fea", type=int, default=1)
    parser.add_argument("--num_edge_bins", type=int, default=80)
    parser.add_argument("--num_angle_bins", type=int, default=40)
    parser.add_argument("--num_clmb_bins", type=int, default=120)
    parser.add_argument("--embedding_dims", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, default=512)
    parser.add_argument("--out_dims", type=int, default=3)
    parser.add_argument("--out_names", nargs="+", default=["bandgap", "HOMO", "LUMO"])

    args = parser.parse_args()

    args.periodic = bool(args.periodic)
    args.process = bool(args.process)

    sanity_check_dataloader(args)
