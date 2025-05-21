import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"

import gc
import time
import pathlib
import argparse
import warnings
import numpy as np
import os.path as osp
import contextlib

import torch
import torch.nn as nn
import torch.backends.cudnn
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.transformer import multiheaded
from model.alignn import EdgeGatedGraphConv
from model.graphformer import Graphformer, encoder
from utils.datasets import StructureDataset
import wandb

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
import warnings

# Ignore exactly those lazyInitCUDA deprecation warnings:
warnings.filterwarnings(
    "ignore",
    message=".*lazyInitCUDA is deprecated.*",
    category=UserWarning
)

def clear_memory():
    """Run full CPU/GPU garbage collection to mitigate memory fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def maybe_reset_peak_stats(device):
    """Reset peak memory stats if running on CUDA to keep profiler values sane."""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

def train(args, model, loader, optimizer, criterion, fabric):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_loss = torch.zeros(2, device=fabric.device)
    maybe_reset_peak_stats(fabric.device)

    with tqdm(total=len(loader), desc="Training", unit="batch", disable=not fabric.is_global_zero) as pbar:
        for iteration, (g, lg, fg, target, _) in enumerate(loader):
            clear_memory()
            g, lg, fg = g.to(fabric.device, non_blocking=True), lg.to(fabric.device, non_blocking=True), fg.to(fabric.device, non_blocking=True)
            # print(g)
            # print(lg)
            # print(fg)
            target = target.to(fabric.device, non_blocking=True)
            is_accumulating = (iteration % args.n_cum) != 0
            try:
                with torch.autocast(device_type=fabric.device.type, dtype=torch.float16, enabled=fabric.device.type == "cuda"):
                    with fabric.no_backward_sync(model, enabled=is_accumulating):
                        output, _, _, _, _ = model(g, lg, fg)
                        loss = criterion(output, target) / args.n_cum
                        if torch.isnan(loss) or torch.isinf(loss):
                            fabric.print(f"[iteration {iteration}] skipping batch—loss is {loss}")
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            print(f"Graph: {g}")
                            print(f"Line Graph: {lg}")
                            print(f"Full Graph: {fg}")
                            continue
                        fabric.backward(loss)
                if not is_accumulating:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if fabric.device.type == "cuda":
                        torch.cuda.synchronize(device=fabric.device)
                epoch_loss[0] += loss.detach().item() * args.n_cum
                epoch_loss[1] += target.size(0)
                del g, lg, fg, target, output, loss
                clear_memory()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    fabric.print(f"Out of memory at iteration {iteration}, skipping batch")
                    print(f"Graph: {g}")
                    print(f"Line Graph: {lg}")
                    print(f"Full Graph: {fg}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Graph: {g}")
                    print(f"Line Graph: {lg}")
                    print(f"Full Graph: {fg}")
                    raise e
            pbar.update(1)
            pbar.set_postfix({"loss": f"{epoch_loss[0] / epoch_loss[1]:.4f}"})
    fabric.all_reduce(epoch_loss, reduce_op="sum")
    epoch_loss_value = (epoch_loss[0] / epoch_loss[1]).item()
    fabric.print(f"Epoch loss: {epoch_loss_value:.4f}")
    return model, optimizer, epoch_loss_value

def validate(args, model, loader, criterion, fabric):
    model.eval()
    epoch_error = torch.zeros(2, device=fabric.device)
    epoch_indiv = [torch.zeros(2, device=fabric.device) for _ in range(args.out_dims)] if args.out_dims > 1 else None
    maybe_reset_peak_stats(fabric.device)
    with torch.no_grad():
        for (g, lg, fg, target, _) in loader:
            clear_memory()
            g, lg, fg = g.to(fabric.device, non_blocking=True), lg.to(fabric.device, non_blocking=True), fg.to(fabric.device, non_blocking=True)
            target = target.to(fabric.device, non_blocking=True)
            with torch.autocast(device_type=fabric.device.type, dtype=torch.float16, enabled=fabric.device.type == "cuda"):
                output, _, _, _, _ = model(g, lg, fg)
            err = criterion(output, target)
            epoch_error[0] += err.item()
            epoch_error[1] += target.size(0)
            if args.out_dims > 1:
                ts = torch.hsplit(target, target.shape[1])
                osplits = torch.hsplit(output, output.shape[1])
                subs = [criterion(o, t) for o, t in zip(osplits, ts)]
                for i, e in enumerate(subs):
                    epoch_indiv[i][0] += e.item()
                    epoch_indiv[i][1] += target.size(0)
            del g, lg, fg, target, output, err
            clear_memory()
    fabric.all_reduce(epoch_error, reduce_op="sum")
    epoch_error_value = (epoch_error[0] / epoch_error[1]).item()
    indiv_values = None
    if args.out_dims > 1:
        indiv_values = []
        for i in range(args.out_dims):
            fabric.all_reduce(epoch_indiv[i], reduce_op="sum")
            indiv_values.append((epoch_indiv[i][0] / epoch_indiv[i][1]).item())
    msg = f"Validation error: {epoch_error_value:.4f}"
    if indiv_values is not None:
        for i, name in enumerate(args.out_names):
            msg += f" | {name} Error: {indiv_values[i]:.4f}"
    fabric.print(msg)
    return epoch_error_value, indiv_values

def main(args):
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    os.makedirs(args.model_path, exist_ok=True)
    logger = CSVLogger(root_dir=args.save_dir, name=args.run_name, flush_logs_every_n_steps=1)
    policy = {encoder, EdgeGatedGraphConv, multiheaded}
    fsdp = FSDPStrategy(auto_wrap_policy=policy,
                        activation_checkpointing_policy=policy,
                        state_dict_type="full")
    if args.accelerator in ("cuda", "gpu") and args.n_devices > 1:
        fabric = Fabric(accelerator=args.accelerator,
                        devices=args.n_devices,
                        num_nodes=args.n_nodes,
                        strategy=fsdp,
                        precision="16-mixed",
                        loggers=logger)
    else:
        fabric = Fabric(accelerator=args.accelerator,
                        devices=args.n_devices,
                        num_nodes=args.n_nodes,
                        precision="16-mixed",
                        loggers=logger)
    fabric.launch()
    dataset = StructureDataset(args, process=args.process)
    n_train = int(len(dataset) * args.train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
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
    train_loader = fabric.setup_dataloaders(train_loader, move_to_device=False)
    val_loader   = fabric.setup_dataloaders(val_loader,   move_to_device=False)
    model = Graphformer(args=args)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print("ARCHITECTURE:\n"
                 f"\tLayers   - {args.num_layers}\n"
                 f"\tMHAs     - {args.n_mha}\n"
                 f"\tALIGNNs  - {args.n_alignn}\n"
                 f"\tGNNs     - {args.n_gnn}\n"
                 f"\tParams   - {num_params}")
    model = fabric.setup_module(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer = fabric.setup_optimizers(optimizer)
    train_loss = nn.MSELoss()
    val_loss = nn.L1Loss()
    if args.load_model:
        ckpt_path = {
            1: osp.join(args.model_path, args.pretrain_model),
            2: osp.join(args.model_path, args.lowest_model)
        }.get(args.load_model, None)
        if ckpt_path and osp.exists(ckpt_path):
            fabric.load(ckpt_path, models=model, optimizers=optimizer)
            fabric.print(f"Loaded model from {ckpt_path}")
    fabric.print("-------------------- Training Started --------------------")
    best_error = float("inf")
    epoch_times = []
    t_all = time.time()
    for epoch in range(args.begin_epoch, args.epochs + 1):
        epoch_start = time.time()
        model, optimizer, tr_loss = train(args, model, train_loader, optimizer, train_loss, fabric)
        fabric.print(f"Training time: {time.time() - epoch_start:.2f} s")
        epoch_time = time.time() - epoch_start
        wandb.log({"train_loss": tr_loss, "epoch": epoch, "epoch_time": epoch_time})
        with open(osp.join(args.model_path, f"train_{epoch}.log"), "a") as f:
            f.write(f"Epoch {epoch} | Loss: {tr_loss:.4f}\n")
        val_err, _ = validate(args, model, val_loader, val_loss, fabric)
        with open(osp.join(args.model_path, f"val_{epoch}.log"), "a") as f:
            f.write(f"Epoch {epoch} | Validation Error: {val_err:.4f}\n")        
        fabric.print(f"Epoch {epoch} validation complete")
        wandb.log({"val_error": val_err, "epoch": epoch})
        if val_err < best_error:
            best_error = val_err
            save_path = osp.join(args.model_path, args.lowest_model)
            fabric.save(save_path, {"models": model, "optimizer": optimizer})
            fabric.print(f"New best model saved to {save_path}")
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        fabric.print(f"Completed Epoch {epoch}/{args.epochs} in {epoch_duration:.2f} s")
        fabric.barrier()
        clear_memory()
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    fabric.print(f"Avg epoch: {avg_epoch_time:.2f} s | Total: {time.time() - t_all:.2f} s")
    final_model_path = osp.join(args.model_path, args.final_model)
    fabric.save(final_model_path, {"models": model, "optimizer": optimizer})
    fabric.print(f"Final model saved to {final_model_path}")
    fabric.print("-------------------- Training Finished --------------------")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cuda",
                        choices=["cpu", "gpu", "mps", "cuda", "tpu"])
    parser.add_argument("--wandb_project", type=str, default="MGT")
    parser.add_argument("--root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--pretrain_model", default="pretrain.ckpt")
    parser.add_argument("--final_model", default="end_model.ckpt")
    parser.add_argument("--lowest_model", default="lowest.ckpt")
    parser.add_argument("--load_model", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--n_cum", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--process", type=int, default=1, choices=[0, 1])
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
    parser.add_argument("--out_names", nargs="+", default=["bandgap", "HOMO", "LUMO"],help="Names for each output dimension")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--n_mha", type=int, default=1)
    parser.add_argument("--n_alignn", type=int, default=2)
    parser.add_argument("--n_gnn", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--residual", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    args.residual = bool(args.residual)
    args.periodic = bool(args.periodic)
    args.process = bool(args.process)
    if args.save_dir is None:
        args.save_dir = osp.join(os.getcwd(), "output", "train")
    if args.run_name is None:
        args.run_name = f"{args.num_layers}_{args.n_mha}_{args.n_alignn}_{args.n_gnn}"
    main(args)
