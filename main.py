#!/usr/bin/env python3
import os
import numpy as np
import sys
import argparse
import glob
import logging
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from source.model import *
from source.utils import save_predictions, build_model, load_checkpoint
from source.loadData import GraphDataset  # type: ignore
from source.utils import set_seed, get_data_loaders, train_model  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict GNN on molecular datasets")
    parser.add_argument("--train_path", type=str, default=None, help="Path to train.json.gz (optional)")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test.json.gz (required)")
    return parser.parse_args()

def extract_folder_name(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))

def setup_logging(folder_name):
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"{folder_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_model_on_dataset(train_path, folder_name, device):
    train_loader, val_loader = get_data_loaders(train_path, batch_size=32, split_val=True)
    logging.info(f"Loaded train and validation data from '{train_path}'")

    model = build_model(num_class=6, gnn_type="gine", num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=True).to(device)
    logging.info("Model instantiated for training")

    checkpoint_dir = "checkpoints"
    best_val_acc = 0.0
    num_epochs = 100
    save_interval = max(1, num_epochs // 5)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, device, epoch)
        val_loss, val_acc = train_model(model, val_loader, device, epoch, validate=True)

        logging.info(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if epoch % save_interval == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"model_{folder_name}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved: {ckpt_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(checkpoint_dir, f"model_{folder_name}_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            logging.info(f"New best model saved: {best_ckpt}")

def predict_on_dataset(test_path, folder_name, device):
    checkpoint_dir = "checkpoints"
    pattern_epoch = os.path.join(checkpoint_dir, f"model_{folder_name}_epoch_*.pth")
    epoch_ckpts = sorted(glob.glob(pattern_epoch), key=os.path.getmtime)

    if not epoch_ckpts:
        logging.error(f"No checkpoints found for folder '{folder_name}'. Cannot predict.")
        sys.exit(1)

    logging.info(f"Found {len(epoch_ckpts)} checkpoint(s) for ensemble prediction.")

    # Load test data
    test_loader = DataLoader(
        GraphDataset(test_path, transform=lambda x: x),
        batch_size=32,
        shuffle=False
    )

    all_probs_list = []
    all_ids = None  # To be filled once

    for ckpt in epoch_ckpts:
        model = build_model(num_class=6, gnn_type="gine", num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=True).to(device)
        load_checkpoint(model, ckpt, device)
        model.eval()
        logging.info(f"Loaded checkpoint: {ckpt}")

        probs_all_batches = []
        ids_this_model = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                probs_all_batches.append(probs.cpu())

                if hasattr(batch, "idx") and all_ids is None:
                    ids_this_model.extend(batch.idx.cpu().tolist())

        all_probs_list.append(torch.cat(probs_all_batches, dim=0).numpy())

        if all_ids is None and ids_this_model:
            all_ids = ids_this_model

    # Stack shape: [num_models, num_samples, num_classes]
    probs_stack = np.stack(all_probs_list, axis=0)
    avg_probs = np.mean(probs_stack, axis=0)
    y_pred = avg_probs.argmax(axis=1)

    # Create and save DataFrame
    if all_ids:
        df = pd.DataFrame({"id": all_ids, "prediction": y_pred})
    else:
        df = pd.DataFrame({"prediction": y_pred})

    out_csv = os.path.join("submission", f"testset_{folder_name}.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Ensemble predictions saved to {out_csv}")


def main():
    args = parse_args()
    folder_name = extract_folder_name(args.test_path)
    setup_logging(folder_name)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("submission", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.train_path:
        train_folder = extract_folder_name(args.train_path)
        if train_folder != folder_name:
            logging.error(f"Mismatch: train set folder '{train_folder}' ≠ test set folder '{folder_name}'")
            sys.exit(1)
        train_model_on_dataset(args.train_path, folder_name, device)

    predict_on_dataset(args.test_path, folder_name, device)

if __name__ == "__main__":
    main()