#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import logging
import torch
import pandas as pd
from torch_geometric.loader import DataLoader  
from source.model import *
from source.utils import save_predictions, build_model
from source.loadData import GraphDataset # type: ignore
from source.utils import set_seed        # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict GNN on molecular datasets")
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Path to train.json.gz (optional; if omitted, only prediction is run)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test.json.gz (always required)",
    )
    return parser.parse_args()

def extract_folder_name(path):
    """
    Given a path like "./datasets/A/test.json.gz", returns "A".
    """
    return os.path.basename(os.path.dirname(os.path.abspath(path)))

def setup_logging(folder_name):
    """
    Creates logs/ if needed and configures Python logging to write to logs/<folder_name>.log.
    """
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

def main():
    args = parse_args()
    folder_name = extract_folder_name(args.test_path)
    setup_logging(folder_name)

    # Ensure required directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("submission", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # If training is requested
    if args.train_path:
        # Verify that train_path corresponds to same folder
        train_folder = extract_folder_name(args.train_path)
        if train_folder != folder_name:
            logging.error(
                f"train_path is in folder '{train_folder}' but test_path is in '{folder_name}'. "
                "They must match."
            )
            sys.exit(1)

        # Load train + validation data loaders
        train_loader, val_loader = get_data_loaders(args.train_path, batch_size=32, split_val=True)
        logging.info(f"Loaded train and validation data from '{args.train_path}'")

        # Build model
        model = build_model(num_class=6, gnn_type="gine", num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=True)
        model = model.to(device)
        logging.info("Model instantiated")

        # Training loop: save checkpoints into checkpoints/model_<folder>_epoch_<n>.pth
        checkpoint_dir = "checkpoints"
        best_val_acc = 0.0
        num_epochs = 100
        save_interval = max(1, num_epochs // 5)  # ensure at least 5 checkpoints

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_model(model, train_loader, device, epoch)
            val_loss, val_acc = train_model(model, val_loader, device, epoch, validate=True)

            logging.info(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save checkpoint at intervals
            if epoch % save_interval == 0 or epoch == num_epochs:
                ckpt_path = os.path.join(checkpoint_dir, f"model_{folder_name}_epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Checkpoint saved: {ckpt_path}")

            # Save best model (by val_acc) at each validation
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_ckpt = os.path.join(checkpoint_dir, f"model_{folder_name}_best.pth")
                torch.save(model.state_dict(), best_ckpt)
                logging.info(f"New best model saved: {best_ckpt}")

    # Prediction mode (either after training or standalone)
    # Find the checkpoint to load:
    checkpoint_dir = "checkpoints"
    pattern_best = os.path.join(checkpoint_dir, f"model_{folder_name}_best.pth")
    pattern_epoch = os.path.join(checkpoint_dir, f"model_{folder_name}_epoch_*.pth")

    if os.path.isfile(pattern_best):
        ckpt_to_load = pattern_best
        logging.info(f"Loading best checkpoint: {ckpt_to_load}")
    else:
        # Fallback: pick latest epoch checkpoint
        epoch_ckpts = sorted(glob.glob(pattern_epoch), key=os.path.getmtime)
        if not epoch_ckpts:
            logging.error(f"No checkpoints found for folder '{folder_name}'. Cannot predict.")
            sys.exit(1)
        ckpt_to_load = epoch_ckpts[-1]
        logging.info(f"No 'best' checkpoint found. Loading latest epoch checkpoint: {ckpt_to_load}")

    # Rebuild the same model architecture before loading
    model = build_model(num_class=6, gnn_type="gine", num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=True)
    model = model.to(device)
    load_checkpoint(model, ckpt_to_load, device)
    model.eval()
    logging.info("Model loaded for prediction")

    # Load test dataset and run predictions
    test_loader = DataLoader(
        GraphDataset(args.test_path, transform=lambda x: x),  # replace with actual transform if needed
        batch_size=32,
        shuffle=False
    )

    all_preds = []
    all_ids = []  # if your dataset returns an ID per graph
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(model, batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            if hasattr(batch, "idx"):
                all_ids.extend(batch.idx.cpu().tolist())

    # Build DataFrame: if you need an "ID" column, use all_ids; otherwise index by row number
    if all_ids:
        df = pd.DataFrame({"id": all_ids, "prediction": all_preds})
    else:
        df = pd.DataFrame({"prediction": all_preds})

    out_csv = os.path.join("submission", f"testset_{folder_name}.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Predictions saved to {out_csv}")

if __name__ == "__main__":
    main()