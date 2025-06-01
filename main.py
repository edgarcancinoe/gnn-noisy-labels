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
from source.utils import save_predictions, build_gnn, load_checkpoint
from source.loadData import GraphDataset  # type: ignore
from source.utils import set_seed, get_data_loaders  # type: ignore
import copy
from source.config import *

# File configuration

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

    model = build_gnn(num_class=6, gnn_type="gine", num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=True).to(device)
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

def predict_on_dataset(test_path, folder_name, device, args):
    """
    Loads all .pth files from the corresponding dataset folder (A, B, C, D),
    runs ensemble prediction using equal weights, and saves the result.
    """
    # Load all checkpoints matching the dataset folder name
    checkpoint_dir = os.path.join("checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    if not ckpt_paths:
        logging.error(f"No .pth files found in '{checkpoint_dir}'.")
        sys.exit(1)
        
    logging.info(f"Found {len(ckpt_paths)} checkpoint(s) for dataset {folder_name}.")

    # Prepare test loader
    test_loader = DataLoader(
        GraphDataset(test_path, transform=lambda x: x),
        batch_size=args.batch_size,
        shuffle=False
    )

    model_test_probs = {}
    y_ids = None

    for ckpt_path in ckpt_paths:
        name = os.path.basename(ckpt_path)
        logging.info(f"Processing {name}")
        
        # Heuristics to select architecture config
        gine = 'gine' in name.lower()
        older300 = 'older' in name.lower()

        argss = copy.deepcopy(args)

        if 'gine' in ckpt_path or gine:
            logging.info("→ GINE-Virtual")
            argss.emb_dim = 128
            argss.num_layer = 2
            argss.gnn = 'gine-virtual'
        elif 'B_054' in ckpt_path or ckpt_path.endswith("model_B_best (1).pth"):
            logging.info("→ Ensemble config")
            argss.gnn = 'ensemble'
            argss.graph_pooling = 'attention'
            argss.emb_dim = 64
            argss.drop_ratio1 = 0.1; argss.num_layer1 = 2; argss.emb_dim1 = 64
            argss.drop_ratio2 = 0.1; argss.num_layer2 = 2; argss.emb_dim2 = 32
            argss.drop_ratio3 = 0.5; argss.num_layer3 = 2; argss.emb_dim3 = 32
            argss.drop_ratio4 = 0.1; argss.num_layer4 = 4; argss.emb_dim4 = 32
            argss.drop_ratio5 = 0.1; argss.num_layer5 = 5; argss.emb_dim5 = 32
        elif older300:
            logging.info("→ Older 300 config")
            argss.emb_dim = 300
            argss.num_layer = 5
            argss.gnn = 'gin'
            argss.singleGNN = False
            argss.simpleCoTeaching = True
        else:
            logging.info("→ Default GIN")
            argss.gnn = 'gin'

        # Load and run model
        model = build_gnn(argss, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        all_probs = []
        ids_this_model = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                if hasattr(batch, "idx") and y_ids is None:
                    ids_this_model.extend(batch.idx.cpu().tolist())

        model_test_probs[name] = torch.cat(all_probs, dim=0).numpy()
        if y_ids is None and ids_this_model:
            y_ids = ids_this_model

    # Stack all model predictions
    model_names = list(model_test_probs.keys())
    probs_stack = np.stack([model_test_probs[n] for n in model_names], axis=0)

    # Equal weights
    weights = np.ones(len(model_names), dtype=np.float32) / len(model_names)
    weighted_probs = np.tensordot(weights, probs_stack, axes=([0], [0]))
    y_pred = weighted_probs.argmax(axis=1)

    # Save output
    df = pd.DataFrame({"id": y_ids, "prediction": y_pred}) if y_ids else pd.DataFrame({"prediction": y_pred})
    out_csv = os.path.join("submission", f"testset_{folder_name}.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Ensemble predictions saved to {out_csv}")
    
def main():
    args = parse_args()
    config_args.test_path = args.test_path
    config_args.train_path = args.train_path
    
    # Test dir
    test_dir_name = os.path.basename(os.path.dirname(config_args.test_path))
 
    # Log configuration
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())


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
    elif args.test_path:
        predict_on_dataset(args.test_path, folder_name, device, config_args)
    else:
        logging.error("No training or testing path provided. Cannot train model.")
        sys.exit(1)
if __name__ == "__main__":
    main()