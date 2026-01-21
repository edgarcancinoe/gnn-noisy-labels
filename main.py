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
from source.utils import save_predictions, build_gnn, load_checkpoint, get_loss, train_gnn, train_coteaching  # type: ignore
from source.loadData import GraphDataset  # type: ignore
from source.utils import set_seed, get_data_loaders, add_zeros  # type: ignore
import copy
from source.config import *

# File configuration

def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict GNN on molecular datasets")
    parser.add_argument("--train_path", type=str, default=None, help="Path to train.json.gz (optional)")
    parser.add_argument("--test_path", type=str, required=False, help="Path to test.json.gz (required)")
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

def train_model_on_dataset(train_path, folder_name, device, args):
    print('Starting training sequence.')
    train_loader, val_loader = get_data_loaders(train_path, batch_size=32, split_val=True)
    logging.info(f"Loaded train and validation data from '{train_path}'")
    print('Data has ben loaded and graph object constructed.')

    logging.info("Model instantiated for training")

    # --- Prepare objects ------------------------------------------------------
    num_epochs = args.epochs
    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]
    training_params = ()

    # --- Prepare objects ------------------------------------------------------
    num_epochs = args.epochs
    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]

    training_params = ()
    checkpoints_folder = os.path.join("checkpoints", folder_name)
    os.makedirs(checkpoints_folder, exist_ok=True)
    train_size = len(train_loader.dataset)

    # Choose models and training algorithm -------------------------------------
    # --------------------------------------------------------------------------
    assert not (args.singleGNN and args.simpleCoTeaching)
    training_fn = lambda: None
    # Case for training a single GNN -------------------------------------------
    if args.singleGNN:
      # Obtain GNN model
      model = build_gnn(args, device)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
      # Choose Loss function according to parameters
      criterion = get_loss(args, num_train_samples=train_size).to(device)
      training_fn = train_gnn
      training_params = (model, train_loader, val_loader, optimizer, criterion, device, num_epochs, checkpoints_folder, checkpoint_intervals, at_least, folder_name)
      print('Training Single GNN with arguments:\n', args)

    # Case for perform simple co-teaching --------------------------------------
    if args.simpleCoTeaching:
      model_f = build_gnn(args, device)
      model_g = build_gnn(args, device)
      opt_f = torch.optim.Adam(model_f.parameters(), lr=adams_lr)
      opt_g = torch.optim.Adam(model_g.parameters(), lr=adams_lr)
      loss_f = get_loss(args, num_train_samples=train_size).to(device)
      loss_g = get_loss(args, num_train_samples=train_size).to(device)
      training_fn = train_coteaching
      training_params = (model_f, model_g, train_loader, val_loader, opt_f, opt_g, loss_f, loss_g, device, \
                         num_epochs, checkpoints_folder, checkpoint_intervals, at_least, folder_name, args.noise_rate, args.ramp_up_epochs)
      print('Training Co-Teaching algorithm with arguments', args)

    results = training_fn(*training_params)


def predict_on_dataset(test_path, folder_name, device, args, submission_models=True):
    """
    Loads all .pth files from the corresponding dataset folder (A, B, C, D),
    runs ensemble prediction using equal weights, and saves the result.
    """
    # Load all checkpoints matching the dataset folder name
    if submission_models:
        logging.info("Using submission models for prediction.")
        checkpoint_dir = os.path.join("submission_models", folder_name)
    else:
        checkpoint_dir = os.path.join("checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    if not ckpt_paths:
        logging.error(f"No .pth files found in '{checkpoint_dir}'.")
        sys.exit(1)
        
    logging.info(f"Found {len(ckpt_paths)} model(s) for dataset {folder_name}.")

    # Prepare test loader
    test_loader = DataLoader(
        GraphDataset(test_path, transform=add_zeros),
        batch_size=args.batch_size,
        shuffle=False
    )

    model_test_probs = {}
    y_ids = None

    for ckpt_path in ckpt_paths:
        name = os.path.basename(ckpt_path)
        logging.info(f"Processing {name}")
        
        # Heuristics to select architecture config
        gine_list = ['model_C_best.pth', 'old_model_C_best.pth']
        gine = 'gine' in name.lower() or (name in gine_list)

        older300 = 'older' in name.lower()

        argss = copy.deepcopy(args)

        if 'gine' in ckpt_path or gine:
            logging.info("→ GINE-Virtual")
            if '/D/' in ckpt_path:
                argss.emb_dim = 512
                argss.gnn = 'gine-virtual'
                argss.num_layer = 5
            elif '/C/' in ckpt_path:
                argss.emb_dim = 128
                argss.gnn = 'gine'
                argss.num_layer = 2
            else:
                argss.emb_dim = 128
                argss.num_layer = 2
                argss.gnn = 'gine-virtual'
        elif name == 'gian-gin-virtual-model_D_best.pth':
            argss.gnn = 'gin-virtual'
            argss.emb_dim = 64
            argss.graph_pooling = 'attention'
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
    save_predictions(y_pred, test_path)

    
def main():
    args = parse_args()
    config_args.test_path = args.test_path
    config_args.train_path = args.train_path
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("submission", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.train_path:
        folder_name = extract_folder_name(args.train_path)
        setup_logging(folder_name)
        train_model_on_dataset(args.train_path, folder_name, device, args=config_args)

    elif args.test_path:
        # Test dir
        test_dir_name = os.path.basename(os.path.dirname(config_args.test_path))
        folder_name = extract_folder_name(args.test_path)
        setup_logging(folder_name)
        # Log configuration
        logs_folder = os.path.join(script_dir, "logs", test_dir_name)
        log_file = os.path.join(logs_folder, "training.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
        predict_on_dataset(args.test_path, folder_name, device, config_args)
    else:
        logging.error("No training or testing path provided. Cannot train model.")
        sys.exit(1)
if __name__ == "__main__":
    main()
