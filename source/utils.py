import os
import torch
import pandas as pd
import tqdm
from source.model import *
from source.model import GNN, EnsembleGNN
from torch.utils.data import random_split, Dataset

import random
import numpy as np
def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_predictions(predictions, test_path):
    script_dir = os.getcwd()
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, ix):
        data = self.base[ix]
        # attach a tensor of the global index
        data.idx = torch.tensor(ix, dtype=torch.long)
        return data
    
from source.loadData import GraphDataset  # type: ignore}
from torch_geometric.loader import DataLoader

def get_data_loaders(train_path, batch_size=32, split_val=True):
    """
    Get train and validation data loaders.
    """
    # Load the dataset
    dataset = GraphDataset(train_path, transform=add_zeros)
    print(f"Dataset loaded with {len(dataset)} samples from {train_path}")
    # Create an indexed version of the dataset
    indexed_dataset = IndexedDataset(dataset)

    # Split the dataset into training and validation sets
    if split_val:
        train_size = int(0.8 * len(indexed_dataset))
        val_size = len(indexed_dataset) - train_size
        train_dataset, val_dataset = random_split(indexed_dataset, [train_size, val_size])
    else:
        train_dataset = indexed_dataset
        val_dataset = None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    return train_loader, None
### Model and loss selection
def get_loss(args, num_train_samples = None):
  if args.baseline_mode == 1:
    return CELossWithIndex()
  elif args.baseline_mode == 2:
    return LabelSmoothingCrossEntropyLoss(args.noise_prob)
  elif args.baseline_mode == 3:
    return ELRLoss(num_train_samples, 6)
  elif args.baseline_mode == 4:
    return SymmetricCrossEntropyLossWithIndex()
  elif args.baseline_mode == 5:
    print("Using Noisy CE LOSS")
    return NoisyCrossEntropyLoss(0.6)
  else:
    raise ValueError('Invalid baseline mode')

def build_gnn(args, device):
    if args.gnn == 'gin':
        return GNN(num_class=6, gnn_type='gin', num_layer=args.num_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=False, graph_pooling=args.graph_pooling).to(device)
    elif args.gnn == 'gin-virtual':
        return GNN(num_class=6, gnn_type='gin', num_layer=args.num_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=True, graph_pooling=args.graph_pooling).to(device)
    elif args.gnn == 'gcn':
        return GNN(num_class=6, gnn_type='gcn', num_layer=args.num_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=False, graph_pooling=args.graph_pooling).to(device)
    elif args.gnn == 'gcn-virtual':
        return GNN(num_class=6, gnn_type='gcn', num_layer=args.num_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=True, graph_pooling=args.graph_pooling).to(device)
    elif args.gnn == 'gine':
        return GNN(num_class=6, gnn_type='gine', num_layer=args.num_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=False, graph_pooling=args.graph_pooling).to(device)
    elif args.gnn == 'gine-virtual':
        return GNN(num_class=6, gnn_type='gine', num_layer=args.num_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=True, graph_pooling=args.graph_pooling).to(device)

    elif args.gnn == 'ensemble':
       #ensemble_paths = ['/kaggle/working/hackaton/checkpoints/model_weights.pth', '/kaggle/working/hackaton/othermodels/checkpoint.pth']
        # Carica i pesi salvati, MA con strict=False
        model1 = GNN(num_class=6, gnn_type='gine', num_layer=args.num_layer1, emb_dim=args.emb_dim1, drop_ratio=args.drop_ratio1, virtual_node=False, graph_pooling=args.graph_pooling).to(device)
        #pretrained_dict = torch.load("/kaggle/working/hackaton/checkpoints/model_weights.pth", map_location=device)
        #missing, unexpected = model1.load_state_dict(pretrained_dict, strict=False)
        #print("Chiavi NON trovate nel modello nuovo (verranno ignorate):\n", missing)
        #print("Chiavi “extra” nel file .pth (verranno ignorate):\n", unexpected)

        #model1.load_state_dict(torch.load("/kaggle/working/hackaton/checkpoints/model_weights.pth", map_location=device))
        model2 = GNN(num_class=6, gnn_type='gine', num_layer=args.num_layer2, emb_dim=args.emb_dim2,drop_ratio=args.drop_ratio2, virtual_node=False, graph_pooling=args.graph_pooling).to(device)
        #model2.load_state_dict(checkpoint['model_state_dict'])
        model3 = GNN(num_class=6, gnn_type='gine', num_layer=args.num_layer3, emb_dim=args.emb_dim3,drop_ratio=args.drop_ratio3, virtual_node=True,graph_pooling=args.graph_pooling).to(device)
        model4 = GNN(num_class=6, gnn_type='gcn', num_layer=args.num_layer4, emb_dim=args.emb_dim4,drop_ratio=args.drop_ratio4, virtual_node=True,graph_pooling=args.graph_pooling).to(device)
        model5 = GNN(num_class=6, gnn_type='gcn', num_layer=args.num_layer5, emb_dim=args.emb_dim5,drop_ratio=args.drop_ratio5, virtual_node=False,graph_pooling=args.graph_pooling).to(device)

        ensemble_weights =[0.2, 0.2, 0.2, 0.2, 0.2]

        return EnsembleGNN(models=[model1, model2, model3, model4, model5],weights=ensemble_weights).to(device)

    else:
        raise ValueError(f'Invalid GNN type: {args.gnn}')

def load_checkpoint(model, ckpt, device):
    """
    Load model checkpoint from the specified path.
    """
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint file '{ckpt}' does not exist.")
    
    # Load the state dict
    state_dict = torch.load(ckpt, map_location=device)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=False)
    
    logging.info(f"Checkpoint loaded successfully from '{ckpt}'")
    return model