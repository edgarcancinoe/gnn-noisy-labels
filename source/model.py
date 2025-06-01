import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import GINEConv as PyG_GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

import tqdm
import os
import torch.nn as nn
import logging

from source.utils import evaluate
## Losses

class SymmetricCrossEntropyLossWithIndex(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction='none')  # per-sample CE

    def per_sample(self, indices, logits, targets):
        ce_loss = self.ce(logits, targets)  # [B]

        # Reverse Cross Entropy (RCE)
        pred_probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)  # [B, C]
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).float().clamp(min=1e-4, max=1.0)  # [B, C]
        rce_loss = -torch.sum(pred_probs * torch.log(target_one_hot), dim=1)  # [B]

        # Combine
        return self.alpha * ce_loss + self.beta * rce_loss  # [B]

    def forward(self, indices, logits, targets):
        return self.per_sample(indices, logits, targets).mean()

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

class CELossWithIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # no reduction so we can get a vector of losses
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def per_sample(self, indices, logits, targets):
        # ignore indices, just return per-sample CE
        return self.ce(logits, targets)         # shape [B]

    def forward(self, indices, logits, targets):
        # mean over the batch
        return self.per_sample(indices, logits, targets).mean()

class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        # label_smoothing gives us a per-sample loss vector
        self.ce_sm = torch.nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=p_noisy
        )

    def per_sample(self, indices, logits, targets):
        return self.ce_sm(logits, targets)     # shape [B]

    def forward(self, indices, logits, targets):
        return self.per_sample(indices, logits, targets).mean()

class ELRLoss(torch.nn.Module):
    """
    Early‐Learning Regularization Loss.
    Adapted from:
      Sheng Liu et al., “Early‐Learning Regularization Prevents Memorization of Noisy Labels”
    """
    def __init__(self, num_samples, num_classes, ema_momentum=0.7, lambda_elr=3.0):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.lambda_elr   = lambda_elr
        # buffer to store per‐sample EMA targets
        self.register_buffer('Q', torch.zeros(num_samples, num_classes))

    def per_sample(self, index, output, label):
        # 1) per-sample CE
        ce_vec = F.cross_entropy(output, label, reduction='none')   # [B]

        # 2) softmax + clamp
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        # 3) update EMA buffer for these indices
        with torch.no_grad():
            y_pred_det = y_pred.detach()
            y_pred_det = y_pred_det / y_pred_det.sum(dim=1, keepdim=True)
            self.Q[index] = (
                self.ema_momentum * self.Q[index]
              + (1.0 - self.ema_momentum) * y_pred_det
            )

        # 4) per-sample ELR regularizer = log(1 - <Q[i], p_i>)
        inner = (self.Q[index] * y_pred).sum(dim=1)   # [B]
        elr_vec = torch.log(1.0 - inner)              # [B]

        # 5) combine
        return ce_vec + self.lambda_elr * elr_vec     # [B]

    def forward(self, index, output, label):
        # mean of the per-sample losses
        return self.per_sample(index, output, label).mean()
    

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINEConvCustom(torch.nn.Module):
    """
    GINEConv + a 7→emb_dim edge encoder.
    """
    def __init__(self, emb_dim):
        super().__init__()
        # 1) embed your 7-dim edge_attr → emb_dim
        self.edge_encoder = torch.nn.Linear(7, emb_dim)
        # 2) the MLP that the original GINE paper uses
        mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim),
            torch.nn.BatchNorm1d(2*emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_dim, emb_dim),
        )
        # 3) build the actual GINE layer
        self.conv = PyG_GINEConv(nn=mlp, train_eps=True)

    def forward(self, x, edge_index, edge_attr):
        # encode edge_attr → same dim as x
        e = self.edge_encoder(edge_attr)
        # delegate to PyG’s GINEConv
        return self.conv(x, edge_index, e)


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'gine':
                self.convs.append(GINEConvCustom(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'gine':
                self.convs.append(GINEConvCustom(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300,
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = None):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


class EnsembleGNN(nn.Module):
    def __init__(self, models: list[nn.Module], weights: list[float]):
        """
        models: lista di modelli nn.Module (es. [model1, model2, ...])
        weights: lista di float, stessa lunghezza di models, che sommano a 1.0
        """
        super().__init__()
        assert len(models) == len(weights), "Deve esserci un peso per ciascun modello"
        assert abs(sum(weights) - 1.0) < 1e-6, "I pesi devono sommare a 1"
        # Registra i modelli in un ModuleList, così PyTorch li scopre tutti
        self.models = nn.ModuleList(models)
        # I pesi non sono parametri, ma li memorizziamo qui
        #self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        init_raw = torch.tensor(weights, dtype=torch.float32)
        self.raw_weights = nn.Parameter(init_raw)

    def forward(self, data):
        # update model importance weights
        normalized_weights = F.softmax(self.raw_weights, dim=0)
        # Calcola le logit di ogni modello
        logits_list = [m(data) for m in self.models]  # ognuno [B, C]
        # Esegui la somma pesata
        ensembled = sum(w * l for w, l in zip(self.raw_weights, logits_list))
        return ensembled

## Co-Teaching functions
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def make_forget_rate_fn(noise_rate, Tk):
    def forget_rate(epoch):
        return noise_rate * min(epoch / Tk, 1.0)
    return forget_rate

def evaluate_ensemble(val_loader, model_f, model_g, device):
    model_f.eval()
    model_g.eval()
    correct = total = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            # get probabilities from each peer
            p_f = F.softmax(model_f(data), dim=1)
            p_g = F.softmax(model_g(data), dim=1)
            # average them
            p_avg = (p_f + p_g) / 2
            pred = p_avg.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total   += data.y.size(0)
    return correct / total

def train_coteaching(
    model_f, model_g, train_loader, val_loader, opt_f, opt_g, crit_f, crit_g, device,
    num_epochs, checkpoints_folder, checkpoint_intervals, at_least, test_dir_name, noise_rate, ramp_up_epochs):
    # move everything once
    model_f.to(device)
    model_g.to(device)
    crit_f .to(device)
    crit_g .to(device)

    forget_rate_fn = make_forget_rate_fn(noise_rate, ramp_up_epochs)

    # storage for plotting
    train_losses_f, train_accs_f = [], []
    val_losses_f,   val_accs_f   = [], []
    train_losses_g, train_accs_g = [], []
    val_losses_g,   val_accs_g   = [], []
    ensemble_accs               = []

    best_val_acc_f = best_val_acc_g = 0.0

    for epoch in range(1, num_epochs + 1):
        model_f.train()
        model_g.train()

        fr = forget_rate_fn(epoch)
        remember_rate = 1.0 - fr

        total_loss_f = total_loss_g = 0.0
        correct_f = correct_g = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for data in pbar:
            data = data.to(device)
            idxs = data.idx
            y    = data.y

            # 1) forward
            out_f = model_f(data)
            out_g = model_g(data)

            # 2) per-sample losses
            losses_f = crit_f.per_sample(idxs, out_f, y)
            losses_g = crit_g.per_sample(idxs, out_g, y)

            # 3) pick small-loss subsets
            batch_size   = y.size(0)
            num_remember = int(remember_rate * batch_size)

            _, sorted_f = torch.sort(losses_f)
            _, sorted_g = torch.sort(losses_g)
            sel_f = sorted_g[:num_remember]   # f trains on g’s small-loss
            sel_g = sorted_f[:num_remember]   # g trains on f’s small-loss

            # 4) compute peer-selected losses
            loss_f = crit_f(idxs[sel_f], out_f[sel_f], y[sel_f])
            loss_g = crit_g(idxs[sel_g], out_g[sel_g], y[sel_g])

            # 5) backward + step
            opt_f.zero_grad(); loss_f.backward(); opt_f.step()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            # 6) accumulate train metrics
            total_loss_f += loss_f.item()
            total_loss_g += loss_g.item()
            pred_f = out_f.argmax(dim=1)
            pred_g = out_g.argmax(dim=1)
            correct_f += (pred_f == y).sum().item()
            correct_g += (pred_g == y).sum().item()
            total     += y.size(0)

            pbar.set_postfix({
                "fr": f"{fr:.3f}",
                "l_f": f"{loss_f.item():.4f}",
                "l_g": f"{loss_g.item():.4f}"
            })

        # end of epoch: compute averages
        train_loss_f = total_loss_f / len(train_loader)
        train_loss_g = total_loss_g / len(train_loader)
        train_acc_f  = correct_f    / total
        train_acc_g  = correct_g    / total

        train_losses_f.append(train_loss_f)
        train_losses_g.append(train_loss_g)
        train_accs_f   .append(train_acc_f)
        train_accs_g   .append(train_acc_g)

        # validation for each peer
        val_loss_f, val_acc_f = evaluate(val_loader, model_f, device, calculate_accuracy=True)
        val_loss_g, val_acc_g = evaluate(val_loader, model_g, device, calculate_accuracy=True)

        val_losses_f.append(val_loss_f)
        val_losses_g.append(val_loss_g)
        val_accs_f   .append(val_acc_f)
        val_accs_g   .append(val_acc_g)

        # **ensemble** on validation
        ens_acc = evaluate_ensemble(val_loader, model_f, model_g, device)
        ensemble_accs.append(ens_acc)

        print(f"[Epoch {epoch:2d}/{num_epochs}] "
              f"F train: {train_loss_f:.3f}/{train_acc_f:.3f} | "
              f"val: {val_loss_f:.3f}/{val_acc_f:.3f}    "
              f"G train: {train_loss_g:.3f}/{train_acc_g:.3f} | "
              f"val: {val_loss_g:.3f}/{val_acc_g:.3f}    "
              f"ENS val_acc: {ens_acc:.3f}")

        # checkpoint peers at specified epochs
        if epoch in checkpoint_intervals:
            ckf = os.path.join(checkpoints_folder, f"{test_dir_name}_f_epoch_{epoch}.pth")
            ckg = os.path.join(checkpoints_folder, f"{test_dir_name}_g_epoch_{epoch}.pth")
            torch.save(model_f.state_dict(), ckf)
            torch.save(model_g.state_dict(), ckg)
            print(f"  → Saved epoch {epoch} checkpoints")

        # save best but only if good enough
        if val_acc_f > best_val_acc_f:
            best_val_acc_f = val_acc_f
            if val_acc_f > at_least:
              torch.save(model_f.state_dict(),
                        os.path.join(checkpoints_folder, f"{test_dir_name}_f_best.pth"))
        if val_acc_g > best_val_acc_g:
            best_val_acc_g = val_acc_g
            if val_acc_g > at_least:
              torch.save(model_g.state_dict(),
                        os.path.join(checkpoints_folder, f"{test_dir_name}_g_best.pth"))

    # return everything for plotting (including ensemble)
    return (
        train_losses_f, train_accs_f, val_losses_f, val_accs_f,
        train_losses_g, train_accs_g, val_losses_g, val_accs_g,
        ensemble_accs
    )

## Self-Supervised learnign functions
class ContrastivePPADataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, aug1, aug2):
        self.base = base_dataset
        self.aug1 = aug1
        self.aug2 = aug2

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        # produce two random “views” of the same graph
        g1 = self.aug1(data.clone())
        g2 = self.aug2(data.clone())
        return g1, g2
import torch
from torch_geometric.utils import dropout_edge, dropout_node, subgraph
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

class RandomEdgeDrop:
    def __init__(self, p: float):
        self.p = p
    def __call__(self, data: Data) -> Data:
        # randomly drop edges
        edge_index, edge_mask = dropout_edge(
            data.edge_index, p=self.p, force_undirected=True, training=self.p>0
        )
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr  = data.edge_attr[edge_mask]
        return data

class RandomNodeDrop:
    def __init__(self, p: float):
        self.p = p
    def __call__(self, data: Data) -> Data:
        # randomly drop nodes (and incident edges), then relabel
        edge_index, edge_mask, node_mask = dropout_node(
            data.edge_index, p=self.p, num_nodes=data.num_nodes,
            training=self.p>0, relabel_nodes=True
        )
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr  = data.edge_attr[edge_mask]
        if data.x is not None:
            data.x          = data.x[node_mask]
        data.num_nodes   = data.x.size(0)
        return data

class RandomFeatureMask:
    def __init__(self, p: float):
        self.p = p
    def __call__(self, data: Data) -> Data:
        # mask out p-fraction of features
        if data.x is not None:
            mask = (torch.rand_like(data.x.float()) > self.p).float()
            data.x = data.x.float() * mask
        return data

class RandomSubgraph:
    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio
    def __call__(self, data: Data) -> Data:
        # sample a subset of nodes, take induced subgraph
        N = data.num_nodes
        k = int(self.keep_ratio * N)
        perm = torch.randperm(N)
        keep = perm[:k]
        edge_index, edge_attr = subgraph(keep, data.edge_index,
                                         data.edge_attr, relabel_nodes=True)
        data.edge_index = edge_index
        data.edge_attr  = edge_attr
        data.x          = data.x[keep] if data.x is not None else None
        data.num_nodes  = k
        return data

aug1 = Compose([
    RandomNodeDrop(p=0.1),
    RandomEdgeDrop(p=0.2),
])

aug2 = Compose([
    RandomSubgraph(keep_ratio=0.8),
])

class GraphCLModel(torch.nn.Module):
    def __init__(self, base_gnn, emb_dim, proj_dim=512):
        super().__init__()
        self.encoder = base_gnn
        # drop the old classification head
        self.encoder.graph_pred_linear = torch.nn.Identity()
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, proj_dim),
        )
    def forward(self, batch):
        h = self.encoder(batch)             # [B, emb_dim]
        z = self.projector(h)               # [B, proj_dim]
        return F.normalize(z, dim=1)


class GraphClassifier(torch.nn.Module):
    def __init__(self, pretrain_encoder: torch.nn.Module, emb_dim: int, num_classes: int):
        super().__init__()
        # reuse the pre-trained encoder
        self.encoder = pretrain_encoder
        # add a fresh linear head
        self.classifier = torch.nn.Linear(emb_dim, num_classes)

    def forward(self, batch):
        # assume self.encoder returns [B, emb_dim]
        h = self.encoder(batch)
        return self.classifier(h)


def nt_xent(z1, z2, temp=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)        # [2B, D]
    sim = (z @ z.T) / temp                # [2B, 2B]

    # create a boolean mask with False on the diagonal, True elsewhere
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)

    # compute exp(similarity) only over off-diagonals
    exp_sim = torch.exp(sim) * mask.float()

    # positive pairs (i, i+B) and (i+B, i)
    pos = torch.cat([sim.diag(B), sim.diag(-B)], dim=0)

    # NT-Xent loss: -log( exp(sim_pos) / sum_j exp(sim_ij) )
    # which is: -pos + log(sum_j exp_sim)
    logsumexp = torch.log(exp_sim.sum(1))
    loss = -pos + logsumexp
    return loss.mean()

## Training and building functions
def gnn_epoch(data_loader, model, optimizer, criterion, device,
          save_checkpoints, checkpoint_path, current_epoch):
    """
      Train GNN for 1 epoch
    """

    # Set model to training mode (enables dropout, batch-norm, etc.)
    model.train()

    # Initialize accumulators for loss and accuracy
    total_loss = correct = total = 0

    # Iterate over each batch (use tqdm for visualization)
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        # Send to device
        data = data.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        output = model(data)
        # Compute loss between predictions and true labels
        loss = criterion(data.idx.to(device), output, data.y)
        # Backward pass: compute gradients
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate loss for this batch
        total_loss += loss.item()
        # Compute predicted classes and compare to true labels
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    if save_checkpoints:
        # Construct checkpoint filename with epoch number (1-based)
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    # Compute average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion,
              device, num_epochs, checkpoints_folder, checkpoint_intervals, at_least, test_dir_name):

  train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
  best_val_accuracy = 0.0
  for epoch in range(num_epochs):
      
      checkpoint_path = os.path.join(checkpoints_folder, f"model_{test_dir_name}")

      train_loss, train_acc = gnn_epoch(
          train_loader, model, optimizer, criterion, device,
          save_checkpoints=(epoch + 1 in checkpoint_intervals),
          checkpoint_path=checkpoint_path,
          current_epoch=epoch
      )

      val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)

      print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
      logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

      train_losses.append(train_loss)
      train_accuracies.append(train_acc)
      val_losses.append(val_loss)
      val_accuracies.append(val_acc)


      if val_acc > best_val_accuracy:
          best_val_accuracy = val_acc
          print('Saving model...', end=' ')
          torch.save(model.state_dict(), checkpoint_path)
          print(f"Best model updated and saved at {checkpoint_path}")

  return (train_losses, train_accuracies, val_losses, val_accuracies)


