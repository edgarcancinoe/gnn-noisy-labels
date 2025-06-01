# NoisyGNN: Robust GNNs Under Noisy Labels

This repository contains a PyTorch Geometric-based framework for training and evaluating Graph Neural Networks (GNNs) under noisy label conditions. The framework supports ensemble learning and model robustness strategies and was developed for molecular graph datasets.

Project developed for the Graph Classification with Noisy Labels Exam Hackaton for the Deep Learning class of the MSc in Artificial Intelligence of Sapienza University of Rome.

Jose Edgar Hernandez Cancino Estrada | 2223606 | hernandezcancinoestrada.2223606@studenti.uniroma1.it
Gianni Regina | 1972467 | regina.1972467@studenti.uniroma1.it

---

## Repository Structure

```bash
.
├── submission_models/    # Folder with our best models used for final submission
│   └── A/, B/, C/, D/    # One folder per dataset
├── checkpoints/          # Folder with trained model checkpoints
│   └── A/, B/, C/, D/    # One folder per dataset
├── datasets/             # Dataset files (.json.gz) for training and testing
├── logs/                 # Training logs
├── submission/           # Output CSV files with predictions
├── source/               # Source code: models, data loading, utils
│   ├── model.py
│   ├── utils.py
│   └── loadData.py
├── main.py               # Entry point for training and inference
├── requirements.txt      # Python and library dependencies
└── README.md             # Project documentation
```

---

## How to Use

> ⚠️ Ensure `torch_geometric` is correctly installed. You can use:

---

### Running Training and Prediction

#### Train a Model

```bash
python main.py --train_path datasets/B/train.json.gz --test_path datasets/B/test.json.gz
```

- This will train a model on the specified training set.
- Checkpoints will be saved to `checkpoints/B/`

#### Predict Only (Ensemble Mode)

```bash
python main.py --test_path datasets/B/test.json.gz
```

- Will automatically load all `.pth` files from `submission_models/B/`
- Performs ensemble averaging across all models present in the folder.
- Saves results to `submission/testset_B.csv`
- Can use checkpoint_models if function parameter `submission_models = False`

---

## Model and Ensemble Logic

- Different models have different architectures GIN/GINE-Virtual variants, ensemble, and different graph pooling mechanisms (mean, attention).
- Different combination or architectures, parameters and strategies were trained and the selected best-performing models were included in the final ensemble predicting model array, which is used to generate the test predictions.
- Different loss functions (CCE, Label-Smoothing CE, ERL, etc) and training algorithms (single GNN, 2-netork co-teaching) were employed. These can be chosen in `config.py`.
- The architecture is adjusted dynamically depending on the checkpoint name.
- Special configurations like ensemble submodels, GINs are automatically detected based on configuration.

---

## Output

- Trained model checkpoints: `checkpoints/B/model_B_epoch_10.pth`, etc.
- Predictions: `submission/testset_B.csv`
- Logs: `logs/B.log`

---

## Requirements

```bash
python
torch
torch-geometric
numpy
scikit-learn
pandas
```

```bash
!gdown --folder https://drive.google.com/drive/folders/1_np6HKijJ_0vaNoXCrtVo7z79vuCMCO_ -O datasets
!git clone --branch baselineCe https://github.com/Graph-Classification-Noisy-Label/hackaton.git
!pip install torch_geometric
!git clone https://github.com/edgarcancinoe/gnn-noisy-labels.git
```

### Some (early) experimentation reported results


#### Dataset A: Single GNN with different loss functions

| Try | Loss Function | Model        | Epochs | LR | Batch Size | Dropout | Num Layers | Emb Dim | Params |Validation Accuracy |
|:---:|:--------------|:-------------|:------:|:---:|:----------:|:-------:|:----------:|:-------:|:-------------------:|:-------------------:|
| 1 | CCE | gin-virtual | 40 | 0.01 | 32 | 0.1 | 5 | 300 | NA| **0.69**|
| 2 | CCE | gcn-virtual | 40 | 0.01 | 32 | 0.1 | 5 | 300 | NA| **0.62**|
| 3 | ELR|gin-virtual|40| 0.01 | 32 | 0.1 | 5 | 300 | β=0.7, λ=3.0| **0.55** |
| 4 | ELR|gcn-virtual|40| 0.01 | 32 | 0.1 | 5 | 300 |  β=0.7, λ=3.0| **0.57** |
| 5 | LS CCE | gin-virtual| 40 | 0.01  | 32 | 0.1 | 5 | 300 | P=0.2| **0.69** |
| 5.1| LS CCE | gin-virtual| `\|60\|` | `\|0.03\|`  | 32 | 0.1 | 5 | 300 | p=0.2| **0.70** |
| 5.1| LS CCE | gin-virtual| `\|60\|` | 0.01  | 32 | 0.1 | 5 | 300 | `\|p=0.35\|`| **XXXX** |
| 6 | LS CCE | gcn-virtual| 40 | 0.01  | 32 | 0.1 | 5 | 300 | p=0.2| **0.68** |

#### Dataset A: Basic Co-Teaching algorithm with different GNN and Loss functions


| Try | Loss Function | Model        | Epochs | LR | Batch Size | Dropout | Num Layers | Emb Dim | Params |Ensemble Validation Accuracy |
|:---:|:--------------|:-------------|:------:|:---:|:----------:|:-------:|:----------:|:-------:|:-------------------:|:-------------------:|
| 1 | LS CCE | gnc-virtual | 40 | 0.01 | 32 | 0.1 | 5 | 300 |  $p_{}=0.2$ $\:$ $T_k$=10, $\:$ τ = 0.2| **0.66**|
| 1 | LS CCE | gic-virtual | `60` | `0.001` | 32 | 0.1 | 5 | 300 |  $p_{}=$ `0.35` $\:$ $T_k$=`5`, $\:$ τ = `0.35`| **0.66**|
| 1 | LS CCE | gic-virtual | `60` | 0.01 | 32 | 0.1 | 5 | 300 |  $p_{}=$ `0.35` $\:$ $T_k$=10, $\:$ τ = `0.35`| **0.66**|


#### Visualization A: GIC - Virutal Ensemble: `noise_prob=0.35`, `noise_rate=0.35`, `ramp_up_epochs=5`, `adam_lr = 0.001`

gnn='gin-virtual', drop_ratio=0.1, num_layer=5, emb_dim=300, batch_size=32, epochs=60, baseline_mode=2, noise_prob=0.35, singleGNN=False, simpleCoTeaching=True, noise_rate=0.35, ramp_up_epochs=5


![](https://github.com/edgarcancinoe/gnn-noisy-labels/blob/main/img/Visualization%20A%20GIC%20-%20Virutal%20Ensemble.png)
![](https://github.com/edgarcancinoe/gnn-noisy-labels/blob/main/img/loss1.png)


#### Visualization A: GIC - Virutal Ensemble: `noise_prob=0.35`, `noise_rate=0.35`, `ramp_up_epochs=10`, `adam_lr=0.01`

gnn='gin-virtual', drop_ratio=0.1, num_layer=5, emb_dim=300, batch_size=32, epochs=60, baseline_mode=2, noise_prob=0.35, singleGNN=False, simpleCoTeaching=True, noise_rate=0.35, ramp_up_epochs=10

![](https://github.com/edgarcancinoe/gnn-noisy-labels/blob/main/img/Visualization%20A%20GIC%20-%20Virutal%20EnsembleLR0.01.png)
![](https://github.com/edgarcancinoe/gnn-noisy-labels/blob/main/img/loss2.png)

#### Dataset D: Co-Teaching with GINEconv implementation. Learning rate increased to 0.05 for ADAM.
![](https://github.com/edgarcancinoe/gnn-noisy-labels/blob/main/img/trainD.png)
![](https://github.com/edgarcancinoe/gnn-noisy-labels/blob/main/img/valD.png)
