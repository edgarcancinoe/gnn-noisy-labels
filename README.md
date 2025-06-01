# MessyGNN: Robust GNNs Under Noisy Labels

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

- Different models have different architectures GIN/GINE-Virtual variants, ensemble, and different graph_pooling mechanisms.
- Different loss functions and training algorithms (single GNN, 2-netork co-teaching) were employed, and can be chosen in `config.py`
- The architecture is adjusted dynamically depending on the checkpoint name.
- Special configurations like ensemble submodels,, GINs are automatically detected based on configuration.

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
