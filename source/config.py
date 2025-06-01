import os
import sys
import argparse
import logging
import torch

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
PRESET_ARGS = {
    'num_checkpoints': 10,
    'device':      0,
    'gnn':         'gin',
    'drop_ratio':  0.2,
    'num_layer':   2,
    'emb_dim':     128,
    'batch_size':  32,
    'epochs':      100,
    'baseline_mode': 2, # 1 = CCE, 2 = LS CCE, 3 = ELR
    'noise_prob':  0.2, # Used for Label smoothing CCE
    'singleGNN'    : True,
    'simpleCoTeaching': False,
    'noise_rate':  0.35, # Used for co-teaching algorithm
    'ramp_up_epochs': 7,
    'divide_mix': False,
    'warmup_epochs': 1,
    'graph_pooling': 'mean',

    'drop_ratio1':  0.1,
    'num_layer1':   2,
    'emb_dim1':     64,

    'drop_ratio2':  0.1,
    'num_layer2':   2,
    'emb_dim2':     32,

    'drop_ratio3':  0.5,
    'num_layer3':   2,
    'emb_dim3':     32,

    'drop_ratio4':  0.1,
    'num_layer4':   4,
    'emb_dim4':     32,

    'drop_ratio5':  0.1,
    'num_layer5':   5,
    'emb_dim5':     32,

}

# -----------------------------------------------------------------------------
# 2) Helper: interactive vs. non‐interactive
# -----------------------------------------------------------------------------
def get_arguments(interactive=True):
    if not interactive:
        # Return a Namespace built from PRESET_ARGS
        return argparse.Namespace(**PRESET_ARGS)

    # Otherwise run your existing prompts
    def get_user_input(prompt, default=None, required=False, type_cast=str):
        while True:
            user_input = input(f"{prompt} [{default}]: ")
            if user_input == "" and required:
                print("This field is required. Please enter a value.")
                continue
            if user_input == "" and default is not None:
                return default
            if user_input == "" and not required:
                return None
            try:
                return type_cast(user_input)
            except ValueError:
                print(f"Invalid input. Please enter a valid {type_cast.__name__}.")

    parser = {}
    parser['num_checkpoints']  = get_user_input("Number of checkpoints to save", default=3, type_cast=int)
    parser['device']           = get_user_input("Which GPU to use if any", default=1, type_cast=int)
    parser['gnn']              = get_user_input("GNN type", default='gin')
    parser['drop_ratio']       = get_user_input("Dropout ratio", default=0.0, type_cast=float)
    parser['num_layer']        = get_user_input("Number of GNN layers", default=5, type_cast=int)
    parser['emb_dim']          = get_user_input("Embedding dim", default=300, type_cast=int)
    parser['batch_size']       = get_user_input("Batch size", default=32, type_cast=int)
    parser['epochs']           = get_user_input("Epochs", default=10, type_cast=int)
    parser['baseline_mode']    = get_user_input("Baseline mode (1=CE,2=ELR)", default=1, type_cast=int)
    parser['noise_prob']       = get_user_input("Noise prob", default=0.2, type_cast=float)
    parser['singleGNN']        = get_user_input("Single GNN", default=False, type_cast=bool)
    parser['simpleCoTeaching'] = get_user_input("Simple Co-Teaching", default=True, type_cast=bool)
    parser['noise_rate']       = get_user_input("Noise rate", default=0.2, type_cast=float)
    parser['ramp_up_epochs']   = get_user_input("Ramp-up epochs", default=10, type_cast=int)
    parser['divide_mix']        = get_user_input("Divide mix", default=True, type_cast=bool)
    parser['warmup_epochs']    = get_user_input("Warmup epochs", default=10, type_cast=int)



    return argparse.Namespace(**parser)


config_args = get_arguments(interactive=False)

# Notebook configuration
manual_seed = 1
script_dir = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adams_lr = 0.01
at_least = 0.45
num_checkpoints = config_args.num_checkpoints

