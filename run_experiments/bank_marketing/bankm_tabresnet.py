import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
from pytorch_widedeep.metrics import Accuracy, F1Score
from pytorch_widedeep.models import TabResnet, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402
from parsers.tabresnet_parser import parse_args  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()

args = parse_args()

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())

PROCESSED_DATA_DIR = ROOTDIR / "/".join(["processed_data", args.bankm_dset])
RESULTS_DIR = WORKDIR / "/".join(["results", args.bankm_dset, "tabresnet"])
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

train = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_val.p")
colnames = [c.replace(".", "_") for c in train.columns]
train.columns = colnames
valid.columns = colnames

# All columns will be treated as categorical. The column with the highest
# number of categories has 308
cat_embed_cols = [c for c in train.columns if c != "target"]

# all columns will be represented by embeddings
prepare_tab = TabPreprocessor(embed_cols=cat_embed_cols)
X_train = prepare_tab.fit_transform(train)
y_train = train.target.values
X_valid = prepare_tab.transform(valid)
y_valid = valid.target.values

if args.blocks_dims == "same":
    n_inp_dim = sum([e[2] for e in prepare_tab.embeddings_input])
    blocks_dims = [n_inp_dim, n_inp_dim, n_inp_dim]
else:
    blocks_dims = eval(args.blocks_dims)

if args.mlp_hidden_dims == "auto":
    n_inp_dim = blocks_dims[-1]
    mlp_hidden_dims = [4 * n_inp_dim, 2 * n_inp_dim]
else:
    mlp_hidden_dims = eval(args.mlp_hidden_dims)

deeptabular = TabResnet(
    embed_input=prepare_tab.embeddings_input,
    column_idx=prepare_tab.column_idx,
    blocks_dims=blocks_dims,
    blocks_dropout=args.blocks_dropout,
    mlp_hidden_dims=mlp_hidden_dims,
    mlp_activation=args.mlp_activation,
    mlp_dropout=args.mlp_dropout,
    mlp_batchnorm=args.mlp_batchnorm,
    mlp_batchnorm_last=args.mlp_batchnorm_last,
    mlp_linear_first=args.mlp_linear_first,
    embed_dropout=args.embed_dropout,
)
model = WideDeep(deeptabular=deeptabular)

optimizers = set_optimizer(model, args)

steps_per_epoch = (X_train.shape[0] // args.batch_size) + 1
lr_schedulers = set_lr_scheduler(optimizers, steps_per_epoch, args)
early_stopping = EarlyStopping(
    monitor=args.monitor,
    min_delta=args.early_stop_delta,
    patience=args.early_stop_patience,
)

trainer = Trainer(
    model,
    objective="binary",
    optimizers=optimizers,
    lr_schedulers=lr_schedulers,
    reducelronplateau_criterion=args.monitor.split("_")[-1],
    callbacks=[early_stopping, LRHistory(n_epochs=args.n_epochs)],
    metrics=[Accuracy, F1Score],
)

start = time()
trainer.fit(
    X_train={"X_tab": X_train, "target": y_train},
    X_val={"X_tab": X_valid, "target": y_valid},
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    validation_freq=args.eval_every,
)
runtime = time() - start

if args.save_results:
    suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
    filename = "_".join(["bankm_tabresnet", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args.__dict__
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
