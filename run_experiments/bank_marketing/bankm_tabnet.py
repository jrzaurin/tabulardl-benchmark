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
from pytorch_widedeep.models import TabNet, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402
from parsers.tabnet_parser import parse_args  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()

args = parse_args()

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())

PROCESSED_DATA_DIR = ROOTDIR / "/".join(["processed_data", args.bankm_dset])
RESULTS_DIR = WORKDIR / "/".join(["results", args.bankm_dset, "tabnet"])
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
prepare_deep = TabPreprocessor(embed_cols=cat_embed_cols)
X_train = prepare_deep.fit_transform(train)
y_train = train.target.values
X_valid = prepare_deep.transform(valid)
y_valid = valid.target.values

args = parse_args()

deeptabular = TabNet(
    column_idx=prepare_deep.column_idx,
    embed_input=prepare_deep.embeddings_input,
    embed_dropout=args.embed_dropout,
    n_steps=args.n_steps,
    step_dim=args.step_dim,
    attn_dim=args.attn_dim,
    dropout=args.dropout,
    n_glu_step_dependent=args.n_glu_step_dependent,
    n_glu_shared=args.n_glu_shared,
    ghost_bn=args.ghost_bn,
    virtual_batch_size=args.virtual_batch_size,
    momentum=args.momentum,
    gamma=args.gamma,
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
    filename = "_".join(["adult_tabnet", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args.__dict__
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
