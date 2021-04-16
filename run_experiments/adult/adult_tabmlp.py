import os
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import ModelCheckpoint, EarlyStopping, LRHistory
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from tabmlp_parser import parse_args
from utils import set_lr_scheduler, set_optimizer

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()

ROOT_DIR = Path(os.getcwd())

RESULTS_DIR = ROOT_DIR / "results/adult/"
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

PROCESSED_DATA_DIR = ROOT_DIR / "processed_data/adult/"

train = pd.read_pickle(PROCESSED_DATA_DIR / "adult_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "adult_val.p")
for df in [train, valid]:
    df.drop("education_num", axis=1, inplace=True)

# 200 is rather arbitraty but one has to make a decision as to how to decide
# if something will be represented as embeddings or continuous in a "kind-of"
# automated way
cat_embed_cols = []
for col in train.columns:
    if train[col].dtype == "O" or train[col].nunique() < 200 and col != "target":
        cat_embed_cols.append(col)

# all columns will be represented by embeddings
prepare_deep = TabPreprocessor(embed_cols=cat_embed_cols)
X_train = prepare_deep.fit_transform(train)
y_train = train.target.values
X_valid = prepare_deep.transform(valid)
y_valid = valid.target.values

args = parse_args()

if args.mlp_hidden_dims == "auto":
    n_inp_dim = sum([e[2] for e in prepare_deep.embeddings_input])
    mlp_hidden_dims = [4 * n_inp_dim, 2 * n_inp_dim]
else:
    mlp_hidden_dims = eval(args.mlp_hidden_dims)

deeptabular = TabMlp(
    column_idx=prepare_deep.column_idx,
    mlp_hidden_dims=mlp_hidden_dims,
    mlp_activation=args.mlp_activation,
    mlp_dropout=args.mlp_dropout,
    mlp_batchnorm=args.mlp_batchnorm,
    mlp_batchnorm_last=args.mlp_batchnorm_last,
    mlp_linear_first=args.mlp_linear_first,
    embed_input=prepare_deep.embeddings_input,
    embed_dropout=args.embed_dropout,
    batchnorm_cont=args.batchnorm_cont,
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
# model_checkpoint = ModelCheckpoint(
#     filepath="models/adult_tabmlp_model",
#     monitor=args.monitor,
#     save_best_only=True,
#     verbose=1,
#     max_save=1,
# )

trainer = Trainer(
    model,
    objective="binary",
    optimizers=optimizers,
    lr_schedulers=lr_schedulers,
    reduce_on="acc",
    # callbacks=[early_stopping, model_checkpoint, LRHistory(n_epochs=args.n_epochs)],
    callbacks=[early_stopping, LRHistory(n_epochs=args.n_epochs)],
    metrics=[Accuracy],
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
    filename = "_".join(["adult_tabmlp", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args.__dict__
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
