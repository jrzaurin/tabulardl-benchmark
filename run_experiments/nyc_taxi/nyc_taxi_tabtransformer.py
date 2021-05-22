import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
from pytorch_widedeep.models import TabTransformer, Wide, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402
from parsers.tabtransformer_parser import parse_args  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()

args = parse_args()

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())

PROCESSED_DATA_DIR = ROOTDIR / "processed_data/nyc_taxi/"
RESULTS_DIR = WORKDIR / "results/nyc_taxi/tabtransformer"
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

train = pd.read_pickle(PROCESSED_DATA_DIR / "nyc_taxi_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "nyc_taxi_val.p")

drop_cols = [
    "pickup_datetime",
    "dropoff_datetime",
    "trip_duration",
]  # trip_duration is "target"
for df in [train, valid]:
    df.drop(drop_cols, axis=1, inplace=True)

upper_trip_duration = train.target.quantile(0.99)
lower_trip_duration = 60  # a minute
train = train[
    (train.target >= lower_trip_duration) & (train.target <= upper_trip_duration)
]
valid = valid[
    (valid.target >= lower_trip_duration) & (valid.target <= upper_trip_duration)
]

# All columns will be treated as categorical. The column with the highest
# number of categories has 308
if args.with_wide:
    cat_embed_cols = []
    for col in train.columns:
        if train[col].nunique() >= 5 and train[col].nunique() < 200 and col != "target":
            cat_embed_cols.append(col)
    num_cols = [c for c in train.columns if c not in cat_embed_cols + ["target"]]

    wide_cols = []
    for col in train.columns:
        if train[col].nunique() < 40 and col != "target":
            wide_cols.append(col)

    prepare_wide = WidePreprocessor(wide_cols)
    X_wide_train = prepare_wide.fit_transform(train)
    X_wide_valid = prepare_wide.transform(valid)

    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols,
        continuous_cols=num_cols,
        for_tabtransformer=True,
        scale=False,
    )
    X_tab_train = prepare_tab.fit_transform(train)
    X_tab_valid = prepare_tab.transform(valid)

    y_train = train.target.values
    y_valid = valid.target.values

    wide = Wide(wide_dim=np.unique(X_wide_train).shape[0])

    X_train = {"X_wide": X_wide_train, "X_tab": X_tab_train, "target": y_train}
    X_val = {"X_wide": X_wide_valid, "X_tab": X_tab_valid, "target": y_valid}

else:
    cat_embed_cols = []
    for col in train.columns:
        if train[col].dtype == "O" or train[col].nunique() < 200 and col != "target":
            cat_embed_cols.append(col)
    num_cols = [c for c in train.columns if c not in cat_embed_cols + ["target"]]

    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols,
        continuous_cols=num_cols,
        for_tabtransformer=True,
        scale=False,
    )
    X_tab_train = prepare_tab.fit_transform(train)
    X_tab_valid = prepare_tab.transform(valid)

    y_train = train.target.values
    y_valid = valid.target.values

    wide = None

    X_train = {"X_tab": X_tab_train, "target": y_train}
    X_val = {"X_tab": X_tab_valid, "target": y_valid}

if args.mlp_hidden_dims == "same":
    mlp_hidden_dims = [
        len(cat_embed_cols) * args.input_dim,
        len(cat_embed_cols) * args.input_dim,
        (len(cat_embed_cols) * args.input_dim) // 2,
    ]
elif args.mlp_hidden_dims == "None":
    mlp_hidden_dims = None
else:
    mlp_hidden_dims = eval(args.mlp_hidden_dims)


deeptabular = TabTransformer(
    column_idx=prepare_tab.column_idx,
    embed_input=prepare_tab.embeddings_input,
    embed_dropout=args.embed_dropout,
    continuous_cols=prepare_tab.continuous_cols,
    full_embed_dropout=args.full_embed_dropout,
    shared_embed=args.shared_embed,
    add_shared_embed=args.add_shared_embed,
    frac_shared_embed=args.frac_shared_embed,
    input_dim=args.input_dim,
    n_heads=args.n_heads,
    n_blocks=args.n_blocks,
    dropout=args.dropout,
    ff_hidden_dim=4 * args.input_dim if not args.ff_hidden_dim else args.ff_hidden_dim,
    transformer_activation=args.transformer_activation,
    mlp_hidden_dims=mlp_hidden_dims,
    mlp_activation=args.mlp_activation,
    mlp_batchnorm=args.mlp_batchnorm,
    mlp_batchnorm_last=args.mlp_batchnorm_last,
    mlp_linear_first=args.mlp_linear_first,
)

model = WideDeep(wide=wide, deeptabular=deeptabular)

optimizers = set_optimizer(model, args)

steps_per_epoch = (X_tab_train.shape[0] // args.batch_size) + 1
lr_schedulers = set_lr_scheduler(optimizers, steps_per_epoch, args)

early_stopping = EarlyStopping(
    monitor=args.monitor,
    min_delta=args.early_stop_delta,
    patience=args.early_stop_patience,
)

trainer = Trainer(
    model,
    objective="regression",
    optimizers=optimizers,
    lr_schedulers=lr_schedulers,
    reducelronplateau_criterion=args.monitor.split("_")[-1],
    callbacks=[early_stopping, LRHistory(n_epochs=args.n_epochs)],
)

start = time()
trainer.fit(
    X_train=X_train,
    X_val=X_val,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    validation_freq=args.eval_every,
)
runtime = time() - start

if args.save_results:
    suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
    filename = "_".join(["nyc_taxi_tabtransformer", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args.__dict__
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
