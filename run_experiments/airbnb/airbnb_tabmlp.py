import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
from pytorch_widedeep.models import DeepText, TabMlp, Wide, WideDeep
from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    TextPreprocessor,
    WidePreprocessor,
)

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402
from parsers.tabmlp_parser import parse_args  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()


def cleanhtml(raw_html):
    # with much gratitude, taken from here:
    # https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    cleanr = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    cleantext = re.sub(cleanr, " ", raw_html)
    return cleantext


args = parse_args()

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())

PROCESSED_DATA_DIR = ROOTDIR / "processed_data/airbnb/"
RESULTS_DIR = WORKDIR / "results/airbnb/tabmlp"
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

word_vectors_path = ROOTDIR / "word_vectors/cc.en.300.vec"

train = pd.read_pickle(PROCESSED_DATA_DIR / "airbnb_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "airbnb_val.p")

# some manual cleaning: only dropping 8 rows
train = train[~train.description.isin([".", "2", "...", "W", "..", "P"])]
valid = valid[~valid.description.isin([".", "2", "...", "W", "..", "P"])]

# And some html cleaning
train["description"] = train.description.apply(lambda x: cleanhtml(x))
valid["description"] = valid.description.apply(lambda x: cleanhtml(x))

drop_cols = ["id", "host_id", "host_since", "latitude", "longitude"]
for df in [train, valid]:
    df.drop(drop_cols, axis=1, inplace=True)
    df.rename(columns={"yield": "target"}, inplace=True)

upper_yield = train.target.quantile(0.99)
train = train[train.target <= upper_yield]
valid = valid[valid.target <= upper_yield]

if args.with_wide:
    cat_embed_cols = []
    for col in train.columns:
        if (train[col].dtype == "O" or train[col].nunique() < 200) and col not in [
            "target",
            "description",
        ]:
            cat_embed_cols.append(col)
    wide_cols = []
    for col in train.columns:
        if train[col].nunique() < 40 and col not in [
            "target",
            "description",
        ]:
            wide_cols.append(col)
    num_cols = [
        c for c in train.columns if c not in cat_embed_cols + ["target", "description"]
    ]

    prepare_wide = WidePreprocessor(wide_cols)
    X_wide_train = prepare_wide.fit_transform(train)
    X_wide_valid = prepare_wide.transform(valid)

    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols, continuous_cols=num_cols, scale=args.scale_cont
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
        if (train[col].dtype == "O" or train[col].nunique() < 200) and col not in [
            "target",
            "description",
        ]:
            cat_embed_cols.append(col)
    num_cols = [
        c for c in train.columns if c not in cat_embed_cols + ["target", "description"]
    ]

    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols, continuous_cols=num_cols, scale=args.scale_cont
    )
    X_tab_train = prepare_tab.fit_transform(train)
    X_tab_valid = prepare_tab.transform(valid)

    y_train = train.target.values
    y_valid = valid.target.values

    wide = None

    X_train = {"X_tab": X_tab_train, "target": y_train}
    X_val = {"X_tab": X_tab_valid, "target": y_valid}

if args.with_text:

    if os.path.isfile("tmp/X_text_train.npy") and not args.prepare_text:
        X_text_train = np.load("tmp/X_text_train.npy")
        X_text_valid = np.load("tmp/X_text_valid.npy")
        with open("tmp/prepare_text.p", "rb") as f:
            prepare_text = pickle.load(f)
    else:
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        prepare_text = TextPreprocessor(
            text_col="description",
            max_vocab=args.max_vocab,
            min_freq=args.min_freq,
            maxlen=args.maxlen,
            pad_first=args.pad_first,
            pad_idx=args.pad_idx,
            word_vectors_path=word_vectors_path if args.use_word_vectors else None,
        )
        X_text_train = prepare_text.fit_transform(train)
        X_text_valid = prepare_text.transform(valid)
        np.save("tmp/X_text_train", X_text_train)
        np.save("tmp/X_text_valid", X_text_valid)
        with open("tmp/prepare_text.p", "wb") as f:
            pickle.dump(prepare_text, f)

    X_train["X_text"] = X_text_train
    X_val["X_text"] = X_text_valid

    deeptext = DeepText(
        vocab_size=len(prepare_text.vocab.itos),
        rnn_type=args.rnn_type,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        rnn_dropout=args.rnn_dropout,
        bidirectional=args.bidirectional,
        use_hidden_state=args.use_hidden_state,
        embed_dim=args.embed_dim,
        embed_matrix=prepare_text.embedding_matrix if args.use_word_vectors else None,
        embed_trainable=args.embed_trainable,
        head_hidden_dims=eval(args.head_hidden_dims),
        head_activation=args.head_activation,
        head_dropout=args.head_dropout,
        head_batchnorm=args.head_batchnorm,
        head_batchnorm_last=args.head_batchnorm_last,
        head_linear_first=args.head_linear_first,
    )
else:
    deeptext = None

if args.mlp_hidden_dims == "auto":
    n_inp_dim = sum([e[2] for e in prepare_tab.embeddings_input])
    mlp_hidden_dims = [4 * n_inp_dim, 2 * n_inp_dim]
else:
    mlp_hidden_dims = eval(args.mlp_hidden_dims)

deeptabular = TabMlp(
    column_idx=prepare_tab.column_idx,
    mlp_hidden_dims=mlp_hidden_dims,
    mlp_activation=args.mlp_activation,
    mlp_dropout=args.mlp_dropout,
    mlp_batchnorm=args.mlp_batchnorm,
    mlp_batchnorm_last=args.mlp_batchnorm_last,
    mlp_linear_first=args.mlp_linear_first,
    embed_input=prepare_tab.embeddings_input,
    embed_dropout=args.embed_dropout,
    continuous_cols=prepare_tab.continuous_cols,
    batchnorm_cont=args.batchnorm_cont,
)
model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext)

optimizers = set_optimizer(model, args)

steps_per_epoch = (X_train["X_tab"].shape[0] // args.batch_size) + 1
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
    warmup=args.warmup,
    warmup_epochs=args.warmup_epochs,
    warmup_max_lr=args.warmup_max_lr,
)
runtime = time() - start

if args.save_results:
    suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
    filename = "_".join(["airbnb_tabmlp", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args.__dict__
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["trainer_lr_history"] = trainer.lr_history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
