import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory, ModelCheckpoint
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import TabNet, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from sklearn.metrics import accuracy_score

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import read_best_model_args  # noqa: E402
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())
PROCESSED_DATA_DIR = ROOTDIR / "processed_data/adult/"

MODELS_DIR = WORKDIR / "best_models/adult/tabnet"
RESULTS_DIR = WORKDIR / "results/adult/tabnet"
for d in [MODELS_DIR, RESULTS_DIR]:
    if not d.is_dir():
        os.makedirs(d)


train = pd.read_pickle(PROCESSED_DATA_DIR / "adult_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "adult_val.p")
test = pd.read_pickle(PROCESSED_DATA_DIR / "adult_test.p")
for df in [train, valid, test]:
    df.drop("education_num", axis=1, inplace=True)
train = pd.concat([train, valid], ignore_index=True)

# 200 is rather arbitraty but one has to make a decision as to how to decide
# if something will be represented as embeddings or continuous in a "kind-of"
# automated way
cat_embed_cols = []
for col in train.columns:
    if train[col].dtype == "O" or train[col].nunique() < 200 and col != "target":
        cat_embed_cols.append(col)

# all columns will be represented by embeddings
prepare_tab = TabPreprocessor(embed_cols=cat_embed_cols)
X_train = prepare_tab.fit_transform(train)
y_train = train.target.values
X_test = prepare_tab.transform(test)
y_test = test.target.values

args = read_best_model_args(RESULTS_DIR)

deeptabular = TabNet(
    column_idx=prepare_tab.column_idx,
    embed_input=prepare_tab.embeddings_input,
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

model_checkpoint = ModelCheckpoint(
    filepath=str(MODELS_DIR / "best_model"),
    monitor=args.monitor,
    save_best_only=True,
    max_save=1,
)

trainer = Trainer(
    model,
    objective="binary",
    optimizers=optimizers,
    lr_schedulers=lr_schedulers,
    reducelronplateau_criterion=args.monitor.split("_")[-1],
    callbacks=[early_stopping, model_checkpoint, LRHistory(n_epochs=args.n_epochs)],
    metrics=[Accuracy],
)

start = time()
trainer.fit(
    X_train={"X_tab": X_train, "target": y_train},
    X_val={"X_tab": X_test, "target": y_test},
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    validation_freq=args.eval_every,
)
runtime = time() - start

y_pred = trainer.predict(X_tab=X_test)
acc = accuracy_score(y_test, y_pred)

if args.save_results:
    suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
    filename = "_".join(["adult_tabnet_best", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args
    results_d["acc"] = acc
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["trainer_lr_history"] = trainer.lr_history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
