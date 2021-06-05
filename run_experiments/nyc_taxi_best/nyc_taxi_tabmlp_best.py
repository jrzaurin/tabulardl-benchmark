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
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory, ModelCheckpoint
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import read_best_model_args  # noqa: E402
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()


ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())
PROCESSED_DATA_DIR = ROOTDIR / "processed_data/nyc_taxi/"


def set_dirs(model_name):

    models_dir = WORKDIR / "/".join(["best_models", "nyc_taxi", model_name])
    results_dir = WORKDIR / "/".join(["results", "nyc_taxi", model_name])
    for d in [models_dir, results_dir]:
        if not d.is_dir():
            os.makedirs(d)

    return results_dir, models_dir


def load_dataset():

    train = pd.read_pickle(PROCESSED_DATA_DIR / "nyc_taxi_train.p")
    valid = pd.read_pickle(PROCESSED_DATA_DIR / "nyc_taxi_val.p")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "nyc_taxi_test.p")

    drop_cols = [
        "pickup_datetime",
        "dropoff_datetime",
        "trip_duration",
    ]  # trip_duration is "target"
    for df in [train, valid, test]:
        df.drop(drop_cols, axis=1, inplace=True)

    # so the cut is the same as with lightgbm
    upper_trip_duration = train.target.quantile(0.99)
    lower_trip_duration = 60  # a minute

    train = pd.concat([train, valid], ignore_index=True)
    train = train[
        (train.target >= lower_trip_duration) & (train.target <= upper_trip_duration)
    ]
    test = test[
        (test.target >= lower_trip_duration) & (test.target <= upper_trip_duration)
    ]

    return train, test


def prepare_data(results_dir):

    train, test = load_dataset()

    cat_embed_cols = []
    for col in train.columns:
        if train[col].dtype == "O" or train[col].nunique() < 200 and col != "target":
            cat_embed_cols.append(col)
    num_cols = [c for c in train.columns if c not in cat_embed_cols + ["target"]]

    args = read_best_model_args(results_dir)
    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols, continuous_cols=num_cols, scale=args.scale_cont
    )
    X_train = prepare_tab.fit_transform(train)
    y_train = train.target.values
    X_test = prepare_tab.transform(test)
    y_test = test.target.values

    return args, prepare_tab, X_train, X_test, y_train, y_test


def set_model(args, prepare_tab):

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
    model = WideDeep(deeptabular=deeptabular)

    return model


def run_experiment_and_save(
    model,
    model_name,
    results_dir,
    models_dir,
    args,
    X_train,
    X_test,
    y_train,
    y_test,
):

    optimizers = set_optimizer(model, args)

    steps_per_epoch = (X_train.shape[0] // args.batch_size) + 1
    lr_schedulers = set_lr_scheduler(optimizers, steps_per_epoch, args)

    early_stopping = EarlyStopping(
        monitor=args.monitor,
        min_delta=args.early_stop_delta,
        patience=args.early_stop_patience,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=str(models_dir / "best_model"),
        monitor=args.monitor,
        save_best_only=True,
        max_save=1,
    )

    trainer = Trainer(
        model,
        objective="regression",
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        reducelronplateau_criterion=args.monitor.split("_")[-1],
        callbacks=[early_stopping, model_checkpoint, LRHistory(n_epochs=args.n_epochs)],
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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"rmse with the best model: {rmse}")

    if args.save_results:
        suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
        filename = "_".join(["nyc_taxi", model_name, "best", suffix]) + ".p"
        results_d = {}
        results_d["args"] = args
        results_d["rmse"] = rmse
        results_d["r2"] = r2
        results_d["early_stopping"] = early_stopping
        results_d["trainer_history"] = trainer.history
        results_d["trainer_lr_history"] = trainer.lr_history
        results_d["runtime"] = runtime
        with open(results_dir / filename, "wb") as f:
            pickle.dump(results_d, f)


if __name__ == "__main__":

    model_name = "tabmlp"

    results_dir, models_dir = set_dirs(model_name)

    args, prepare_tab, X_train, X_test, y_train, y_test = prepare_data(results_dir)

    model = set_model(args, prepare_tab)

    run_experiment_and_save(
        model,
        model_name,
        results_dir,
        models_dir,
        args,
        X_train,
        X_test,
        y_train,
        y_test,
    )
