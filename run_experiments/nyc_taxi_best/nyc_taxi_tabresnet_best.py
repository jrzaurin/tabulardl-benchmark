import os
import sys
from pathlib import Path

from pytorch_widedeep.models import TabResnet, WideDeep

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from nyc_taxi_tabmlp_best import prepare_data  # noqa: E402
from nyc_taxi_tabmlp_best import run_experiment_and_save, set_dirs  # noqa: E402

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())
PROCESSED_DATA_DIR = ROOTDIR / "processed_data/nyc_taxi/"


def set_model(args, prepare_tab):

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
        continuous_cols=prepare_tab.continuous_cols,
        batchnorm_cont=args.batchnorm_cont,
        concat_cont_first=args.concat_cont_first,
    )
    model = WideDeep(deeptabular=deeptabular)

    return model


if __name__ == "__main__":

    model_name = "tabresnet"

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
