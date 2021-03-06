import os
import sys
from pathlib import Path

from pytorch_widedeep.models import TabTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from bankm_tabmlp_best import load_dataset  # noqa: E402
from bankm_tabmlp_best import run_experiment_and_save, set_dirs  # noqa: E402
from general_utils.utils import read_best_model_args  # noqa: E402

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())
PROCESSED_DATA_DIR = ROOTDIR / "processed_data/bank_marketing/"


def prepare_data(results_dir):

    train, test = load_dataset()

    # All columns will be treated as categorical. The column with the highest
    # number of categories has 308
    cat_embed_cols = [c for c in train.columns if c != "target"]

    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols, for_tabtransformer=True, scale=False
    )
    X_train = prepare_tab.fit_transform(train)
    y_train = train.target.values
    X_test = prepare_tab.transform(test)
    y_test = test.target.values

    return len(cat_embed_cols), prepare_tab, X_train, X_test, y_train, y_test


def set_model(args, prepare_tab, mlp_hidden_dims_same):

    if args.mlp_hidden_dims == "same":
        mlp_hidden_dims = [
            mlp_hidden_dims_same * args.input_dim,
            mlp_hidden_dims_same * args.input_dim,
            (mlp_hidden_dims_same * args.input_dim) // 2,
        ]
    elif args.mlp_hidden_dims == "None":
        mlp_hidden_dims = None
    else:
        mlp_hidden_dims = eval(args.mlp_hidden_dims)

    deeptabular = TabTransformer(
        column_idx=prepare_tab.column_idx,
        embed_input=prepare_tab.embeddings_input,
        embed_dropout=args.embed_dropout,
        full_embed_dropout=args.full_embed_dropout,
        shared_embed=args.shared_embed,
        add_shared_embed=args.add_shared_embed,
        frac_shared_embed=args.frac_shared_embed,
        input_dim=args.input_dim,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        ff_hidden_dim=4 * args.input_dim
        if not args.ff_hidden_dim
        else args.ff_hidden_dim,
        transformer_activation=args.transformer_activation,
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_activation=args.mlp_activation,
        mlp_batchnorm=args.mlp_batchnorm,
        mlp_batchnorm_last=args.mlp_batchnorm_last,
        mlp_linear_first=args.mlp_linear_first,
    )

    model = WideDeep(deeptabular=deeptabular)

    return model


if __name__ == "__main__":

    model_name = "tabtransformer"

    results_dir, models_dir = set_dirs(model_name)

    mlp_hidden_dims_same, prepare_tab, X_train, X_test, y_train, y_test = prepare_data(
        results_dir
    )

    args = read_best_model_args(results_dir)

    model = set_model(args, prepare_tab, mlp_hidden_dims_same)

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
