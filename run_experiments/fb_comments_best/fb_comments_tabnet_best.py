import os
import sys
from pathlib import Path

from pytorch_widedeep.models import TabNet, WideDeep

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from fb_comments_tabmlp_best import prepare_data  # noqa: E402
from fb_comments_tabmlp_best import run_experiment_and_save, set_dirs  # noqa: E402

ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
WORKDIR = Path(os.getcwd())
PROCESSED_DATA_DIR = ROOTDIR / "processed_data/fb_comments/"


def set_model(args, prepare_tab):

    deeptabular = TabNet(
        column_idx=prepare_tab.column_idx,
        embed_input=prepare_tab.embeddings_input,
        embed_dropout=args.embed_dropout,
        continuous_cols=prepare_tab.continuous_cols,
        batchnorm_cont=args.batchnorm_cont,
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

    return model


if __name__ == "__main__":

    model_name = "tabnet"

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
