import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="TabRenNet parameters")

    # data set
    parser.add_argument(
        "--bankm_dset",
        type=str,
        default="bank_marketing",
        help="bank_marketing or bank_marketing_kaggle",
    )

    # model parameters
    parser.add_argument(
        "--n_steps",
        type=int,
        default=5,
        help="number of decision steps",
    )
    parser.add_argument(
        "--step_dim",
        type=int,
        default=16,
        help="Step's output dimension",
    )
    parser.add_argument(
        "--attn_dim",
        type=int,
        default=16,
        help="Attention dimension",
    )
    parser.add_argument(
        "--n_glu_step_dependent",
        type=int,
        default=2,
        help="number of GLU Blocks [FC -> BN -> GLU] that are step dependen",
    )
    parser.add_argument(
        "--n_glu_shared",
        type=int,
        default=2,
        help="number of GLU Blocks [FC -> BN -> GLU] that will be shared across decision steps",
    )
    parser.add_argument(
        "--ghost_bn",
        action="store_true",
        help="Boolean indicating if Ghost Batch Normalization will be used",
    )
    parser.add_argument(
        "--virtual_batch_size",
        type=int,
        default=128,
        help="Batch size when using Ghost Batch Normalization",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.98,
        help="Ghost Batch Normalization's momentum",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.5, help="Relaxation parameter in the paper"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Glu block dropout")
    parser.add_argument(
        "--embed_dropout", type=float, default=0.0, help="Embedding dropout"
    )

    # train/eval parameters
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epoch.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 reg.")
    parser.add_argument(
        "--lambda_sparse",
        type=float,
        default=0.0001,
        help="Tabnet sparse regularization factor",
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=0.0,
        help="Min delta for early stopping",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=30,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="(val_)loss or (val_)metric name to monitor",
    )

    # Optimizer parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Only Adam, AdamW, and RAdam are considered. UseDefault is AdamW with default values",
    )

    # Scheduler parameters
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="ReduceLROnPlateau",
        help="one of 'ReduceLROnPlateau', 'CyclicLR' or 'OneCycleLR', NoScheduler",
    )
    # ReduceLROnPlateau (rop) params
    parser.add_argument(
        "--rop_mode",
        type=str,
        default="min",
        help="One of min, max",
    )
    parser.add_argument(
        "--rop_factor",
        type=float,
        default=0.2,
        help="Factor by which the learning rate will be reduced",
    )
    parser.add_argument(
        "--rop_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which learning rate will be reduced",
    )
    parser.add_argument(
        "--rop_threshold",
        type=float,
        default=0.001,
        help="Threshold for measuring the new optimum",
    )
    parser.add_argument(
        "--rop_threshold_mode",
        type=str,
        default="abs",
        help="One of rel, abs",
    )
    # CyclicLR and OneCycleLR params
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.001,
        help="base_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.01,
        help="max_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--div_factor",
        type=float,
        default=25,
        help="Determines the initial learning rate via initial_lr = max_lr/div_factor",
    )
    parser.add_argument(
        "--final_div_factor",
        type=float,
        default=1e4,
        help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
    )
    parser.add_argument(
        "--n_cycles",
        type=float,
        default=5,
        help="number of cycles for CyclicLR",
    )
    parser.add_argument(
        "--cycle_momentum",
        action="store_true",
    )
    parser.add_argument(
        "--pct_step_up",
        type=float,
        default=0.3,
        help="Percentage of the cycle (in number of steps) spent increasing the learning rate",
    )

    # save parameters
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()
