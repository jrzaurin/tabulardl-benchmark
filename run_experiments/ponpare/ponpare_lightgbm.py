import sys
import os
import pickle
from copy import copy
from datetime import datetime
from pathlib import Path
from time import time
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Dataset as lgbDataset
from pytorch_widedeep.utils import LabelEncoder
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score

sys.path.append(
    os.path.abspath("/Users/javier/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.lightgbm_optimizer import (  # isort:skipimport pickle  # noqa: E402
    LGBOptimizerHyperopt,
    LGBOptimizerOptuna,
)

pd.options.display.max_columns = 100


# ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
ROOTDIR = Path("/Users/javier/Projects/tabulardl-benchmark/")
# WORKDIR = Path(os.getcwd())
WORKDIR = Path("/Users/javier/Projects/tabulardl-benchmark/run_experiments")

PROCESSED_DATA_DIR = ROOTDIR / "processed_data/ponpare/"

RESULTS_DIR = WORKDIR / "results/ponpare/lightgbm"
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

MODELS_DIR = WORKDIR / "models/ponpare/lightgbm"
if not MODELS_DIR.is_dir():
    os.makedirs(MODELS_DIR)

OPTIMIZE_WITH = "optuna"
WITH_USER_AND_ITEM_ID = False

train = pd.read_pickle(PROCESSED_DATA_DIR / "ponpare_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "ponpare_val.p")
test = pd.read_pickle(PROCESSED_DATA_DIR / "ponpare_test.p")

train["target"] = train.target.apply(lambda x: 2 if x >= 2 else x)
valid["target"] = valid.target.apply(lambda x: 2 if x >= 2 else x)
test["target"] = test.target.apply(lambda x: 2 if x >= 2 else x)

cat_cols = []
for col in train.columns:
    if train[col].dtype == "O" or train[col].nunique() < 200 and col != "target":
        cat_cols.append(col)

if not WITH_USER_AND_ITEM_ID:
    cat_cols.remove("user_id_hash")
    cat_cols.remove("coupon_id_hash")
    train.drop(["user_id_hash", "coupon_id_hash"], axis=1, inplace=True)
    valid.drop(["user_id_hash", "coupon_id_hash"], axis=1, inplace=True)
    test.drop(["user_id_hash", "coupon_id_hash"], axis=1, inplace=True)

num_cols = [c for c in train.columns if c not in cat_cols + ["target"]]

#  TRAIN/VALID for hyperparam optimization
label_encoder = LabelEncoder(cat_cols)
train_le = label_encoder.fit_transform(train)
valid_le = label_encoder.transform(valid)

lgbtrain = lgbDataset(
    train_le[cat_cols + num_cols],
    train_le.target,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgbvalid = lgbDataset(
    valid_le[cat_cols + num_cols],
    valid_le.target,
    reference=lgbtrain,
    free_raw_data=False,
)

if OPTIMIZE_WITH == "optuna":
    optimizer: Union[LGBOptimizerHyperopt, LGBOptimizerOptuna] = LGBOptimizerOptuna(
        objective="multiclass", num_class=train.target.nunique()
    )
elif OPTIMIZE_WITH == "hyperopt":
    optimizer = LGBOptimizerHyperopt(
        objective="multiclass", num_class=train.target.nunique(), verbose=True
    )

optimizer.optimize(lgbtrain, lgbvalid)

# Final TRAIN/TEST

ftrain = pd.concat([train, valid]).reset_index(drop=True)
flabel_encoder = LabelEncoder(cat_cols)
ftrain_le = flabel_encoder.fit_transform(ftrain)
test_le = flabel_encoder.transform(test)

params = {"n_estimators": 1000, "objective": "multiclass", "num_class": ftrain.target.nunique(), "is_unbalance": True}
# params = copy(optimizer.best)
# params["n_estimators"] = 1000

flgbtrain = lgbDataset(
    ftrain_le[cat_cols + num_cols],
    ftrain_le.target,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgbtest = lgbDataset(
    test_le[cat_cols + num_cols],
    test_le.target,
    reference=flgbtrain,
    free_raw_data=False,
)

start = time()
model = lgb.train(
    params,
    flgbtrain,
    valid_sets=[lgbtest],
    early_stopping_rounds=50,
    verbose_eval=True,
)
runtime = time() - start

preds = model.predict(lgbtest.data)
logloss = log_loss(lgbtest.label, preds)

preds_cat = np.argmax(preds, axis=1)
acc = accuracy_score(lgbtest.label, preds_cat)
print(confusion_matrix(lgbtest.label, preds_cat))
print(f"Accuracy: {acc}")

# SAVE
suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
results_filename = "_".join(["ponpare_lightgbm", suffix]) + ".p"
results_d = {}
results_d["best_params"] = optimizer.best
results_d["runtime"] = runtime
results_d["acc"] = acc
with open(RESULTS_DIR / results_filename, "wb") as f:
    pickle.dump(results_d, f)

model_filename = "_".join(["model_ponpare_lightgbm", suffix]) + ".p"
with open(MODELS_DIR / model_filename, "wb") as f:
    pickle.dump(model, f)
