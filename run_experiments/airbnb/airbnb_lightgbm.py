import os
import pickle
import sys
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
from sklearn.metrics import mean_squared_error

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

PROCESSED_DATA_DIR = ROOTDIR / "processed_data/airbnb/"

RESULTS_DIR = WORKDIR / "results/airbnb/lightgbm"
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

MODELS_DIR = WORKDIR / "models/airbnb/lightgbm"
if not MODELS_DIR.is_dir():
    os.makedirs(MODELS_DIR)

OPTIMIZE_WITH = "optuna"


train = pd.read_pickle(PROCESSED_DATA_DIR / "airbnb_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "airbnb_val.p")
test = pd.read_pickle(PROCESSED_DATA_DIR / "airbnb_test.p")

drop_cols = ["id", "host_id", "host_since", "description", "latitude", "longitude"]
rename_target = {"yield": "target"}
for df in [train, valid, test]:
    df.drop(drop_cols, axis=1, inplace=True)
    df.rename(columns=rename_target, inplace=True)

upper_yield = train.target.quantile(0.99)
train = train[train.target <= upper_yield]
valid = valid[valid.target <= upper_yield]
test = test[test.target <= upper_yield]

cat_cols = []
for col in train.columns:
    if train[col].dtype == "O" or train[col].nunique() < 200 and col != "target":
        cat_cols.append(col)
num_cols = [c for c in train.columns if c not in cat_cols + ["target"]]

#  TRAIN/VALID for hyperparam optimization

# this does not chaneg results, but, let's not encode them
cols_to_encode = [c for c in cat_cols if train[c].dtype != int]
label_encoder = LabelEncoder(cols_to_encode)
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
        objective="regression"
    )
elif OPTIMIZE_WITH == "hyperopt":
    optimizer = LGBOptimizerHyperopt(objective="regression", verbose=True)

optimizer.optimize(lgbtrain, lgbvalid)

# Final TRAIN/TEST

ftrain = pd.concat([train, valid]).reset_index(drop=True)
flabel_encoder = LabelEncoder(cols_to_encode)
ftrain_le = flabel_encoder.fit_transform(ftrain)
test_le = flabel_encoder.transform(test)

params = copy(optimizer.best)
params["n_estimators"] = 1000

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
rmse = np.sqrt(mean_squared_error(lgbtest.label, preds))
print(f"RMSE: {rmse}")

# SAVE
suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
results_filename = "_".join(["airbnb_lightgbm", suffix]) + ".p"
results_d = {}
results_d["best_params"] = optimizer.best
results_d["runtime"] = runtime
results_d["rmse"] = rmse
with open(RESULTS_DIR / results_filename, "wb") as f:
    pickle.dump(results_d, f)

model_filename = "_".join(["model_airbnb_lightgbm", suffix]) + ".p"
with open(MODELS_DIR / model_filename, "wb") as f:
    pickle.dump(model, f)
