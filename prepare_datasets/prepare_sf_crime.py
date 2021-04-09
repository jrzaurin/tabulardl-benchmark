import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / "raw_data/sf_crime/"
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data/sf_crime/"

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

sfc = pd.read_csv(RAW_DATA_DIR / "train.csv.zip", parse_dates=["Dates"])
sfc = sfc.drop_duplicates().reset_index(drop=True)
sfc.drop(["Descript", "DayOfWeek", "Resolution"], axis=1, inplace=True)
sfc.columns = [c.lower() for c in sfc.columns]
sfc.rename(columns={"x": "lon", "y": "lat"}, inplace=True)

has_na = all([sfc[c].isna().sum().astype("bool") for c in sfc.columns])

na_counts = []
for c in sfc.columns:
    na_counts.append(sfc[c].isna().sum())

for c in sfc.columns:
    try:
        sfc[c] = sfc[c].str.lower()
    except Exception:
        pass

# only 3
sfc = sfc[sfc.category != "trea"]

# Chronological split
sfc = sfc.sort_values("dates").reset_index(drop=True)
test_size = int(np.ceil(sfc.shape[0] * 0.1))
train_size = sfc.shape[0] - test_size * 2

# train
sfc_train = sfc.iloc[:train_size].reset_index(drop=True)
tmp = sfc.iloc[train_size:].reset_index(drop=True)

# valid and test
sfc_val = tmp.iloc[:test_size].reset_index(drop=True)
sfc_test = tmp.iloc[test_size:].reset_index(drop=True)

sfc_train["dset"] = 0
sfc_val["dset"] = 1
sfc_test["dset"] = 2

sfc = pd.concat([sfc_train, sfc_val, sfc_test])

del (sfc_train, sfc_val, sfc_test)


def date_feats(df):
    df["hourofday"] = df["dates"].dt.hour
    df["dayofweek"] = df["dates"].dt.dayofweek
    df["dayofmonth"] = df["dates"].dt.day
    df["monthofyear"] = df["dates"].dt.month
    df["year"] = df["dates"].dt.year
    return df


sfc = date_feats(sfc)
sfc.drop("dates", axis=1, inplace=True)

address_types_counts = Counter([ad.split()[-1] for ad in sfc.address.tolist()])
address_types = [k for k, v in address_types_counts.items() if v > 1000] + ["block"]


def address_feat(address):
    address_split = address.split()
    tps = set([tp for tp in address_split if tp in address_types])
    if len(tps) == 0:
        return "other"
    else:
        return "_".join(sorted(tps))


sfc["address_type"] = sfc.address.apply(lambda x: address_feat(x))
sfc.drop("address", axis=1, inplace=True)
sfc = sfc.drop_duplicates().reset_index(drop=True)


sfc.replace({"lon": -120.5, "lat": 90.0}, np.nan, inplace=True)
coord_per_district = (
    sfc[["pddistrict", "lon", "lat"]].groupby("pddistrict").mean().reset_index()
)
coord_per_district.columns = ["pddistrict", "avg_lon", "avg_lat"]
sfc = pd.merge(sfc, coord_per_district, how="left", on="pddistrict")
sfc.lon.fillna(sfc["avg_lon"], inplace=True)
sfc.lat.fillna(sfc["avg_lat"], inplace=True)
sfc.drop(["avg_lon", "avg_lat"], axis=1, inplace=True)

sfc["x"] = np.cos(sfc.lat) * np.cos(sfc.lon)
sfc["y"] = np.cos(sfc.lat) * np.sin(sfc.lon)
sfc["z"] = np.sin(sfc.lat)
sfc["lon"] = sfc.lon / 180
sfc["lat"] = sfc.lat / 90

sfc["category"] = sfc.category.apply(lambda x: x.replace("/", "_").replace(" ", "_"))

sfc_train = sfc[sfc.dset == 0].drop("dset", axis=1)
sfc_val = sfc[sfc.dset == 1].drop("dset", axis=1)
sfc_test = sfc[sfc.dset == 2].drop("dset", axis=1)

sfc_train.to_pickle(PROCESSED_DATA_DIR / "sfc_train.p")
sfc_val.to_pickle(PROCESSED_DATA_DIR / "sfc_val.p")
sfc_test.to_pickle(PROCESSED_DATA_DIR / "sfc_test.p")
