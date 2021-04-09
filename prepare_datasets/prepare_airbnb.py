import os
import re
import warnings
from pathlib import Path
from functools import reduce
from itertools import chain
from collections import Counter

import umap
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


warnings.filterwarnings("ignore")

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / "raw_data/airbnb/"
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data/airbnb/"

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

fname = "listings.csv.gz"

airbnb_raw = pd.read_csv(RAW_DATA_DIR / fname, parse_dates=["host_since"])

print(airbnb_raw.shape)
airbnb_raw.head()

# this is just subjective. One can choose some other columns
keep_cols = [
    "id",
    "host_id",
    "host_since",
    "description",
    "host_name",
    "host_neighbourhood",
    "host_listings_count",
    "host_verifications",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "minimum_nights",
    "instant_bookable",
    "reviews_per_month",
]


airbnb = airbnb_raw[keep_cols]
airbnb = airbnb[~airbnb.reviews_per_month.isna()]
airbnb = airbnb[~airbnb.description.isna()]
airbnb = airbnb[~airbnb.host_listings_count.isna()]
airbnb = airbnb[airbnb.host_has_profile_pic == "t"].reset_index(drop=True)
airbnb.drop("host_has_profile_pic", axis=1, inplace=True)
print(airbnb.shape)

# Chronological split
airbnb = airbnb.sort_values("host_since").reset_index(drop=True)
test_size = int(np.ceil(airbnb.shape[0] * 0.1))
train_size = airbnb.shape[0] - test_size * 2

# train
airbnb_train = airbnb.iloc[:train_size].reset_index(drop=True)
tmp = airbnb.iloc[train_size:].reset_index(drop=True)

# valid and test
airbnb_val = tmp.iloc[:test_size].reset_index(drop=True)
airbnb_test = tmp.iloc[test_size:].reset_index(drop=True)

airbnb_train["dset"] = 0
airbnb_val["dset"] = 1
airbnb_test["dset"] = 2

airbnb = pd.concat([airbnb_train, airbnb_val, airbnb_test]).reset_index(drop=True)

# some cols to lower
cols_to_lower = [
    "host_neighbourhood",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "bathrooms_text",
]
for col in cols_to_lower:
    airbnb[col] = airbnb[col].str.lower()

# host_name
host_name = airbnb.host_name.tolist()
d = gender.Detector()
host_gender = [d.get_gender(n) for n in host_name]
replace_dict = {"mostly_male": "male", "mostly_female": "female", "andy": "unknown"}
host_gender = [replace_dict.get(item, item) for item in host_gender]
Counter(host_gender)
airbnb["host_gender"] = host_gender
airbnb.drop("host_name", axis=1, inplace=True)
airbnb.head()

#  host neighbourhood
airbnb.host_neighbourhood.fillna("unknown", inplace=True)

# long and lat
airbnb["x"] = np.cos(airbnb.latitude) * np.cos(airbnb.longitude)
airbnb["y"] = np.cos(airbnb.latitude) * np.sin(airbnb.longitude)
airbnb["z"] = np.sin(airbnb.latitude)
airbnb["longitude"] = airbnb.longitude / 180
airbnb["latitude"] = airbnb.latitude / 90

# property_type and bathrooms_text
airbnb.loc[
    airbnb.property_type == "private room in townhouse", "property_type"
] = "private room in house"


main_property_types = [
    "entire apartment",
    "private room in apartment",
    "private room in house",
    "entire house",
]

airbnb["property_type"] = airbnb.property_type.apply(
    lambda x: "other" if x not in main_property_types else x
)

airbnb["bathrooms_text"][
    (airbnb.bathrooms_text.isna()) & (airbnb.room_type == "private_room")
] = "0 baths"
airbnb["bathrooms_text"][
    (airbnb.bathrooms_text.isna()) & (airbnb.room_type == "entire_home/apt")
] = "1 baths"

main_bathrooms = [
    "1 bath",
    "1 shared bath",
    "2 baths",
    "1 private bath",
    "1.5 baths",
    "1.5 shared baths",
    "2.5 baths",
    "2 shared baths",
    "3 baths",
]

airbnb["bathrooms_text"] = airbnb.bathrooms_text.apply(
    lambda x: "other" if x not in main_bathrooms else x
)

airbnb.bedrooms.fillna(1, inplace=True)
airbnb["bedrooms"] = airbnb.bedrooms.apply(lambda x: -1 if x > 4 else x)
airbnb.beds.fillna(1, inplace=True)
airbnb["beds"] = airbnb.beds.apply(lambda x: -1 if x > 4 else x)

min_nights_stats_per_host = (
    airbnb[["host_id", "minimum_nights"]]
    .groupby("host_id")
    .agg({"minimum_nights": ["min", "max", "median"]})
    .reset_index()
)
min_nights_stats_per_host.columns = ["host_id"] + [
    "_".join(col) for col in min_nights_stats_per_host.columns[1:]
]
airbnb = airbnb.merge(min_nights_stats_per_host, on="host_id")

# host_verifications and amenities


def rm_useless_spaces(s):
    return re.sub(" {2,}", " ", s)


def list_to_multilabel(df, colname, repls, top_n=None, top_n_list=None, mlb=None):

    raw_str = df[colname].str.lower().tolist()
    cleaned_str = [
        rm_useless_spaces(reduce(lambda a, kv: a.replace(*kv), repls, s)).split(", ")
        for s in raw_str
    ]
    if top_n_list is None:
        all_cleaned_str = list(chain(*cleaned_str))
        all_cleaned_str = Counter(all_cleaned_str).most_common()
        top_n_list = [s for s, count in all_cleaned_str[:top_n]]

    list_of_lengths = []
    top_n_lists = []
    for cs in cleaned_str:
        list_of_lengths.append(len(cs))
        top_n_lists.append([s.replace(" ", "_") for s in cs if s in top_n_list])

    top_n_lists = [
        ["_".join([colname, cs]) for cs in cleaned_str] for cleaned_str in top_n_lists
    ]

    if mlb is not None:
        out_df = pd.DataFrame(
            mlb.transform(top_n_lists), columns=mlb.classes_, index=df.index
        )
    else:
        mlb = MultiLabelBinarizer()
        out_df = pd.DataFrame(
            mlb.fit_transform(top_n_lists), columns=mlb.classes_, index=df.index
        )

    out_df["_".join(["n", colname])] = list_of_lengths

    return top_n_list, mlb, out_df


def add_umap_feat_eng(df, colname, mapper):

    try:
        check_is_fitted(mapper, ["embedding_"])
    except NotFittedError:
        # last col is 'list_of_lengths'. We don't use it for UMAP
        mapper.fit(df.values[:, :-1])

    colnames = [
        "_".join([colname, "umap", str(i + 1)]) for i in range(mapper.n_components)
    ]
    df_umap = pd.DataFrame(mapper.transform(df.values[:, :-1]), columns=colnames)
    return mapper, pd.concat([df, df_umap], axis=1)


airbnb_train = airbnb[airbnb.dset == 0].reset_index(drop=True)
airbnb_val = airbnb[airbnb.dset == 1].reset_index(drop=True)
airbnb_test = airbnb[airbnb.dset == 2].reset_index(drop=True)

host_verifications_repls = (("'", ""), ("[", ""), ("]", ""))
top_n_host_verifications = 11
(
    top_host_verifications,
    host_verifications_mlb,
    train_host_verifications,
) = list_to_multilabel(
    airbnb_train,
    "host_verifications",
    host_verifications_repls,
    top_n=top_n_host_verifications,
)
_, _, valid_host_verifications = list_to_multilabel(
    airbnb_val,
    "host_verifications",
    host_verifications_repls,
    top_n_list=top_host_verifications,
    mlb=host_verifications_mlb,
)
_, _, test_host_verifications = list_to_multilabel(
    airbnb_test,
    "host_verifications",
    host_verifications_repls,
    top_n_list=top_host_verifications,
    mlb=host_verifications_mlb,
)

host_verifications_umapper = umap.UMAP(random_state=1)
host_verifications_umapper, train_host_verifications = add_umap_feat_eng(
    train_host_verifications, "host_verifications", host_verifications_umapper
)
_, valid_host_verifications = add_umap_feat_eng(
    valid_host_verifications, "host_verifications", host_verifications_umapper
)
_, test_host_verifications = add_umap_feat_eng(
    test_host_verifications, "host_verifications", host_verifications_umapper
)

host_verifications_df = pd.concat(
    [
        train_host_verifications,
        valid_host_verifications,
        test_host_verifications,
    ]
).reset_index(drop=True)


amenities_repls = (
    ('"', ""),
    ("{", ""),
    ("}", ""),
    ("[", ""),
    ("]", ""),
    (" / ", ""),
    ("/", " "),
    ("(s)", ""),
    ("\\u2019s", ""),
    ("\\u2019n", ""),
    ("\\u2013\\u00a0", ""),
    ("\\u2013", ""),
)
top_n_amenities = 65
(top_amenities, amenities_mlb, train_amenities,) = list_to_multilabel(
    airbnb_train, "amenities", amenities_repls, top_n=top_n_amenities
)
_, _, valid_amenities = list_to_multilabel(
    airbnb_val,
    "amenities",
    amenities_repls,
    top_n_list=top_amenities,
    mlb=amenities_mlb,
)
_, _, test_amenities = list_to_multilabel(
    airbnb_test,
    "amenities",
    amenities_repls,
    top_n_list=top_amenities,
    mlb=amenities_mlb,
)

amenities_umapper = umap.UMAP(random_state=2, n_components=5)
amenities_umapper, train_amenities = add_umap_feat_eng(
    train_amenities, "amenities", amenities_umapper
)
_, valid_amenities = add_umap_feat_eng(valid_amenities, "amenities", amenities_umapper)
_, test_amenities = add_umap_feat_eng(test_amenities, "amenities", amenities_umapper)

amenities_df = pd.concat(
    [
        train_amenities,
        valid_amenities,
        test_amenities,
    ]
).reset_index(drop=True)

airbnb = pd.concat([airbnb, host_verifications_df, amenities_df], axis=1)
airbnb.drop(["host_verifications", "amenities"], axis=1, inplace=True)

# Price
airbnb.price.fillna("$0", inplace=True)
airbnb["price"] = airbnb.price.apply(
    lambda x: x.replace("$", "").replace(",", "")
).astype(float)

# let's make sure there are no nan left
has_nan = airbnb.isnull().any(axis=0)
has_nan = [airbnb.columns[i] for i in np.where(has_nan)[0]]
if not has_nan:
    print("no NaN, all OK")

# Computing a proxi for yield

# Yield is defined as price * occupancy rate. Occupancy rate can be calculated
# by multiplying ((reviews / review rate) * average length of stay), where
# review rate and average length of stay are normally taken as a factor based
# in some model.  For example, in the San Francisco model a review rate of 0.5
# is used to convert reviews to estimated bookings (i.e. we assume that only
# half of the guests will leave a review). An average length of stay of 3
# nights  multiplied by the estimated bookings over a period gives the
# occupancy rate. Therefore, in the expression I have used below, if you want
# to turn my implementation of 'yield' into a "proper" one under the San
# Francisco model assumptions simply multiply my yield by 6 (3 * (1/0.5)) or
# by 72 (3 * 2 * 12) if you prefer per year.
airbnb["yield"] = airbnb["price"] * airbnb["reviews_per_month"]
airbnb.drop(["price", "reviews_per_month"], axis=1, inplace=True)

# Save
airbnb_train = airbnb[airbnb.dset == 0].drop("dset", axis=1)
airbnb_val = airbnb[airbnb.dset == 1].drop("dset", axis=1)
airbnb_test = airbnb[airbnb.dset == 2].drop("dset", axis=1)

airbnb_train.to_pickle(PROCESSED_DATA_DIR / "airbnb_train.p")
airbnb_val.to_pickle(PROCESSED_DATA_DIR / "airbnb_val.p")
airbnb_test.to_pickle(PROCESSED_DATA_DIR / "airbnb_test.p")
