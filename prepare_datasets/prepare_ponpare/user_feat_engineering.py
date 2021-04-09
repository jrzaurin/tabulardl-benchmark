import os
from collections import Counter
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

DATA_PRODUCTS_DIR = ROOT_DIR / "prepare_datasets/prepare_ponpare/data_products/"
TRANSLATED_DATA_DIR = DATA_PRODUCTS_DIR / "ponpare_translated/"
PROCESSED_DATA_DIR = DATA_PRODUCTS_DIR / "processed_data/"


def nan_with_unknown_imputer(df, columns):
    for c in columns:
        df[c] = df[c].fillna("unknown")
    return df


def nan_with_minus_one_imputer(df, columns):
    for c in columns:
        df[c] = df[c].fillna(-1.0).astype("float")
    return df


def top_values(row, top_n=2):
    counts = [c[0] for c in Counter(row).most_common()]
    row_len = len(set(row))
    if row_len < top_n:
        top_vals = counts + [counts[-1]] * (top_n - row_len)
    else:
        top_vals = counts[:top_n]
    return top_vals


def time_diff_stats(row, all_metrics=False):
    if len(row) == 1:
        min_diff = 0
        max_diff = 0
        median_diff = 0
    else:
        row = sorted(row, reverse=True)
        diff = [t - s for t, s in zip(row, row[1:])]
        min_diff = np.min(diff)
        max_diff = np.max(diff)
        median_diff = np.median(diff)
    if all_metrics:
        return [min_diff, max_diff, median_diff]
    else:
        return median_diff


def read_and_filter_train_sets(verbose=True):

    # User features. Since I am going to build features based on how they
    # behave, I will only use those for the benchmarking excercise here. If we
    # were building a recsys, this processing step would be different
    df_users = pd.read_pickle(PROCESSED_DATA_DIR / "df_users.p")
    df_users_tr = df_users[df_users.dset == 2]

    # train coupons features
    df_coupon_feat = pd.read_pickle(PROCESSED_DATA_DIR / "df_coupons_feat.p")

    # Interactions: visits and purchases
    df_visits = pd.read_pickle(PROCESSED_DATA_DIR / "df_visits.p")
    df_visits_tr = df_visits[df_visits.dset == 2]
    df_purchases = pd.read_pickle(PROCESSED_DATA_DIR / "df_purchases.p")
    df_purchases_tr = df_purchases[df_purchases.dset == 2]

    active_users = list(
        set(
            list(df_visits_tr.user_id_hash.unique())
            + list(df_purchases_tr.user_id_hash.unique())
        )
    )
    if verbose:
        inactive_users = np.setdiff1d(
            list(df_users_tr.user_id_hash.unique()), active_users
        )
        withdraw_users = df_users_tr[~df_users_tr.withdraw_date.isna()][
            "user_id_hash"
        ].unique()
        all_users = df_users.user_id_hash.nunique()
        tr_users = df_users_tr.user_id_hash.nunique()
        print(
            f"{tr_users} out of {all_users} users are seen during the training period"
        )
        print(
            f"there are {inactive_users.shape[0]} inactive users (did not buy or visit)."
            " These are dropped"
        )
        print(
            f"there are {withdraw_users.shape[0]} users that did withdraw. These are kept so "
            "that we can learn from there behaviour"
        )

    # drop some not used cols
    df_users_tr.drop(
        ["reg_date", "withdraw_date", "days_to_present"],
        axis=1,
        inplace=True,
    )

    # Focus on active_users
    df_users_tr = df_users_tr[df_users_tr.user_id_hash.isin(active_users)]
    df_visits_tr = df_visits_tr[df_visits_tr.user_id_hash.isin(active_users)]
    df_purchases_tr = df_purchases_tr[df_purchases_tr.user_id_hash.isin(active_users)]

    df_users_tr.pref_name.fillna("unknown", inplace=True)

    return (
        df_users_tr,
        df_coupon_feat,
        df_purchases_tr,
        df_visits_tr,
    )


def agg_feat_eng(df, agg_functions):
    df_agg = df.groupby("user_id_hash").agg(agg_functions).reset_index()
    df_agg.columns = ["user_id_hash"] + ["_".join(pair) for pair in df_agg.columns[1:]]
    return df_agg


def top_n_feat_eng(df, colname, top_n):

    topn_colname = "_".join(["top", colname])
    topn_colnames = ["_".join(["top", str(n), colname]) for n in range(1, top_n + 1)]

    top_n_df = df.groupby("user_id_hash")[colname].apply(list).reset_index()
    top_n_df[topn_colname] = top_n_df[colname].apply(
        lambda x: top_values(x, top_n=top_n)
    )
    for i in range(top_n):
        top_n_df[topn_colnames[i]] = top_n_df[topn_colname].apply(lambda x: x[i])

    top_n_df.drop([colname, topn_colname], axis=1, inplace=True)

    return top_n_df


def time_diff_stats_feat_eng(df, colname, all_metrics):

    time_diff_df = df.groupby("user_id_hash")[colname].apply(list).reset_index()
    time_diff_df["time_diff"] = time_diff_df[colname].apply(
        lambda x: time_diff_stats(x, all_metrics)
    )

    if all_metrics:
        time_diff_colnames = [
            "_".join([colname, suffix])
            for suffix in ["min_diff", "max_diff", "median_diff"]
        ]
        time_diff_df[time_diff_colnames[0]] = time_diff_df.time_diff.apply(
            lambda x: x[0]
        )
        time_diff_df[time_diff_colnames[1]] = time_diff_df.time_diff.apply(
            lambda x: x[1]
        )
        time_diff_df[time_diff_colnames[2]] = time_diff_df.time_diff.apply(
            lambda x: x[2]
        )
        time_diff_df.drop(["time_diff", colname], axis=1, inplace=True)
    else:
        time_diff_colname = "_".join([colname, "median_diff"])
        time_diff_df = time_diff_df.rename(columns={"time_diff": time_diff_colname})
        time_diff_df.drop(colname, axis=1, inplace=True)

    return time_diff_df


def interactions_based_feat_eng(
    purchases_df,
    agg_functions,
    all_metrics,
    top_n,
    interaction_type,
):

    dfc = purchases_df.copy()

    dfc["dayofweek"] = dfc.i_date.dt.dayofweek
    df_dayofweek_agg = agg_feat_eng(dfc, agg_functions)

    df_time_diff = time_diff_stats_feat_eng(
        dfc, colname="days_to_present", all_metrics=all_metrics
    )

    df_top_dayofweek = top_n_feat_eng(dfc, colname="dayofweek", top_n=top_n)
    dfL = [df_dayofweek_agg, df_time_diff, df_top_dayofweek]
    if interaction_type == "purchase":
        df_top_small_areas = top_n_feat_eng(dfc, colname="small_area_name", top_n=top_n)
        dfL.append(df_top_small_areas)

    df_user_feat = reduce(
        lambda left, right: pd.merge(left, right, on=["user_id_hash"]), dfL
    )
    del dfL

    return df_user_feat


def remove_purchases_from_visits(visits_df):

    df_visits_tr_only_visits = visits_df.copy()

    df_visits_tr_only_visits["activity_hash"] = (
        df_visits_tr_only_visits["user_id_hash"]
        + "_"
        + df_visits_tr_only_visits["view_coupon_id_hash"]
    )

    purchases = df_visits_tr_only_visits[
        ~df_visits_tr_only_visits.purchaseid_hash.isna()
    ]["activity_hash"].unique()

    df_visits_tr_only_visits = df_visits_tr_only_visits[
        ~df_visits_tr_only_visits.activity_hash.isin(purchases)
    ][["user_id_hash", "view_coupon_id_hash"]]

    df_visits_tr_only_visits.columns = ["user_id_hash", "coupon_id_hash"]

    return df_visits_tr_only_visits


def unique_coupons_bought_per_cat(df, colname):

    unique_coupons_df = df.pivot_table(
        values="coupon_id_hash",
        index="user_id_hash",
        columns=colname,
        aggfunc=lambda x: len(x.unique()),
    )
    root_colname = colname.replace("_cat", "")
    colnames = [
        "_".join([root_colname, str(cat)])
        for cat in unique_coupons_df.columns.categories
    ]
    unique_coupons_df.columns = colnames
    unique_coupons_df.reset_index(inplace=True)
    unique_coupons_df.fillna(0, inplace=True)

    return unique_coupons_df


def top_n_coupons_bought_per_cat(df, colname, top_n):

    topn_df = df.groupby("user_id_hash")[colname].apply(list).reset_index()
    topn_colname = "top_" + colname
    colnames = ["top" + str(i) + "_" + colname for i in range(1, top_n + 1)]
    topn_df[topn_colname] = topn_df[colname].apply(lambda x: top_values(x, top_n=top_n))
    for i, cn in enumerate(colnames):
        topn_df[cn] = topn_df[topn_colname].apply(lambda x: x[i])
    topn_df.drop([colname, topn_colname], axis=1, inplace=True)

    return topn_df


def coupon_cat_based_feat_eng(df_interactions, df_coupon_features):
    coupons_cols = [
        "capsule_text",
        "genre_name",
        "coupon_id_hash",
        "catalog_price",
        "discount_price",
        "catalog_price_cat",
        "discount_price_cat",
    ]
    df_c = pd.merge(
        df_interactions, df_coupon_features[coupons_cols], on="coupon_id_hash"
    )
    agg_functions_p_c = {
        "catalog_price": ["mean", "median", "min", "max"],
        "discount_price": ["mean", "median", "min", "max"],
    }
    df_price_num = df_c.groupby("user_id_hash").agg(agg_functions_p_c).reset_index()
    df_price_num.columns = ["user_id_hash"] + [
        "_".join(pair) for pair in df_price_num.columns[1:]
    ]

    counts_dfL = [
        unique_coupons_bought_per_cat(df_c, col)
        for col in ["catalog_price_cat", "discount_price_cat"]
    ]
    df_counts_per_cat = reduce(
        lambda left, right: pd.merge(left, right, on=["user_id_hash"]), counts_dfL
    )

    topn_dfL = [
        top_n_coupons_bought_per_cat(df_c, col, top_n=3)
        for col in ["capsule_text", "genre_name"]
    ]
    df_topn_per_cat = reduce(
        lambda left, right: pd.merge(left, right, on=["user_id_hash"]), topn_dfL
    )

    dfL = [df_price_num, df_counts_per_cat, df_topn_per_cat]
    coupon_based_df = reduce(
        lambda left, right: pd.merge(left, right, on=["user_id_hash"]), dfL
    )

    return coupon_based_df


def rename_visits_columns(purchases_df, visits_df):

    rename_cols = {
        col: "view_" + col
        for col in visits_df.columns
        if col in purchases_df.columns and "id_hash" not in col
    }

    visits_df = visits_df.rename(columns=rename_cols)

    return visits_df


def float_to_int(df):
    for c in df.columns:
        if df[c].dtype == "float" and (df[c] - df[c].astype(int) == 0).all():
            df[c] = df[c].astype(int)
        else:
            continue


def user_feature_eng(verbose=False):

    (
        df_users_tr,
        df_coupon_feat,
        df_purchases_tr,
        df_visits_tr,
    ) = read_and_filter_train_sets(verbose=verbose)

    # User features based on purchase behaviour (_p)
    agg_functions_p = {
        "purchaseid_hash": ["count"],
        "coupon_id_hash": ["nunique"],
        "item_count": ["sum"],
        "small_area_name": ["nunique"],
        "dayofweek": ["nunique"],
    }
    df_user_feat_p = interactions_based_feat_eng(
        df_purchases_tr,
        agg_functions=agg_functions_p,
        all_metrics=False,
        top_n=2,
        interaction_type="purchase",
    )

    # User features based on visit behaviour (_v):
    agg_functions_v = {
        "view_coupon_id_hash": ["count", "nunique"],
        "session_id_hash": ["nunique"],
        "dayofweek": ["nunique"],
    }
    df_user_feat_v = interactions_based_feat_eng(
        df_visits_tr,
        agg_functions=agg_functions_v,
        all_metrics=True,
        top_n=2,
        interaction_type="visits",
    )

    # User features based on the coupons they bought (_c). Here possibilities
    # are nearly endless. I am going to focus on price features (catalogue
    # price and price rate) and coupon category features (capsule_text_cat and
    # genre_name_cat)
    df_visits_tr_only_visits = remove_purchases_from_visits(df_visits_tr)
    df_user_feat_c_p = coupon_cat_based_feat_eng(
        df_purchases_tr[["user_id_hash", "coupon_id_hash"]], df_coupon_feat
    )
    df_user_feat_c_v = coupon_cat_based_feat_eng(
        df_visits_tr_only_visits, df_coupon_feat
    )

    # remane columns before merging
    df_user_feat_v = rename_visits_columns(df_user_feat_p, df_user_feat_v)
    df_user_feat_c_v = rename_visits_columns(df_user_feat_c_p, df_user_feat_c_v)

    # Final merge for a DF with all user feats
    user_feat_dfL = [
        df_users_tr,
        df_user_feat_p,
        df_user_feat_v,
        df_user_feat_c_p,
        df_user_feat_c_v,
    ]
    df_user_feat = reduce(
        lambda left, right: pd.merge(left, right, on=["user_id_hash"], how="outer"),
        user_feat_dfL,
    )
    df_user_feat = df_user_feat[~df_user_feat.dset.isna()]

    # Simple NaN imputation
    cat_cols = [
        c
        for c in df_user_feat.columns
        if df_user_feat[c].dtype == "O" and "user_id_hash" not in c
    ]
    num_cols = [
        c
        for c in df_user_feat.columns
        if df_user_feat[c].dtype == "float" and c != "dset"
    ]

    df_user_feat = nan_with_unknown_imputer(df_user_feat, cat_cols)
    df_user_feat = nan_with_minus_one_imputer(df_user_feat, num_cols)

    #  count variables to adequate type
    float_to_int(df_user_feat)

    # save
    df_user_feat.to_pickle(PROCESSED_DATA_DIR / "df_users_feat.p")


if __name__ == "__main__":
    user_feature_eng()
