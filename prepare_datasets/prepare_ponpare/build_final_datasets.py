import os
from pathlib import Path

import pandas as pd

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

DATA_PRODUCTS_DIR = ROOT_DIR / "prepare_datasets/prepare_ponpare/data_products/"
PROCESSED_DATA_DIR = DATA_PRODUCTS_DIR / "processed_data/"

MAIN_PROCESSED_DATA_DIR = ROOT_DIR / "processed_data/ponpare"
if not os.path.isdir(MAIN_PROCESSED_DATA_DIR):
    os.makedirs(MAIN_PROCESSED_DATA_DIR)


def read_and_filtered_train_users():

    # Interactions
    df_purchases = pd.read_pickle(PROCESSED_DATA_DIR / "df_purchases.p")
    df_visits = pd.read_pickle(PROCESSED_DATA_DIR / "df_visits.p")
    df_visits.rename(
        index=str, columns={"view_coupon_id_hash": "coupon_id_hash"}, inplace=True
    )
    df_purchases = df_purchases[["user_id_hash", "coupon_id_hash", "dset"]]
    df_visits = df_visits[["user_id_hash", "coupon_id_hash", "dset"]]

    # users and coupons features
    df_coupons_feat = pd.read_pickle(PROCESSED_DATA_DIR / "df_coupons_feat.p")
    df_user_feat = pd.read_pickle(PROCESSED_DATA_DIR / "df_users_feat.p")

    # remember, we only computed features for users seen during training, and
    # for this excercise here we only focus on those
    df_tr_users_purchases = df_purchases[
        df_purchases.user_id_hash.isin(df_user_feat.user_id_hash.unique())
        & df_purchases.coupon_id_hash.isin(df_coupons_feat.coupon_id_hash.unique())
    ]
    df_tr_users_visits = df_visits[
        df_visits.user_id_hash.isin(df_user_feat.user_id_hash.unique())
        & df_visits.coupon_id_hash.isin(df_coupons_feat.coupon_id_hash.unique())
    ]

    return df_tr_users_purchases, df_tr_users_visits, df_coupons_feat, df_user_feat


def remove_purchases_from_visits(purchases_df, visits_df):

    # remove from the visits table those pairs user-coupon that ended up in
    # purchases
    purchased_id_hash = (
        purchases_df["user_id_hash"] + "_" + purchases_df["coupon_id_hash"]
    ).unique()
    visits_df["purchased_id_hash"] = (
        visits_df["user_id_hash"] + "_" + visits_df["coupon_id_hash"]
    )
    visits_df = visits_df[~visits_df.purchased_id_hash.isin(purchased_id_hash)]
    visits_df.drop("purchased_id_hash", axis=1, inplace=True)

    return visits_df


def split_and_add_target_purchases(df):

    # for purchases catg target = 0
    train_purchases = (
        df[df.dset == 2].drop("dset", axis=1).reset_index(drop=True).drop_duplicates()
    )
    valid_purchases = (
        df[df.dset == 1].drop("dset", axis=1).reset_index(drop=True).drop_duplicates()
    )
    test_purchases = (
        df[df.dset == 0].drop("dset", axis=1).reset_index(drop=True).drop_duplicates()
    )
    train_purchases["target"], valid_purchases["target"], test_purchases["target"] = (
        0,
        0,
        0,
    )

    return train_purchases, valid_purchases, test_purchases


def split_and_add_target_visits(df):

    # for visits target will depend on number of visits
    train_visits = df[df.dset == 2].drop("dset", axis=1).reset_index(drop=True)
    valid_visits = df[df.dset == 1].drop("dset", axis=1).reset_index(drop=True)
    test_visits = df[df.dset == 0].drop("dset", axis=1).reset_index(drop=True)

    train_visits_nviews = (
        train_visits.groupby(["user_id_hash", "coupon_id_hash"]).size().reset_index()
    )
    train_visits_nviews.columns = [
        "user_id_hash",
        "coupon_id_hash",
        "target",
    ]

    valid_visits_nviews = (
        valid_visits.groupby(["user_id_hash", "coupon_id_hash"]).size().reset_index()
    )
    valid_visits_nviews.columns = [
        "user_id_hash",
        "coupon_id_hash",
        "target",
    ]

    test_visits_nviews = (
        test_visits.groupby(["user_id_hash", "coupon_id_hash"]).size().reset_index()
    )
    test_visits_nviews.columns = [
        "user_id_hash",
        "coupon_id_hash",
        "target",
    ]

    return train_visits_nviews, valid_visits_nviews, test_visits_nviews


def add_coupon_and_user_feat(
    df_train, df_valid, df_test, df_coupons_feat, df_user_feat
):

    df_train = pd.merge(df_train, df_coupons_feat, on="coupon_id_hash")
    df_train = pd.merge(df_train, df_user_feat, on="user_id_hash")
    df_valid = pd.merge(df_valid, df_coupons_feat, on="coupon_id_hash")
    df_valid = pd.merge(df_valid, df_user_feat, on="user_id_hash")
    df_test = pd.merge(df_test, df_coupons_feat, on="coupon_id_hash")
    df_test = pd.merge(df_test, df_user_feat, on="user_id_hash")

    return df_train, df_valid, df_test


def build_train_valid_and_test():

    (
        df_tr_users_purchases,
        df_tr_users_visits,
        df_coupons_feat,
        df_user_feat,
    ) = read_and_filtered_train_users()

    df_tr_users_visits = remove_purchases_from_visits(
        df_tr_users_purchases, df_tr_users_visits
    )

    train_purchases, valid_purchases, test_purchases = split_and_add_target_purchases(
        df_tr_users_purchases
    )
    train_visits, valid_visits, test_visits = split_and_add_target_visits(
        df_tr_users_visits
    )

    train_purchases, valid_purchases, test_purchases = add_coupon_and_user_feat(
        train_purchases, valid_purchases, test_purchases, df_coupons_feat, df_user_feat
    )

    train_visits, valid_visits, test_visits = add_coupon_and_user_feat(
        train_visits, valid_visits, test_visits, df_coupons_feat, df_user_feat
    )

    train = (
        pd.concat([train_purchases, train_visits]).sample(frac=1).reset_index(drop=True)
    )
    valid = (
        pd.concat([valid_purchases, valid_visits]).sample(frac=1).reset_index(drop=True)
    )
    test = (
        pd.concat([test_purchases, test_visits]).sample(frac=1).reset_index(drop=True)
    )

    train.to_pickle(MAIN_PROCESSED_DATA_DIR / "ponpare_train.p")
    valid.to_pickle(MAIN_PROCESSED_DATA_DIR / "ponpare_val.p")
    test.to_pickle(MAIN_PROCESSED_DATA_DIR / "ponpare_test.p")


if __name__ == "__main__":
    build_train_valid_and_test()
