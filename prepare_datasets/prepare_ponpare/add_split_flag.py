import os
from pathlib import Path

import pandas as pd

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

DATA_PRODUCTS_DIR = ROOT_DIR / "prepare_datasets/prepare_ponpare/data_products/"
TRANSLATED_DATA_DIR = DATA_PRODUCTS_DIR / "ponpare_translated/"
PROCESSED_DATA_DIR = DATA_PRODUCTS_DIR / "processed_data/"
if not PROCESSED_DATA_DIR.is_dir():
    PROCESSED_DATA_DIR.mkdir()

TESTING_PERIOD = 28


def flag_dset(df, date_column, present, TESTING_PERIOD):
    """
    flag data (0,1,2) depending on whether the dataset is (test,val,train)
    """
    df = days_to_present_col(df, present, date_column)
    df["dset"] = df.days_to_present.apply(
        lambda x: 0
        if x <= TESTING_PERIOD - 1
        else 1
        if ((x > TESTING_PERIOD - 1) and (x <= (TESTING_PERIOD * 2) - 1))
        else 2
    )
    return df


def days_to_present_col(df, present, date_column):
    tmp_df = pd.DataFrame({"present": [present] * df.shape[0]})
    df["days_to_present"] = tmp_df["present"] - df[date_column]
    df["days_to_present"] = df.days_to_present.dt.days
    return df


def read_and_parse_dates():

    df_users = pd.read_csv(
        TRANSLATED_DATA_DIR / "user_list.csv", parse_dates=["reg_date"]
    )
    df_coupons = pd.read_csv(
        TRANSLATED_DATA_DIR / "coupon_list_train.csv",
        parse_dates=["dispfrom", "dispend", "validfrom", "validend"],
    )
    df_purchases = pd.read_csv(
        TRANSLATED_DATA_DIR / "coupon_detail_train.csv", parse_dates=["i_date"]
    )
    df_visits = pd.read_csv(
        TRANSLATED_DATA_DIR / "coupon_visit_train.csv", parse_dates=["i_date"]
    )

    return df_users, df_coupons, df_purchases, df_visits


def add_flag_and_save_dset(df, df_name, date_column, present):
    fname = PROCESSED_DATA_DIR.joinpath(df_name + ".p")
    print(f"INFO: Saving {df_name} to {fname}")
    flagged_df = flag_dset(df, date_column, present, TESTING_PERIOD)
    flagged_df.to_pickle(fname)


def add_split_flag():

    df_users, df_coupons, df_purchases, df_visits = read_and_parse_dates()

    present = max([df["i_date"].max() for df in [df_visits, df_purchases]])

    add_flag_and_save_dset(df_visits, "df_visits", "i_date", present)
    add_flag_and_save_dset(df_purchases, "df_purchases", "i_date", present)
    add_flag_and_save_dset(df_users, "df_users", "reg_date", present)
    add_flag_and_save_dset(df_coupons, "df_coupons", "dispfrom", present)


if __name__ == "__main__":
    add_split_flag()
