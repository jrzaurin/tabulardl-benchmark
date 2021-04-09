import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

DATA_PRODUCTS_DIR = ROOT_DIR / "prepare_datasets/prepare_ponpare/data_products/"
TRANSLATED_DATA_DIR = DATA_PRODUCTS_DIR / "ponpare_translated/"
PROCESSED_DATA_DIR = DATA_PRODUCTS_DIR / "processed_data/"


def validperiod_as_cat(df, q, suffix, bins=None):
    r"""
    df mutates inside...is ok, the function takes a copy

    Method 1: NaN as another category. Maybe we want to consider coupons with
    a validperiod of 0 as a class. For now, we will simply use quantiles
    """

    colname = "_".join(["validperiod", suffix])
    labels = np.arange(q)

    if bins is not None:
        df[colname] = pd.cut(
            df["validperiod"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    else:
        df[colname], bins = pd.qcut(df["validperiod"], q=q, labels=labels, retbins=True)
        df[colname].cat.add_categories([q], inplace=True)
        df[colname].fillna(q, inplace=True)

    return df, bins


def validperiod_nan_impute(df):
    """
    df mutates inside...is ok, the function takes a copy

    imputation logic:
    for capsule_text in capsule_texts:

        if there are more than 50 non nan locations and one or more
        non nan validperiods, replace with random sampling

        if there are less than or equal to 50 non nan locations and one or
        more non nan validperiods, replace with median

        if there are no non nan validperiods, replace with overall mode
    """

    validperiod_mode = df.validperiod.mode().values[0]
    capsule_texts = list(df[df.validperiod.isna()]["capsule_text"].value_counts().index)

    for ct in capsule_texts:
        validperiods = df[(df.capsule_text == ct) & (~df.validperiod.isna())][
            "validperiod"
        ].values
        nan_idx = list(df[(df.capsule_text == ct) & (df.validperiod.isna())].index)
        if (len(nan_idx) > 50) & (validperiods.size > 0):
            np.random.seed(1)
            replace_vals = np.random.choice(validperiods, len(nan_idx))
            df.loc[nan_idx, "validperiod"] = replace_vals
        elif (len(nan_idx) <= 50) & (validperiods.size > 0):
            median = np.median(validperiods)
            df.loc[nan_idx, "validperiod"] = median
        elif validperiods.size == 0:
            df.loc[nan_idx, "validperiod"] = validperiod_mode

    return df


def validfrom_day_of_week_nan_impute(df, new_colname, replace_constant):
    """
    df mutates inside...is ok, the function takes a copy

    whenever validfrom is NaN, validend is also NaN, therefore, we are only
    going to process validfrom, since validfrom + the already processed
    validperiod are validend

    imputation logic:
    for capsule_text in capsule_texts:

        if there are more than 50 non nan locations and one or more non nan
        new_colname (validfrom_dayofweek_method2_cat) values, replace with
        random sampling

        if there are less than or equal to 50 non nan locations and one or
        more non nan new_colname values, replace with mode per capsule_text

        if there are no non nan new_colname value, replace with overall mode
        referred as 'replace_constant'
    """

    # new_colname = "validfrom_dayofweek" + "_method2_cat"
    df[new_colname] = df.validfrom.dt.dayofweek
    capsule_texts = list(
        df[df[new_colname].isna()]["capsule_text"].value_counts().index
    )
    for ct in capsule_texts:
        non_nan_values = df[(df.capsule_text == ct) & (~df[new_colname].isna())][
            new_colname
        ].values
        nan_idx = list(df[(df.capsule_text == ct) & (df[new_colname].isna())].index)
        if (len(nan_idx) > 50) & (non_nan_values.size > 0):
            replace_vals = np.random.choice(non_nan_values, len(nan_idx))
            df.loc[nan_idx, new_colname] = replace_vals
        elif (len(nan_idx) <= 50) & (non_nan_values.size > 0):
            mode = stats.mode(non_nan_values).mode[0]
            df.loc[nan_idx, new_colname] = mode
        elif non_nan_values.size == 0:
            df.loc[nan_idx, new_colname] = replace_constant

    df[new_colname] = df[new_colname].astype("int")

    return df


def validperiod_feat_eng(df):
    """
    Engineers features from validperiod.

    Wrapper around validperiod_as_cat and validperiod_nan_impute
    """

    dfc = df.copy()

    dfc, bins = validperiod_as_cat(dfc, q=4, suffix="method1")

    dfc = validperiod_nan_impute(dfc)
    dfc, _ = validperiod_as_cat(dfc, q=4, suffix="method2", bins=bins)

    dfc["validperiod"] = dfc["validperiod"].astype("int")

    return dfc, bins


def validfrom_feat_eng(df):
    """
    Engineers features from validfrom.
    """

    dfc = df.copy()

    validfrom_dayofweek_mode = dfc["validfrom"].dt.dayofweek.mode().values[0]

    new_colname = "validfrom_dayofweek_method1"
    dfc[new_colname] = dfc.validfrom.dt.dayofweek
    dfc[new_colname] = dfc[new_colname].fillna(7).astype("int")

    dfc = validfrom_day_of_week_nan_impute(
        dfc,
        new_colname="validfrom_dayofweek_method2",
        replace_constant=validfrom_dayofweek_mode,
    )

    return dfc


def dispfrom_dispend_feat_eng(df):
    """
    Engineers features from dispfrom and dispend.
    """

    dfc = df.copy()

    dfc["dispfrom_dayofweek"] = dfc.dispfrom.dt.dayofweek
    dfc["dispend_dayofweek"] = dfc.dispend.dt.dayofweek

    dfc["dispperiod_cat"], dispperiod_bins = pd.qcut(
        dfc.dispperiod, q=4, labels=[0, 1, 2, 3], retbins=True
    )

    return dfc, dispperiod_bins


def price_feat_eng(df):
    """
    Engineers features price related features
    """

    dfc = df.copy()

    dfc["price_rate_cat"], price_rate_bins = pd.qcut(
        dfc["price_rate"], q=3, labels=[0, 1, 2], retbins=True
    )

    dfc["catalog_price_cat"], catalog_price_bins = pd.qcut(
        dfc["catalog_price"], q=3, labels=[0, 1, 2], retbins=True
    )

    dfc["discount_price_cat"], discount_price_bins = pd.qcut(
        dfc["discount_price"], q=3, labels=[0, 1, 2], retbins=True
    )

    return dfc, price_rate_bins, catalog_price_bins, discount_price_bins


def coupon_feature_eng():

    # We will assume that we know the coupons that will go live beforehand and
    # that we have the time to compute the features using the whole dataset of
    # coupons

    # Coupon features
    df_coupons = pd.read_pickle(PROCESSED_DATA_DIR / "df_coupons.p")

    # columns with NaN
    # has_nan = df_coupons.isnull().any(axis=0)
    # has_nan = [df_coupons.columns[i] for i in np.where(has_nan)[0]]
    has_nan = [
        "validfrom",
        "validend",
        "validperiod",
        "usable_date_mon",
        "usable_date_tue",
        "usable_date_wed",
        "usable_date_thu",
        "usable_date_fri",
        "usable_date_sat",
        "usable_date_sun",
        "usable_date_holiday",
        "usable_date_before_holiday",
    ]

    # All features with Nan are time related features. usable_date_day has
    # values of 0,1 and 2, so I am going to replace NaN with another value: 3,
    # as if this was another category of coupons.
    for col in has_nan[3:]:
        df_coupons[col] = df_coupons[col].fillna(3).astype("int")

    # 6147 of the validperiod entries are NaN. 5821 are "Delivery service". In
    # the case of "Correspondence course" all observations have validperiod
    # NaN and "Other" and "Leisure" only have one entry with NaN. There is an
    # option that those coupons with no validperiod last "forever". Therefore,
    # to start with, we will create a categorical feature grouping and
    # defining a new category coupons with no valid period
    df_validperiod, validperiod_bins = validperiod_feat_eng(df_coupons)

    # Let's now take care of validfrom. Whenever validfrom is NaN, validend is
    # also NaN, therefore, we are only going to process validfrom, since
    # validfrom + the already processed validperiod are validend
    df_validfrom = validfrom_feat_eng(df_validperiod)

    # let's do dispfrom/dispend and dispperiod
    df_dispperiod, dispperiod_bins = dispfrom_dispend_feat_eng(df_validfrom)

    # Finally price related features
    (
        df_coupons_feat,
        price_rate_bins,
        catalog_price_bins,
        discount_price_bins,
    ) = price_feat_eng(df_dispperiod)

    # Add bins to a dict_of_mappings
    dict_of_mappings = {}
    dict_of_mappings["validperiod_cat"] = validperiod_bins
    dict_of_mappings["dispperiod_cat"] = dispperiod_bins
    dict_of_mappings["price_rate_cat"] = price_rate_bins
    dict_of_mappings["catalog_price_cat"] = catalog_price_bins
    dict_of_mappings["discount_price_cat"] = discount_price_bins

    # drop useless cols
    drop_cols = ["dispfrom", "dispend", "validfrom", "validend", "days_to_present"]
    df_coupons_feat.drop(drop_cols, axis=1, inplace=True)

    # save objects
    df_coupons_feat.to_pickle(PROCESSED_DATA_DIR / "df_coupons_feat.p")
    pickle.dump(dict_of_mappings, open(PROCESSED_DATA_DIR / "dict_of_mappings.p", "wb"))


if __name__ == "__main__":
    coupon_feature_eng()
