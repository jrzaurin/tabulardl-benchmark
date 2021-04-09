import os
from pathlib import Path

import pandas as pd

pd.options.display.max_columns = 100

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / "raw_data/ponpare/"
DOCUMENTATION_DIR = RAW_DATA_DIR / "documentation/"

DATA_PRODUCTS_DIR = ROOT_DIR / "prepare_datasets/prepare_ponpare/data_products/"
TRANSLATED_DATA_DIR = DATA_PRODUCTS_DIR / "ponpare_translated/"

TRANSLATE_FNAME = DOCUMENTATION_DIR / "CAPSULE_TEXT_Translation.xlsx"
PREFECTURE_FNAME = RAW_DATA_DIR / "prefecture.txt"

if not os.path.isdir(DATA_PRODUCTS_DIR):
    os.makedirs(DATA_PRODUCTS_DIR)

if not os.path.isdir(TRANSLATED_DATA_DIR):
    os.makedirs(TRANSLATED_DATA_DIR)

translate_df = pd.read_excel(
    os.path.join(DOCUMENTATION_DIR, TRANSLATE_FNAME), skiprows=5
)

caps_col_idx = [i for i, c in enumerate(translate_df.columns) if "CAPSULE" in c]
engl_col_idx = [i for i, c in enumerate(translate_df.columns) if "English" in c]

capsule_text_df = translate_df.iloc[:, [caps_col_idx[0], engl_col_idx[0]]]
capsule_text_df.columns = ["capsule_text", "english_translation"]

genre_name_df = translate_df.iloc[:, [caps_col_idx[1], engl_col_idx[1]]]
genre_name_df.columns = ["genre_name", "english_translation"]
genre_name_df = genre_name_df[~genre_name_df.genre_name.isna()]

# create capsule_text and genre_name dictionaries
capsule_text_dict = dict(
    zip(capsule_text_df.capsule_text, capsule_text_df.english_translation)
)
genre_name_dict = dict(zip(genre_name_df.genre_name, genre_name_df.english_translation))

# create prefecture dictionary for region/area translation
prefecture_dict = {}
prefecture_path = os.path.join(RAW_DATA_DIR, PREFECTURE_FNAME)
with open(prefecture_path, "r") as f:
    pref_names = f.readlines()
    for pname in pref_names:
        jp2eng = pname.rstrip().split(",")
        prefecture_dict[jp2eng[0]] = jp2eng[1]

csv_files = [
    f.name for f in RAW_DATA_DIR.glob("*.csv") if f.name != "sample_submission.csv"
]

# define a dictionary with the columns to replace and the dictionary to
# replace them
replace_cols = {
    "capsule_text": "capsule_text_dict",
    "genre_name": "genre_name_dict",
    "pref_name": "prefecture_dict",
    "large_area_name": "prefecture_dict",
    "ken_name": "prefecture_dict",
    "small_area_name": "prefecture_dict",
}

for fname in csv_files:
    print(
        "INFO: translating {} into {}".format(
            str(RAW_DATA_DIR / fname), str(TRANSLATED_DATA_DIR / fname)
        )
    )

    tmp_df = pd.read_csv(RAW_DATA_DIR / fname)
    tmp_df.columns = [c.lower() for c in tmp_df]
    for col in tmp_df.columns:
        if col in replace_cols.keys():
            replace_dict = eval(replace_cols[col])
            tmp_df[col].replace(replace_dict, inplace=True)
    tmp_df.to_csv(TRANSLATED_DATA_DIR / fname, index=False)
