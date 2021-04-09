from add_split_flag import add_split_flag
from coupon_feat_engineering import coupon_feature_eng
from user_feat_engineering import user_feature_eng
from build_final_datasets import build_train_valid_and_test


def full_process():
    add_split_flag()
    coupon_feature_eng()
    user_feature_eng()
    build_train_valid_and_test()


if __name__ == '__main__':
    full_process()
