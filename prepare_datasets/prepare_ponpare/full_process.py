from add_split_flag import add_split_flag
from build_final_datasets import build_train_valid_and_test
from coupon_feat_engineering import coupon_feature_eng
from user_feat_engineering import user_feature_eng


def full_process():
    # after running translate
    add_split_flag()
    coupon_feature_eng()
    user_feature_eng()
    build_train_valid_and_test()


if __name__ == "__main__":
    full_process()
