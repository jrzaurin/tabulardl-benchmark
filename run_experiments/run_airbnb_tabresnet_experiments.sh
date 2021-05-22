python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --early_stop_patience 20 --batchnorm_cont --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dropout 0.4 --mlp_hidden_dims [100,50] --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dims same --mlp_hidden_dims auto --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dims same --mlp_hidden_dims auto --early_stop_patience 20 --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_activation leaky_relu --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_dropout 0.4 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_batchnorm --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_batchnorm_last --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_linear_first --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dims [100,100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --early_stop_patience 20 --batchnorm_cont --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dropout 0.4 --mlp_hidden_dims [100,50] --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dims same --mlp_hidden_dims auto --early_stop_patience 20 --save_results
python airbnb/airbnb_tabresnet.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dims same --mlp_hidden_dims auto --early_stop_patience 20 --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_activation leaky_relu --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_dropout 0.4 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_batchnorm --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_batchnorm_last --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --mlp_linear_first --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer AdamW  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --blocks_dims [100,100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5  --n_epochs 100 --save_results

python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python airbnb/airbnb_tabresnet.py --concat_cont_first --optimizer Adam  --early_stop_patience 20 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [200,100] --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 100 --save_results
