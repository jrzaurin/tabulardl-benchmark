python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --optimizer Adam  --lr 0.01 --rop_threshold_mode rel --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --optimizer AdamW --lr 0.01 --rop_threshold_mode rel --scale_cont --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --optimizer Adam  --lr 0.01 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --optimizer AdamW --lr 0.01 --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --optimizer Adam  --lr 0.01 --rop_threshold_mode rel --scale_cont --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --optimizer AdamW --lr 0.01 --rop_threshold_mode rel --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims [100,50] --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims [400,200] --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims [400,200,100] --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims [100,50] --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims [400,200] --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims [400,200,100] --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_dropout 0.5 --mlp_hidden_dims auto --optimizer Adam --lr 0.01 --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_dropout 0.2 --mlp_hidden_dims auto --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_dropout 0.2 --mlp_hidden_dims auto --optimizer AdamW --lr 0.01 --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --embed_dropout 0.1 --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_activation leaky_relu --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm_last --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm --mlp_batchnorm_last --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_linear_first --optimizer Adam --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm --mlp_linear_first --optimizer Adam --lr 0.01 --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --embed_dropout 0.1 --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_activation leaky_relu --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm_last --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm --mlp_batchnorm_last --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_linear_first --optimizer AdamW --lr 0.01 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --scale_cont --mlp_hidden_dims auto --mlp_batchnorm --mlp_linear_first --optimizer AdamW --lr 0.01 --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --mlp_hidden_dims auto --mlp_dropout 0.2 --optimizer Adam --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.05 --n_cycles 10 --n_epochs 100 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --mlp_hidden_dims auto --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.05 --n_cycles 10 --n_epochs 100 --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --mlp_hidden_dims auto --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 10 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --mlp_hidden_dims auto --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 10 --save_results

python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --mlp_hidden_dims auto --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 100 --early_stop_patience 100 --save_results
python nyc_taxi/nyc_taxi_tabmlp.py --batch_size 1024 --mlp_hidden_dims auto --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 100 --early_stop_patience 100 --save_results
