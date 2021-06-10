python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.01 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.01 --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer AdamW --rop_threshold_mode rel --lr 0.01 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer AdamW --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer AdamW --rop_threshold_mode rel --lr 0.01 --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --ghost_bn --virtual_batch_size 128 --optimizer Adam --dropout 0.5 --rop_threshold_mode rel --lr 0.01 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --ghost_bn --virtual_batch_size 128 --optimizer Adam --dropout 0.5 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --ghost_bn --virtual_batch_size 128 --optimizer Adam --dropout 0.5 --rop_threshold_mode rel --lr 0.01 --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --momentum 0.25 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --momentum 0.50 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --momentum 0.75 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --gamma 1.25 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --gamma 1.75 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --gamma 2.00 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --lambda_sparse 0.01 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --lambda_sparse 0.001 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam --batch_size 1024 --step_dim 8 --attn_dim 8 --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --lambda_sparse 0.00001 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.005 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.005 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.005 --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.01 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.01 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --lr 0.01 --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --lr 0.005 --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --lr 0.005 --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --lr 0.005 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --lr 0.01 --scale_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --lr 0.01 --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --lr 0.01 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 3 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 3 --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --optimizer Adam  --batch_size 8192 --ghost_bn --virtual_batch_size 128 --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --momentum 0.75  --lambda_sparse 0.001 --gamma 2.00  --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 3 --step_dim 8 --attn_dim 8 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 3 --step_dim 8 --attn_dim 8  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 3 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 8 --attn_dim 8 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 8 --attn_dim 8  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --momentum 0.1 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --momentum 0.2 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --momentum 0.4 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --momentum 0.6 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --gamma 1.25 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --gamma 1.75 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --gamma 2.00 --save_results

python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --lambda_sparse 0. --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --lambda_sparse 0.001 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --lambda_sparse 0.00001 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --lambda_sparse 0. --weight_decay 0.001 --save_results
python nyc_taxi/nyc_taxi_tabnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --n_steps 7 --step_dim 32 --attn_dim 32  --dropout 0.2 --scale_cont --batchnorm_cont --rop_threshold_mode rel --rop_patience 4 --early_stop_patience 15 --lambda_sparse 0. --weight_decay 0.0001 --save_results
