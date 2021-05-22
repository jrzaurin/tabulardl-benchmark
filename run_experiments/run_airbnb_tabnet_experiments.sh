python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --batch_size 512 --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 512 --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 512 --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 1024 --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --batch_size 512 --optimizer Adam --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 512 --optimizer Adam --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 512 --optimizer Adam --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --batch_size 1024 --optimizer Adam --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 1024 --optimizer Adam --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --batch_size 1024 --optimizer Adam --ghost_bn --virtual_batch_size 128 --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 8 --attn_dim 8 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 8  --attn_dim 8  --n_steps 3 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 32 --attn_dim 32 --n_steps 3 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 3 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 8  --attn_dim 8  --n_steps 4 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 32 --attn_dim 32 --n_steps 4 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 32 --attn_dim 32 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --momentum 0.25 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --momentum 0.50 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --momentum 0.75 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --gamma 1.25 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --gamma 1.75 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --gamma 2.00 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --lambda_sparse 0.01    --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --lambda_sparse 0.001   --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --lambda_sparse 0.00001 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --momentum 0.25 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --momentum 0.50 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --momentum 0.75 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --gamma 1.25 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --gamma 1.75 --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --gamma 2.00 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --lambda_sparse 0.01    --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --lambda_sparse 0.001   --scale_cont --batchnorm_cont --save_results
python airbnb/airbnb_tabnet.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --step_dim 16 --attn_dim 16 --n_steps 4 --lambda_sparse 0.00001 --scale_cont --batchnorm_cont --save_results
