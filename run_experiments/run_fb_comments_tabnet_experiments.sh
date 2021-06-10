python fb_comments/fb_comments_tabnet.py --optimizer Adam --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer RAdam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 5 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 7 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --n_steps 5 --dropout 0.2  --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --n_steps 7 --dropout 0.2  --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --step_dim 8 --attn_dim 8 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 5 --step_dim 8 --attn_dim 8 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 7 --step_dim 8 --attn_dim 8 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --n_steps 3 --step_dim 8 --attn_dim 8 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --n_steps 5 --step_dim 8 --attn_dim 8 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --n_steps 7 --step_dim 8 --attn_dim 8 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 128 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_con --rop_patience 4 --rop_threshold_mode rel --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 256 --scale_cont --batchnorm_cont --rop_patience 4 --rop_threshold_mode rel --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.1 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.3 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.5 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.7 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --gamma 1.25 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --gamma 1.75 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --gamma 2.00 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0. --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0.01 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0.001 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0.0001 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0. --weight_decay 0.001 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0. --weight_decay 0.0001 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.1 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.3 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.5 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.7 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --gamma 1.25 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --gamma 1.75 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --gamma 2.00 --save_results

python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0. --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0.01 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0.001 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0.0001 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0. --weight_decay 0.001 --save_results
python fb_comments/fb_comments_tabnet.py --optimizer Adam --batch_size 512 --n_steps 3 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --momentum 0.9 --lambda_sparse 0. --weight_decay 0.0001 --save_results

