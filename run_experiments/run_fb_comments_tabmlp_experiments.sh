python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --rop_threshold_mode rel --batchnorm_cont --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer AdamW --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer AdamW --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4  --batch_size 256 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4  --batch_size 256 --rop_threshold_mode rel --batchnorm_cont --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer AdamW --rop_patience 4 --batch_size 256 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer AdamW --rop_patience 4 --batch_size 256 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 256 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 256 --rop_threshold_mode rel --scale_cont --batchnorm_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4  --batch_size 512 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4  --batch_size 1024 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer AdamW --rop_patience 4  --batch_size 512 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer AdamW --rop_patience 4  --batch_size 1024 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4  --batch_size 512 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4  --batch_size 1024 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims auto --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims [100,50] --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims [400,200] --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims [400,200,100] --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --rop_patience 4 --batch_size 256 --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims auto --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims [100,50] --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims [400,200] --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims [400,200,100] --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --rop_patience 4 --batch_size 512 --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam --embed_dropout 0.1 --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --mlp_activation leaky_relu --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --mlp_batchnorm --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --mlp_batchnorm_last --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --mlp_batchnorm --mlp_batchnorm_last --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --mlp_linear_first --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam --mlp_batchnorm --mlp_batchnorm_last --batch_size 256 --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --embed_dropout 0.1 --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --mlp_activation leaky_relu --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --mlp_batchnorm --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --mlp_batchnorm_last --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --mlp_batchnorm --mlp_batchnorm_last --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --mlp_linear_first --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --mlp_batchnorm --mlp_linear_first --batch_size 512 --mlp_hidden_dims [100,50] --scale_cont --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5 --n_epochs 100 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --batch_size 512 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5 --n_epochs 100 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam --batch_size 512 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 32 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 32 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 32 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 32 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 1 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 5 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer Adam  --batch_size 256 --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 50 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam  --batch_size 32 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 1 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam  --batch_size 32 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam  --batch_size 32 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results

python fb_comments/fb_comments_tabmlp.py --optimizer RAdam  --batch_size 512 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 5 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam  --batch_size 512 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python fb_comments/fb_comments_tabmlp.py --optimizer RAdam  --batch_size 512 --mlp_hidden_dims [100,50] --rop_patience 4 --rop_threshold_mode rel --scale_cont --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 50 --save_results

