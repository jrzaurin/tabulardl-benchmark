python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 4096 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer RAdam --batch_size 512 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer RAdam --batch_size 1024 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer RAdam --batch_size 4096 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 4 --n_blocks 4 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --n_blocks 4 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 4 --n_blocks 4 --n_blocks 4 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 32 --n_heads 4 --n_blocks 4 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 32 --n_heads 8 --n_blocks 4 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 32 --n_heads 4 --n_blocks 6 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 32 --n_heads 8 --n_blocks 6 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 4 --n_blocks 4 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --n_blocks 4 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 4 --n_blocks 4 --n_blocks 4 --dropout 0.2 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 32 --n_heads 4 --n_blocks 4 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 32 --n_heads 8 --n_blocks 4 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 32 --n_heads 4 --n_blocks 6 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 32 --n_heads 8 --n_blocks 6 --rop_patience 4 --dropout 0.2 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --mlp_hidden_dims same --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 32 --n_heads 8 --n_blocks 6 --shared_embed --frac_shared_embed 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 32 --n_heads 8 --n_blocks 6 --shared_embed --frac_shared_embed 8 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5 --n_epochs 100 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 5e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 1 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 5e-4 --max_lr 0.05 --final_div_factor 1e3 --n_epochs 5 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer Adam --batch_size 1024 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 5e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --mlp_hidden_dims same --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 1024 --input_dim 32 --n_heads 8 --n_blocks 6 --shared_embed --frac_shared_embed 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 1024 --input_dim 32 --n_heads 8 --n_blocks 6 --shared_embed --frac_shared_embed 8 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --save_results

python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5 --n_epochs 100 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 5e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python fb_comments/fb_comments_tabtransformer.py --optimizer AdamW --batch_size 4096 --input_dim 16 --n_heads 2 --n_blocks 4 --rop_patience 4 --rop_threshold_mode rel --early_stop_patience 15 --lr_scheduler OneCycleLR --lr 5e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 20 --save_results
