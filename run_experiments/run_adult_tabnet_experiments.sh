# python adult/adult_tabnet.py --optimizer Adam --save_results
# python adult/adult_tabnet.py --optimizer AdamW --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --save_results

# python adult/adult_tabnet.py --optimizer Adam --monitor val_acc --rop_mode max --save_results
# python adult/adult_tabnet.py --optimizer AdamW --monitor val_acc --rop_mode max --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --monitor val_acc --rop_mode max --save_results
# python adult/adult_tabnet.py --optimizer RAdam --monitor val_acc --rop_mode max --save_results

# python adult/adult_tabnet.py --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 4096 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 4096 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 4096 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 4096 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --batch_size 4096 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --batch_size 1024 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --batch_size 4096 --ghost_bn --virtual_batch_size 128 --save_results

# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --n_steps 7 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --n_steps 7 --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --dropout 0.2 --n_steps 7 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --n_steps 7 --dropout 0.2 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --n_steps 7 --dropout 0.2 --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results

# python adult/adult_tabnet.py --optimizer Adam --step_dim 32 --attn_dim 32 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --step_dim 32 --attn_dim 32 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --step_dim 32 --attn_dim 32 --lr 0.05 --save_results
# python adult/adult_tabnet.py --optimizer Adam --step_dim 64 --attn_dim 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --step_dim 64 --attn_dim 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --step_dim 64 --attn_dim 64 --lr 0.05 --save_results
# python adult/adult_tabnet.py --optimizer Adam --dropout 0.1 --step_dim 32 --attn_dim 32 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --dropout 0.1 --step_dim 32 --attn_dim 32 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --dropout 0.1 --step_dim 32 --attn_dim 32 --lr 0.05 --save_results
# python adult/adult_tabnet.py --optimizer Adam --dropout 0.2 --step_dim 64 --attn_dim 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --dropout 0.2 --step_dim 64 --attn_dim 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --dropout 0.2 --step_dim 64 --attn_dim 64 --lr 0.05 --save_results

# python adult/adult_tabnet.py --optimizer Adam --step_dim 32 --attn_dim 32 --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --step_dim 32 --attn_dim 32 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --step_dim 32 --attn_dim 32 --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --step_dim 64 --attn_dim 64 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --step_dim 64 --attn_dim 64 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --step_dim 64 --attn_dim 64 --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --dropout 0.1 --step_dim 32 --attn_dim 32 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --dropout 0.1 --step_dim 32 --attn_dim 32 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05  --dropout 0.1 --step_dim 32 --attn_dim 32 --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer Adam --dropout 0.2 --step_dim 64 --attn_dim 64 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --dropout 0.2 --step_dim 64 --attn_dim 64 --batch_size 512 --ghost_bn --virtual_batch_size 64 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --dropout 0.2 --step_dim 64 --attn_dim 64 --batch_size 1024 --ghost_bn --virtual_batch_size 64 --save_results

# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --momentum 0.25 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --momentum 0.50 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --momentum 0.75 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --gamma 1.25 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --gamma 1.75 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --gamma 2.00 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --lambda_sparse 0.001 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --lambda_sparse 0.00001 --save_results

# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --momentum 0.25 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --momentum 0.50 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --momentum 0.75 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --gamma 1.25 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --gamma 1.75 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --gamma 2.00 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --lambda_sparse 0.001 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 64 --lambda_sparse 0.00001 --save_results

# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --momentum 0.25 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --momentum 0.50 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --momentum 0.75 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --gamma 1.25 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --gamma 1.75 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --gamma 2.00 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --lambda_sparse 0.001 --save_results
# python adult/adult_tabnet.py --optimizer RAdam --lr 0.05 --n_steps 7 --lambda_sparse 0.00001 --save_results

# python adult/adult_tabnet.py --optimizer RAdam --n_steps 7 --gamma 1.25 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.05 --n_cycles 10 --n_epochs 100 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.25 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.05 --n_cycles 10 --n_epochs 100 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.50 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.05 --n_cycles 10 --n_epochs 100 --save_results

# python adult/adult_tabnet.py --optimizer RAdam --n_steps 7 --gamma 1.25 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.1 --final_div_factor 1e3 --n_epochs 1 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.25 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.1 --final_div_factor 1e3 --n_epochs 1 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.50 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.1 --final_div_factor 1e3 --n_epochs 1 --save_results

# python adult/adult_tabnet.py --optimizer RAdam --n_steps 7 --gamma 1.25 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.07 --final_div_factor 1e3 --n_epochs 5 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.25 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.07 --final_div_factor 1e3 --n_epochs 5 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.50 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.07 --final_div_factor 1e3 --n_epochs 5 --save_results

# python adult/adult_tabnet.py --optimizer RAdam --n_steps 7 --gamma 1.25 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.07 --final_div_factor 1e3 --n_epochs 10 --save_results
# python adult/adult_tabnet.py --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.25 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.07 --final_div_factor 1e3 --n_epochs 10 --save_results
# python adult/adult_tabnet.py --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 128 --n_steps 5 --gamma 1.50 --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.07 --final_div_factor 1e3 --n_epochs 10 --save_results

python adult/adult_tabnet.py --lr 0.03 --optimizer Adam --save_results
python adult/adult_tabnet.py --lr 0.03 --optimizer AdamW --save_results

python adult/adult_tabnet.py --lr 0.03 --optimizer Adam --batch_size 512 --save_results
python adult/adult_tabnet.py --lr 0.03 --optimizer AdamW --batch_size 512 --save_results

python adult/adult_tabnet.py --lr 0.03 --optimizer Adam --batch_size 1024 --save_results
python adult/adult_tabnet.py --lr 0.03 --optimizer AdamW --batch_size 1024 --save_results

python adult/adult_tabnet.py --lr 0.03 --optimizer Adam --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results
python adult/adult_tabnet.py --lr 0.03 --optimizer AdamW --batch_size 512 --ghost_bn --virtual_batch_size 128 --save_results

python adult/adult_tabnet.py --lr 0.03 --optimizer Adam --batch_size 1024 --ghost_bn --virtual_batch_size 256 --save_results
python adult/adult_tabnet.py --lr 0.03 --optimizer AdamW --batch_size 1024 --ghost_bn --virtual_batch_size 256 --save_results
