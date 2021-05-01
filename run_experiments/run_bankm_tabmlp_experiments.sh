python bank_marketing/bankm_tabmlp.py --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --optimizer RAdam --lr 0.03 --save_results
python bank_marketing/bankm_tabmlp.py --optimizer RAdam --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims auto --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200] --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200,100] --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims auto --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200] --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200,100] --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims auto --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200] --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200,100] --optimizer RAdam --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims auto --mlp_dropout 0.5 --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims auto --mlp_dropout 0.5 --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims auto --mlp_dropout 0.5 --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --optimizer RAdam --save_results

python bank_marketing/bankm_tabmlp.py --optimizer Adam  --batch_size 512 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer Adam --batch_size 512 --save_results
python bank_marketing/bankm_tabmlp.py --optimizer Adam  --batch_size 1024 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer Adam --batch_size 1024 --save_results
python bank_marketing/bankm_tabmlp.py --optimizer AdamW  --batch_size 512 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer AdamW --batch_size 512 --save_results
python bank_marketing/bankm_tabmlp.py --optimizer AdamW  --batch_size 1024 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer AdamW --batch_size 1024 --save_results
python bank_marketing/bankm_tabmlp.py --optimizer RAdam  --batch_size 512 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer RAdam --batch_size 512 --save_results
python bank_marketing/bankm_tabmlp.py --optimizer RAdam  --batch_size 1024 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --optimizer RAdam --batch_size 1024 --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_dropout 0.2 --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --embed_dropout 0.1 --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_activation leaky_relu --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm_last --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_batchnorm_last --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_linear_first --optimizer Adam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_linear_first --optimizer Adam --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_dropout 0.2 --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --embed_dropout 0.1 --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_activation leaky_relu --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm_last --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_batchnorm_last --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_linear_first --optimizer AdamW --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_linear_first --optimizer AdamW --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_dropout 0.2 --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --embed_dropout 0.1 --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_activation leaky_relu --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm_last --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_batchnorm_last --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_linear_first --optimizer RAdam --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_linear_first --optimizer RAdam --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler CyclicLR --batch_size 64 --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler CyclicLR --batch_size 64 --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler CyclicLR --batch_size 64 --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [200,100] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results

python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_batchnorm_last --optimizer Adam --focal_loss --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --mlp_batchnorm --mlp_batchnorm_last --optimizer AdamW --focal_loss --save_results
python bank_marketing/bankm_tabmlp.py --mlp_hidden_dims [100,50] --batch_size 512 --optimizer RAdam --focal_loss --save_results
