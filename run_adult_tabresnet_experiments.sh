# python run_experiments/adult/adult_tabresnet.py --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --optimizer RAdam --lr 0.03 --save_results
# python run_experiments/adult/adult_tabresnet.py --optimizer RAdam --save_results

# python run_experiments/adult/adult_tabresnet.py --optimizer Adam --monitor val_acc --rop_mode max --save_results
# python run_experiments/adult/adult_tabresnet.py --optimizer AdamW --monitor val_acc --rop_mode max --save_results
# python run_experiments/adult/adult_tabresnet.py --optimizer RAdam --lr 0.03 --monitor val_acc --rop_mode max --save_results
# python run_experiments/adult/adult_tabresnet.py --optimizer RAdam --monitor val_acc --rop_mode max --save_results

# python run_experiments/adult/adult_tabresnet.py --blocks_dims same --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims same --mlp_hidden_dims auto --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50] --mlp_hidden_dims [50,50] --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims [50,50] --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,100] --mlp_dropout 0.5 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [400,400,400] --blocks_dropout 0.5 --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims same --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims same --mlp_hidden_dims auto --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50] --mlp_hidden_dims [50,50] --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims [50,50] --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,100] --mlp_dropout 0.5 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [400,400,400] --blocks_dropout 0.5 --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims same --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims same --mlp_hidden_dims auto --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50] --mlp_hidden_dims [50,50] --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims [50,50] --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,100] --mlp_dropout 0.5 --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [400,400,400] --blocks_dropout 0.5 --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer RAdam --save_results

# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --optimizer RAdam --save_results

# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --batch_size 512 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --batch_size 512 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,200] --mlp_dropout 0.5 --batch_size 512 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --batch_size 1024 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --batch_size 1024 --optimizer Adam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,200] --mlp_dropout 0.5 --batch_size 1024 --optimizer Adam --save_results

# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --batch_size 512 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --batch_size 512 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,200] --mlp_dropout 0.5 --batch_size 512 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --batch_size 1024 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --batch_size 1024 --optimizer AdamW --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,200] --mlp_dropout 0.5 --batch_size 1024 --optimizer AdamW --save_results

# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --batch_size 512 --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --batch_size 512 --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,200] --mlp_dropout 0.5 --batch_size 512 --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [50,50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims None --batch_size 1024 --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --mlp_dropout 0.2 --batch_size 1024 --optimizer RAdam --save_results
# python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims [200,200] --mlp_dropout 0.5 --batch_size 1024 --optimizer RAdam --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer Adam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer Adam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer Adam --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer AdamW --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer AdamW --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer AdamW --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer RAdam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [100,100,100] --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer RAdam --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims [200,200,200] --blocks_dropout 0.5 --mlp_hidden_dims None --batch_size 1024 --optimizer RAdam --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results

python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python run_experiments/adult/adult_tabresnet.py --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims None --optimizer RAdam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results