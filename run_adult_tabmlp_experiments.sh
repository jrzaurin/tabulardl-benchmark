python run_experiments/adult/adult_tabmlp.py --optimizer Adam --lr_scheduler ReduceLROnPlateau --save_results
python run_experiments/adult/adult_tabmlp.py --optimizer AdamW --lr_scheduler ReduceLROnPlateau --save_results
python run_experiments/adult/adult_tabmlp.py --optimizer RAdam --lr 0.03 --lr_scheduler ReduceLROnPlateau --save_results
python run_experiments/adult/adult_tabmlp.py --optimizer RAdam --lr_scheduler ReduceLROnPlateau --save_results

python run_experiments/adult/adult_tabmlp.py --optimizer Adam --monitor val_acc --rop_mode max --lr_scheduler ReduceLROnPlateau --save_results
python run_experiments/adult/adult_tabmlp.py --optimizer AdamW --monitor val_acc --rop_mode max --lr_scheduler ReduceLROnPlateau --save_results
python run_experiments/adult/adult_tabmlp.py --optimizer RAdam --monitor val_acc --rop_mode max --lr_scheduler ReduceLROnPlateau --save_results