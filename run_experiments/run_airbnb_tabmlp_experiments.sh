python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabmlp.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --scale_cont --batchnorm_cont --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims auto --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims auto --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims auto --mlp_dropout 0.4 --save_results

python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims auto --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims auto --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims auto --mlp_dropout 0.4 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_activation leaky_relu --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm_last --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_linear_first --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.1 --embed_dropout 0.1 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --embed_dropout 0.1 --save_results

python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_activation leaky_relu --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm_last --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_linear_first --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.1 --embed_dropout 0.1 --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --embed_dropout 0.1 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam --batch_size 512 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam --batch_size 512 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --batch_size 512 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam --batch_size 1024 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam --batch_size 1024 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --batch_size 1024 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam --batch_size 64 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam --batch_size 64 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --batch_size 64 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_wide --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --with_wide --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --with_wide --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_wide --warmup --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --with_wide --warmup --save_results
python airbnb/airbnb_tabmlp.py --optimizer AdamW --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --mlp_batchnorm --with_wide --warmup --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --prepare_text --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --bidirectional --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --head_hidden_dims [128,64] --bidirectional --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --embed_dim 100 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --bidirectional --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --head_hidden_dims [128,64] --bidirectional --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --embed_dim 100 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --bidirectional --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --head_hidden_dims [128,64] --bidirectional --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --embed_dim 100 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --rnn_type gru --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --rnn_type gru --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_dropout 0.4 --prepare_text --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --embed_dim 100 --use_hidden_state --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --weight_decay 0.0001 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --weight_decay 0.001 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --rnn_dropout 0.4 --prepare_text --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --embed_dim 100 --use_hidden_state --rnn_type gru --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --weight_decay 0.0001 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --weight_decay 0.001 --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_hidden_state --rnn_type gru --rnn_dropout 0.4 --head_hidden_dims [64,64] --head_dropout 0.4 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --embed_dim 100 --use_hidden_state --hidden_dim 256 --prepare_text --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --hidden_dim 256 --head_hidden_dims [256,128] --head_dropout 0.2 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --embed_dim 100 --use_hidden_state --bidirectional --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --bidirectional --head_hidden_dims [128,64] --head_dropout 0.2 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --warmup --embed_dim 100 --use_hidden_state --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --warmup --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --prepare_text --use_word_vectors --use_hidden_state --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --use_hidden_state --hidden_dim 128 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --use_hidden_state --hidden_dim 256 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --warmup --use_hidden_state --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --warmup --use_hidden_state --hidden_dim 128 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --warmup --use_hidden_state --hidden_dim 256 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --prepare_text --use_word_vectors --embed_trainable --warmup --use_hidden_state --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --embed_trainable --warmup --use_hidden_state --hidden_dim 128 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --embed_trainable --warmup --use_hidden_state --hidden_dim 256 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --embed_trainable --warmup --use_hidden_state --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --embed_trainable --warmup --use_hidden_state --head_hidden_dims [128,128] --hidden_dim 128 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --use_word_vectors --embed_trainable --warmup --use_hidden_state --head_hidden_dims [256,256] --hidden_dim 256 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --prepare_text --warmup --embed_dim 100 --use_hidden_state --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5  --n_epochs 100 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --warmup --embed_dim 100 --use_hidden_state --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --warmup --embed_dim 100 --use_hidden_state --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --warmup --embed_dim 100 --use_hidden_state --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python airbnb/airbnb_tabmlp.py --optimizer Adam  --rop_threshold_mode rel --rop_threshold 0.01 --batchnorm_cont --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --with_text --warmup --embed_dim 100 --use_hidden_state --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 100 --save_results
