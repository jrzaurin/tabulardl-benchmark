python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 32 --n_heads 8  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 64 --n_heads 16 --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 64 --n_heads 4 --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 64 --n_heads 4 --n_blocks 6 --rop_threshold 0.01 --rop_threshold_mode rel --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --shared_embed --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --mlp_hidden_dims same --dropout 0.2 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --mlp_hidden_dims [171,100,50] --dropout 0.2 --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --shared_embed --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims [171,100,50] --dropout 0.2 --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_hidden_state --prepare_text  --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_hidden_state --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_hidden_state --prepare_text --head_hidden_dims [128,64] --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_hidden_state --bidirectional --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_hidden_state --embed_dim 100 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_hidden_state --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --use_hidden_state --prepare_text  --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --use_hidden_state --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --use_hidden_state --head_hidden_dims [128,64] --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --use_hidden_state --bidirectional --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --use_hidden_state --embed_dim 100 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --use_hidden_state --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --prepare_text  --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --head_hidden_dims [128,64] --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --bidirectional --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --embed_dim 100 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --prepare_text  --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --head_hidden_dims [64,64] --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --head_hidden_dims [128,64] --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --bidirectional --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --embed_dim 100 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --rnn_type gru --embed_dim 100 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_hidden_state --embed_dim 100 -rnn_dropout 0.2 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_hidden_state --embed_dim 100 -rnn_dropout 0.2 --head_hidden_dims [64,64] --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --prepare_text --use_word_vectors --use_hidden_state --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_word_vectors --use_hidden_state --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_word_vectors --use_hidden_state --hidden_dim 256 --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_word_vectors --use_hidden_state --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_word_vectors --use_hidden_state --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_word_vectors --use_hidden_state --hidden_dim 256 --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --prepare_text --use_word_vectors --embed_trainable --use_hidden_state --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_word_vectors --use_hidden_state --embed_trainable --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --use_word_vectors --use_hidden_state --embed_trainable --hidden_dim 256 --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_word_vectors --embed_trainable --use_hidden_state --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_word_vectors --use_hidden_state --embed_trainable --hidden_dim 128 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_word_vectors --use_hidden_state --embed_trainable --hidden_dim 256 --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --prepare_text --warmup --use_hidden_state --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 5  --n_epochs 100  --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_hidden_state --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10  --n_epochs 100  --save_results

python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_hidden_state --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_hidden_state --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 10 --save_results
python airbnb/airbnb_tabtransformer.py --optimizer Adam --input_dim 16 --n_heads 4  --n_blocks 4 --rop_threshold 0.01 --rop_threshold_mode rel --transformer_activation gelu --mlp_hidden_dims same --dropout 0.2 --with_text --warmup --use_hidden_state --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 100 --save_results