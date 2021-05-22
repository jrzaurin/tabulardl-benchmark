python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 256 --optimizer Adam --lr 0.01 --input_dim 16 --dropout 0.1 --n_heads 4 --n_blocks 4 --rop_threshold_mode rel --save_results
python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 256 --optimizer Adam --lr 0.01 --input_dim 32 --dropout 0.4 --n_heads 8 --n_blocks 4 --rop_threshold_mode rel --save_results
python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 256 --optimizer Adam --lr 0.01 --input_dim 64 --dropout 0.4 --n_heads 16 --n_blocks 4 --rop_threshold_mode rel --save_results

python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 512 --optimizer Adam --lr 0.01 --input_dim 16 --dropout 0.1 --n_heads 4 --n_blocks 4 --rop_threshold_mode rel --save_results
python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 512 --optimizer Adam --lr 0.01 --input_dim 32 --dropout 0.4 --n_heads 8 --n_blocks 4 --rop_threshold_mode rel --save_results
python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 512 --optimizer Adam --lr 0.01 --input_dim 64 --dropout 0.4 --n_heads 16 --n_blocks 4 --rop_threshold_mode rel --save_results

python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 1024 --optimizer Adam --lr 0.01 --input_dim 16 --dropout 0.1 --n_heads 4 --n_blocks 4 --rop_threshold_mode rel --save_results
python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 1024 --optimizer Adam --lr 0.01 --input_dim 32 --dropout 0.4 --n_heads 8 --n_blocks 4 --rop_threshold_mode rel --save_results
python nyc_taxi/nyc_taxi_tabtransformer.py --batch_size 1024 --optimizer Adam --lr 0.01 --input_dim 64 --dropout 0.4 --n_heads 16 --n_blocks 4 --rop_threshold_mode rel --save_results
