python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --concat_cont_first --optimizer Adam  --lr 0.01 --rop_threshold_mode rel --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --concat_cont_first --optimizer Adam  --lr 0.01 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --concat_cont_first --optimizer Adam  --lr 0.01 --rop_threshold_mode rel --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --concat_cont_first --optimizer AdamW --lr 0.01 --rop_threshold_mode rel --scale_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --concat_cont_first --optimizer AdamW --lr 0.01 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --concat_cont_first --optimizer AdamW --lr 0.01 --rop_threshold_mode rel --scale_cont --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims auto --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --embed_dropout 0.1 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims auto --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer Adam --lr 0.01 --embed_dropout 0.1 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer AdamW --lr 0.01 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims auto --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer AdamW --lr 0.01 --embed_dropout 0.1 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims auto --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer AdamW --lr 0.01 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer AdamW --lr 0.01 --embed_dropout 0.1 --blocks_dims same --blocks_dropout 0.2 --mlp_hidden_dims [100,100] --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 512 --optimizer Adam --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer Adam --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 512 --optimizer Adam --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.5 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer Adam --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.5 --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 512 --optimizer AdamW --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer AdamW --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 512 --optimizer AdamW --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.5 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer AdamW --lr 0.01 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.5 --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer Adam --lr 0.04 --blocks_dims same --blocks_dropout 0.5 --mlp_hidden_dims auto --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer Adam --lr 0.01 --blocks_dims [50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims [50,50] --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer Adam --lr 0.01 --blocks_dims [50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims [50,30] --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 2048 --optimizer Adam --lr 0.01 --blocks_dims [50,50,50] --blocks_dropout 0.4 --mlp_hidden_dims [50,50] --mlp_dropout 0.4 --rop_threshold_mode rel --batchnorm_cont --save_results

python nyc_taxi/nyc_taxi_tabresnet.py --batch_size 1024 --optimizer AdamW --lr 0.01 --blocks_dims [50,50,50] --blocks_dropout 0.2 --mlp_hidden_dims [50,50] --mlp_dropout 0.2 --rop_threshold_mode rel --batchnorm_cont --save_results
