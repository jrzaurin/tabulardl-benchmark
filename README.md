# Deep Learning vs `LightGBM` for tabular data

This repo contains the code to run over 1500 experiments that compare the
performance of Deep Learning algorithms for tabular data with [`LightGBM`](https://lightgbm.readthedocs.io/en/latest/).

Deep Learning models for tabular data are run via the [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) library.

Companion post: [pytorch-widedeep, deep learning for tabular data IV: Deep Learning vs LightGBM](https://jrzaurin.github.io/infinitoml/2021/05/28/pytorch-widedeep_iv.html)

For the experiments in this repo I have used four datasets:

1. [Adult Census](https://archive.ics.uci.edu/ml/datasets/adult) (binary classification)
2. [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (binary classification)
3. [NYC taxi ride duration](https://www.kaggle.com/neomatrix369/nyc-taxi-trip-duration-extended) (regression)
4. [Facebook Comment Volume](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset) (regression)

And mainly four deep learning models:

1. TabMlp: a simple MLP very similar to the tabular api implementation in the fastai library
2. TabResnet: similar to the MLP but instead of dense layers I use Resnet blocks
3. [Tabnet](https://arxiv.org/pdf/1908.07442.pdf)
4. [TabTransformer](https://arxiv.org/pdf/2012.06678.pdf)

## RESULTS

**ADULT CENSUS**

|  model 	        |  acc 	    |   runtime	  |   best_epoch_or_ntrees	|
|---	            |---	    |---	      |---		    |
|   lightgbm	    |  0.878178 |  0.908639   |   408.0		|
|   tabmlp	        |  0.872209 |  205.357588 |   62.0		|
|   tabtransformer	|  0.871767 |  288.640581 |   32.0		|
|   tabnet	        |  0.870440 |  422.296659 |   26.0		|
|   tabresnet	    |  0.869777 |  388.932547 |   25.0		|


**BANK MARKETING**

|  model 	        |  f1 	    |  auc 	    |   runtime	  |   best_epoch_or_ntrees	|
|---	            |---	    |---	    |---	      |---		    |
|   tabresnet	    |  0.429799 |  0.650147 |  92.517464  |   11.0		|
|   tabtransformer  |  0.419971 |  0.643972 |  31.693761  |   4.0		|
|   tabmlp       	|  0.385542 |  0.628082 |  9.572095   |   7.0		|
|   lightgbm        |  0.385208 |  0.626490 |  0.461398   |   57.0		|
|   tabnet  	    |  0.308703 |  0.594316 |  77.878060  |   13.0		|


**NYC TAXI RIDE DURATION**

|  model 	        |  rmse 	  |  r2 	  |   runtime	 |   best_epoch_or_ntrees	|
|---	            |---	      |---	      |---	         |---		    |
|   lightgbm	    |  262.709865 |  0.804393 |  42.721136	 |   504.0		|
|   tabmlp	        |  271.342218 |  0.791327 |  568.430923	 |   24.0		|
|   tabresnet	    |  292.890792 |  0.756867 |  471.264983	 |   24.0		|
|   tabtransformer  |  336.582554 |  0.678919 |  5779.031367 |   54.0		|
|   tabnet	        |  376.053004 |  0.599198 |  1844.472289 |   15.0		|


**FACEBOOK COMMENT VOLUME**

|  model 	        |  rmse	    |  r2 	    |   runtime	  |   best_epoch_or_ntrees	|
|---	            |---	    |---	    |---	      |---	    |
|   lightgbm	    |  5.528963 |  0.823208 |  6.525877	  |   687.0	|
|   tabmlp	        |  5.908498 |  0.798103 |  250.476762 |   43.0	|
|   tabtransformer	|  5.925587 |  0.796933 |  533.390816 |   27.0	|
|   tabresnet	    |  6.213813 |  0.776698 |  70.466089  |   9.0	|
|   tabnet	        |  6.428503 |  0.761001 |  935.020483 |   59.0	|

For more results on all the experiments run see [here](https://github.com/jrzaurin/tabulardl-benchmark/tree/master/analyze_experiments/leaderboards)