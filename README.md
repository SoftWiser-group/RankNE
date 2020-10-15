# RankNE
implementation of paper **Bringing Order to Network Embedding: A Relative Ranking based Approach** (CIKM 2020)

## Requirements
	Python >= 3.6
	pytorch >= 1.4
	
## Quick Start
We give a basic setup to use the LiWine by run
 
	# python LiWine/train.py
	...
	ppi multiview Results, using embeddings of dimensionality 128
	-------------------
	Train percent: 0.05
	Average score: {'micro': 0.15644274014924833, 'macro': 0.10813561271513757}
	-------------------
	Train percent: 0.07
	Average score: {'micro': 0.1679065272187365, 'macro': 0.12340636947323862}
	-------------------
	Train percent: 0.1
	Average score: {'micro': 0.18120811157993072, 'macro': 0.14101778252002523}
	-------------------
	Train percent: 0.5
	Average score: {'micro': 0.23657523629669872, 'macro': 0.20124535320050524}
	-------------------
	Train percent: 0.9
	Average score: {'micro': 0.2503478417565816, 'macro': 0.20457823695412025}
	-------------------

## Usage

To use LiWine, use the following command

	# python LiWine/train.py -h
	usage: train.py [-h] [--prefix PREFIX] [--dataset DATASET]
                [--embedding_dim EMBEDDING_DIM] [--batch_size BATCH_SIZE]
                [--epochs EPOCHS] [--lr LR] [--fin_lr FIN_LR]
                [--weight_decay WEIGHT_DECAY] [--neg_sample NEG_SAMPLE]
                [-init INIT] [--init_method INIT_METHOD] [--context CONTEXT]
                [--random_seed RANDOM_SEED]

	optional arguments:
	  -h, --help            show this help message and exit
	  --prefix PREFIX       dir prefix
	  --dataset DATASET     dataset name
	  --embedding_dim 		EMBEDDING_DIM
	                        the output dim of final layer
	  --batch_size BATCH_SIZE
	                        batch_size
	  --epochs EPOCHS       epochs
	  --lr LR               learning rate
	  --fin_lr FIN_LR       final learning rate
	  --weight_decay WEIGHT_DECAY
	                        weight_decay
	  --neg_sample NEG_SAMPLE
	                        number of negative samples
	  --init INIT            whether to initialization
	  --init_method INIT_METHOD
	                        init method
	  --context CONTEXT     context
	  --random_seed RANDOM_SEED
	                        whether to use random seed
	                        
For example

	python LiWine/train.py -h --dataset ppi --embedding_dim 128 --batch_size 256 --epochs 300 --lr 0.005 --fin_lr 5e-6 --neg_sample 1000 --context N1 --random_seed True

## Repository contents

| file | description |
|------|------|
|LiWine/train.py| This is the main training file|
|LiWine/models.py| This define the model of LiWine |
|LiWine/utils.py| This file used to load data|
|LiWine/scoring.py| This file used to evalute the learned network embedding|
	
## Datasets
We do not provide all the datasets but PPI dataset in 'mat' format.

## Citing

If you find either PaWine or LiWine useful in your research, we ask that you cite the following paper

	@inproceedings{
	author = {Yaojing Wang, Guosheng Pan, Yuan Yao, Hanghang Tong, Hongxia Yang, Feng Xu, Jian Lu},
	title = {Bringing Order to Network Embedding: A Relative Ranking based Approach},
	booktitle = {29th ACM International Conference on Information and Knowledge Management},
	year = {2020},
	}

