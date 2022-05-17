#!/bin/sh -x

python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name TiSASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
