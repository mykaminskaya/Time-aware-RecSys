#!/bin/sh -x


python3.7 main.py --model_name aaaa --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --epoch 10 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --log_file '../../m.y.kaminskaya/ReChorus/log/one_try.txt'




