#!/bin/sh -x


python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --time_features 'day' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --time_features 'weekday' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --time_features 'month' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20'  --time_diffs 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'


python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --time_features 'day' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --time_features 'weekday' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20' --time_features 'month' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python main.py --model_name SASRec --metric 'NDCG, HR, MNAP' --topk '1,5,10,15,20'  --time_diffs 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'