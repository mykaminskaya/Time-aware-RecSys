#!/bin/sh -x


python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'


python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'

python3.7 main.py --model_name SASRec --metric 'NDCG,HR,MNAP' --continuous_time 1 --topk '1,5,10,15,20' --time_diffs 1 --emb_size 64 --time_features 'day,weekday,month' --epoch 200 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'amazon_electronics'