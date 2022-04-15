# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" GRU4Rec
Reference:
    "Session-based Recommendations with Recurrent Neural Networks"
    Hidasi et al., ICLR'2016.
CMD example:
    python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 128 --lr 1e-3 --l2 1e-4 --history_max 20 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel


class GRU4Rec(SequentialModel):
    extra_log_args = ['emb_size', 'hidden_size', 'time_features']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--time_features', type=str, default='',
                            help='')
        parser.add_argument('--continuous_time', type=int, default=0,
                            help='')
        parser.add_argument('--time_diffs', type=int, default=0,
                            help='')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.time_features = [m.strip().upper() for m in args.time_features.split(',')]
        if self.time_features[0] == '':
            self.time_features = []
        self.continuous_time = args.continuous_time
        self.time_diffs = args.time_diffs
        self.sz = self.emb_size*(len(self.time_features)+1)+self.continuous_time+self.time_diffs
        super().__init__(args, corpus)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        for f in self.time_features:
            if f == 'HOUR':
                self.hours_embeddings = nn.Embedding(24, self.emb_size)
            elif f == 'MONTH':
                self.months_embeddings = nn.Embedding(12, self.emb_size)
            elif f == 'DAY':
                self.days_embeddings = nn.Embedding(31, self.emb_size)
            elif f == 'WEEKDAY':
                self.weekdays_embeddings = nn.Embedding(7, self.emb_size)
            else:
                raise ValueError('Undefined time feature: {}.'.format(f))
        
        self.rnn = nn.GRU(input_size=self.sz, hidden_size=self.hidden_size, batch_first=True)
        # self.pred_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']
        hours = feed_dict['history_hours']
        days = feed_dict['history_days']
        months = feed_dict['history_months']
        weekdays = feed_dict['history_weekdays']
        norm_time = feed_dict['history_normalized_times'].float()
        norm_diff = feed_dict['history_normalized_diffs'].float()
        
        
        his_vectors = self.i_embeddings(history)
        for f in self.time_features:
            if f == 'HOUR':
                hours_vectors = self.hours_embeddings(hours)
                his_vectors = torch.cat((his_vectors, hours_vectors), 2)
            elif f == 'MONTH':
                months_vectors = self.months_embeddings(months)
                his_vectors = torch.cat((his_vectors, months_vectors), 2)
            elif f == 'DAY':
                days_vectors = self.days_embeddings(days)
                his_vectors = torch.cat((his_vectors, days_vectors), 2)
            elif f == 'WEEKDAY':
                weekdays_vectors = self.weekdays_embeddings(weekdays)
                his_vectors = torch.cat((his_vectors, weekdays_vectors), 2)
            else:
                raise ValueError('Undefined time feature: {}.'.format(f))
        if self.continuous_time == 1:
            his_vectors = torch.cat((his_vectors, norm_time.reshape(norm_time.shape[0], norm_time.shape[1], 1)), 2)
        if self.time_diffs == 1:
            his_vectors = torch.cat((his_vectors, norm_diff.reshape(norm_diff.shape[0], norm_diff.shape[1], 1)), 2)
            

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)

        # RNN
        output, hidden = self.rnn(history_packed, None)

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)

        # Predicts
        # pred_vectors = self.pred_embeddings(i_ids)
        pred_vectors = self.i_embeddings(i_ids)
        rnn_vector = self.out(rnn_vector)
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
