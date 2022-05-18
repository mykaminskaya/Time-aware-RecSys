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
import numpy as np

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
        parser.add_argument('--norm_timestamps', type=int, default=0,
                            help='')
        parser.add_argument('--norm_diffs', type=int, default=0,
                            help='')
        parser.add_argument('--clear_timestamps', type=int, default=0,
                            help='')
        parser.add_argument('--clear_diffs', type=int, default=0,
                            help='')
        parser.add_argument('--disc_method', type=int, default=1,
                            help='')
        parser.add_argument('--cont_method', type=int, default=1,
                            help='')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.time_features = [m.strip().upper() for m in args.time_features.split(',')]
        if self.time_features[0] == '':
            self.time_features = []
        self.norm_timestamps = args.norm_timestamps
        self.norm_diffs = args.norm_diffs
        self.clear_timestamps = args.clear_timestamps
        self.clear_diffs = args.clear_diffs
        self.disc_method = args.disc_method
        self.cont_method = args.cont_method
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

        self.sz = self.emb_size*(len(self.time_features)*(self.disc_method // 2)+1)+(self.norm_timestamps+self.norm_diffs+self.clear_timestamps+self.clear_diffs) * (self.cont_method -1 )
        self.sz1 = self.emb_size*(len(self.time_features)*int(bool(self.disc_method))+1)+(self.norm_timestamps+self.norm_diffs+self.clear_timestamps+self.clear_diffs)

        self.rnn = nn.GRU(input_size=self.sz, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.sz)
        if self.disc_method == 1 or self.cont_method == 1:
            self.lin = nn.Linear(self.sz1, self.sz).to(self.device)

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
        time = feed_dict['history_times'].float()
        diff = feed_dict['history_diffs'].float()

        his_vectors = self.i_embeddings(history)
        pred_vectors = self.i_embeddings(i_ids)
        for f in self.time_features:
            if f == 'MONTH':
                months_vectors = self.months_embeddings(months)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, months_vectors), 2)
                else:
                    his_vectors = his_vectors + months_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['months'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.months_embeddings(d)
                    pred_vectors = torch.cat((pred_vectors, d), 2)
            elif f == 'DAY':
                days_vectors = self.days_embeddings(days)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, days_vectors), 2)
                else:
                    his_vectors = his_vectors + days_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['days'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.days_embeddings(d)
                    pred_vectors = torch.cat((pred_vectors, d), 2)
            elif f == 'HOUR':
                hours_vectors = self.hours_embeddings(hours)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, hours_vectors), 2)
                else:
                    his_vectors = his_vectors + hours_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['hours'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.hours_embeddings(d)
                    pred_vectors = torch.cat((pred_vectors, d), 2)
            elif f == 'WEEKDAY':
                weekdays_vectors = self.weekdays_embeddings(weekdays)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, weekdays_vectors), 2)
                else:
                    his_vectors = his_vectors + weekdays_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['weekdays'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.weekdays_embeddings(d)
                    pred_vectors = torch.cat((pred_vectors, d), 2)
            else:
                raise ValueError('Undefined time feature: {}.'.format(f))
        if self.norm_timestamps == 1:
            his_vectors = torch.cat((his_vectors, norm_time.reshape(norm_time.shape[0], norm_time.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['normalized_times'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                pred_vectors = torch.cat((pred_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)
        if self.clear_timestamps == 1:
            his_vectors = torch.cat((his_vectors, time.reshape(time.shape[0], time.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['times'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                pred_vectors = torch.cat((pred_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)
        if self.clear_diffs == 1:
            his_vectors = torch.cat((his_vectors, diff.reshape(diff.shape[0], diff.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['diffs'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                pred_vectors = torch.cat((pred_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)
        if self.norm_diffs == 1:
            his_vectors = torch.cat((his_vectors, norm_diff.reshape(norm_diff.shape[0], norm_diff.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['normalized_diffs'].cpu(), (pred_vectors.shape[1], 1)).transpose()).to(self.device)
                pred_vectors = torch.cat((pred_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)

        if self.disc_method == 1 or self.cont_method == 1:
            his_vectors = self.lin(his_vectors)
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
        rnn_vector = self.out(rnn_vector)
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
