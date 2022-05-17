# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" SASRec
Reference:
    "Self-attentive Sequential Recommendation"
    Kang et al., IEEE'2018.
Note:
    When incorporating position embedding, we make the position index start from the most recent interaction.
CMD example:
    python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class SASRec(SequentialModel):
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
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
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
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
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.sz = self.emb_size*(len(self.time_features)*(self.disc_method // 2)+1)+(self.norm_timestamps+self.norm_diffs+self.clear_timestamps+self.clear_diffs) * (self.cont_method -1 )
        self.sz1 = self.emb_size*(len(self.time_features)*int(bool(self.disc_method))+1)+(self.norm_timestamps+self.norm_diffs+self.clear_timestamps+self.clear_diffs)

        for f in self.time_features:
            if f == 'MONTH':
                self.months_embeddings = nn.Embedding(12, self.emb_size)
            elif f == 'DAY':
                self.days_embeddings = nn.Embedding(31, self.emb_size)
            elif f == 'WEEKDAY':
                self.weekdays_embeddings = nn.Embedding(7, self.emb_size)
            else:
                raise ValueError('Undefined time feature: {}.'.format(f))

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.sz, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

        if self.disc_method == 1 or self.cont_method == 1:
            self.lin = nn.Linear(self.sz1, self.sz).to(self.device)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        days = feed_dict['history_days']
        months = feed_dict['history_months']
        weekdays = feed_dict['history_weekdays']
        norm_time = feed_dict['history_normalized_times'].float()
        norm_diff = feed_dict['history_normalized_diffs'].float()
        times = feed_dict['history_times'].float()
        diff = feed_dict['history_diffs'].float()
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # features
        i_vectors = self.i_embeddings(i_ids)
        for f in self.time_features:
            if f == 'MONTH':
                months_vectors = self.months_embeddings(months)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, months_vectors), 2)
                else:
                    his_vectors = his_vectors + months_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['months'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.months_embeddings(d)
                    i_vectors = torch.cat((i_vectors, d), 2)
            elif f == 'DAY':
                days_vectors = self.days_embeddings(days)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, days_vectors), 2)
                else:
                    his_vectors = his_vectors + days_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['days'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.days_embeddings(d)
                    i_vectors = torch.cat((i_vectors, d), 2)
            elif f == 'WEEKDAY':
                weekdays_vectors = self.weekdays_embeddings(weekdays)
                if self.disc_method > 0:
                    his_vectors = torch.cat((his_vectors, weekdays_vectors), 2)
                else:
                    his_vectors = his_vectors + weekdays_vectors

                if self.disc_method == 2:
                    d = torch.tensor(np.tile(feed_dict['weekdays'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                    d = self.weekdays_embeddings(d)
                    i_vectors = torch.cat((i_vectors, d), 2)
            else:
                raise ValueError('Undefined time feature: {}.'.format(f))
        if self.norm_timestamps == 1:
            his_vectors = torch.cat((his_vectors, norm_time.reshape(norm_time.shape[0], norm_time.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['normalized_times'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                i_vectors = torch.cat((i_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)
        if self.clear_timestamps == 1:
            his_vectors = torch.cat((his_vectors, time.reshape(time.shape[0], time.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['times'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                i_vectors = torch.cat((i_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)
        if self.clear_diffs == 1:
            his_vectors = torch.cat((his_vectors, diff.reshape(diff.shape[0], diff.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['diffs'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                i_vectors = torch.cat((i_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)
        if self.norm_diffs == 1:
            his_vectors = torch.cat((his_vectors, norm_diff.reshape(norm_diff.shape[0], norm_diff.shape[1], 1)), 2)
            if self.cont_method == 2:
                t = torch.tensor(np.tile(feed_dict['normalized_diffs'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(self.device)
                i_vectors = torch.cat((i_vectors, t.reshape(t.shape[0], t.shape[1], 1)), 2)

        # Self-attention

        if self.disc_method == 1 or self.cont_method == 1:
            his_vectors = self.lin(his_vectors)
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # Ã¢â€â€˜ average pooling is shown to be more effective than the most recent embedding

        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}
