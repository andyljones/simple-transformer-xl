# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Dropout's zeroed since the task data I'm interested in is all infinite
DROPOUT = .0
LAYER_NORM_EPS = 1e-5

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner):
        super().__init__()

        self.core = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(d_inner, d_model),
            nn.Dropout(DROPOUT))

        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

    def forward(self, inp):
        h = self.core(inp)
        output = self.layer_norm(inp + h)
        return output


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.scale = 1/d_head**.5

        self.qkv_net = nn.Linear(d_model, 3*n_head*d_head, bias=False)
        self.r_net = nn.Linear(d_model, self.n_head*self.d_head, bias=False)
        self.o_net = nn.Linear(n_head*d_head, d_model, bias=False)

        self.drop = nn.Dropout(DROPOUT)
        self.layer_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self.r_r_bias = nn.Parameter(torch.empty((self.n_head, self.d_head)))
        self.r_w_bias = nn.Parameter(torch.empty((self.n_head, self.d_head)))

        # Doesn't really matter how we initialize; LayerNorm will stop things from blowing up
        torch.nn.init.normal_(self.r_r_bias)
        torch.nn.init.normal_(self.r_w_bias)


    def _rel_shift(self, x):
        """Explanation: https://github.com/kimiyoung/transformer-xl/issues/8#issuecomment-454458852"""
        (T, C), tail = x.shape[:2], x.shape[2:]

        zero_pad = x.new_zeros((T, 1) + tail)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view((C+1, T) + tail)

        return x_padded[1:].view_as(x)

    def forward(self, h, m, r, mask):
        T, C, B = h.size(0), r.size(0), h.size(1)
        N, D = self.n_head, self.d_head

        cat = torch.cat([m, h], 0)
        w_heads = self.qkv_net(cat)
        r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-T:]

        w_head_q = w_head_q.view(T, B, N, D)
        w_head_k = w_head_k.view(C, B, N, D)
        w_head_v = w_head_v.view(C, B, N, D)

        r_head_k = r_head_k.view(C, N, D)

        #### compute attention score
        rw_head_q = w_head_q + self.r_w_bias                                    # T x B x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # T x C x B x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # T x C x B x n_head
        BD = self._rel_shift(BD)

        # [T x C x B x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        attn_score = (attn_score
                        .float()
                        # -65k is a very, very small number for a 16-bit float
                        .masked_fill(mask.bool()[:, :, :, None], -65000)
                        .type_as(attn_score))

        # [T x C x B x n_head]
        attn_prob = F.softmax(attn_score, dim=1)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [T x B x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(T, B, N*D)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        return self.layer_norm(h + attn_out)


class Decoder(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner):
        super().__init__()

        self.attn = MultiHeadAttn(n_head, d_model, d_head)
        self.ff = PositionwiseFF(d_model, d_inner)

    def forward(self, h, m, r, mask):
        attn = self.attn(h, m, r, mask)
        ff = self.ff(attn)
        return ff


class TransformerXL(nn.Module):

    def __init__(self, d_model=128, d_head=32, d_inner=256, n_head=4, n_layer=4, mem_len=16):
        super().__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.mem_len = mem_len
        assert self.mem_len > 0, 'Memory length must be greater than zero, else the attention softmax will break'

        self.drop = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([Decoder(n_head, d_model, d_head, d_inner) for _ in range(n_layer)])
        self.pos_emb = PositionalEmbedding(self.d_model)

    def initialize(self, h):
        return h.new_zeros((self.n_layer, self.mem_len, h.size(1), self.d_model)).detach()

    def forward(self, h, ms):
        """
        Args/returns:
            h: (T, B, d_model)
            ms: (n_layers, M, B, d_model)
        """
        T, B, d_model = h.size()
        M = ms.size(1)
        assert d_model == self.d_model
        assert M == self.mem_len

        all_ones = h.new_ones((T, M+T), dtype=torch.uint8)
        mask = (torch.triu(all_ones, M+1) + torch.tril(all_ones, 0))[:, :, None] 

        pos_seq = torch.arange(M+T-1, -1, -1., device=h.device, dtype=h.dtype)
        r = self.pos_emb(pos_seq)

        h = self.drop(h)
        r = self.drop(r)

        hids = []
        for m, layer in zip(ms, self.layers):
            hids.append(h)
            h = layer(h, m, r, mask)
        h = self.drop(h)

        hids = torch.stack(hids)
        ms = torch.cat([ms, hids], dim=1)[:, -M:].detach()

        return h, ms
