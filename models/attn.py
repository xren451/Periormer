import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from models.Period import fftTransfer1#fftTransfer3D
from data.data_loader import mydataset1
import os
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)


        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class PeriorProbAttention(nn.Module):
    def __init__(self,root_path,data_path,mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(PeriorProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
  #      self.root_path=root_path
  #      self.data_path=data_path
    def _periorprob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        import numpy as np
        import math
        B, H, L_K, E = K.shape  #Q.shape=K.shape=32*8*96*64#B: Batch H:Head;L_K:The number of K;E:Dimension of K
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)##Before: (32,8,96,64);After:K_expand(32,8,96,96,64),96 number of 96*64 matrices. Each matrix is the same.
        outputFFT = torch.load("./outputFFT.pt")
        #print(outputFFT.shape)##[7*5*2]=#(feature*n*FFTout_channel)

        for m in range(outputFFT.size(0)):
            P = outputFFT[m, :, 1]
            Amp = outputFFT[m, :, 0]
            P_eli=[]
            Amp_eli=[]
            index_P = []
            ##This loop is to get the index within the length.
            for i in range(len(P)):
                if P[i] <= L_K:
                    index_P.append(i)
            #print("index_P is :",index_P)
            #This loop is to get the value of P and Amp based on the index.
            for i in index_P:
                P_eli.append(P[i])
                Amp_eli.append(Amp[i])

            index_K = []
            Amp_K = []
            for j in range(len(Amp_eli)):
                for i in range(math.ceil(L_K / P_eli[j])):
                    if (L_K - (i + 1) * P_eli[j]) >= 0:
                        index_K.append(L_K - (i + 1) * P_eli[j])
                        Amp_K.append(Amp_eli[j])

        Amp_K=np.array(Amp_K)
        import numpy as np
        index_K = index_K[:sample_k]
        #Amp_K= np.concatenate((np.array(Amp_K1), np.array(Amp_K2), np.array(Amp_K3)), 0)

        tensor = torch.tensor(index_K)
        tensor = tensor.unsqueeze(-2).expand(L_K, len(index_K))
        index_sample=[]
        index_sample=tensor
       # print("P after elimination is :",P_eli)
       # print("Amp after elimination is :",Amp_eli)


        # j = 0
        # index_K1 = []
        # Amp_K1=[]
        # Amp_K2 = []
        # Amp_K3 = []
        # for i in range(math.ceil(L_K / P_eli[j])):
        #     if (L_K - (i + 1) * P_eli[j]) >= 0:
        #         index_K1.append(L_K - (i + 1) * P_eli[j])
        # index_K1 = list(reversed(index_K1))
        # Amp_K1=[Amp_eli[0] for x in range (len(index_K1))]
        # # print(index_K1)   [92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0]
        # # print(list(reversed(index_K1)))  [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92]
        # index_K2 = []
        # index_K3 = []
        # j = 1
        # index_K2 = []
        # for i in range(math.ceil(L_K / P_eli[j])):
        #     if (L_K - (i + 1) * P_eli[j]) >= 0:
        #         index_K2.append(L_K - (i + 1) * P_eli[j])
        # index_K2 = list(reversed(index_K2))
        # Amp_K2 = [Amp_eli[1] for x in range(len(index_K2))]
        # # print(index_K2) [64,32,0]
        # # print(list(reversed(index_K2)))  [0,32,64]
        # j = 2
        # index_K3 = []
        # for i in range(math.ceil(L_K / P_eli[j])):
        #     if (L_K - (i + 1) * P_eli[j]) >= 0:
        #         index_K3.append(L_K - (i + 1) * P_eli[j])
        # index_K3 = list(reversed(index_K3))
        # Amp_K3 = [Amp_eli[2] for x in range(len(index_K3))]


        # print(index_K3)  [21]
        # print(list(reversed(index_K3)))  [21]
        # This loop is to get the index of K based on the P_eli




        #index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q### Generate random index
        #Multiple corresponding amplitudes

        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample.long(), :]#(32,8,96,16,64)
        K_sample_amp = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), Amp_K, :]  #This sentence may be problematic
        K_sample_amp=K_sample
        K_sample_amp_K=[]
        K_sample_amp_K=K_sample_amp*K_sample
        ##K_sample * corresponding 16 amplitudes!!!!
        ##2023.3.31--->Afternoon
        Q_K_sample = torch.matmul(Q.unsqueeze(-2),  K_sample_amp_K.transpose(-2, -1)).squeeze(-2)##shape: [32,8,96,16]
#Q.unqueeze(32,8,96,1,64);K_sample.transpose(-2,-1).shape;
        #Do Attention---->The number of Q:96 K:25.

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)#[32,8,96]
        M_top = M.topk(n_top, sorted=False)[1]#[32,8,25]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],#[32,8,25,64]
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) ##[32,8,25,96] factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._periorprob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn