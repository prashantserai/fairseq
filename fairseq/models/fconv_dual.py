'''
This file provides templates useful for a dual encoder seq2seq model, including a template
for a decoder that can attend to two encoders. It also implements the Dual Attention and
Encoder Dropout schemes as as described in the below work:

Serai, Prashant et al. “Hallucination of speech recognition errors with sequence to sequence learning.”
Submitted to IEEE Transactions on Audio, Speech, and Language Processing, 2021.
arXiv preprint arXiv:2103.12258
https://arxiv.org/pdf/2103.12258.pdf

The below code is an adaptation of fconv.py provided as part of Facebook's Fairseq toolkit.

Author: Prashant Serai
Last updated: April 29, 2021
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax, BeamableMM, GradMultiply, LearnedPositionalEmbedding,
    LinearizedConvolution,
)

from . import fconv

import sys

class FConvDecoderWithDualAttention(fconv.FConvDecoder):
    """Convolutional decoder"""

    def __init__(
        self, dictionary, embed_dim=512, embed_dict=None, out_embed_dim=256,
        max_positions=1024, convolutions=((512, 3),) * 20, attention=True,
        dropout=0.1, share_embed=False, positional_embeddings=True,
        adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, 
        encoder_output_dropout=0,
    ):
        super(fconv.FConvDecoder,self).__init__(dictionary) # super of fconv.FConvDecoder
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout
        self.encoder_output_dropout = encoder_output_dropout
        self.need_attn = True

        convolutions = fconv.extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = fconv.Embedding(num_embeddings, embed_dim, padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)

        self.embed_positions = fconv.PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
        ) if positional_embeddings else None

        self.fc1 = fconv.Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(fconv.Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            self.convolutions.append(
                fconv.LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            self.attention.append(DualAttentionLayer(out_channels, embed_dim)
                                  if attention[i] else None)
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, in_channels, adaptive_softmax_cutoff,
                                                    dropout=adaptive_softmax_dropout)
        else:
            self.fc2 = fconv.Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, \
                    "Shared embed weights implies same dimensions " \
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = fconv.Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out1=None, encoder_out2=None, incremental_state=None, **unused):
        # encoder_out1,encoder_out2 = encoder_out2,encoder_out1
        if encoder_out1 is not None:
            encoder_padding_mask1 = encoder_out1['encoder_padding_mask']
            encoder_out1 = encoder_out1['encoder_out']

            # split and transpose encoder outputs
            encoder_a1, encoder_b1 = self._split_encoder_out(encoder_out1, incremental_state, 'encoder_out1')

        if encoder_out2 is not None:
            encoder_padding_mask2 = encoder_out2['encoder_padding_mask']
            encoder_out2 = encoder_out2['encoder_out']

            # split and transpose encoder outputs
            encoder_a2, encoder_b2 = self._split_encoder_out(encoder_out2, incremental_state, 'encoder_out2')

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        encoder_output_dropout_random_values = torch.rand(x.size()[0]) # B values uniformly chosen from [0, 1)


        #randX1d and randX2d are tensors of sizes [b], essentially each is a list of multiplying factors
        #corresponding to each element in the batch respectively
        # eg. randX1d = [2,1,0] means the first batch element gets multiplied by 2, second by 1, and third by 0
        # here multiply by 2 means x2 is dropped and thus x1 is compensating
        # multiply by 1 means neither is dropped and both will be multipled by 1 (not affected)
        # and multiply by 0 means that x1 is dropped and thus x2 would compensate
        # randX1d = 1 * (encoder_output_dropout_random_values < (0.5 * self.encoder_output_dropout)).float() #_debug
        randX1d = 2 * (encoder_output_dropout_random_values < (0.5 * self.encoder_output_dropout)).float() \
                + 1 * (encoder_output_dropout_random_values >= self.encoder_output_dropout).float()
        randX2d = -1 * (randX1d - 2) # 1s remain 1s, 0s turn into 2s, 2s turn into 0s
        randX1 = torch.diag(randX1d).to(device='cuda') #mask for encoder 1 along batch dimension
        randX2 = torch.diag(randX2d).to(device='cuda') #mask for encoder 2 along batch dimension

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(self.projections, self.convolutions, self.attention,
                                                    self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                x, attn_scores = attention(x, target_embedding, (encoder_a1, encoder_b1), (encoder_a2, encoder_b2),
                                            encoder_padding_mask1, encoder_padding_mask2,
                                            randX1, randX2)
                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

        return x, avg_attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        encoder_out1 = utils.get_incremental_state(self, incremental_state, 'encoder_out1')
        if encoder_out1 is not None:
            encoder_out1 = tuple(eo.index_select(0, new_order) for eo in encoder_out1)
            utils.set_incremental_state(self, incremental_state, 'encoder_out1', encoder_out1)
        encoder_out2 = utils.get_incremental_state(self, incremental_state, 'encoder_out2')
        if encoder_out2 is not None:
            encoder_out2 = tuple(eo.index_select(0, new_order) for eo in encoder_out2)
            utils.set_incremental_state(self, incremental_state, 'encoder_out2', encoder_out2)

    def _split_encoder_out(self, encoder_out, incremental_state, key='encoder_out'):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = utils.get_incremental_state(self, incremental_state, key)
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, key, result)
        return result

class DualAttentionLayer(fconv.AttentionLayer):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super(fconv.AttentionLayer, self).__init__() #super of fconv.AttentionLayer
        # projects from output of convolution layer of decoder to embedding dimension (query to encoder 1)
        self.in_projection1 = fconv.Linear(conv_channels, embed_dim)
        # projects from output of convolution layer of decoder to embedding dimension (query to encoder 2)
        self.in_projection2 = fconv.Linear(conv_channels, embed_dim)

        # projects from DOUBLE OF embedding dimension to convolution size
        self.out_projection = fconv.Linear(embed_dim * 2, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def attn_compute(self, x, encoder_out, encoder_padding_mask):
        x = self.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = x.float().masked_fill(
                encoder_padding_mask.unsqueeze(1),
                float('-inf')
            ).type_as(x)  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(dim=1, keepdim=True)  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())
        return x, attn_scores

    def forward(self, x, target_embedding, encoder_out1, encoder_out2, encoder_padding_mask1, encoder_padding_mask2,
        randX1, randX2): #encoder_output_dropout, encoder_output_dropout_random_values
        residual = x

        # attention
        x1 = (self.in_projection1(x) + target_embedding) * math.sqrt(0.5)
        x2 = (self.in_projection2(x) + target_embedding) * math.sqrt(0.5)
        
    
        x1, attn_scores1 = self.attn_compute(x1, encoder_out1, encoder_padding_mask1)
        x2, attn_scores2 = self.attn_compute(x2, encoder_out2, encoder_padding_mask2)

        ## ENCODER OUTPUT DROPOUT IMPLEMENTATION
        ## IF encoder_output_dropout = 0.5, there's a 50% chance that
        ## EXACTLY one of the encoder outputs will be dropped out,
        ## AND correspondingly the other encoder's outputs will be doubled.
        ## The decision is now made independently for each element of the minibatch.
        ## Some reading about dropout implementation: https://wiseodd.github.io/techblog/2016/06/25/dropout/

        if self.training:
            x1size = x1.size()
            x2size = x2.size()
            #flatten to 2D, multiply with randX tensors, back to 3D
            x1 = torch.matmul(randX1, x1.view([x1size[0], x1size[1] * x1size[2]])).view(x1size)  
            x2 = torch.matmul(randX2, x2.view([x2size[0], x2size[1] * x2size[2]])).view(x2size)  
        x = torch.cat((x1, x2), -1)
        attn_scores = torch.cat((attn_scores1, attn_scores2), -1)

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)

        return x, attn_scores

