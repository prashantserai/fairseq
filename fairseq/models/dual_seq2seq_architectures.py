'''
This file provides a template for a dual encoder sequence to sequence model. It also registers the parameters
for end to end models for speech recognition error prediction that are described in the below work.

Serai, Prashant et al. “Hallucination of speech recognition errors with sequence to sequence learning.”
Submitted to IEEE Transactions on Audio, Speech, and Language Processing, 2021.
arXiv preprint arXiv:2103.12258
https://arxiv.org/pdf/2103.12258.pdf

The below code inherits and adapts from fconv.py provided as part of Facebook's Fairseq toolkit.

Author: Prashant Serai
Last updated: April 29, 2021
'''

from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq.models.fconv as fconv

from fairseq import utils
from fairseq.data import Dictionary

from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    AdaptiveSoftmax, BeamableMM, GradMultiply, LearnedPositionalEmbedding,
    LinearizedConvolution,
)

from . import fconv_dual

from fairseq.models.fconv import FConvEncoder

@register_model('fconv_dual_single')
class FConvModelDualEncoderSingleDecoder(fconv.FConvModel):
    def __init__(self, encoder1, encoder2, decoder):
        super(FairseqEncoderDecoderModel, self).__init__() # super of FairseqEncoderDecoderModel

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder
        #assert isinstance(self.encoder, FairseqEncoder)
        #assert isinstance(self.decoder, FairseqDecoder)
        self.encoder1.num_attention_layers = sum(layer is not None for layer in decoder.attention)
        self.encoder2.num_attention_layers = sum(layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        fconv.FConvModel.add_args(parser)
        parser.add_argument('--encoder-output-dropout', type=float, metavar='EOD',
                            help='encoder output dropout probability', default=0.0)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        fconv.base_architecture(args)

        encoder_embed_dict = None
        assert args.encoder_embed_path == None
        # if args.encoder_embed_path:
        #     encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
        #     utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        encoder1 = FConvEncoder(
            dictionary=task.source_dictionary1,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        encoder2 = FConvEncoder(
            dictionary=task.source_dictionary2,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        decoder = fconv_dual.FConvDecoderWithDualAttention(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
            encoder_output_dropout=args.encoder_output_dropout,
        )
        return FConvModelDualEncoderSingleDecoder(encoder1, encoder2, decoder)


    def forward(self, src_tokens1, src_lengths1, src_tokens2, src_lengths2, prev_output_tokens, **kwargs):
        encoder_out1 = self.encoder1(src_tokens1, src_lengths=src_lengths1, **kwargs)
        encoder_out2 = self.encoder2(src_tokens2, src_lengths=src_lengths2, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out1=encoder_out1,
                                    encoder_out2=encoder_out2, **kwargs)
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder1.max_positions(), self.encoder2.max_positions(), self.decoder.max_positions())

@register_model('fconv_dual_dual')
class FConvModelDualEncoderDualDecoder(FConvModelDualEncoderSingleDecoder):
    def __init__(self, encoder1, encoder2, decoder, decoder_extra):
        super().__init__(encoder1, encoder2, decoder)
        self.decoder_extra = decoder_extra

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        fconv.base_architecture(args)

        encoder_embed_dict = None
        assert args.encoder_embed_path == None

        decoder_embed_dict = None
        assert args.decoder_embed_path == None

        encoder1 = FConvEncoder(
            dictionary=task.source_dictionary1,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        encoder2 = FConvEncoder(
            dictionary=task.source_dictionary2,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        decoder = fconv_dual.FConvDecoderWithDualAttention(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
            encoder_output_dropout=args.encoder_output_dropout,
        )

        decoder_extra = fconv_dual.FConvDecoderWithDualAttention(
            dictionary=task.target_dictionary_extra,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
            encoder_output_dropout=args.encoder_output_dropout,
        )
        return FConvModelDualEncoderDualDecoder(encoder1, encoder2, decoder, decoder_extra)

    def forward(self, src_tokens1, src_lengths1, src_tokens2, src_lengths2, prev_output_tokens,
            prev_output_tokens_extra, decode_extra=False, **kwargs):
        encoder_out1 = self.encoder1(src_tokens1, src_lengths=src_lengths1, **kwargs)
        encoder_out2 = self.encoder2(src_tokens2, src_lengths=src_lengths2, **kwargs)
        if not decode_extra:
            return self.decoder(prev_output_tokens, encoder_out1=encoder_out1,
                                        encoder_out2=encoder_out2, **kwargs)
        else:
            return self.decoder_extra(prev_output_tokens_extra, encoder_out1=encoder_out1,
                                    encoder_out2=encoder_out2, **kwargs)

@register_model_architecture('fconv', 'fconv_ww1')
def fconv_ww1(args):
    encoder_convs = '[(256, 3)] * 4'
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', encoder_convs)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.encoder_output_dropout = getattr(args, 'encoder_output_dropout', 0)
    fconv.base_architecture(args)

@register_model_architecture('fconv', 'fconv_pw1')
def fconv_pw1(args):
    encoder_convs = '[(64, 11)] * 3'
    encoder_convs += ' + [(128, 7)] * 2'
    encoder_convs += ' + [(256, 5)] * 1'
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', encoder_convs)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.encoder_output_dropout = getattr(args, 'encoder_output_dropout', 0)
    fconv.base_architecture(args)

@register_model_architecture('fconv_dual_single', 'fconv_dual_wpw1')
def fconv_dual_wpw1(args):
    fconv_pw1(args)

@register_model_architecture('fconv_dual_dual', 'fconv_dual_wpwp1')
def fconv_dual_wpwp1(args):
    fconv_pw1(args)