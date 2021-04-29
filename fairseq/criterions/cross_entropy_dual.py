# This file implements a dual cross entropy criterion for training a seq2seq model with two decoders. 
# The basic implementation here applies the cross entropy loss two times using the targets 
# for each of the two decoders, and simply sums them.
#
# TODO: incorporate a command line argument lambda to enable a weighted sum of the two losses.
#
# The below code inherits and adapts from cross_entropy.py provided as part of Facebook's Fairseq toolkit.
#
# Author: Prashant Serai
# Last updated: April 29, 2021

from fairseq import utils

from . import cross_entropy, register_criterion
import math

@register_criterion('cross_entropy_dual')
class DualCrossEntropyCriterion(cross_entropy.CrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        basic_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample['target'], sample['target_extra'] = sample['target_extra'], sample['target']
        net_output_extra = model(**sample['net_input'], decode_extra=True)
        extra_loss, _ = self.compute_loss(model, net_output_extra, sample, reduce=reduce)
        sample['target'], sample['target_extra'] = sample['target_extra'], sample['target']   

        loss = basic_loss + extra_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        sample_size_extra = sample['target_extra'].size(0) if self.args.sentence_avg else sample['ntokens_extra']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'basic_loss': utils.item(basic_loss.data) if reduce else basic_loss.data,
            'extra_loss': utils.item(extra_loss.data) if reduce else extra_loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'sample_size_extra': sample_size_extra,
        }
        #    'ntokens_extra': sample['ntokens_extra'],
        #    'nsentences_extra': sample['target_extra'].size(0),
        return loss, sample_size, logging_output
        
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        basic_loss_sum = sum(log.get('basic_loss', 0) for log in logging_outputs)
        extra_loss_sum = sum(log.get('extra_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_extra = sum(log.get('sample_size_extra', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'basic_loss': basic_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'extra_loss': extra_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'extra_loss_v2': extra_loss_sum / sample_size_extra / math.log(2) if sample_size_extra > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

