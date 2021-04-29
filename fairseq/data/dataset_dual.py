# This file implements a dataset template and an associated function for the purposes of working with
# a seq2seq model that has two encoders (or/and two decoders). It enables jointly processing an example
# for training or inference in two different 'languages' (eg. words, phonemes, etc.) simultaneously.
#
# The below code inherits and adapts from language_pair_dataset.py provided as part of Facebook's Fairseq toolkit.
#
# Author: Prashant Serai
# Last updated: April 29, 2021

import numpy as np
import torch

from . import data_utils, LanguagePairDataset


def dual_collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens1 = merge('source1', left_pad=left_pad_source)
    src_tokens2 = merge('source2', left_pad=left_pad_source)

    # sort by descending source length
    src_lengths1 = torch.LongTensor([s['source1'].numel() for s in samples])
    src_lengths2 = torch.LongTensor([s['source2'].numel() for s in samples])
    src_lengths1, sort_order = src_lengths1.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_lengths2 = src_lengths2.index_select(0, sort_order)
    src_tokens1 = src_tokens1.index_select(0, sort_order)
    src_tokens2 = src_tokens2.index_select(0, sort_order)

    prev_output_tokens = None
    prev_output_tokens_extra = None
    target = None
    target_extra = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source1']) + len(s['source2']) for s in samples)

    if samples[0].get('target_extra', None) is not None:
        target_extra = merge('target_extra', left_pad=left_pad_target)
        target_extra = target_extra.index_select(0, sort_order)
        tgt_lengths_extra = torch.LongTensor([s['target_extra'].numel() for s in samples]).index_select(0, sort_order)
        ntokens_extra = sum(len(s['target_extra']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens_extra = merge(
                'target_extra',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens_extra = prev_output_tokens_extra.index_select(0, sort_order)

            batch = {
                'id': id,
                'nsentences': len(samples),
                'ntokens': ntokens,
                'ntokens_extra': ntokens_extra,
                'net_input': {
                    'src_tokens1': src_tokens1,
                    'src_lengths1': src_lengths1,
                    'src_tokens2': src_tokens2,
                    'src_lengths2': src_lengths2,
                },
                'target': target,
                'target_extra': target_extra,
            }
    else:
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens1': src_tokens1,
                'src_lengths1': src_lengths1,
                'src_tokens2': src_tokens2,
                'src_lengths2': src_lengths2,
            },
            'target': target,
        }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if prev_output_tokens_extra is not None:
        batch['net_input']['prev_output_tokens_extra'] = prev_output_tokens_extra

    # The below code is not adapted for dual_dataset, hence the print statement and the skip.
    if samples[0].get('alignment', None) is not None:
        print("| ALIGNMENT NOT SUPPORTED FOR THE CLASS DatasetDual, skipping alignment!")

    # if samples[0].get('alignment', None) is not None:
    #     bsz, tgt_sz = batch['target'].shape
    #     src_sz = batch['net_input']['src_tokens'].shape[1]

    #     offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
    #     offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
    #     if left_pad_source:
    #         offsets[:, 0] += (src_sz - src_lengths)
    #     if left_pad_target:
    #         offsets[:, 1] += (tgt_sz - tgt_lengths)

    #     alignments = [
    #         alignment + offset
    #         for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
    #         for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
    #         if check_alignment(alignment, src_len, tgt_len)
    #     ]

    #     if len(alignments) > 0:
    #         alignments = torch.cat(alignments, dim=0)
    #         align_weights = compute_alignment_weights(alignments)

    #         batch['alignments'] = alignments
    #         batch['align_weights'] = align_weights

    return batch


class DatasetDual(LanguagePairDataset):
    def __init__(self, d1, d2, dual_decoder = False):
        assert len(d1) == len(d2)
        self.d1 = d1
        self.d2 = d2
        self.dual_decoder = dual_decoder

    def __getitem__(self, index):
        ex1 = self.d1[index]
        ex2 = self.d2[index]
        if self.dual_decoder:
            example = {
                'id': index,
                'source1': ex1['source'],
                'source2': ex2['source'],
                'target': ex1['target'],
                'target_extra': ex2['target']
            }
        else:
            example = {
                'id': index,
                'source1': ex1['source'],
                'source2': ex2['source'],
                'target': ex2['target']
            }
        return example

    def __len__(self):
        return len(self.d1)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""

        return dual_collate(
            samples, pad_idx=self.d1.src_dict.pad(), eos_idx=self.d1.src_dict.eos(),
            left_pad_source=self.d1.left_pad_source, left_pad_target=self.d1.left_pad_target,
            input_feeding=self.d1.input_feeding,
        )

        #prev_output_tokens doesn't match!
        #id doesn't match
        #both of these keys are lengths 248 for both dictionaries
        #length only captures the first dimension of a multidimensional tensor
        #248 is likely the batch size here
        #error occurs because of the sorting by descending source length in the collate method
        #may be possible to fix by replace the sort_order line with:  sort_order = torch.LongTensor(range(len(id)))
        #also it seems like there's more keys in c1 and c2 than we explicitly account for here 
        #also fix DualSourceSequenceGenerator.generate

        indexes = [sample['id'] for sample in samples]

        c1 = self.d1.collater([self.d1[index] for index in indexes])
        c2 = self.d2.collater([self.d2[index] for index in indexes])

        # c1 = self.d1.collater([self.d1[sample['id']] for sample in samples])
        # c2 = self.d2.collater([self.d2[sample['id']] for sample in samples])

        net_input1 = c1['net_input']; net_input2 = c2['net_input']
        net_input = {}
        for key in net_input1.keys():
            if 'src_' in key:
                net_input[key+'1'] = net_input1[key]
            elif key == 'prev_output_tokens':
                net_input[key] = net_input1[key]
            # elif key == 'ntokens':
            #     net_input[key] = net_input1[key]
            else:
                raise AssertionError
        for key in net_input2.keys():
            if 'src_' in key:
                net_input[key+'2'] = net_input2[key]
            elif key == 'prev_output_tokens':
                if self.dual_decoder:
                    net_input[key+'_extra'] = net_input2[key]
                else:
                    # net_input[key] = net_input2[key]
                    pass
                    # err = "NET_INPUT ASSERTION: "+str(len(indexes))+";\n"
                    # err += str(len(net_input[key])) + "\t" + str(net_input[key]) + "\n"
                    # err += str(len(net_input2[key])) + "\t" + str(net_input2[key]) + "\n"
                    # assert False, err
                    # if not net_input[key] == net_input2[key]:
                    #     print("NET_INPUT ASSERTION:")
                    #     print(net_input[key])
                    #     print(net_input2[key])
                    #     raise AssertionError
            else:
                raise AssertionError

        c = {'net_input': net_input}
        for key in c1.keys():
            if key == 'target':
                c[key] = c1[key]
            elif key == 'ntokens':
                c[key] = c1[key]
            elif key == 'id' or key == 'nsentences':
                c[key] = c1[key]
            else:
                assert key == 'net_input',key
        for key in c2.keys():
            if key == 'target':
                c[key] = c2[key]
            elif key == 'ntokens':
                if 'target' not in samples[0]:
                    c[key] += c2[key] # source tokens
                elif self.dual_decoder:
                    c[key+'_extra'] = c2[key] # target tokens for decoder 2
                else:
                    assert c[key] == c2[key], "NTOKENS:\n"+str(c[key])+"\n"+str(c2[key]) # target tokens for decoder
            elif key == 'id':
                # set1 = set(c[key])
                # set2 = set(c2[key])
                # assert set1 == set2
                assert False, "ID: lengths: "+str(len(indexes))+"; "+str(len(c[key]))+", "+str(len(c2[key]))+"\n"+str(c[key][:10])+"...\n"+str(c2[key][:10])+"...\n"            
                assert c[key] == c2[key], "ID:\n"+str(c[key])+"\n"+str(c2[key])
            elif key == 'nsentences':
                assert c[key] == c2[key], "NSENT:\n"+str(c[key])+"\n"+str(c2[key])
            else:
                assert key == 'net_input',key
        return c



        net_input1['src_tokens1'] = net_input1.pop('src_tokens')   
        net_input1['src_lengths1'] = net_input1.pop('src_lengths')
        net_input1['src_tokens2'] = net_input2['src_tokens']   
        net_input1['src_lengths2'] = net_input2['src_lengths']

        if self.dual_decoder:
            net_input1['prev_output_tokens_extra'] = net_input2['prev_output_tokens']
            c1['target_extra'] = c2['target']
            c1['ntokens_extra'] = c2['ntokens']
        if 'target' not in samples[0]:
            #ntokens and ntokens_extra represent the total number of source tokens
            c1['ntokens'] = c1['ntokens'] + c2['ntokens']
            if 'ntokens_extra' in c1:
                 c1['ntokens_extra'] = c1['ntokens']
        #else ntokens is the total number of target tokens
        return c1

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.d1.num_tokens(index), self.d2.num_tokens(index))

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.d1.size(index)
        # FILTER BASED ON D1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.d1.ordered_indices()
        # RETURN BASED ON D1's sizes

    @property
    def supports_prefetch(self):
        return self.d1.supports_prefetch and self.d2.supports_prefetch

    def prefetch(self, indices):
        self.d1.prefetch(indices)
        self.d2.prefetch(indices)
