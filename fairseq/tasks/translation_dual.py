# This file implements a version of the translation task with two source languages and one main
# target language (along with possibly an auxiliary or 'extra' target language). We parse
# args.source_lang to dissect the two input languages separated by commas. If args.tgt_lang contains
# a comma, then the first language is the primary target language and the second is the 'extra'.
# Otherwise, args.tgt_lang contains a single string that is taken as the main target language which
# we finally use at inference time.  The 'extra' target language (if any) is used only at train time.
#
# The below code inherits and adapts from translation.py provided as part of Facebook's Fairseq toolkit.
#
# Author: Prashant Serai
# Last updated: April 29, 2021


from . import translation as t
from . import FairseqTask, register_task
from fairseq.data.dataset_dual import DatasetDual
from fairseq import options, utils
from fairseq.sequence_generator_dual import DualSourceSequenceGenerator
import os
import torch

@register_task('translation_dual')
class TranslationDual(t.TranslationTask):
    def __init__(self, args, src_dict1, src_dict2, tgt_dict, tgt_dict_extra):
        super(t.TranslationTask, self).__init__(args) #super of t.TranslationTask
        self.src_dict1 = src_dict1
        self.src_dict2 = src_dict2
        self.tgt_dict = tgt_dict
        if tgt_dict_extra is not None:
            # self.dual_decoder = True
            self.tgt_dict_extra = tgt_dict_extra
        else:
            # self.dual_decoder = False
            self.tgt_dict_extra = self.tgt_dict #just duplicate
        assert self.tgt_dict.pad() == self.tgt_dict_extra.pad()
        assert self.target_dictionary is not None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly') # See comment below?
            # TRY ADDING: --source-lang true_w,true_p --target-lang reco_w
            # OR ENABLING BELOW HARDCODED DEFAULTS.
            # args.source_lang = "true_w,true_p"
            # args.target_lang = "reco_w"
        #     args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])

        assert type(args.source_lang) == str
        if ',' not in args.source_lang:
            raise Exception("source-lang is " + args.source_lang + " source-lang needs to contain two comma separated strings")
        # load dictionaries
        src_lang1, src_lang2 = args.source_lang.split(',')
        src_dict1 = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(src_lang1)))
        src_dict2 = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(src_lang2)))
        if ',' in args.target_lang:
            assert args.criterion == 'cross_entropy_dual'
            target_lang, target_lang_extra = args.target_lang.split(',')
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(target_lang)))
            tgt_dict_extra = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(target_lang_extra)))
            dual_decoder = True
        else:
            tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
            dual_decoder = False
            tgt_dict_extra = None

        assert src_dict1.pad() == tgt_dict.pad()
        assert src_dict1.eos() == tgt_dict.eos()
        assert src_dict1.unk() == tgt_dict.unk()
        assert src_dict2.pad() == tgt_dict.pad()
        assert src_dict2.eos() == tgt_dict.eos()
        assert src_dict2.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(src_lang1, len(src_dict1)))
        print('| [{}] dictionary: {} types'.format(src_lang2, len(src_dict2)))

        if dual_decoder:
            print('| [{}] dictionary: {} types'.format(target_lang, len(tgt_dict)))
            print('| [{}] dictionary: {} types'.format(target_lang_extra, len(tgt_dict_extra)))
        else:
            print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        return cls(args, src_dict1, src_dict2, tgt_dict, tgt_dict_extra)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src1, src2 = self.args.source_lang.split(',')
        if ',' in self.args.target_lang:
            # assert self.dual_decoder
            tgt, tgt_extra = self.args.target_lang.split(',')
            dual_decoder = True
        else:
            tgt = self.args.target_lang
            tgt_extra = tgt
            dual_decoder = False
        d2 = t.load_langpair_dataset(
            data_path, split, src2, self.src_dict2, tgt_extra, self.tgt_dict_extra,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
        )
        d1 = t.load_langpair_dataset(
            data_path, split, src1, self.src_dict1, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
        )
        self.datasets[split] = DatasetDual(d1, d2, dual_decoder = dual_decoder)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator_dual import DualSourceSequenceGenerator
            if getattr(args, 'print_alignment', False):
                assert False #no dual source version made for this case
                # seq_gen_cls = SequenceGeneratorWithAlignment
            else:
                seq_gen_cls = DualSourceSequenceGenerator
            return seq_gen_cls(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        raise NotImplementedError
        #return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.src_dict1, self.src_dict2

    @property
    def target_dictionary_extra(self):
        return self.tgt_dict_extra

    @property
    def source_dictionary1(self):
        """Return the source 1 :class:`~fairseq.data.Dictionary`."""
        return self.src_dict1

    @property
    def source_dictionary2(self):
        """Return the source 2 :class:`~fairseq.data.Dictionary`."""
        return self.src_dict2
