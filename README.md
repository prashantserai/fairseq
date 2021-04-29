This repository includes code and parameter specifications for the sequence to sequence models used for predicting ASR errors in the belowmentioned manuscripts. The repository is based on a fork of Fairseq(-py) where the existing Fairseq code is kept as is, and the author's implementations are in the belowmentioned new files:

* new file:   fairseq/models/dual_seq2seq_architectures.py
           >> Implements the class for a dual encoder sequence to sequence model, and "registers" the hyperparameters for all the models described in the papers.
* new file:   fairseq/models/fconv_dual.py
           >> Implements the class for a decoder that can attend to two encoders.
* new file:   fairseq/data/dataset_dual.py
           >> Implements a dataset class that enables jointly processing an example in two different 'languages' (eg. words, phonemes, etc.) in sync for training or inference.
* new file:   fairseq/tasks/translation_dual.py
           >> Specifies a translation "task" where there are two source languages and one main target language (along with possibly an auxiliary or 'extra' target language, tried in an ablation study).
* new file:   fairseq/criterions/cross_entropy_dual.py 
           >> Specifies a two target cross-entropy criterion for the abovementioned ablation study where an auxiliary decoder was employed.    
* new file:   fairseq/sequence_generator_dual.py
           >> Provides helpers for performing inference with a dual encoder seq2seq model.

# Papers implemented in this repository

```bibtex
@inproceedings{serai2020end,
  title = {END TO END SPEECH RECOGNITION ERROR PREDICTION WITH SEQUENCE TO SEQUENCE LEARNING},
  author = {Prashant Serai and Adam Stiff and Eric Fosler-Lussier},
  booktitle = {ICASSP},
  year = {2020},
  organization={IEEE}
}
@article{serai2021hallucination,
  title={Hallucination of speech recognition errors with sequence to sequence learning},
  author={Serai, Prashant and Sunder, Vishal and Fosler-Lussier, Eric},
  journal={arXiv preprint arXiv:2103.12258},
  year={2021}
}
```

# Fairseq: About, Requirements, and Installation
Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks. 

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

**Fairseq: Installing from source**

To install fairseq from source and develop locally:
```bash
git clone https://github.com/OSU-slatelab/asr-errors
cd asr-errors
pip install --editable .
```
