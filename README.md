This repository includes code for predicting ASR errors, and is based on a fork of Fairseq(-py).

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks. 

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

**Installing from source**

To install fairseq from source and develop locally:
```bash
git clone https://github.com/OSU-slatelab/asr-errors
cd asr-errors
pip install --editable .
```

# Papers implemented

```bibtex
@inproceedings{serai2020end,
  title = {END TO END SPEECH RECOGNITION ERROR PREDICTION WITH SEQUENCE TO SEQUENCE LEARNING},
  author = {Prashant Serai and Adam Stiff and Eric Fosler-Lussier},
  booktitle = {ICASSP},
  year = {2020},
  organization={IEEE}
}
```