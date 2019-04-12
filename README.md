# Dynamic evaluation of Transformer-XL
## Introduction

This code applies [dynamic evaluation](https://arxiv.org/abs/1709.07432) to pretrained Transformer-XL models from this [paper](https://arxiv.org/abs/1901.02860). Our codebase is a modified version of [their codebase](https://github.com/kimiyoung/transformer-xl). We used this code to obtain state of the art results on WikiText-103 (perplexity: 16.4), enwik8 (bits/char: 0.94), and text8 (bits/char: 1.04).

## Requirements

- Python 2.7
- Tensorflow, these results used version [1.13.1](https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1), but similar versions should work as well



## Obtain and dynamically evaluate pretrained SoTA models

#### 1. Download preprocessed data (vocab) & pretrained models

(a) cd to the `tf` folder

(b) Download the models with `bash sota/download.sh`



#### 2. Run dynamic evaluation scripts to replicate SoTA results on GPUs



- **Wiki-Text 103**:  run `bash sota/wt103.sh` with the default parameters. Takes < 20 minutes. Should get a perplexity of 16.4.

- **enwik8**: run `bash sota/enwik8.sh` with the default parameters. Takes a few hours. Should get a bpc of 0.940.

- **text8**:  run `bash sota/text8.sh` with the default parameters. Takes a few hours. Should get a bpc of 1.038.


## Other Notes:

-  By default, scripts set CUDA_VISABLE_DEVICES=0 . This contains run to a single gpu. 

-  Not currently set up to run on multiple threads. This easiest way to do this would probably be to divide the test set up into chunks and dynamically evaluate separate runs of `dynamiceval_tf.py`, and combine results.

-  SGD dynamic evaluation can be applied by setting `--rms=False` in the arguments of dynamiceval_tf.py . This will require a higher learning rate to get best results.

- If you are doing hyper parameter tuning for dynamic evaluation, be sure to do this on the validation set. Use the validation set by setting `--eval_split=valid` in the arguments of dynamiceval_tf.py.

- The `ratio` of dynamiceval_tf.py argument allows you to only use a subset of (1/ratio) of the evaluation set. This can speed up hyper-parameter tuning.

- Setting the learning rate to 0 is equivalent to static/normal evaluation.