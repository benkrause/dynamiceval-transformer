#!/bin/bash

URL=http://curtis.ml.cmu.edu/datasets/pretrained_xl

DATA_ROOT=./

function download () {
  fileurl=${1}
  filename=${fileurl##*/}
  if [ ! -f ${filename} ]; then
    echo ">>> Download '${filename}' from '${fileurl}'."
    wget --quiet ${fileurl}
  else
    echo "*** File '${filename}' exists. Skip."
  fi
}

cd $DATA_ROOT
mkdir -p pretrained_xl && cd pretrained_xl

# enwik8
mkdir -p tf_enwik8 && cd tf_enwik8

mkdir -p data && cd data
download ${URL}/tf_enwiki8/data/cache.pkl
download ${URL}/tf_enwiki8/data/corpus-info.json
cd ..

mkdir -p model && cd model
download ${URL}/tf_enwiki8/model/checkpoint
download ${URL}/tf_enwiki8/model/model.ckpt-0.data-00000-of-00001
download ${URL}/tf_enwiki8/model/model.ckpt-0.index
download ${URL}/tf_enwiki8/model/model.ckpt-0.meta
cd ..

cd ..

# text8
mkdir -p tf_text8 && cd tf_text8

mkdir -p data && cd data
download ${URL}/tf_text8/data/cache.pkl
download ${URL}/tf_text8/data/corpus-info.json
cd ..

mkdir -p model && cd model
download ${URL}/tf_text8/model/checkpoint
download ${URL}/tf_text8/model/model.ckpt-0.data-00000-of-00001
download ${URL}/tf_text8/model/model.ckpt-0.index
download ${URL}/tf_text8/model/model.ckpt-0.meta
cd ..

cd ..

# wt103
mkdir -p tf_wt103 && cd tf_wt103

mkdir -p data && cd data
download ${URL}/tf_wt103/data/cache.pkl
download ${URL}/tf_wt103/data/corpus-info.json
cd ..

mkdir -p model && cd model
download ${URL}/tf_wt103/model/checkpoint
download ${URL}/tf_wt103/model/model.ckpt-0.data-00000-of-00001
download ${URL}/tf_wt103/model/model.ckpt-0.index
download ${URL}/tf_wt103/model/model.ckpt-0.meta
cd ..

cd ..

