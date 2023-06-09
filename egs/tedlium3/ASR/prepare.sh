#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=0
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/tedlium3
#      You can find data, doc, legacy, LM, etc, inside it.
#      You can download them from https://www.openslr.org/51
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=/DB

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  5000
  2000
  1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/tedlium3,
  # you can create a symlink
  #
  # ln -sfv /path/to/tedlium3 $dl_dir/tedlium3
  #
  if [ ! -d $dl_dir/tedlium3 ]; then
    lhotse download tedlium $dl_dir
    mv $dl_dir/TEDLIUM_release-3 $dl_dir/tedlium3
  fi

  # Download big and small 4 gram lanuage models
  if [ ! -d $dl_dir/lm ]; then
    wget --continue http://kaldi-asr.org/models/5/4gram_small.arpa.gz -P $dl_dir/lm
    wget --continue http://kaldi-asr.org/models/5/4gram_big.arpa.gz -P $dl_dir/lm
    gzip -d $dl_dir/lm/4gram_small.arpa.gz $dl_dir/lm/4gram_big.arpa.gz
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #ln -sfv /path/to/musan $dl_dir/musan

  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare tedlium3 manifests"
  if [ ! -f data/manifests/.tedlium3.done ]; then
    # We assume that you have downloaded the tedlium3 corpus
    # to $dl_dir/tedlium3
    mkdir -p data/manifests
    lhotse prepare tedlium $dl_dir/tedlium3 data/manifests
    touch data/manifests/.tedlium3.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifests"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -e data/manifests/.musan.done ]; then
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for tedlium3"

  if [ ! -e data/fbank/.tedlium3.done ]; then
    mkdir -p data/fbank

    python3 ./local/compute_fbank_tedlium.py

    gunzip -c data/fbank/tedlium_cuts_train.jsonl.gz | shuf | \
    gzip -c > data/fbank/tedlium_cuts_train-shuf.jsonl.gz
    mv data/fbank/tedlium_cuts_train-shuf.jsonl.gz \
       data/fbank/tedlium_cuts_train.jsonl.gz

    touch data/fbank/.tedlium3.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  if [ ! -e data/fbank/.musan.done ]; then
    mkdir -p data/fbank
    python3 ./local/compute_fbank_musan.py
    touch data/fbank/.musan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare BPE train data and set of words"
  lang_dir=data/lang
  mkdir -p $lang_dir

  if [ ! -f $lang_dir/train.txt ]; then
    gunzip -c $dl_dir/tedlium3/LM/*.en.gz | sed 's: <\/s>::g' > $lang_dir/train_orig.txt

    ./local/prepare_transcripts.py \
      --input-text-path $lang_dir/train_orig.txt \
      --output-text-path $lang_dir/train.txt
  fi

  if [ ! -f $lang_dir/words.txt ]; then

    awk '{print $1}' $dl_dir/tedlium3/TEDLIUM.152k.dic |
    sed 's:([0-9])::g' | sort | uniq > $lang_dir/words_orig.txt

    ./local/prepare_words.py --lang-dir $lang_dir
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang/words.txt $lang_dir

    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --vocab-size $vocab_size \
      --transcript data/lang/train.txt

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir --oov "<unk>"
    fi
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_4_gram_small.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      --max-arpa-warnings=-1 \
      $dl_dir/lm/4gram_small.arpa > data/lm/G_4_gram_small.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram_big.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      --max-arpa-warnings=-1 \
      $dl_dir/lm/4gram_big.arpa > data/lm/G_4_gram_big.fst.txt
  fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compile HLG"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/HLG.pt ]; then
      ./local/compile_hlg.py \
        --lang-dir $lang_dir \
        --lm G_4_gram_small
    fi
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Split cuts by speaker id"
  gzip -d data/fbank/tedlium_cuts_test.jsonl.gz

  i=0
  for spk in $dl_dir/tedlium3/legacy/test/sph/*; do
	  spk_id=${spk#*sph\/}
	  spk_id=${spk_id%.sph}
	  echo $spk_id
	  cat data/fbank/tedlium_cuts_test.jsonl | grep speaker\":\ \"$spk_id\" > tedlium_cuts_test_$i.jsonl
	  gzip tedlium_cuts_test_$i.jsonl
	  i=`expr $i+1` 
  done
  #cat data/fbank/tedlium_cuts_test.jsonl.gz | grep 

fi
