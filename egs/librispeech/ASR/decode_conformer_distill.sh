CUDA_VISIBLE_DEVICES=$2 python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--epoch $3 \
	--avg 10 \
	--use-averaged-model True \
	--max-duration 1200 \
	--num-decoder-layers 0 \
	--method ctc-greedy-search \
	--distill True \
	--quant False \
	--lm-name gpt2_medium \
	--wandb False \
	--lm-tune False \
	--kernel-size 15 \
	--dim-model 256 \
	--manifest-dir /workspace/icefall_kt/egs/librispeech/ASR/data/fbank \
	--lang-dir /workspace/icefall_kt/egs/librispeech/ASR/data/lang_bpe_500
#--dim-feedforward 1024 \
