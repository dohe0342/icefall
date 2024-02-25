CUDA_VISIBLE_DEVICES=$2 python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--epoch $3 \
	--avg 1 \
	--use-averaged-model True \
	--max-duration 1200 \
	--num-decoder-layers 0 \
	--method ctc-greedy-search \
	--distill True \
	--quant True \
	--kernel-size 15 \
	--lm-name gpt2 \
	--wandb False \
	--lm-tune True
