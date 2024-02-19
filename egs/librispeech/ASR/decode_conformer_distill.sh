CUDA_VISIBLE_DEVICES=$2 python3 ./lm2am/decode_multiple.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--max-duration 1200 \
	--num-decoder-layers 0 \
	--distill False \
	--method ctc-greedy-search \
	--quant False \
	--kernel-size 31 \
	--dim-model 512 \
	--dim-feedforward 2048 \
	--lm-name gpt2 \
	--wandb True
