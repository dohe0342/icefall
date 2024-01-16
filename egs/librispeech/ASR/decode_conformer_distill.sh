python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--epoch $2 \
	--avg 10 \
	--max-duration 600 \
	--num-decoder-layers 0 \
	--distill True \
	--method ctc-greedy-search \
	--lm-name gpt2-medium
