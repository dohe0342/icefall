python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--epoch $2 \
	--avg 1 \
	--max-duration 200 \
	--num-decoder-layers 0 \
	--distill True \
	--method ctc-greedy-search 
