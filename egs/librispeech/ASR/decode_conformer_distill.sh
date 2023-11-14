python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--epoch 60 \
	--avg 10 \
	--max-duration 200 \
	--num-decoder-layers 0 \
	--method ctc-greedy-search 
