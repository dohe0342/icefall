python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--use-averaged-model False \
	--epoch 27 \
	--avg 1 \
	--max-duration 200 \
	--method ctc-greedy-search 
