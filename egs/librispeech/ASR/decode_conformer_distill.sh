python3 ./lm2am/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--epoch 60 \
	--avg 10 \
	--max-duration 200 \
	--method ctc-greedy-search 
