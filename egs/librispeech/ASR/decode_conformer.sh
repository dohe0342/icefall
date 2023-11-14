python3 ./conformer_ctc2/decode.py \
	--exp-dir $1 \
	--use-averaged-model False \
	--epoch 14 \
	--avg 1 \
	--max-duration 200 \
	--method ctc-greedy-search
