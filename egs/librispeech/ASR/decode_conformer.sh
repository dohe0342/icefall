python3 ./conformer_ctc2/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--epoch 39 \
	--avg 9 \
	--max-duration 200 \
	--num-decoder-layers 0 \
	--method ctc-greedy-search
