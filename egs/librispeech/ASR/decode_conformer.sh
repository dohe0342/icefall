python3 ./conformer_ctc2/decode.py \
	--exp-dir conformer_ctc2/exp \
	--use-averaged-model False \
	--epoch 13 \
	--avg 1 \
	--max-duration 200 \
	--method ctc-greedy-search
