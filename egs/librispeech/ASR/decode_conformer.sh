python3 ./conformer_ctc2/decode.py \
	--exp-dir conformer_ctc2/exp \
	--use-averaged-model True \
	--epoch 30 \
	--avg 8 \
	--max-duration 200 \
	--method ctc-greedy-search
