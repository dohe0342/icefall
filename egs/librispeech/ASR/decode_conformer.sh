python3 ./conformer_ctc2/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--lang-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/lang_bpe_500 \
	--epoch $2 \
	--ted2 True \
	--avg 10 \
	--max-duration 200 \
	--num-decoder-layers 0 \
	--method ctc-greedy-search
