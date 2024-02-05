python3 ./conformer_ctc2/decode.py \
	--exp-dir $1 \
	--use-averaged-model True \
	--epoch $2 \
	--avg 10 \
	--max-duration 1200 \
	--num-decoder-layers 0 \
	--method ctc-greedy-search

#--manifest-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/fbank \
#--lang-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/lang_bpe_500 \
#--ted2 True \

