# greedy search
for i in {0..4}; do
	./pruned_transducer_stateless2_prompt/decode.py \
	  --avg 1 \
	  --epoch $i \
	  --decoding-method greedy_search \
	  --exp-dir pruned_transducer_stateless2_prompt/exp \
	  --bpe-model data/lang_bpe_500/bpe.model \
	  --max-duration 600 \
	  --prompt False \
	  --input-strategy PrecomputedFeatures
done
# fast beam search
#./pruned_transducer_stateless2/decode.py \
#  --avg 1 \
#  --epoch 20 \
#  --decoding-method fast_beam_search \
#  --exp-dir pruned_transducer_stateless2/exp \
#  --bpe-model data/lang_bpe_500/bpe.model \
#  --input-strategy PrecomputedFeatures \
#  --max-duration 600
#
## modified beam search
#./pruned_transducer_stateless2/decode.py \
#  --avg 1 \
#  --epoch 15 \
#  --decoding-method modified_beam_search \
#  --exp-dir pruned_transducer_stateless2/exp \
#  --bpe-model data/lang_bpe_500/bpe.model \
#  --input-strategy PrecomputedFeatures \
#  --max-duration 600
