# greedy search
for i in 100 200 300 400 500 600 700 800 900; do
	./pruned_transducer_stateless2_prompt/decode.py \
	  --avg 1 \
	  --iter 1900 \
	  --dataset ted2 \
	  --decoding-method greedy_search \
	  --exp-dir pruned_transducer_stateless2_prompt/$1 \
	  --bpe-model data/lang_bpe_500/bpe.model \
	  --max-duration 600 \
	  --prompt True 
  done
  #--prompt True 
  #--input-strategy PrecomputedFeatures
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
