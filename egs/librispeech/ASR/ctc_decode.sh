for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
  python3 ./lm2am/decode.py \
  --exp-dir lm2am/conformer-18layer-256dim_gpt2-small \
  --result-dir lm2am/conformer-18layer-256dim_gpt2-small \
  --use-averaged-model True --epoch 30 --avg 1 --max-duration 1200 --method $method --distill True --num-decoder-layers 0 
done
