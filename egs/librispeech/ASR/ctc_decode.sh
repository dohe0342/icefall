for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
  python3 ./lm2am/decode.py \
  --exp-dir conformer_ctc2/exp \
  --use-averaged-model False --epoch 18 --avg 1 --max-duration 200 --method $method
done
