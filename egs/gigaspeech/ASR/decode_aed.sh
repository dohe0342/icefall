./conformer_ctc/decode.py \
  --epoch 20 \
  --avg 1 \
  --method 1best \
  --num-paths 1000 \
  --exp-dir conformer_ctc/exp \
  --lang-dir data/lang_bpe_500 \
  --max-duration 300 \
  --input-strategy PrecomputedFeatures \
  --num-workers 10
