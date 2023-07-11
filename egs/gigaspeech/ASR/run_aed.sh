export CUDA_VISIBLE_DEVICES="0,1,2,3"
./conformer_ctc/train_prompt.py \
  --max-duration 600 \
  --num-workers 1 \
  --world-size 8 \
  --exp-dir conformer_ctc/exp_500 \
  --lang-dir data/lang_bpe_500
