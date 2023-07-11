export CUDA_VISIBLE_DEVICES="0,1,2,3"
./conformer_ctc/train_prompt.py \
  --max-duration 600 \
  --num-workers 9 \
  --world-size 4 \
  --exp-dir conformer_ctc/exp \
  --model-name ../epoch-20.pt \
  --lang-dir data/lang_bpe_500
