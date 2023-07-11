export CUDA_VISIBLE_DEVICES="0,1,2,3"
./conformer_ctc_prompt/train_prompt.py \
  --max-duration 600 \
  --num-workers 9 \
  --world-size 4 \
  --exp-dir conformer_ctc_prompt/prompt \
  --model-name ../epoch-20.pt \
  --lang-dir data/lang_bpe_500 \
  --prompt True
