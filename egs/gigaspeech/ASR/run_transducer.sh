export CUDA_VISIBLE_DEVICES="0,1,2,3"
./pruned_transducer_stateless2/train.py \
  --max-duration 600 \
  --num-workers 9 \
  --world-size 4 \
  --exp-dir pruned_transducer_stateless2_prompt/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --use-fp16 True
