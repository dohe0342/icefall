export CUDA_VISIBLE_DEVICES="0,1,2,3"
./pruned_transducer_stateless2_prompt/train_prompt.py \
  --max-duration 600 \
  --num-workers 9 \
  --world-size 4 \
  --exp-dir pruned_transducer_stateless2_prompt/$1 \
  --model-name ../epoch-15.pt \
  --bpe-model data/lang_bpe_500/bpe.model \
  --num-epochs 5 \
  --prompt True \
  --initial-lr 0.01 \
  --use-fp16 True
  #--input-strategy PrecomputedFeatures \
