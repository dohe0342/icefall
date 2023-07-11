export CUDA_VISIBLE_DEVICES="0,1,2,3"
./pruned_transducer_stateless2_prompt/train_prompt.py \
  --max-duration 600 \
  --num-workers 9 \
  --world-size 4 \
  --exp-dir pruned_transducer_stateless2_prompt/exp \
  --model-name epoch-20.pt \
  --bpe-model data/lang_bpe_500/bpe.model \
  --num-epochs 3 \
  --prompt True \
  --input-strategy PrecomputedFeatures \
  --initial-lr 0.0001 \
  --use-fp16 True
