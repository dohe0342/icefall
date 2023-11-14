export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless5/test \
  --max-duration 600 \
  --use-fp16 1
