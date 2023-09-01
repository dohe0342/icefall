export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless5/exp-L \
  --max-duration 300 \
  --use-fp16 0 \
  --num-encoder-layers 18 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 512
