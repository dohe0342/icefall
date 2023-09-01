export CUDA_VISIBLE_DEVICES="0,1,2,3"

./encodec/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --input-strategy AudioSamples \
  ----enable-spec-aug False \
  --exp-dir pruned_transducer_stateless5/exp-L \
  --max-duration 450 \
  --use-fp16 1 \
  --num-encoder-layers 18 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 512
