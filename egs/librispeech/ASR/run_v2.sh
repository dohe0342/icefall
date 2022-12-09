#export CUDA_VISIBLE_DEVICES="0,1,2,3"

git pull

./pruned_transducer_stateless_d2v_v2/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --max-duration 300 \
  --exp-dir pruned_transducer_stateless_d2v_v2/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --ctc-loss-scale 0.2 \
  --master-port 12535
