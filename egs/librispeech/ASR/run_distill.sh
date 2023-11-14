    WORLD_SIZE=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    ./conformer_ctc2/train.py \
    --manifest-dir data/fbank \
    --exp-dir lm2am/exp \
    --full-libri 1 \
    --spec-aug-time-warp-factor 80 \
    --max-duration 600 \
    --world-size ${WORLD_SIZE} \
    --start-epoch 1 \
    --num-epochs 30 \
    --att-rate 0.0 \
    --num-decoder-layers 0 
