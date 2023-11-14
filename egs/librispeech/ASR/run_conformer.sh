    WORLD_SIZE=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    ./conformer_ctc2/train.py \
    --manifest-dir data/fbank \
    --exp-dir conformer_ctc2/$1 \
    --full-libri 1 \
	--use-fp16 True \
	--num-workers 9 \
    --spec-aug-time-warp-factor 80 \
    --max-duration 1200 \
    --world-size ${WORLD_SIZE} \
    --start-epoch 1 \
    --num-epochs 30 \
    --att-rate 0.0 \
    --num-decoder-layers 0 
