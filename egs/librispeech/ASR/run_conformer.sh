:<< 'END'
WORLD_SIZE=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    ./conformer_ctc2/train.py \
    --manifest-dir data/fbank \
    --exp-dir conformer_ctc2/$1 \
    --full-libri 0 \
	--use-fp16 True \
	--num-workers 9 \
    --spec-aug-time-warp-factor 80 \
    --max-duration 1200 \
    --world-size ${WORLD_SIZE} \
    --start-epoch 61 \
    --num-epochs 70 \
    --att-rate 0.0 \
    --num-decoder-layers 0 \
	--master-port 12355
END

WORLD_SIZE=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    ./conformer_ctc2_origin/train.py \
    --exp-dir conformer_ctc2_origin/$1 \
    --full-libri 1 \
    --use-fp16 True \
    --num-workers 9 \
    --spec-aug-time-warp-factor 80 \
    --max-duration 1200 \
    --world-size ${WORLD_SIZE} \
    --start-epoch 1 \
    --num-epochs 40 \
    --att-rate 0.0 \
    --num-decoder-layers 0 \
    --master-port 12355
#--manifest-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/fbank \
#--lang-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/lang_bpe_500 \

