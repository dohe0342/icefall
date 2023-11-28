    WORLD_SIZE=4
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    ./distill_l2s/train_distill.py \
	--ted2 False \
	--master-port 12356 \
    --exp-dir distill_l2s/$1 \
    --full-libri 0 \
	--use-fp16 True \
	--num-workers 9 \
    --spec-aug-time-warp-factor 80 \
    --max-duration 1200 \
    --world-size ${WORLD_SIZE} \
    --start-epoch 1 \
    --num-epochs 40 \
    --att-rate 0.0 \
    --num-decoder-layers 0 \
	--distill True \
	--distill-rate 0.2 \
	--unused-params True
	#--manifest-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/fbank \
	#--lang-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/lang_bpe_500 \

