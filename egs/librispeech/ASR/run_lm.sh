WORLD_SIZE=4
if [ $2 -eq 0 ];then
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

if [ $2 -eq 1 ];then
	export CUDA_VISIBLE_DEVICES="4,5,6,7"
fi

port=$(($RANDOM% 601+12300))
./lm2am/train_distill.py \
	--exp-dir lm2am/$1 \
	--wandb False \
	--master-port $port \
	--full-libri 1 \
	--pure-libri 1 \
	--use-fp16 True \
	--num-workers 9 \
	--spec-aug-time-warp-factor 80 \
	--max-duration 800 \
	--world-size ${WORLD_SIZE} \
	--start-epoch 1 \
	--num-epochs 30 \
	--att-rate 0.0 \
	--num-decoder-layers 0 \
	--distill True \
	--distill-rate 0.1 \
	--unused-params True \
	--kernel-size 15 \
	--lm-tune True \
	--lm-name gpt2
	#--lm-name gpt2
#--lm-name gpt2-medium
#--lm-name mistralai/Mistral-7B-v0.1
#--lm-name gpt2
#--manifest-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/fbank \
#--lang-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/lang_bpe_500 \
#--ted2 True \
#--manifest-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/fbank \
#--lang-dir /home/work/workspace/icefall/egs/tedlium2/ASR/data/lang_bpe_500 \

