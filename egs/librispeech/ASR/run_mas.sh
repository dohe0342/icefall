WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./lm2am/train_distill.py \
	--exp-dir lm2am/$1 \
	--master-port 12355 \
	--full-libri 1 \
	--use-fp16 True \
	--num-workers 9 \
	--spec-aug-time-warp-factor 80 \
	--max-duration 800 \
	--world-size ${WORLD_SIZE} \
	--start-epoch 5 \
	--num-epochs 30 \
	--distill True \
	--quant False \
	--distill-rate 0.5 \
	--unused-params True \
	--dim-model 512 \
	--dim-feedforward 2048 \
	--kernel-size 31 \
	--att-rate 0.0 \
	--num-decoder-layers 0 \
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

