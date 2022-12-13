git pull

workstation="bear"

if [ $workstation = "whale" ]; then
	export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
	./pruned_transducer_stateless_d2v_v2/train.py \
		--wandb true \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--multi-optim True \
		--world-size 8 \
		--num-epochs 30 \
		--start-epoch 1 \
		--full-libri 1 \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
		--max-duration 250 \
		--freeze-finetune-updates 2000 \
		--use-fp16 1 \
		--peak-enc-lr 0.03 \
		--peak-dec-lr 0.1 \
		--accum-grads 1 \
		--encoder-type d2v \
		--additional-block True \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--prune-range 10 \
		--context-size 2 \
		--ctc-loss-scale 0.2 
else
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
	./pruned_transducer_stateless_d2v_v2/train.py \
		--wandb true \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--multi-optim True \
		--world-size 4 \
		--start-batch 34000 \
		--num-epochs 30 \
		--full-libri 1 \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
		--max-duration 150 \
		--freeze-finetune-updates 2000 \
		--use-fp16 1 \
		--peak-enc-lr 0.001 \
		--peak-dec-lr 0.5 \
		--accum-grads 3 \
		--encoder-type d2v \
		--additional-block True \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--prune-range 10 \
		--context-size 2 \
		--ctc-loss-scale 0.2 
fi

#--start-epoch 6 \
