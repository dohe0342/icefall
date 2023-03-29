workstation=$3

if [ $workstation = "whale" ]; then
	export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
	if [ ! -e ./pruned_transducer_stateless_d2v_v2/$1/.train.done ]; then
		./pruned_transducer_stateless_d2v_v2/train_adapter.py \
			--add-adapter True \
			--adapter-lr 0.02 \
			--gender male \
			--wandb False \
			--input-strategy AudioSamples \
			--enable-spec-aug False \
			--multi-optim False \
			--world-size 8 \
			--num-epochs 10000 \
			--num-updates 2510 \
			--save-every-n 500 \
			--full-libri 1 \
			--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
			--max-duration 200 \
			--encoder-dim 768 \
			--decoder-dim 768 \
			--joiner-dim 768 \
			--use-fp16 0 \
			--accum-grads 1 \
			--encoder-type d2v \
			--additional-block True \
			--prune-range 10 \
			--spk-id $2 
		touch ./pruned_transducer_stateless_d2v_v2/$1/.train.done
	fi

#	./pruned_transducer_stateless_d2v_v2/train_adapter.py \
#		--add-adapter True \
#		--adapter-lr 0.001 \
#		--gender female
#		--wandb False \
#		--input-strategy AudioSamples \
#		--enable-spec-aug False \
#		--multi-optim False \
#		--world-size 8 \
#		--num-epochs 10 \
#		--full-libri 1 \
#		--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
#		--max-duration 200 \
#		--use-fp16 0 \
#		--encoder-type d2v \
#		--additional-block True \
#		--encoder-dim 768 \
#		--decoder-dim 768 \
#		--joiner-dim 768 \
#		--prune-range 10 
else
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
	./pruned_transducer_stateless_d2v_v2/train_adapter.py \
		--num-buckets 2 \
		--add-adapter True \
		--adapter-lr 0.02472 \
		--wandb False \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--multi-optim False \
		--world-size 4 \
		--num-epochs 31 \
		--full-libri 1 \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
		--max-duration 150 \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--use-fp16 0 \
		--accum-grads 4 \
		--encoder-type d2v \
		--additional-block True \
		--prune-range 10 \
		--ctc-loss-scale 0.1924 \
		--lm-scale 0.1254 \
		--simple-loss-scale 0.2869 \
		--spk-id $2 
fi
