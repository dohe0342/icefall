for epoch in "30" "40"; do
	for avg in "1" "10"; do
		python3 ./lm2am/decode.py \
			--exp-dir $1 \
			--use-averaged-model True \
			--epoch $epoch \
			--avg $avg \
			--max-duration 2400 \
			--num-decoder-layers 0 \
			--distill True \
			--method ctc-greedy-search \
			--quant False \
			--kernel-size 15 \
			--lm-name gpt2 \
			--wandb True
	done
done
