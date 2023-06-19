spk_id=$1
dir=$2
#for i in 100 200 300
for i in 50
#for i in 10 20 30 40 50 60 70 80 90 100
#for i in 50 100 150 200 250 300 350 400 450 500
#for i in 50 100 #150 200 250 300 350 400 450 500
#for i in 127800 127850
do
	for method in greedy_search
	do
	  ./pruned_transducer_stateless_d2v_v2/decode_and_adapt.py \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--additional-block True \
		--exp-dir ./pruned_transducer_stateless_d2v_v2/$2 \
		--max-duration 300 \
		--iter $i \
		--model-name ../d2v-base-T.pt \
		--decoding-method $method \
		--max-sym-per-frame 1 \
		--encoder-type d2v \
		--encoder-dim 768 \
		--decoder-dim 768 \
		--joiner-dim 768 \
		--avg 1 \
		--use-averaged-model True \
		--bpe-model ../../librispeech/ASR/data/lang_bpe_500/bpe.model \
		--res-name test
	done
done
