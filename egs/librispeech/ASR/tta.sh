spk_id=$1
dir=$2


for method in greedy_search
do
  ./tta/decode_and_adapt.py \
	--input-strategy AudioSamples \
	--enable-spec-aug False \
	--additional-block True \
	--exp-dir ./pruned_transducer_stateless_d2v_v2/$2 \
	--max-duration 600 \
	--model-name ../d2v-base-T.pt \
	--decoding-method $method \
	--max-sym-per-frame 1 \
	--encoder-type d2v \
	--encoder-dim 768 \
	--decoder-dim 768 \
	--joiner-dim 768 \
	--avg 1 \
	--use-averaged-model True \
	--spk-id $spk_id \
	--prompt False \
	--manifest-dir /DB/data/fbank \
	--bpe-model /DB/data/lang_bpe_500/bpe.model \
	--res-name test \
	--bucketing-sampler False \
	--base-lr 1e-3
	#--res-name bitfit_q_fc1_check$i
	#--res-name fullft_check$i
done

#--prompt True \
#--exp-dir ./pruned_transducer_stateless_d2v_v2/"$spk_id"_adapter_10m \
#--model-name epoch-$i.pt \
#--model-name ../d2v-base-T.pt \
#--model-name checkpoint-$i.pt \
