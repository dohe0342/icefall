#dl_dir=/DB/LibriSpeech_tar/vox
subset=$1
dl_dir=/DB/LibriSpeech_tar/$subset
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

for dest in "test-clean" "test-other"; do
#for dest in "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./adapter.sh "$spk_id"_adapter_30m $spk_id bear $subset
	done
done
