#dl_dir=/DB/LibriSpeech_tar/vox
dl_dir=/DB/LibriSpeech_tar/vox_v3
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

#for dest in "test-clean" "test-other"; do
for dest in "test-clean"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./run_adapter.sh "$spk_id"_adapter $spk_id $1
	done
done
