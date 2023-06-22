#dl_dir=/DB/LibriSpeech_tar/vox
dl_dir=/DB/LibriSpeech
subset=$1
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./tta.sh $spk_id tta
	done
done

