#dl_dir=/DB/LibriSpeech_tar/vox
dl_dir=/DB/LibriSpeech
subset=$1
res_name=$2
#dl_dir=/home/work/workspace/LibriSpeech/vox_v3

for dest in $subset; do
#for dest in "test-clean" "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		./tta_suta.sh $spk_id $res_name
	done
done

