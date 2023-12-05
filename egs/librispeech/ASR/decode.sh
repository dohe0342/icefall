#dl_dir=/DB/LibriSpeech
dl_dir=/home/work/workspace/LibriSpeech
for dest in "test-other"; do
	for spk in $dl_dir/$dest/*; do
		spk_id=${spk#*$dest\/}
		echo $spk_id
		#./test.sh $spk_id prompt_tuning_10_$spk_id
		#./test.sh $spk_id prompt_tuning_100_$spk_id
		#./test.sh $spk_id encoderfreeze_$spk_id
		#./test.sh $spk_id encoderlast2_$spk_id
		#./test.sh $spk_id bitfit_"$spk_id"_q_fc1
		./test2.sh $spk_id baseline
		#./test.sh $spk_id self_init_"$spk_id"
		#./test.sh $spk_id prompt_tuning_"$spk_id"
		#./test.sh $spk_id "$spk_id"_adapter_30m
	done
done
