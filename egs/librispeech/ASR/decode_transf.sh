./pruned_transducer_transf/decode.py --input-strategy AudioSamples --enable-spec-aug False --exp-dir ./pruned_transducer_transf/230627_exp --max-duration 300 --epoch 6 --decoding-method greedy_search --max-sym-per-frame 1 --avg 1 --num-encoder-layers 18 --num-decoder-layers 6 --dim-feedforward 2048 --nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 512 --use-transf-pred True --use-averaged-model True --max-sym-per-frame 2
