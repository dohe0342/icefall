from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import numpy as np
import soundfile as sf

bandwidth = 24
# load a demonstration datasets
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[0]["audio"]["array"]
sf.write('original.wav', np.array(audio_sample), 24000)

# pre-process the inputs
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=bandwidth)
print(encoder_outputs.keys())
print(encoder_outputs.audio_codes.size())
print(encoder_outputs.audio_codes[0][0][0][0])

codebook = model.quantizer.decode(encoder_outputs.audio_codes[0])
print(codebook.size())
print(codebook)
audio_values = model.decode(encoder_outputs.audio_codes[:,:,:4,:], encoder_outputs.audio_scales, inputs["padding_mask"])[0]

audio_values = audio_values.detach().numpy()[0][0]
sf.write(f'encodec_{bandwidth}kb_test.wav', audio_values, 24000)
