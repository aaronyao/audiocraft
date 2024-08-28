import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from tempfile import NamedTemporaryFile

SAMPLE_RATE = 32000

# 读取 WAV 文件
wav_path = 'get-lucky-cuts0-14.837551020408164.wav'
waveform, sr = torchaudio.load(wav_path)

# 如果采样率不匹配，需要进行重采样
if sr != SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resampler(waveform)

MODEL = MusicGen.get_pretrained('facebook/musicgen-small')
MODEL.set_generation_params(duration=20) 

# outputs = MODEL.generate(['disco progressive funk with groovy rhythm and spontaneous ad-libs'], 
#                           progress=True)
descriptions = ['disco progressive funk with groovy rhythm and spontaneous ad-libs']
generated_outputs = MODEL.generate_continuation(prompt=waveform, 
                                              prompt_sample_rate=SAMPLE_RATE,
                                              descriptions=descriptions,
                                              progress=True)

#outputs = convert_audio(outputs, SAMPLE_RATE, SAMPLE_RATE, 1)

for output in generated_outputs:
  with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
      audio_write(
          file.name, output, 32000, strategy="loudness",
          loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)

      print(f'wav: {file.name}')
