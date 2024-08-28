
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from tempfile import NamedTemporaryFile

sample_rate = 32000
MODEL = MusicGen.get_pretrained('facebook/musicgen-small', device='cpu')
MODEL.set_generation_params(duration=20) 

outputs = MODEL.generate(['disco progressive funk with groovy rhythm and spontaneous ad-libs'], 
                          progress=True)

#outputs = convert_audio(outputs, sample_rate, sample_rate, 1)

for output in outputs:
  with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
      audio_write(
          file.name, output, 32000, strategy="loudness",
          loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)

      print(f'wav: {file.name}')
